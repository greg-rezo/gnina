/*
 * ligand_batch_manager.cpp
 *
 * Multi-ligand batch docking implementation.
 */

#include "ligand_batch_manager.h"
#include "cache_gpu.h"
#include "gpu_util.h"
#include <algorithm>
#include <unordered_map>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>
#include <queue>
#include <cmath>
#include "parsing.h"
#include "parse_pdbqt.h"
#include "GninaConverter.h"
#include "molgetter.h"

// RDKit includes for parallel 3D generation (thread-safe)
#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <Geometry/point.h>

// OpenBabel includes for SDF parsing
#include <openbabel/obconversion.h>
#include <openbabel/mol.h>

// ============================================================================
// LigandBatchGroup Methods
// ============================================================================

void LigandBatchGroup::compute_max_dimensions() {
    max_atoms = 0;
    max_torsions = 0;
    max_nodes = 0;
    max_conf_size = 0;
    max_change_size = 0;
    total_optimizers = 0;

    for (const LigandDescriptor* lig : ligands) {
        if (lig->num_atoms > (unsigned)max_atoms) max_atoms = lig->num_atoms;
        if (lig->num_torsions > (unsigned)max_torsions) max_torsions = lig->num_torsions;
        if (lig->num_nodes > (unsigned)max_nodes) max_nodes = lig->num_nodes;
        if (lig->n_conf > max_conf_size) max_conf_size = lig->n_conf;
        if (lig->n_change > max_change_size) max_change_size = lig->n_change;

        total_optimizers += exhaustiveness;
    }
}

size_t LigandBatchGroup::estimate_memory() const {
    if (ligands.empty()) return 0;

    int n_opt = total_optimizers;
    int hess_size = max_change_size * (max_change_size + 1) / 2;

    // Per-optimizer memory breakdown (matching bfgs_parallel.cu allocate_batch_memory)
    size_t per_optimizer =
        3 * max_conf_size * sizeof(float) +     // x, x_new, best_confs
        4 * max_change_size * sizeof(float) +   // g, g_new, p, y
        hess_size * sizeof(float) +             // h (Hessian)
        2 * max_atoms * 3 * sizeof(float) +     // coords, forces
        2 * max_nodes * 3 * sizeof(float) +     // node_forces, node_torques
        2 * sizeof(float);                      // energies, best_energies

    // Additional per-ligand memory (contexts, mappings)
    size_t per_ligand = sizeof(ScoringContext);

    // Optimizer-to-ligand mapping
    size_t mapping_bytes = n_opt * sizeof(int);

    return n_opt * per_optimizer + ligands.size() * per_ligand + mapping_bytes;
}

bool LigandBatchGroup::is_compatible(const LigandDescriptor& lig, float max_ratio) const {
    if (ligands.empty()) return true;

    // Check if size mismatch is too large
    // Compare atoms as primary metric
    int min_atoms = max_atoms;
    int max_atoms_new = max_atoms;

    for (const LigandDescriptor* existing : ligands) {
        if ((int)existing->num_atoms < min_atoms) min_atoms = existing->num_atoms;
    }

    if ((int)lig.num_atoms < min_atoms) min_atoms = lig.num_atoms;
    if ((int)lig.num_atoms > max_atoms_new) max_atoms_new = lig.num_atoms;

    // Ratio check: max atoms in batch should be no more than max_ratio times min
    if (min_atoms > 0 && (float)max_atoms_new / (float)min_atoms > max_ratio) {
        return false;
    }

    return true;
}

// ============================================================================
// LigandBatchManager Methods
// ============================================================================

void LigandBatchManager::extract_size_metrics(
    const model& m,
    unsigned int& num_atoms,
    unsigned int& num_torsions,
    unsigned int& num_nodes,
    int& n_conf,
    int& n_change
) {
    num_atoms = m.num_movable_atoms();

    // Get size from conf_size
    // conf_size::ligands is szv (vector of sizes), where each element is the torsion count
    conf_size sz = m.get_size();
    unsigned nlig_roots = sz.ligands.size();

    // Count torsions (each ligand entry in sz.ligands IS the torsion count)
    num_torsions = 0;
    for (size_t i = 0; i < sz.ligands.size(); i++) {
        num_torsions += sz.ligands[i];  // sz.ligands[i] is already the torsion count
    }

    // num_nodes = roots + torsions
    num_nodes = nlig_roots + num_torsions;

    // Conf and change sizes
    n_conf = 7 * nlig_roots + num_torsions;
    n_change = 6 * nlig_roots + num_torsions;
}

size_t LigandBatchManager::load_all_ligands(
    MolGetter& mols,
    const std::vector<std::string>& ligand_names,
    tee& log,
    int verbosity
) {
    all_ligands.clear();
    unsigned int lig_id = 0;

    unsigned int skipped = 0;
    for (const std::string& fname : ligand_names) {
        mols.setInputFile(fname);

        // Read all molecules from this file
        while (true) {
            auto m = std::make_unique<model>();
            try {
                if (!mols.readMoleculeIntoModel(*m)) {
                    break;
                }

                LigandDescriptor desc;
                desc.ligand_id = lig_id++;
                desc.name = m->get_name();
                desc.m = std::move(m);

                // Extract size metrics
                extract_size_metrics(*desc.m, desc.num_atoms, desc.num_torsions,
                                   desc.num_nodes, desc.n_conf, desc.n_change);

                // Skip molecules that are too large for GPU processing
                // (peptides, macrocycles, etc. cause memory issues)
                const unsigned int MAX_ATOMS = 150;
                const unsigned int MAX_TORSIONS = 30;
                if (desc.num_atoms > MAX_ATOMS || desc.num_torsions > MAX_TORSIONS) {
                    skipped++;
                    if (verbosity >= 1) {
                        log << "Warning: Skipping molecule " << lig_id << " (" << desc.name
                            << ") - too large (atoms=" << desc.num_atoms
                            << ", torsions=" << desc.num_torsions << ")\n";
                    }
                    lig_id++;
                    continue;
                }

                if (verbosity >= 2) {
                    log << "Loaded ligand " << desc.ligand_id << ": " << desc.name
                        << " (atoms=" << desc.num_atoms
                        << ", torsions=" << desc.num_torsions << ")\n";
                }

                // Log atom smina types for first ligand
                if (all_ligands.empty() && verbosity >= 1) {
                    const auto& atoms = desc.m->get_movable_atoms();
                    std::cout << "SDF path - All atom smina types: [";
                    for (size_t i = 0; i < atoms.size(); i++) {
                        if (i > 0) std::cout << ",";
                        std::cout << atoms[i].sm;
                    }
                    std::cout << "]\n";

                    // Log ALL atom charges for debugging
                    std::cout << std::fixed << std::setprecision(4);
                    std::cout << "SDF path - All atom charges: [";
                    for (size_t i = 0; i < atoms.size(); i++) {
                        if (i > 0) std::cout << ",";
                        std::cout << atoms[i].charge;
                    }
                    std::cout << "]\n";
                    std::cout << std::defaultfloat;

                    // Log number of intramolecular pairs
                    if (!desc.m->ligands.empty()) {
                        std::cout << "SDF path - Intramolecular pairs: " << desc.m->ligands[0].pairs.size() << "\n";
                        // Log first 5 pairs
                        const auto& pairs = desc.m->ligands[0].pairs;
                        std::cout << "SDF path - First 5 pairs (a,b,t1,t2): ";
                        for (size_t i = 0; i < std::min((size_t)5, pairs.size()); i++) {
                            if (i > 0) std::cout << " | ";
                            std::cout << "(" << pairs[i].a << "," << pairs[i].b << "," << pairs[i].t1 << "," << pairs[i].t2 << ")";
                        }
                        std::cout << "\n";
                    }
                }

                all_ligands.push_back(std::move(desc));
            } catch (const internal_error& e) {
                // Skip molecules with degenerate geometry (e.g., from SMILES 3D generation)
                skipped++;
                if (verbosity >= 1) {
                    log << "Warning: Skipping molecule " << lig_id << " - internal error at "
                        << e.file << ":" << e.line << "\n";
                }
                lig_id++;
            } catch (const std::exception& e) {
                // Skip molecules that fail to load for other reasons
                skipped++;
                if (verbosity >= 1) {
                    log << "Warning: Skipping molecule " << lig_id << " - " << e.what() << "\n";
                }
                lig_id++;
            }
        }
    }

    if (skipped > 0 && verbosity >= 0) {
        log << "Skipped " << skipped << " molecules due to loading errors\n";
    }

    if (verbosity >= 1) {
        log << "Loaded " << all_ligands.size() << " ligands for batch docking\n";
    }

    return all_ligands.size();
}

// Helper struct for parallel SMILES processing
struct SmilesEntry {
    std::string smiles;
    std::string name;
    unsigned int original_index;
};

// Bond length lookup table: key = (min_atomic_num, max_atomic_num, bond_order)
// Bond order: 1=single, 2=double, 3=triple, 4=aromatic
struct BondKey {
    int a1, a2, order;
    bool operator==(const BondKey& o) const { return a1 == o.a1 && a2 == o.a2 && order == o.order; }
};
struct BondKeyHash {
    size_t operator()(const BondKey& k) const { return k.a1 * 10000 + k.a2 * 100 + k.order; }
};

static const std::unordered_map<BondKey, double, BondKeyHash> BOND_LENGTHS = {
    // Single bonds (order=1): H=1, C=6, N=7, O=8, F=9, P=15, S=16, Cl=17, Br=35, I=53
    {{1, 6, 1}, 1.09},   // C-H
    {{1, 7, 1}, 1.01},   // N-H
    {{1, 8, 1}, 0.96},   // O-H
    {{1, 16, 1}, 1.34},  // S-H
    {{6, 6, 1}, 1.54},   // C-C
    {{6, 7, 1}, 1.47},   // C-N
    {{6, 8, 1}, 1.43},   // C-O
    {{6, 9, 1}, 1.35},   // C-F
    {{6, 15, 1}, 1.84},  // C-P
    {{6, 16, 1}, 1.82},  // C-S
    {{6, 17, 1}, 1.77},  // C-Cl
    {{6, 35, 1}, 1.94},  // C-Br
    {{6, 53, 1}, 2.14},  // C-I
    {{7, 7, 1}, 1.45},   // N-N
    {{7, 8, 1}, 1.40},   // N-O
    {{8, 8, 1}, 1.48},   // O-O
    {{8, 15, 1}, 1.63},  // P-O
    {{8, 16, 1}, 1.58},  // S-O
    {{16, 16, 1}, 2.05}, // S-S
    // Double bonds (order=2)
    {{6, 6, 2}, 1.34},   // C=C
    {{6, 7, 2}, 1.29},   // C=N
    {{6, 8, 2}, 1.23},   // C=O
    {{6, 16, 2}, 1.60},  // C=S
    {{7, 7, 2}, 1.25},   // N=N
    {{7, 8, 2}, 1.21},   // N=O
    {{8, 15, 2}, 1.48},  // P=O
    {{8, 16, 2}, 1.43},  // S=O
    // Triple bonds (order=3)
    {{6, 6, 3}, 1.20},   // C≡C
    {{6, 7, 3}, 1.16},   // C≡N
    {{7, 7, 3}, 1.10},   // N≡N
    // Aromatic bonds (order=4)
    {{6, 6, 4}, 1.40},   // C:C
    {{6, 7, 4}, 1.34},   // C:N
    {{6, 8, 4}, 1.36},   // C:O (furan, etc)
    {{6, 16, 4}, 1.74},  // C:S (thiophene)
    {{7, 7, 4}, 1.35},   // N:N
    {{7, 8, 4}, 1.30},   // N:O
};

static const double DEFAULT_BOND_LENGTHS[] = {0.0, 1.50, 1.34, 1.20, 1.40}; // indexed by order

static double get_bond_length(int atomic_num1, int atomic_num2, RDKit::Bond::BondType bond_type) {
    // Convert bond type to order (1=single, 2=double, 3=triple, 4=aromatic)
    int order;
    switch (bond_type) {
        case RDKit::Bond::SINGLE: order = 1; break;
        case RDKit::Bond::DOUBLE: order = 2; break;
        case RDKit::Bond::TRIPLE: order = 3; break;
        case RDKit::Bond::AROMATIC: order = 4; break;
        default: order = 1; break;
    }

    // Normalize key so a1 <= a2 (handles both permutations)
    int a1 = std::min(atomic_num1, atomic_num2);
    int a2 = std::max(atomic_num1, atomic_num2);

    auto it = BOND_LENGTHS.find({a1, a2, order});
    if (it != BOND_LENGTHS.end()) {
        return it->second;
    }

    // Fallback to default for this bond order
    std::cerr << "Warning: Unknown bond " << a1 << "-" << a2 << " order=" << order
              << ", using default " << DEFAULT_BOND_LENGTHS[order] << " A\n";
    return DEFAULT_BOND_LENGTHS[order];
}

// Get ideal bond angle based on hybridization
static double get_bond_angle(RDKit::Atom::HybridizationType hyb) {
    switch (hyb) {
        case RDKit::Atom::SP3: return 109.5 * M_PI / 180.0;
        case RDKit::Atom::SP2: return 120.0 * M_PI / 180.0;
        case RDKit::Atom::SP:  return 180.0 * M_PI / 180.0;
        default: return 109.5 * M_PI / 180.0;
    }
}

// Fast template-based 3D coordinate generation (thread-safe)
// Builds coordinates outward from first atom using bond lengths and angles
// Much faster than distance geometry, sufficient for docking since poses get randomized
static std::unique_ptr<RDKit::RWMol> generate_3d_fast(
    const std::string& smiles, const std::string& name) {

    // Parse SMILES
    std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smiles));
    if (!mol || mol->getNumAtoms() == 0) {
        return nullptr;
    }

    // Add hydrogens
    RDKit::MolOps::addHs(*mol);

    unsigned int nAtoms = mol->getNumAtoms();

    // Create conformer
    auto *conf = new RDKit::Conformer(nAtoms);
    conf->set3D(true);

    // BFS to assign coordinates
    std::vector<bool> visited(nAtoms, false);
    std::vector<int> parent(nAtoms, -1);
    std::queue<unsigned int> queue;

    // Start from atom 0 at origin
    conf->setAtomPos(0, RDGeom::Point3D(0, 0, 0));
    visited[0] = true;
    queue.push(0);

    // Track placement direction for each atom
    std::vector<RDGeom::Point3D> directions(nAtoms);
    directions[0] = RDGeom::Point3D(1, 0, 0);

    while (!queue.empty()) {
        unsigned int curr = queue.front();
        queue.pop();

        RDGeom::Point3D currPos = conf->getAtomPos(curr);
        RDKit::Atom* atom = mol->getAtomWithIdx(curr);
        RDKit::Atom::HybridizationType hyb = atom->getHybridization();
        double angle = get_bond_angle(hyb);

        // Get direction we came from (for angle placement)
        RDGeom::Point3D inDir = directions[curr];

        // Count unvisited neighbors
        std::vector<unsigned int> unvisitedNbrs;
        for (const auto& nbr : mol->atomNeighbors(atom)) {
            unsigned int nbrIdx = nbr->getIdx();
            if (!visited[nbrIdx]) {
                unvisitedNbrs.push_back(nbrIdx);
            }
        }

        // Place each unvisited neighbor
        int nbrCount = 0;
        for (unsigned int nbrIdx : unvisitedNbrs) {
            RDKit::Atom* nbrAtom = mol->getAtomWithIdx(nbrIdx);
            RDKit::Bond* bond = mol->getBondBetweenAtoms(curr, nbrIdx);

            double bondLen = get_bond_length(
                atom->getAtomicNum(),
                nbrAtom->getAtomicNum(),
                bond->getBondType()
            );

            // Calculate direction for this neighbor
            // Rotate around incoming direction based on neighbor index
            double theta = angle;  // Angle from incoming direction
            double phi = nbrCount * (2.0 * M_PI / std::max((int)unvisitedNbrs.size(), 1));  // Rotation around axis

            // Create orthogonal basis
            RDGeom::Point3D up(0, 0, 1);
            if (std::abs(inDir.z) > 0.9) up = RDGeom::Point3D(1, 0, 0);

            RDGeom::Point3D right = inDir.crossProduct(up);
            right.normalize();
            up = right.crossProduct(inDir);
            up.normalize();

            // Calculate new direction
            double sinTheta = sin(theta);
            double cosTheta = cos(theta);
            RDGeom::Point3D outDir =
                inDir * (-cosTheta) +
                right * (sinTheta * cos(phi)) +
                up * (sinTheta * sin(phi));
            outDir.normalize();

            // Place atom
            RDGeom::Point3D newPos = currPos + outDir * bondLen;
            conf->setAtomPos(nbrIdx, newPos);
            directions[nbrIdx] = outDir;

            visited[nbrIdx] = true;
            parent[nbrIdx] = curr;
            queue.push(nbrIdx);
            nbrCount++;
        }
    }

    mol->addConformer(conf, true);
    mol->setProp("_Name", name);

    return mol;
}

// Process a single SMILES to RDKit RWMol with 3D coords (thread-safe)
// Uses RDKit EmbedMolecule with ETKDGv3 - no force field optimization
// num_conformers: if > 1, generates multiple conformers with different ring puckerings
static std::unique_ptr<RDKit::RWMol> generate_3d_from_smiles_rdkit(
    const std::string& smiles, const std::string& name, bool fast_embed = false,
    int num_conformers = 1) {

    // Use fast template-based generation if requested (doesn't support multi-conformer)
    if (fast_embed) {
        return generate_3d_fast(smiles, name);
    }

    // Parse SMILES
    std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smiles));
    if (!mol || mol->getNumAtoms() == 0) {
        return nullptr;
    }

    // Add hydrogens
    RDKit::MolOps::addHs(*mol);

    // Generate 3D coordinates with ETKDGv3 (fast, good quality)
    RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
    params.randomSeed = 42;  // Fixed seed for reproducibility
    params.useSmallRingTorsions = true;  // Important for ring puckering diversity

    if (num_conformers > 1) {
        // Generate multiple conformers in same mol
        // RDKit will generate diverse conformers with different ring puckerings
        std::vector<int> cids;
        RDKit::DGeomHelpers::EmbedMultipleConfs(*mol, cids, num_conformers, params);

        // If we didn't get any conformers, try fallback
        if (cids.empty()) {
            RDKit::DGeomHelpers::EmbedParameters fallback;
            fallback.useRandomCoords = true;
            fallback.randomSeed = 42;
            int result = RDKit::DGeomHelpers::EmbedMolecule(*mol, fallback);
            if (result == -1) {
                return nullptr;
            }
        }
    } else {
        int result = RDKit::DGeomHelpers::EmbedMolecule(*mol, params);

        // Fallback to random coordinates if embedding fails
        if (result == -1) {
            RDKit::DGeomHelpers::EmbedParameters fallback;
            fallback.useRandomCoords = true;
            fallback.randomSeed = 42;
            result = RDKit::DGeomHelpers::EmbedMolecule(*mol, fallback);
        }

        if (result == -1) {
            return nullptr;
        }
    }

    // Set molecule name
    mol->setProp("_Name", name);

    // Keep hydrogens from RDKit embedding - don't remove them
    // This makes the SDF we write identical to loading an SDF file with Hs
    // (Previously we removed Hs and let OpenBabel re-add them, which changes atom ordering)

    return mol;
}

size_t LigandBatchManager::load_smiles_parallel(
    MolGetter& mols,
    const std::vector<std::string>& ligand_names,
    tee& log,
    int num_threads,
    int verbosity
) {
    // Get receptor model for grid_atoms (critical for correct scoring!)
    const model& receptor_model = mols.getInitModel();
    all_ligands.clear();

    // Step 1: Read all SMILES entries (serial, fast)
    std::vector<SmilesEntry> entries;
    unsigned int idx = 0;

    for (const std::string& fname : ligand_names) {
        // Check if file is SMILES format
        std::string ext = fname.substr(fname.find_last_of(".") + 1);
        if (ext != "smi" && ext != "smiles") {
            log << "Warning: load_smiles_parallel only supports .smi files, skipping: " << fname << "\n";
            continue;
        }

        std::ifstream infile(fname);
        if (!infile) {
            log << "Warning: Cannot open file: " << fname << "\n";
            continue;
        }

        std::string line;
        while (std::getline(infile, line)) {
            if (line.empty()) continue;

            std::istringstream iss(line);
            SmilesEntry entry;
            iss >> entry.smiles;

            // Name is optional (second column or after tab)
            if (iss >> entry.name) {
                // Got name
            } else {
                entry.name = "mol_" + std::to_string(idx);
            }

            entry.original_index = idx++;
            entries.push_back(std::move(entry));
        }
    }

    if (entries.empty()) {
        log << "No SMILES entries found\n";
        return 0;
    }

    if (verbosity >= 1) {
        log << "Read " << entries.size() << " SMILES entries\n";
    }

    // Step 2: Set up thread count
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    if (verbosity >= 1) {
        log << "Generating 3D coordinates with " << num_threads << " threads";
        if (fast_embed) {
            log << " (fast template-based)";
        }
        if (parallel_embed > 1) {
            log << " (" << parallel_embed << " conformers/mol)";
        }
        log << "...\n";
    }

    // Step 3: Generate 3D coordinates in parallel using RDKit (thread-safe)
    std::vector<std::unique_ptr<RDKit::RWMol>> rdkit_mols(entries.size());

    auto start_time = std::chrono::high_resolution_clock::now();

    bool use_fast = fast_embed;  // Capture for OpenMP
    int n_conformers = parallel_embed;  // Capture for OpenMP
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 10)
    for (size_t i = 0; i < entries.size(); i++) {
        const SmilesEntry& entry = entries[i];
        try {
            rdkit_mols[i] = generate_3d_from_smiles_rdkit(entry.smiles, entry.name, use_fast, n_conformers);
        } catch (...) {
            // Failed - leave as nullptr
        }
    }

    auto embed_time = std::chrono::high_resolution_clock::now();
    double embed_elapsed = std::chrono::duration<double>(embed_time - start_time).count();

    if (verbosity >= 1) {
        size_t success_count = 0;
        for (const auto& mol : rdkit_mols) {
            if (mol) success_count++;
        }
        double rate = entries.size() / embed_elapsed;
        log << (use_fast ? "Fast 3D embedding: " : "RDKit 3D embedding: ")
            << std::fixed << std::setprecision(1)
            << embed_elapsed << "s, " << rate << " mol/s, "
            << success_count << "/" << entries.size() << " succeeded\n";
    }

    // Step 4: Convert to gnina models serially (OpenBabel not thread-safe)
    // For multi-conformer mode: create separate model for each conformer
    // Each model captures the ring puckering of that conformer in its reference coords
    struct ModelInfo {
        std::unique_ptr<model> m;
        unsigned int original_index;  // Which SMILES entry
        int conformer_index;          // Which conformer (0 for single-conformer mode)
        std::string name;
    };
    std::vector<ModelInfo> all_models;
    all_models.reserve(entries.size() * parallel_embed);

    int models_created = 0;
    for (size_t i = 0; i < entries.size(); i++) {
        if (!rdkit_mols[i]) continue;

        const SmilesEntry& entry = entries[i];
        RDKit::RWMol* rdmol = rdkit_mols[i].get();
        int num_confs = rdmol->getNumConformers();

        // For multi-conformer mode: create model for each conformer
        // For single-conformer mode: just use the first (only) conformer
        int confs_to_process = (parallel_embed > 1) ? num_confs : 1;

        for (int cid = 0; cid < confs_to_process; cid++) {
            try {
                // Create a copy of the mol with only this conformer
                // This ensures we use the correct ring puckering
                RDKit::RWMol mol_copy(*rdmol);

                // Get conformer IDs and remove all except the one we want
                std::vector<unsigned int> conf_ids;
                for (auto it = mol_copy.beginConformers(); it != mol_copy.endConformers(); ++it) {
                    conf_ids.push_back((*it)->getId());
                }

                // Remove all conformers except cid
                for (unsigned int conf_id : conf_ids) {
                    if ((int)conf_id != cid) {
                        mol_copy.removeConformer(conf_id);
                    }
                }

                std::string conf_name = entry.name;
                if (parallel_embed > 1 && num_confs > 1) {
                    conf_name += "_conf" + std::to_string(cid);
                }

                // Write RDKit mol to SDF string (with hydrogens)
                std::string sdf_block = RDKit::MolToMolBlock(mol_copy);
                sdf_block += "$$$$\n";

                // DEBUG: Save first molecule's intermediate SDF for debugging
                static bool saved_intermediate = false;
                if (!saved_intermediate) {
                    std::ofstream debug_sdf("intermediate_smi_pose.sdf");
                    if (debug_sdf) {
                        debug_sdf << sdf_block;
                        std::cerr << "DEBUG: Saved intermediate SDF to intermediate_smi_pose.sdf" << std::endl;
                    }
                    saved_intermediate = true;
                }

                // Parse with OpenBabel - replicate EXACT code from molgetter.cpp OB case
                std::istringstream sdf_stream(sdf_block);
                OpenBabel::OBConversion conv;
                conv.SetInFormat("sdf");
                conv.SetInStream(&sdf_stream);

                OpenBabel::OBMol obmol;
                if (!conv.Read(&obmol)) {
                    std::cerr << "Warning: OpenBabel failed to parse SDF for " << conf_name << std::endl;
                    continue;
                }

                // Match molgetter.cpp OB case exactly
                obmol.SetTitle(conf_name.c_str());
                obmol.StripSalts();

                if (obmol.NumAtoms() == 0) {
                    std::cerr << "Warning: Empty molecule " << conf_name << std::endl;
                    continue;
                }

                // Convert using same function as molgetter
                auto m = std::make_unique<model>();
                m->set_name(conf_name);

                if (!convertOBMolToModel(obmol, *m, true, false)) {
                    std::cerr << "Warning: Failed to convert " << conf_name << " to model" << std::endl;
                    continue;
                }

                // Copy receptor's grid_atoms for proper grid-based scoring
                // This is CRITICAL - without this, the scoring grid is empty!
                m->grid_atoms = receptor_model.grid_atoms;

                ModelInfo info;
                info.m = std::move(m);
                info.original_index = entry.original_index;
                info.conformer_index = cid;
                info.name = conf_name;
                all_models.push_back(std::move(info));
                models_created++;
            } catch (const std::exception& e) {
                // Failed - skip this conformer and log reason
                std::cerr << "Warning: Skipping " << entry.name << ": " << e.what() << std::endl;
            } catch (...) {
                // Unknown error - skip this conformer
                std::cerr << "Warning: Skipping " << entry.name << ": unknown error" << std::endl;
            }
        }
    }

    auto convert_time = std::chrono::high_resolution_clock::now();
    double convert_elapsed = std::chrono::duration<double>(convert_time - embed_time).count();

    if (verbosity >= 1) {
        log << "Model conversion: " << std::fixed << std::setprecision(1)
            << convert_elapsed << "s, " << models_created << " models created";
        if (parallel_embed > 1) {
            log << " (" << parallel_embed << " conformers/mol)";
        }
        log << "\n";
    }

    // Clear RDKit mols to free memory and avoid double-free on destruction
    // (RDKit mol destruction must complete before OpenBabel-based models are destroyed)
    rdkit_mols.clear();
    rdkit_mols.shrink_to_fit();

    // Step 5: Collect successful results
    // For multi-conformer mode: each conformer is a separate ligand with parent tracking
    unsigned int skipped = 0;
    const unsigned int MAX_ATOMS = 150;
    const unsigned int MAX_TORSIONS = 30;

    for (size_t i = 0; i < all_models.size(); i++) {
        ModelInfo& info = all_models[i];
        if (!info.m) {
            skipped++;
            continue;
        }

        LigandDescriptor desc;
        desc.ligand_id = all_ligands.size();  // Sequential ID for batch processing
        desc.name = info.name;
        desc.m = std::move(info.m);

        extract_size_metrics(*desc.m, desc.num_atoms, desc.num_torsions,
                           desc.num_nodes, desc.n_conf, desc.n_change);

        // Skip molecules that are too large
        if (desc.num_atoms > MAX_ATOMS || desc.num_torsions > MAX_TORSIONS) {
            skipped++;
            if (verbosity >= 1) {
                log << "Warning: Skipping molecule " << desc.name
                    << " - too large (atoms=" << desc.num_atoms
                    << ", torsions=" << desc.num_torsions << ")\n";
            }
            continue;
        }

        // Track parent ligand for multi-conformer mode
        if (parallel_embed > 1) {
            desc.parent_ligand_id = info.original_index;
            desc.conformer_index = info.conformer_index;
        } else {
            desc.parent_ligand_id = -1;  // Not using multi-conformer
            desc.conformer_index = 0;
        }

        desc.num_embedded_conformers = 1;  // Each LigandDescriptor is now one conformer

        all_ligands.push_back(std::move(desc));
    }

    if (verbosity >= 1) {
        double total_elapsed = embed_elapsed + convert_elapsed;
        log << "Loaded " << all_ligands.size() << " ligands in "
            << std::fixed << std::setprecision(1) << total_elapsed << "s";
        if (skipped > 0) {
            log << " (skipped " << skipped << ")";
        }
        log << "\n";
    }

    return all_ligands.size();
}

void LigandBatchManager::sort_and_group_ligands(int verbosity) {
    batch_groups.clear();

    if (all_ligands.empty()) return;

    // Create sorted indices by (num_atoms, num_torsions)
    std::vector<size_t> indices(all_ligands.size());
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
        const auto& la = all_ligands[a];
        const auto& lb = all_ligands[b];
        if (la.num_atoms != lb.num_atoms) {
            return la.num_atoms < lb.num_atoms;
        }
        return la.num_torsions < lb.num_torsions;
    });

    // Group into batches
    LigandBatchGroup current_batch;
    current_batch.exhaustiveness = exhaustiveness;

    for (size_t idx : indices) {
        LigandDescriptor* lig = &all_ligands[idx];

        // Check if adding this ligand would exceed limits
        int new_total = current_batch.total_optimizers + exhaustiveness;
        bool would_exceed_poses = (new_total > target_total_poses && !current_batch.ligands.empty());

        // Estimate memory for potential new batch
        LigandBatchGroup test_batch = current_batch;
        test_batch.ligands.push_back(lig);
        test_batch.compute_max_dimensions();
        bool would_exceed_memory = (max_gpu_memory > 0 && test_batch.estimate_memory() > max_gpu_memory);

        // Check size compatibility
        bool size_mismatch = !current_batch.is_compatible(*lig, 2.0f);

        if ((would_exceed_poses || would_exceed_memory || size_mismatch) && !current_batch.ligands.empty()) {
            // Finalize current batch
            current_batch.compute_max_dimensions();
            batch_groups.push_back(std::move(current_batch));

            // Start new batch
            current_batch = LigandBatchGroup();
            current_batch.exhaustiveness = exhaustiveness;
        }

        current_batch.ligands.push_back(lig);
        current_batch.total_optimizers += exhaustiveness;
    }

    // Finalize last batch
    if (!current_batch.ligands.empty()) {
        current_batch.compute_max_dimensions();
        batch_groups.push_back(std::move(current_batch));
    }

    if (verbosity >= 1) {
        std::cerr << "Created " << batch_groups.size() << " batch groups:\n";
        for (size_t i = 0; i < batch_groups.size(); i++) {
            const auto& batch = batch_groups[i];
            std::cerr << "  Batch " << i << ": "
                      << batch.ligands.size() << " ligands, "
                      << batch.total_optimizers << " poses, "
                      << "max_atoms=" << batch.max_atoms
                      << ", max_torsions=" << batch.max_torsions
                      << ", est_mem=" << std::fixed << std::setprecision(1)
                      << (float)batch.estimate_memory() / (1024 * 1024) << " MB\n";
        }
    }
}

void LigandBatchManager::process_batch(
    LigandBatchGroup& group,
    cache_gpu& cgpu,
    const gfloat3& box_min,
    const gfloat3& box_max,
    int max_iterations,
    unsigned int seed,
    int verbosity
) {
    if (group.ligands.empty()) return;

    const GPUCacheInfo& cacheInfo = cgpu.get_info();

    // Timing
    cudaEvent_t start, end_setup, end_bfgs, end_collect;
    CUDA_CHECK_GNINA(cudaEventCreate(&start));
    CUDA_CHECK_GNINA(cudaEventCreate(&end_setup));
    CUDA_CHECK_GNINA(cudaEventCreate(&end_bfgs));
    CUDA_CHECK_GNINA(cudaEventCreate(&end_collect));
    CUDA_CHECK_GNINA(cudaEventRecord(start));

    // Initialize GPU for each ligand and create scoring contexts
    std::vector<ScoringContext> host_contexts(group.ligands.size());

    for (size_t i = 0; i < group.ligands.size(); i++) {
        LigandDescriptor* lig = group.ligands[i];

        // Initialize GPU data for this ligand
        if (!lig->gpu_initialized) {
            lig->m->initialize_gpu();
            lig->gpu_initialized = true;
        }

        // Create scoring context
        create_scoring_context(host_contexts[i], lig->m->gdata, cacheInfo);
    }

    // Upload contexts to GPU
    ScoringContext* d_contexts;
    CUDA_CHECK_GNINA(cudaMalloc(&d_contexts, group.ligands.size() * sizeof(ScoringContext)));
    CUDA_CHECK_GNINA(cudaMemcpy(d_contexts, host_contexts.data(),
                                group.ligands.size() * sizeof(ScoringContext),
                                cudaMemcpyHostToDevice));

    // Build optimizer-to-ligand mapping
    std::vector<int> optimizer_to_ligand(group.total_optimizers);
    int offset = 0;
    for (size_t i = 0; i < group.ligands.size(); i++) {
        for (int j = 0; j < group.exhaustiveness; j++) {
            optimizer_to_ligand[offset++] = i;
        }
    }

    int* d_optimizer_to_ligand;
    CUDA_CHECK_GNINA(cudaMalloc(&d_optimizer_to_ligand, group.total_optimizers * sizeof(int)));
    CUDA_CHECK_GNINA(cudaMemcpy(d_optimizer_to_ligand, optimizer_to_ligand.data(),
                                group.total_optimizers * sizeof(int), cudaMemcpyHostToDevice));

    // Set up batch
    LigandBatch batch;
    batch.ligand_contexts = d_contexts;
    batch.optimizer_to_ligand = d_optimizer_to_ligand;
    batch.num_ligands = group.ligands.size();
    batch.total_optimizers = group.total_optimizers;

    // Allocate batch memory
    BFGSBatchMemory mem;
    allocate_batch_memory(mem, group.total_optimizers,
                         group.max_conf_size, group.max_change_size,
                         group.max_atoms, group.max_nodes);

    CUDA_CHECK_GNINA(cudaEventRecord(end_setup));

    // Log box and initial ligand centroid for debugging
    if (verbosity >= 1) {
        std::cout << "Docking box: min=(" << box_min.x << "," << box_min.y << "," << box_min.z
                  << ") max=(" << box_max.x << "," << box_max.y << "," << box_max.z << ")\n";

        // Compute and log center of mass for first ligand's reference coordinates
        if (!group.ligands.empty()) {
            LigandDescriptor* lig = group.ligands[0];
            const atomv& atoms = lig->m->atoms;
            double cx = 0, cy = 0, cz = 0;
            int n_heavy = 0;
            for (size_t i = 0; i < atoms.size(); i++) {
                if (!atoms[i].is_hydrogen()) {
                    // atoms[i].coords are the LOCAL/RELATIVE coordinates stored in the model
                    cx += atoms[i].coords[0];
                    cy += atoms[i].coords[1];
                    cz += atoms[i].coords[2];
                    n_heavy++;
                }
            }
            if (n_heavy > 0) {
                cx /= n_heavy;
                cy /= n_heavy;
                cz /= n_heavy;
            }
            std::cout << "First ligand '" << lig->name << "' stored (local) coords centroid: ("
                      << cx << "," << cy << "," << cz << ") [" << n_heavy << " heavy atoms]\n";

            // Log first 5 atom local coords for debugging
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "First 5 atom local coords (heavy atoms only):\n";
            int count = 0;
            for (size_t i = 0; i < atoms.size() && count < 5; i++) {
                if (!atoms[i].is_hydrogen()) {
                    std::cout << "  atom " << i << " (sm=" << atoms[i].sm
                              << " Z=" << smina_atom_type::data[atoms[i].sm].anum
                              << "): (" << atoms[i].coords[0] << ", " << atoms[i].coords[1]
                              << ", " << atoms[i].coords[2] << ")\n";
                    count++;
                }
            }
            std::cout << std::defaultfloat;

            // Log ALL atom smina types for debugging
            std::cout << "All atom smina types: [";
            for (size_t i = 0; i < atoms.size(); i++) {
                if (i > 0) std::cout << ",";
                std::cout << atoms[i].sm;
            }
            std::cout << "]\n";

            // Log ALL atom charges for debugging
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "All atom charges: [";
            for (size_t i = 0; i < atoms.size(); i++) {
                if (i > 0) std::cout << ",";
                std::cout << atoms[i].charge;
            }
            std::cout << "]\n";
            std::cout << std::defaultfloat;

            // Log number of intramolecular pairs
            if (!lig->m->ligands.empty()) {
                std::cout << "Intramolecular pairs: " << lig->m->ligands[0].pairs.size() << "\n";
                // Log first 5 pairs
                const auto& pairs = lig->m->ligands[0].pairs;
                std::cout << "First 5 pairs (a,b,t1,t2): ";
                for (size_t i = 0; i < std::min((size_t)5, pairs.size()); i++) {
                    if (i > 0) std::cout << " | ";
                    std::cout << "(" << pairs[i].a << "," << pairs[i].b << "," << pairs[i].t1 << "," << pairs[i].t2 << ")";
                }
                std::cout << "\n";
            }

            // Also log the ligand root origin (absolute position)
            if (!lig->m->ligands.empty()) {
                const vec& origin = lig->m->ligands[0].node.get_origin();
                std::cout << "First ligand root origin (absolute): ("
                          << origin[0] << "," << origin[1] << "," << origin[2] << ")\n";

                // Log segment structure: which atoms belong to which segment
                std::cout << "Segment structure (node -> atom range, atomic nums):\n";

                // Helper to print atomic numbers for atom range
                auto print_segment_atoms = [&atoms](sz begin, sz end) {
                    std::cout << " [";
                    for (sz i = begin; i < end && i < atoms.size(); i++) {
                        if (i > begin) std::cout << ",";
                        // Get atomic number from smina type
                        std::cout << smina_atom_type::data[atoms[i].sm].anum;
                    }
                    std::cout << "]";
                };

                // Root segment
                std::cout << "  Root: atoms [" << lig->m->ligands[0].node.begin
                          << ", " << lig->m->ligands[0].node.end << ")";
                print_segment_atoms(lig->m->ligands[0].node.begin, lig->m->ligands[0].node.end);
                std::cout << "\n";

                // Count torsion segments by traversing children
                std::function<void(const branch&, int)> print_branch = [&](const branch& b, int depth) {
                    std::cout << "  Torsion " << depth << ": atoms [" << b.node.begin
                              << ", " << b.node.end << ")";
                    print_segment_atoms(b.node.begin, b.node.end);
                    std::cout << "\n";
                    for (const auto& child : b.children) {
                        print_branch(child, depth + 1);
                    }
                };
                int torsion_idx = 0;
                for (const auto& child : lig->m->ligands[0].children) {
                    print_branch(child, torsion_idx++);
                }
            }
        }
    }

    // Launch BFGS
    launch_parallel_bfgs(batch, mem, max_iterations, box_min, box_max, seed, verbosity);
    CUDA_CHECK_GNINA(cudaDeviceSynchronize());
    CUDA_CHECK_GNINA(cudaEventRecord(end_bfgs));

    // Collect results
    std::vector<float> all_energies;
    std::vector<std::vector<float>> all_conformations;
    collect_bfgs_results(mem, group.total_optimizers, all_energies, all_conformations);

    // Distribute results to ligands
    offset = 0;
    for (size_t i = 0; i < group.ligands.size(); i++) {
        LigandDescriptor* lig = group.ligands[i];
        lig->energies.resize(group.exhaustiveness);
        lig->conformations.resize(group.exhaustiveness);

        for (int j = 0; j < group.exhaustiveness; j++) {
            lig->energies[j] = all_energies[offset];
            lig->conformations[j] = all_conformations[offset];
            offset++;
        }
    }

    CUDA_CHECK_GNINA(cudaEventRecord(end_collect));
    CUDA_CHECK_GNINA(cudaEventSynchronize(end_collect));

    // Print timing
    float setup_ms, bfgs_ms, collect_ms;
    CUDA_CHECK_GNINA(cudaEventElapsedTime(&setup_ms, start, end_setup));
    CUDA_CHECK_GNINA(cudaEventElapsedTime(&bfgs_ms, end_setup, end_bfgs));
    CUDA_CHECK_GNINA(cudaEventElapsedTime(&collect_ms, end_bfgs, end_collect));
    float total_ms = setup_ms + bfgs_ms + collect_ms;

    if (verbosity >= 1) {
        std::cerr << "Batch processing timing:\n";
        std::cerr << "  Setup:           " << std::fixed << std::setprecision(2)
                  << setup_ms << " ms (" << std::setprecision(1) << 100.0f * setup_ms / total_ms << "%)\n";
        std::cerr << "  BFGS kernel:     " << std::setprecision(2) << bfgs_ms << " ms ("
                  << std::setprecision(1) << 100.0f * bfgs_ms / total_ms << "%)\n";
        std::cerr << "  Result collect:  " << std::setprecision(2) << collect_ms << " ms ("
                  << std::setprecision(1) << 100.0f * collect_ms / total_ms << "%)\n";
        std::cerr << "  Total:           " << std::setprecision(2) << total_ms << " ms\n";
        std::cerr << "  Throughput:      " << std::setprecision(1)
                  << 1000.0f * group.total_optimizers / total_ms << " poses/sec, "
                  << 1000.0f * group.ligands.size() / total_ms << " ligands/sec\n";
    }

    // Cleanup timing events
    cudaEventDestroy(start);
    cudaEventDestroy(end_setup);
    cudaEventDestroy(end_bfgs);
    cudaEventDestroy(end_collect);

    // Cleanup
    free_batch_memory(mem);
    cudaFree(d_contexts);
    cudaFree(d_optimizer_to_ligand);
}

size_t LigandBatchManager::get_available_gpu_memory() {
    size_t free_mem, total_mem;
    CUDA_CHECK_GNINA(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

void LigandBatchManager::clear_gpu_memory() {
    // Free GPU state for all ligands
    for (auto& lig : all_ligands) {
        if (lig.gpu_initialized && lig.m) {
            lig.m->deallocate_gpu();
            lig.gpu_initialized = false;
        }
    }

    // Synchronize and compact GPU memory
    cudaDeviceSynchronize();
}
