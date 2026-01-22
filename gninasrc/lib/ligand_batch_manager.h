/*
 * ligand_batch_manager.h
 *
 * Multi-ligand batch docking support for GPU kernel.
 * Groups similar-sized ligands together for efficient GPU batch processing.
 */

#ifndef LIGAND_BATCH_MANAGER_H
#define LIGAND_BATCH_MANAGER_H

#include <vector>
#include <string>
#include <memory>
#include "model.h"
#include "molgetter.h"
#include "tee.h"
#include "bfgs_parallel.h"

// Forward declarations
struct GPUCacheInfo;
class cache_gpu;
class weighted_terms;
class precalculate;

// Descriptor for a single ligand in batch docking
struct LigandDescriptor {
    unsigned int ligand_id;        // Original input order (for output ordering)
    std::string name;              // Ligand name from file
    std::unique_ptr<model> m;      // The model object (owned)
    grid_dims gd;                  // Grid dimensions for this ligand

    // Size metrics for grouping
    unsigned int num_atoms;
    unsigned int num_torsions;
    unsigned int num_nodes;
    int n_conf;   // 7 * nlig_roots + n_torsions
    int n_change; // 6 * nlig_roots + n_torsions

    // GPU state
    bool gpu_initialized;

    // Multiple embedded conformers (for --parallel_embed)
    // Each conformer stores full atom coordinates [num_atoms * 3]
    // Different conformers have different ring puckerings
    std::vector<std::vector<float>> embedded_coords;
    int num_embedded_conformers = 1;

    // Parent ligand tracking (for multi-conformer mode)
    // When parallel_embed > 1, each conformer becomes a separate LigandDescriptor
    // with the same parent_ligand_id, allowing results to be grouped together
    int parent_ligand_id = -1;  // -1 means this is the original (or only) conformer
    int conformer_index = 0;    // Which conformer this is (0-indexed)

    // Results per pose
    std::vector<float> energies;
    std::vector<std::vector<float>> conformations;

    LigandDescriptor()
        : ligand_id(0), num_atoms(0), num_torsions(0), num_nodes(0),
          n_conf(0), n_change(0), gpu_initialized(false), num_embedded_conformers(1),
          parent_ligand_id(-1), conformer_index(0) {}

    // Move constructor (models are non-copyable)
    LigandDescriptor(LigandDescriptor&& other) = default;
    LigandDescriptor& operator=(LigandDescriptor&& other) = default;

    // Prevent copying
    LigandDescriptor(const LigandDescriptor&) = delete;
    LigandDescriptor& operator=(const LigandDescriptor&) = delete;
};

// A group of ligands to be processed in a single GPU batch
struct LigandBatchGroup {
    std::vector<LigandDescriptor*> ligands;  // Pointers to ligands in this group (not owned)

    // Max dimensions across all ligands in batch (for memory allocation)
    int max_atoms;
    int max_torsions;
    int max_nodes;
    int max_conf_size;
    int max_change_size;

    // Batch statistics
    int total_optimizers;  // Sum of exhaustiveness across all ligands
    int exhaustiveness;    // Poses per ligand (uniform)

    LigandBatchGroup()
        : max_atoms(0), max_torsions(0), max_nodes(0),
          max_conf_size(0), max_change_size(0),
          total_optimizers(0), exhaustiveness(0) {}

    // Compute max dimensions from ligands
    void compute_max_dimensions();

    // Estimate GPU memory needed for this batch
    size_t estimate_memory() const;

    // Check if adding a ligand would cause too much size mismatch
    bool is_compatible(const LigandDescriptor& lig, float max_ratio = 2.0f) const;
};

// Manager class for multi-ligand batch docking
class LigandBatchManager {
public:
    // Configuration
    int target_total_poses;    // Target number of poses per GPU batch (~50000)
    int exhaustiveness;        // Poses per ligand (uniform across all ligands)
    size_t max_gpu_memory;     // Maximum GPU memory to use (bytes)
    bool fast_embed;           // Use fast template-based 3D generation (skips distance geometry)
    int parallel_embed;        // Number of conformers per SMILES (for --parallel_embed)
    bool skip_torsion_randomize; // Use embedded coords directly (preserve ring geometry)
    float prune_rms_thresh;    // RMSD threshold for pruning similar conformers (negative = disabled)

    // Loaded ligands
    std::vector<LigandDescriptor> all_ligands;

    // Grouped batches for GPU processing
    std::vector<LigandBatchGroup> batch_groups;

    LigandBatchManager()
        : target_total_poses(50000), exhaustiveness(1024),
          max_gpu_memory(0), fast_embed(false), parallel_embed(1),
          skip_torsion_randomize(false), prune_rms_thresh(-0.5f) {}

    // Phase 1: Load all ligands from input files into CPU memory
    // Returns number of ligands loaded
    size_t load_all_ligands(
        MolGetter& mols,
        const std::vector<std::string>& ligand_names,
        tee& log,
        int verbosity = 1
    );

    // Phase 1 (parallel): Load SMILES files with parallel 3D generation
    // Much faster than serial OpenBabel for SMILES input
    // mols provides the receptor model (with grid_atoms) for proper scoring
    size_t load_smiles_parallel(
        MolGetter& mols,
        const std::vector<std::string>& ligand_names,
        tee& log,
        int num_threads = 0,  // 0 = auto-detect
        int verbosity = 1
    );

    // Phase 2: Sort ligands by size and group into batches
    void sort_and_group_ligands(int verbosity = 1);

    // Phase 3: Process a single batch on GPU
    // Sets energies and conformations on each LigandDescriptor in the group
    void process_batch(
        LigandBatchGroup& group,
        cache_gpu& cgpu,
        const gfloat3& box_min,
        const gfloat3& box_max,
        int max_iterations,
        unsigned int seed,
        int verbosity = 1
    );

    // Get total number of ligands loaded
    size_t num_ligands() const { return all_ligands.size(); }

    // Get total number of batches created
    size_t num_batches() const { return batch_groups.size(); }

    // Utility: Query available GPU memory
    static size_t get_available_gpu_memory();

    // Free GPU memory after batch processing (before CNN scoring)
    void clear_gpu_memory();

private:
    // Helper: Extract size metrics from a model
    static void extract_size_metrics(
        const model& m,
        unsigned int& num_atoms,
        unsigned int& num_torsions,
        unsigned int& num_nodes,
        int& n_conf,
        int& n_change
    );
};

#endif // LIGAND_BATCH_MANAGER_H
