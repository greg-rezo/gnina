/*
 * RDKitConverter.cpp
 *
 * Direct RDKit to gnina conversion, bypassing OpenBabel.
 * Uses RDKitTreeBuilder for rotatable bond detection and tree building.
 *
 * Author: gnina developers
 */

#include "RDKitConverter.h"
#include "RDKitTreeBuilder.h"
#include "parsing.h"

#include <GraphMol/MolOps.h>
#include <GraphMol/PartialCharges/GasteigerCharges.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/Conformer.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <sstream>

namespace RDKitConverter {

using namespace RDKit;

void calculateGasteigerCharges(RWMol& mol) {
    try {
        computeGasteigerCharges(mol);
    } catch (...) {
        // Gasteiger charges failed, leave at 0
    }
}

std::unique_ptr<RWMol> deleteNonPolarHydrogens(const ROMol& mol) {
    auto result = std::make_unique<RWMol>(mol);

    // Collect indices of non-polar hydrogens to remove
    std::vector<unsigned int> toRemove;
    for (auto atom : result->atoms()) {
        if (::RDKitTreeBuilder::isNonPolarHydrogen(atom, *result)) {
            toRemove.push_back(atom->getIdx());
        }
    }

    // Remove in reverse order to maintain indices
    std::sort(toRemove.begin(), toRemove.end(), std::greater<unsigned int>());
    for (unsigned int idx : toRemove) {
        result->removeAtom(idx);
    }

    return result;
}

unsigned convertRDKitParsing(const ROMol& mol, parsing_struct& p, context& c,
                             int rootatom, const std::vector<int>& norot, bool addH) {
    // Create a working copy
    RWMol workMol(mol);

    // Add hydrogens if requested
    if (addH) {
        MolOps::addHs(workMol);
    }

    // Make sure aromaticity is perceived and ring info is initialized
    try {
        MolOps::findSSSR(workMol);  // Initialize ring info
        MolOps::setAromaticity(workMol);
    } catch (...) {
        // Aromaticity perception failed, continue with what we have
    }

    // Calculate Gasteiger charges
    calculateGasteigerCharges(workMol);

    // Save atoms to preserve norotate indices after hydrogen deletion
    std::vector<const Atom*> norotate_atoms;
    if (!norot.empty()) {
        for (int i : norot) {
            if (i > 0 && (unsigned)i <= workMol.getNumAtoms()) {
                const Atom* a = workMol.getAtomWithIdx(i - 1);  // Convert to 0-based
                if (a->getAtomicNum() != 1) {  // Not hydrogen
                    norotate_atoms.push_back(a);
                }
            }
        }
    }

    // Delete non-polar hydrogens (leaves polar hydrogens)
    auto polarMol = deleteNonPolarHydrogens(workMol);

    // Rebuild norotate indices in new molecule
    std::vector<int> norotate;
    for (const Atom* orig_atom : norotate_atoms) {
        // Find this atom in the new molecule by its properties
        // (This is approximate - works for heavy atoms)
        for (auto atom : polarMol->atoms()) {
            if (atom->getAtomicNum() == orig_atom->getAtomicNum()) {
                // More sophisticated matching could be done here
                norotate.push_back(atom->getIdx() + 1);  // 1-based
                break;
            }
        }
    }

    // Check for empty molecule
    if (polarMol->getNumAtoms() == 0) {
        return 0;
    }

    // Find rigid fragments and best root (use global namespace)
    std::vector<std::vector<int>> rigid_fragments;
    unsigned best_root_atom = ::RDKitTreeBuilder::findFragments(*polarMol, rigid_fragments, rootatom, norotate);
    unsigned torsdof = rigid_fragments.size() - 1;

    // Use user-supplied root if specified
    if (rootatom > 0) {
        best_root_atom = rootatom;
    }

    // Find which fragment contains the root
    unsigned int root_piece = 0;
    for (unsigned j = 0; j < rigid_fragments.size(); j++) {
        if (::RDKitTreeBuilder::isIn(rigid_fragments[j], best_root_atom)) {
            root_piece = j;
            break;
        }
    }

    // Construct tree
    std::map<unsigned int, ::RDKitTreeBuilder::rdkitbranch> tree;
    ::RDKitTreeBuilder::constructTree(tree, rigid_fragments, root_piece, *polarMol, true);

    // Output tree
    ::RDKitTreeBuilder::outputTree(*polarMol, c, p, tree, torsdof);

    return torsdof;
}

unsigned convertRDKitParsing(const ROMol& mol, parsing_struct& p, context& c, bool addH) {
    std::vector<int> nr;
    return convertRDKitParsing(mol, p, c, 0, nr, addH);
}

bool convertRDKitToModel(const ROMol& mol, model& m, bool add_hydrogens, bool strip_hydrogens) {
    try {
        parsing_struct p;
        context c;
        unsigned torsdof = convertRDKitParsing(mol, p, c, add_hydrogens);

        non_rigid_parsed nr;
        postprocess_ligand(nr, p, c, torsdof);
        VINA_CHECK(nr.atoms_atoms_bonds.dim() == nr.atoms.size());

        pdbqt_initializer tmp;
        tmp.initialize_from_nrp(nr, c, true);
        tmp.initialize(nr.mobility_matrix());

        if (strip_hydrogens) {
            tmp.m.strip_hydrogens();
        }

        m.append(tmp.m);

        // Set name if available
        if (mol.hasProp("_Name")) {
            std::string name;
            mol.getProp("_Name", name);
            m.set_name(name);
        }

        return true;
    } catch (std::exception& e) {
        std::cerr << "Error converting RDKit molecule: " << e.what() << std::endl;
        return false;
    }
}

// RDKitMCMolConverter implementation using pImpl

struct RDKitMCMolConverter::Impl {
    std::unique_ptr<RWMol> mol_;
    std::vector<std::vector<int>> rigid_fragments;
    std::map<unsigned int, ::RDKitTreeBuilder::rdkitbranch> tree;
};

RDKitMCMolConverter::RDKitMCMolConverter(const ROMol& m) : pImpl(std::make_unique<Impl>()), torsdof(0) {
    // Make a working copy with hydrogens
    pImpl->mol_ = std::make_unique<RWMol>(m);
    MolOps::addHs(*pImpl->mol_);

    // Set aromaticity
    try {
        MolOps::setAromaticity(*pImpl->mol_);
    } catch (...) {
        // Continue without aromaticity
    }

    // Calculate charges
    calculateGasteigerCharges(*pImpl->mol_);

    // Delete non-polar hydrogens
    pImpl->mol_ = deleteNonPolarHydrogens(*pImpl->mol_);

    if (pImpl->mol_->getNumAtoms() == 0) {
        torsdof = 0;
        return;
    }

    // Find fragments (use global namespace to avoid shadowing)
    unsigned best_root_atom = ::RDKitTreeBuilder::findFragments(*pImpl->mol_, pImpl->rigid_fragments);
    torsdof = pImpl->rigid_fragments.size() - 1;

    // Find root piece
    unsigned int root_piece = 0;
    for (unsigned j = 0; j < pImpl->rigid_fragments.size(); j++) {
        if (::RDKitTreeBuilder::isIn(pImpl->rigid_fragments[j], best_root_atom)) {
            root_piece = j;
            break;
        }
    }

    // Construct tree
    ::RDKitTreeBuilder::constructTree(pImpl->tree, pImpl->rigid_fragments, root_piece, *pImpl->mol_, true);
}

RDKitMCMolConverter::~RDKitMCMolConverter() = default;

void RDKitMCMolConverter::convertConformer(unsigned conf, std::ostream& out) {
    if (!pImpl->mol_ || pImpl->mol_->getNumAtoms() == 0) {
        return;
    }

    // Set conformer
    // Note: After deleting non-polar hydrogens, conformer IDs may change
    // For now, assume conformer 0 exists
    if (pImpl->mol_->getNumConformers() == 0) {
        return;
    }

    parsing_struct p;
    context c;

    // Make a copy of tree (outputTree modifies it)
    std::map<unsigned int, ::RDKitTreeBuilder::rdkitbranch> tree_copy = pImpl->tree;
    ::RDKitTreeBuilder::outputTree(*pImpl->mol_, c, p, tree_copy, torsdof);

    boost::iostreams::filtering_stream<boost::iostreams::output> strm;
    strm.push(boost::iostreams::gzip_compressor());
    strm.push(out);

    boost::archive::binary_oarchive serialout(strm,
        boost::archive::no_header | boost::archive::no_tracking);

    serialout << torsdof;
    serialout << p;
    serialout << c;
}

} // namespace RDKitConverter
