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

std::unique_ptr<RWMol> deleteAllHydrogens(const ROMol& mol) {
    // Use RDKit's built-in function to remove all hydrogens
    // This properly handles conformer coordinates
    auto result = MolOps::removeAllHs(mol);
    return std::make_unique<RWMol>(*result);
}

unsigned convertRDKitParsing(const ROMol& mol, parsing_struct& p, context& c,
                             int rootatom, const std::vector<int>& norot, bool /*addH*/) {
    // Remove all hydrogens - gnina works with heavy atoms only
    // This properly handles conformer coordinates via RDKit's removeAllHs
    auto heavyMol = deleteAllHydrogens(mol);

    // Make sure aromaticity is perceived and ring info is initialized
    try {
        MolOps::findSSSR(*heavyMol);  // Initialize ring info
        MolOps::setAromaticity(*heavyMol);
    } catch (...) {
        // Aromaticity perception failed, continue with what we have
    }

    // Calculate Gasteiger charges on heavy atoms
    calculateGasteigerCharges(*heavyMol);

    // Convert norotate indices (they should already refer to heavy atoms)
    // Since we removed hydrogens, indices may have shifted
    // For now, just filter to valid range
    std::vector<int> norotate;
    for (int i : norot) {
        if (i > 0 && (unsigned)i <= heavyMol->getNumAtoms()) {
            norotate.push_back(i);
        }
    }

    // Check for empty molecule
    if (heavyMol->getNumAtoms() == 0) {
        return 0;
    }

    // Find rigid fragments and best root (use global namespace)
    std::vector<std::vector<int>> rigid_fragments;
    unsigned best_root_atom = ::RDKitTreeBuilder::findFragments(*heavyMol, rigid_fragments, rootatom, norotate);
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
    ::RDKitTreeBuilder::constructTree(tree, rigid_fragments, root_piece, *heavyMol, true);

    // Output tree
    ::RDKitTreeBuilder::outputTree(*heavyMol, c, p, tree, torsdof);

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
    // Remove all hydrogens - gnina works with heavy atoms only
    pImpl->mol_ = deleteAllHydrogens(m);

    // Set aromaticity and ring info
    try {
        MolOps::findSSSR(*pImpl->mol_);
        MolOps::setAromaticity(*pImpl->mol_);
    } catch (...) {
        // Continue without aromaticity
    }

    // Calculate charges on heavy atoms
    calculateGasteigerCharges(*pImpl->mol_);

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
