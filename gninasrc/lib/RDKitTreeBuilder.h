/*
 * RDKitTreeBuilder.h
 *
 * RDKit-based rotatable bond detection and torsion tree building.
 * Replaces OpenBabel-dependent PDBQTUtilities for ligand processing.
 *
 * Author: gnina developers
 */

#ifndef RDKIT_TREE_BUILDER_H_
#define RDKIT_TREE_BUILDER_H_

#include <GraphMol/GraphMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/MolOps.h>
#include <vector>
#include <map>
#include <set>
#include "parsing.h"
#include "atom_constants.h"

namespace RDKitTreeBuilder {

// Branch structure for tree representation (mirrors obbranch from PDBQTUtilities)
struct rdkitbranch {
    std::vector<int> atoms;  // Atom indices in this fragment
    bool done;
    unsigned int index;
    std::set<unsigned int> children;
    std::vector<unsigned int> parents;
    unsigned int depth;
    unsigned int connecting_atom_parent;
    unsigned int connecting_atom_branch;
    unsigned int how_many_atoms_moved;
    std::set<unsigned int> rigid_with;

    rdkitbranch() : done(false), index(0), depth(0),
                    connecting_atom_parent(0), connecting_atom_branch(0),
                    how_many_atoms_moved(0) {
        parents.push_back(0);
    }

    void clear() {
        done = false;
        index = 0;
        depth = 0;
        connecting_atom_parent = 0;
        connecting_atom_branch = 0;
        how_many_atoms_moved = 0;
        children.clear();
        parents.clear();
        atoms.clear();
        rigid_with.clear();
        parents.push_back(0);
    }

    unsigned int UpOne() const {
        if (parents.size() >= 2) {
            return parents[parents.size() - 2];
        }
        return 0;
    }
};

// Check if a bond is rotatable according to PDBQT rules
// Criteria:
// - Single bond
// - Not in a ring
// - Not an amide bond
// - Both atoms have at least one other heavy atom neighbor (except if desired_root)
bool isRotatableBond(const RDKit::ROMol& mol, const RDKit::Bond* bond,
                     unsigned int desired_root = 0);

// Check if a bond is an amide bond (C(=O)-N)
bool isAmideBond(const RDKit::ROMol& mol, const RDKit::Bond* bond);

// Get the number of heavy (non-hydrogen) neighbors for an atom
unsigned int getHeavyDegree(const RDKit::Atom* atom);

// Find rigid fragments by identifying rotatable bonds
// Returns the best root atom index
// desired_root: optionally specify a root atom (0 = auto-select)
// norotate: atom indices that should not have rotatable bonds around them
unsigned int findFragments(const RDKit::ROMol& mol,
                          std::vector<std::vector<int>>& rigid_fragments,
                          unsigned int desired_root = 0,
                          const std::vector<int>& norotate = std::vector<int>());

// Construct the tree structure from rigid fragments
void constructTree(std::map<unsigned int, rdkitbranch>& tree,
                  std::vector<std::vector<int>> rigid_fragments,
                  unsigned int root_piece,
                  const RDKit::ROMol& mol,
                  bool flexible);

// Find connecting atoms between two fragments
bool findBondedPiece(const std::vector<int>& root,
                    const std::vector<int>& branch,
                    unsigned int& root_atom,
                    unsigned int& branch_atom,
                    unsigned int& root_atom_rank,
                    unsigned int& branch_atom_rank,
                    const RDKit::ROMol& mol,
                    unsigned int& atoms_moved);

// Check if a vector contains a value
bool isIn(const std::vector<int>& vec, int num);

// Get smina atom type from RDKit atom
smt rdkitAtomToSminaType(const RDKit::Atom* atom, const RDKit::ROMol& mol);

// Check if atom is a hydrogen bond acceptor
bool isHBondAcceptor(const RDKit::Atom* atom, const RDKit::ROMol& mol);

// Check if atom is a hydrogen bond donor (has H attached)
bool isHBondDonor(const RDKit::Atom* atom, const RDKit::ROMol& mol);

// Check if atom has polar hydrogen attached
bool hasPolarHydrogen(const RDKit::Atom* atom, const RDKit::ROMol& mol);

// Check if hydrogen is non-polar (attached to carbon)
bool isNonPolarHydrogen(const RDKit::Atom* hatom, const RDKit::ROMol& mol);

// Delete non-polar hydrogens from atom index list
// Returns new list of atom indices after removing non-polar H
std::vector<int> removeNonPolarHydrogens(const RDKit::ROMol& mol,
                                         const std::vector<int>& atoms);

// Output the tree structure to parsing_struct and context
// mol: molecule (with polar H only)
// lines: context for output
// p: parsing struct to populate
// tree: tree structure from constructTree
// torsdof: number of torsional degrees of freedom
bool outputTree(const RDKit::ROMol& mol, context& lines, parsing_struct& p,
               std::map<unsigned int, rdkitbranch>& tree, unsigned int torsdof);

// Create SDF context from RDKit molecule
void createSDFContext(const RDKit::ROMol& mol,
                     const std::vector<unsigned int>& atomorder,
                     sdfcontext& sc);

} // namespace RDKitTreeBuilder

#endif /* RDKIT_TREE_BUILDER_H_ */
