/*
 * RDKitTreeBuilder.cpp
 *
 * RDKit-based rotatable bond detection and torsion tree building.
 * Replaces OpenBabel-dependent PDBQTUtilities for ligand processing.
 *
 * Author: gnina developers
 */

#include "RDKitTreeBuilder.h"
#include <GraphMol/RingInfo.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/AtomIterators.h>
#include <GraphMol/BondIterators.h>
#include <queue>
#include <algorithm>
#include <cassert>
#include <sstream>

namespace RDKitTreeBuilder {

using namespace RDKit;

bool isIn(const std::vector<int>& vec, int num) {
    return std::find(vec.begin(), vec.end(), num) != vec.end();
}

unsigned int getHeavyDegree(const Atom* atom) {
    unsigned int count = 0;
    for (const auto& nbr : atom->getOwningMol().atomNeighbors(atom)) {
        if (nbr->getAtomicNum() != 1) {
            count++;
        }
    }
    return count;
}

bool isAmideBond(const ROMol& mol, const Bond* bond) {
    // Amide bond: N-C(=O)
    // Check if one atom is N and the other is C with a double-bonded O
    const Atom* atom1 = bond->getBeginAtom();
    const Atom* atom2 = bond->getEndAtom();

    // Need N and C
    const Atom* nitrogen = nullptr;
    const Atom* carbon = nullptr;

    if (atom1->getAtomicNum() == 7 && atom2->getAtomicNum() == 6) {
        nitrogen = atom1;
        carbon = atom2;
    } else if (atom2->getAtomicNum() == 7 && atom1->getAtomicNum() == 6) {
        nitrogen = atom2;
        carbon = atom1;
    } else {
        return false;
    }

    // Check if carbon has a double-bonded oxygen
    for (const auto& bond : mol.atomBonds(carbon)) {
        if (bond->getBondType() == Bond::DOUBLE) {
            const Atom* other = bond->getOtherAtom(carbon);
            if (other->getAtomicNum() == 8) {
                return true;  // Found C(=O)-N
            }
        }
    }

    return false;
}

bool isRotatableBond(const ROMol& mol, const Bond* bond, unsigned int desired_root) {
    // Must be single bond
    if (bond->getBondType() != Bond::SINGLE) {
        return false;
    }

    // Must not be in a ring - use RDKit's ring info system
    // Ring info should be initialized before calling this function
    const RingInfo* ringInfo = mol.getRingInfo();
    if (ringInfo && ringInfo->isInitialized() && ringInfo->numBondRings(bond->getIdx()) > 0) {
        return false;
    }

    // Must not be an amide bond
    if (isAmideBond(mol, bond)) {
        return false;
    }

    // Both atoms must have at least one other heavy neighbor
    // (except if one of them is the desired root)
    unsigned int hvyDegree1 = getHeavyDegree(bond->getBeginAtom());
    unsigned int hvyDegree2 = getHeavyDegree(bond->getEndAtom());

    unsigned int beginIdx = bond->getBeginAtomIdx() + 1;  // 1-based like OpenBabel
    unsigned int endIdx = bond->getEndAtomIdx() + 1;

    if (hvyDegree1 == 1 || hvyDegree2 == 1) {
        // Terminal bond - only rotatable if it's the desired root
        if (beginIdx == desired_root || endIdx == desired_root) {
            return true;
        }
        return false;
    }

    return true;
}

bool isHBondAcceptor(const Atom* atom, const ROMol& mol) {
    // N, O, S with lone pairs can be acceptors
    int atomicNum = atom->getAtomicNum();

    if (atomicNum == 7) {  // Nitrogen
        // Check if it's not positively charged and has lone pair
        if (atom->getFormalCharge() >= 0) {
            // Count total bonds + formal charge
            int totalDegree = atom->getTotalDegree();
            int formalCharge = atom->getFormalCharge();
            // sp3 N with 3 bonds or less, or sp2 N with 2 bonds or less
            if (totalDegree + formalCharge < 4) {
                return true;
            }
        }
    } else if (atomicNum == 8) {  // Oxygen
        // Most oxygens are acceptors unless positively charged
        return atom->getFormalCharge() <= 0;
    } else if (atomicNum == 16) {  // Sulfur
        // Sulfur can be acceptor in some cases
        // Similar logic to oxygen
        return atom->getFormalCharge() <= 0;
    }

    return false;
}

bool isHBondDonor(const Atom* atom, const ROMol& mol) {
    // An atom is a donor if it has hydrogen attached
    // Check for explicit hydrogens
    for (const auto& nbr : mol.atomNeighbors(atom)) {
        if (nbr->getAtomicNum() == 1) {
            return true;
        }
    }
    // Check for implicit hydrogens
    return atom->getTotalNumHs() > 0;
}

bool hasPolarHydrogen(const Atom* atom, const ROMol& mol) {
    // Polar H is attached to N, O, S
    int atomicNum = atom->getAtomicNum();
    if (atomicNum == 7 || atomicNum == 8 || atomicNum == 16) {
        return isHBondDonor(atom, mol);
    }
    return false;
}

bool isNonPolarHydrogen(const Atom* hatom, const ROMol& mol) {
    if (hatom->getAtomicNum() != 1) {
        return false;
    }
    // Get the atom this H is attached to
    for (const auto& nbr : mol.atomNeighbors(hatom)) {
        // If attached to carbon, it's non-polar
        if (nbr->getAtomicNum() == 6) {
            return true;
        }
    }
    return false;
}

std::vector<int> removeNonPolarHydrogens(const ROMol& mol, const std::vector<int>& atoms) {
    std::vector<int> result;
    for (int idx : atoms) {
        const Atom* atom = mol.getAtomWithIdx(idx);
        if (!isNonPolarHydrogen(atom, mol)) {
            result.push_back(idx);
        }
    }
    return result;
}

smt rdkitAtomToSminaType(const Atom* atom, const ROMol& mol) {
    using namespace smina_atom_type;

    int atomicNum = atom->getAtomicNum();

    // Determine base type from element and aromaticity
    std::string ename;

    if (atomicNum == 1) {
        ename = "HD";  // Hydrogen, potentially polar
    } else if (atomicNum == 6 && atom->getIsAromatic()) {
        ename = "A";  // Aromatic carbon
    } else if (atomicNum == 8) {
        ename = "OA";  // Oxygen acceptor
    } else if (atomicNum == 7 && isHBondAcceptor(atom, mol)) {
        ename = "NA";  // Nitrogen acceptor
    } else if (atomicNum == 16 && isHBondAcceptor(atom, mol)) {
        ename = "SA";  // Sulfur acceptor
    } else {
        // Use element symbol
        ename = atom->getSymbol();
    }

    smt atype = string_to_smina_type(ename);

    // Check neighborhood for adjustment
    bool hbonded = false;
    bool heteroBonded = false;

    for (const auto& nbr : mol.atomNeighbors(atom)) {
        if (nbr->getAtomicNum() == 1) {
            hbonded = true;
        } else if (nbr->getAtomicNum() != 6) {
            heteroBonded = true;
        }
    }

    return adjust_smina_type(atype, hbonded, heteroBonded);
}

unsigned int findFragments(const ROMol& mol,
                          std::vector<std::vector<int>>& rigid_fragments,
                          unsigned int desired_root,
                          const std::vector<int>& norotate) {
    unsigned int numAtoms = mol.getNumAtoms();
    if (numAtoms == 0) {
        return 1;
    }

    // Find best root atom (minimize largest remaining subgraph)
    unsigned int best_root_atom = 1;  // 1-based
    unsigned int shortest_maximal = numAtoms;

    for (unsigned int i = 0; i < numAtoms; i++) {
        // Create adjacency list without atom i
        std::vector<std::vector<int>> adj(numAtoms);
        for (auto bondIt = mol.beginBonds(); bondIt != mol.endBonds(); ++bondIt) {
            unsigned int a = (*bondIt)->getBeginAtomIdx();
            unsigned int b = (*bondIt)->getEndAtomIdx();
            if (a != i && b != i) {
                adj[a].push_back(b);
                adj[b].push_back(a);
            }
        }

        // Find connected components
        std::vector<bool> visited(numAtoms, false);
        visited[i] = true;
        unsigned int max_component = 0;

        for (unsigned int j = 0; j < numAtoms; j++) {
            if (!visited[j]) {
                unsigned int component_size = 0;
                std::queue<unsigned int> q;
                q.push(j);
                visited[j] = true;
                while (!q.empty()) {
                    unsigned int cur = q.front();
                    q.pop();
                    component_size++;
                    for (int nbr : adj[cur]) {
                        if (!visited[nbr]) {
                            visited[nbr] = true;
                            q.push(nbr);
                        }
                    }
                }
                if (component_size > max_component) {
                    max_component = component_size;
                }
            }
        }

        if (max_component < shortest_maximal) {
            shortest_maximal = max_component;
            best_root_atom = i + 1;  // 1-based
        }
    }

    // Create set of norotate atoms
    std::set<int> norotSet(norotate.begin(), norotate.end());

    // Find rotatable bonds
    std::vector<std::pair<int, int>> rotBonds;
    for (auto bondIt = mol.beginBonds(); bondIt != mol.endBonds(); ++bondIt) {
        const Bond* bond = *bondIt;
        int src = bond->getBeginAtomIdx() + 1;  // 1-based
        int dst = bond->getEndAtomIdx() + 1;

        // Check if both atoms are in norotate set
        if (norotSet.count(src) && norotSet.count(dst)) {
            continue;  // Not rotatable
        }

        if (isRotatableBond(mol, bond, desired_root)) {
            rotBonds.push_back({bond->getBeginAtomIdx(), bond->getEndAtomIdx()});
        }
    }

    // Create fragments by removing rotatable bonds
    // Build adjacency list without rotatable bonds
    std::vector<std::vector<int>> adj(numAtoms);
    std::set<std::pair<int, int>> rotBondSet;
    for (const auto& rb : rotBonds) {
        rotBondSet.insert({std::min(rb.first, rb.second), std::max(rb.first, rb.second)});
    }

    for (auto bondIt = mol.beginBonds(); bondIt != mol.endBonds(); ++bondIt) {
        int a = (*bondIt)->getBeginAtomIdx();
        int b = (*bondIt)->getEndAtomIdx();
        std::pair<int, int> key = {std::min(a, b), std::max(a, b)};
        if (rotBondSet.find(key) == rotBondSet.end()) {
            adj[a].push_back(b);
            adj[b].push_back(a);
        }
    }

    // Find connected components (rigid fragments)
    std::vector<bool> visited(numAtoms, false);
    rigid_fragments.clear();

    for (unsigned int i = 0; i < numAtoms; i++) {
        if (!visited[i]) {
            std::vector<int> fragment;
            std::queue<unsigned int> q;
            q.push(i);
            visited[i] = true;
            while (!q.empty()) {
                unsigned int cur = q.front();
                q.pop();
                fragment.push_back(cur + 1);  // 1-based indices
                for (int nbr : adj[cur]) {
                    if (!visited[nbr]) {
                        visited[nbr] = true;
                        q.push(nbr);
                    }
                }
            }
            rigid_fragments.push_back(fragment);
        }
    }

    return best_root_atom;
}

bool findBondedPiece(const std::vector<int>& root,
                    const std::vector<int>& branch,
                    unsigned int& root_atom,
                    unsigned int& branch_atom,
                    unsigned int& root_atom_rank,
                    unsigned int& branch_atom_rank,
                    const ROMol& mol,
                    unsigned int& atoms_moved) {
    for (unsigned int i = 0; i < root.size(); i++) {
        for (unsigned int j = 0; j < branch.size(); j++) {
            int r_idx = root[i] - 1;  // Convert to 0-based
            int b_idx = branch[j] - 1;

            // Check if there's a bond between these atoms
            const Bond* bond = mol.getBondBetweenAtoms(r_idx, b_idx);
            if (bond != nullptr) {
                root_atom = root[i];
                branch_atom = branch[j];
                root_atom_rank = i;
                branch_atom_rank = j;

                // Count atoms moved (atoms reachable from branch side after removing bond)
                // This is approximated by the branch size
                atoms_moved = branch.size();
                return true;
            }
        }
    }
    return false;
}

void constructTree(std::map<unsigned int, rdkitbranch>& tree,
                  std::vector<std::vector<int>> rigid_fragments,
                  unsigned int root_piece,
                  const ROMol& mol,
                  bool flexible) {
    unsigned int first_atom = 0;
    unsigned int second_atom = 0;
    unsigned int first_atom_rank = 0;
    unsigned int second_atom_rank = 0;

    rdkitbranch sprog;
    sprog.atoms = rigid_fragments[root_piece];
    sprog.rigid_with.insert(0);

    tree.clear();
    tree.insert({0, sprog});

    rigid_fragments.erase(rigid_fragments.begin() + root_piece);

    unsigned int position = 0;
    unsigned int atoms_moved = 0;
    bool fecund;

    while (!tree[0].done) {
        fecund = !tree[position].done;
        if (fecund) {
            bool sterile = true;
            for (unsigned int i = 0; i < rigid_fragments.size(); i++) {
                if (findBondedPiece(tree[position].atoms,
                                   rigid_fragments[i],
                                   first_atom, second_atom,
                                   first_atom_rank, second_atom_rank,
                                   mol, atoms_moved)) {
                    sprog.clear();
                    sprog.connecting_atom_parent = first_atom;
                    sprog.connecting_atom_branch = second_atom;
                    sprog.how_many_atoms_moved = atoms_moved;
                    sprog.atoms = rigid_fragments[i];

                    sprog.depth = tree[position].depth + 1;
                    sprog.parents = tree[position].parents;
                    sprog.parents.push_back(tree.size());
                    sprog.index = tree.size();
                    sprog.rigid_with.clear();
                    sprog.rigid_with.insert(sprog.index);

                    tree[position].children.insert(tree.size());
                    tree.insert({tree.size(), sprog});

                    rigid_fragments.erase(rigid_fragments.begin() + i);
                    sterile = false;
                    position = tree.size() - 1;
                    break;
                }
            }
            if (sterile) {
                tree[position].done = true;
            }
        } else {
            position--;
        }
    }
}

void createSDFContext(const ROMol& mol,
                     const std::vector<unsigned int>& atomorder,
                     sdfcontext& sc) {
    sc.atoms.clear();
    sc.bonds.clear();
    sc.properties.clear();

    // Get molecule name
    if (mol.hasProp("_Name")) {
        mol.getProp("_Name", sc.name);
    } else {
        sc.name = "ligand";
    }

    // Create mapping from atom index to position in atomorder
    std::map<unsigned int, unsigned int> idx2pos;
    for (unsigned int i = 0; i < atomorder.size(); i++) {
        idx2pos[atomorder[i]] = i;
    }

    // Add atoms
    for (unsigned int i = 0; i < atomorder.size(); i++) {
        unsigned int atomIdx = atomorder[i];
        const Atom* atom = mol.getAtomWithIdx(atomIdx);
        std::string symbol = atom->getSymbol();
        sc.atoms.push_back(sdfcontext::sdfatom(symbol.c_str(), false));

        // Handle special properties
        if (atom->getFormalCharge() != 0) {
            sdfcontext::sdfprop prop(i, 'c', atom->getFormalCharge());
            sc.properties.push_back(prop);
        }
        if (atom->getIsotope() != 0) {
            sdfcontext::sdfprop prop(i, 'i', atom->getIsotope());
            sc.properties.push_back(prop);
        }
    }

    // Add bonds
    for (auto bondIt = mol.beginBonds(); bondIt != mol.endBonds(); ++bondIt) {
        const Bond* bond = *bondIt;
        unsigned int a = bond->getBeginAtomIdx();
        unsigned int b = bond->getEndAtomIdx();

        // Only include bonds where both atoms are in atomorder
        if (idx2pos.count(a) && idx2pos.count(b)) {
            unsigned int first = idx2pos[a];
            unsigned int second = idx2pos[b];

            // Convert bond type to order
            int order = 1;
            switch (bond->getBondType()) {
                case Bond::SINGLE: order = 1; break;
                case Bond::DOUBLE: order = 2; break;
                case Bond::TRIPLE: order = 3; break;
                case Bond::AROMATIC: order = 4; break;
                default: order = 1; break;
            }

            sc.bonds.push_back(sdfcontext::sdfbond(first, second, order));
        }
    }

    // Handle SDF data (properties) - not implemented for now
    sc.datastr = "";
}

// Helper function to output an atom
static void outputAtom(const Atom* atom, const Conformer& conf,
                      context& lines, std::vector<unsigned int>& atomorder,
                      parsing_struct& p, unsigned int index, unsigned int immobile_num,
                      const ROMol& mol) {

    // Get smina type
    smt sm = rdkitAtomToSminaType(atom, mol);
    if (sm >= smina_atom_type::NumTypes) {
        sm = smina_atom_type::Hydrogen;  // Fallback
    }

    // Get coordinates
    const RDGeom::Point3D& pos = conf.getAtomPos(atom->getIdx());
    vec coords(pos.x, pos.y, pos.z);

    // Get partial charge (use Gasteiger if available, otherwise 0)
    double charge = 0.0;
    if (atom->hasProp("_GasteigerCharge")) {
        atom->getProp("_GasteigerCharge", charge);
        if (!std::isfinite(charge)) {
            charge = 0.0;
        }
    }

    parsed_atom patom(sm, charge, coords, index);

    // Check for immobile atom
    if (patom.number == immobile_num) {
        p.immobile_atom = p.atoms.size();
    }

    p.add(patom, lines, atomorder.size());
    atomorder.push_back(atom->getIdx());
}

// Helper function to output a group of atoms
static void outputGroup(const ROMol& mol, const Conformer& conf,
                       context& lines, std::vector<unsigned int>& atomorder,
                       parsing_struct& p, unsigned int immobile_num,
                       const std::vector<int>& group,
                       const std::map<unsigned int, unsigned int>& new_indexes) {
    for (int atomIdx : group) {
        const Atom* atom = mol.getAtomWithIdx(atomIdx - 1);  // Convert from 1-based
        unsigned int newIndex = new_indexes.at(atomIdx);
        outputAtom(atom, conf, lines, atomorder, p, newIndex, immobile_num, mol);
    }
}

bool outputTree(const ROMol& mol, context& lines, parsing_struct& p,
               std::map<unsigned int, rdkitbranch>& tree, unsigned int torsdof) {
    if (tree.empty()) {
        return false;
    }

    // Get conformer for coordinates
    if (mol.getNumConformers() == 0) {
        return false;
    }
    const Conformer& conf = mol.getConformer();

    set_fixed_rotable_hydrogens(true);

    std::vector<unsigned int> atomorder;
    std::map<unsigned int, unsigned int> new_order;

    // Generate new ordering (1-based output indices)
    unsigned int current_atom_index = 1;
    for (unsigned int i = 0; i < tree.size(); i++) {
        assert(tree.count(i));
        for (unsigned int rigid_idx : tree[i].rigid_with) {
            const std::vector<int>& atoms = tree[rigid_idx].atoms;
            for (int atomIdx : atoms) {
                new_order.insert({atomIdx, current_atom_index});
                current_atom_index++;
            }
        }
    }

    std::stack<parsing_struct> pstack;
    pstack.push(parsing_struct());
    std::stack<std::pair<unsigned int, unsigned int>> bnumstack;

    // Output root
    for (unsigned int rigid_idx : tree[0].rigid_with) {
        outputGroup(mol, conf, lines, atomorder, pstack.top(), UINT_MAX,
                   tree[rigid_idx].atoms, new_order);
    }

    // Output branches
    for (unsigned int i = 1; i < tree.size(); i++) {
        unsigned int parent_atom = tree[i].connecting_atom_parent;
        unsigned int child_atom = tree[i].connecting_atom_branch;
        unsigned int parnum = new_order[parent_atom];
        unsigned int childnum = new_order[child_atom];

        pstack.push(parsing_struct());
        bnumstack.push({parnum, childnum});

        for (unsigned int rigid_idx : tree[i].rigid_with) {
            outputGroup(mol, conf, lines, atomorder, pstack.top(), childnum,
                       tree[rigid_idx].atoms, new_order);
        }

        // Close branches
        for (auto it = tree[i].parents.end(); it != tree[i].parents.begin();) {
            it--;
            if (*it == 0) break;

            auto it_parent = it;
            it_parent--;

            if (tree[*it].children.empty()) {
                parsing_struct branch = pstack.top();
                pstack.pop();
                unsigned int pnum = bnumstack.top().first;
                bnumstack.pop();

                // Find parent fragment position
                unsigned int pos = 0;
                for (unsigned int n = pstack.top().atoms.size(); pos < n; pos++) {
                    if (pstack.top().atoms[pos].a.number == pnum) break;
                }
                assert(pos < pstack.top().atoms.size());

                if (branch.mobile_hydrogens_only()) {
                    pstack.top().mergeInto(branch);
                } else {
                    pstack.top().atoms[pos].ps.push_back(branch);
                }

                tree[*it_parent].children.erase(*it);
            }
        }
    }

    createSDFContext(mol, atomorder, lines.sdftext);
    assert(pstack.size() == 1);
    assert(bnumstack.empty());
    p = pstack.top();
    return true;
}

} // namespace RDKitTreeBuilder
