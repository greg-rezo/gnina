/*
 * RDKitConverter.h
 *
 * Direct RDKit to gnina conversion, bypassing OpenBabel.
 * Uses RDKitTreeBuilder for rotatable bond detection and tree building.
 *
 * Author: gnina developers
 */

#ifndef RDKIT_CONVERTER_H_
#define RDKIT_CONVERTER_H_

#include <GraphMol/GraphMol.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <iostream>
#include <vector>
#include "parsing.h"

namespace RDKitConverter {

// Main conversion function: RDKit mol to smina parsing struct and context
// Returns number of torsional degrees of freedom
// mol: RDKit molecule (should have 3D coordinates)
// p: parsing struct to populate
// c: context to populate
// rootatom: optional root atom (1-based index, 0 = auto-select)
// norotate: atoms that should not have rotatable bonds (1-based indices)
// addH: add hydrogens before processing
unsigned convertRDKitParsing(const RDKit::ROMol& mol, parsing_struct& p, context& c,
                             int rootatom = 0,
                             const std::vector<int>& norotate = std::vector<int>(),
                             bool addH = true);

// Simplified version with just mol and basic outputs
unsigned convertRDKitParsing(const RDKit::ROMol& mol, parsing_struct& p, context& c,
                             bool addH);

// Convert RDKit mol to gnina model
// mol: RDKit molecule with 3D coordinates
// m: model to populate
// add_hydrogens: add hydrogens before processing
// strip_hydrogens: remove non-polar hydrogens after processing
// Returns true on success
bool convertRDKitToModel(const RDKit::ROMol& mol, model& m,
                         bool add_hydrogens = true, bool strip_hydrogens = true);

// Helper: Delete non-polar hydrogens from an RDKit molecule
// Returns a new molecule with non-polar H removed
std::unique_ptr<RDKit::RWMol> deleteNonPolarHydrogens(const RDKit::ROMol& mol);

// Helper: Calculate Gasteiger charges for a molecule
void calculateGasteigerCharges(RDKit::RWMol& mol);

// Note: RDKitTreeBuilder types are fully defined in RDKitTreeBuilder.h
// They are accessed via ::RDKitTreeBuilder:: to avoid namespace issues

// Class for efficiently converting multi-conformer molecule
class RDKitMCMolConverter {
public:
    RDKitMCMolConverter(const RDKit::ROMol& mol);
    ~RDKitMCMolConverter();  // Need destructor for pImpl pattern

    // Output smina data for specified conformer
    void convertConformer(unsigned conf, std::ostream& out);

    unsigned getTorsDof() const { return torsdof; }

private:
    struct Impl;  // pImpl to avoid exposing rdkitbranch in header
    std::unique_ptr<Impl> pImpl;
    unsigned torsdof;
};

} // namespace RDKitConverter

#endif /* RDKIT_CONVERTER_H_ */
