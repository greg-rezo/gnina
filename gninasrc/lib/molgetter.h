/*
 * molgetter.h
 *
 *  Created on: Jun 5, 2014
 *      Author: dkoes
 */

#ifndef MOLGETTER_H_
#define MOLGETTER_H_

#include "model.h"
#include "covinfo.h"
#include "flexinfo.h"
#include "model.h"
#include "obmolopener.h"

// RDKit includes needed for SDMolSupplier (complete type required for unique_ptr destructor)
#include <GraphMol/FileParsers/MolSupplier.h>

// this class abstracts reading molecules from a file
// we have five means of input:
// RDKit for SDF/MOL2 ligands (preferred - better geometry)
// openbabel for covalent docking and PDB ligands (still needed for flex residues)
// vina parse_pdbqt for pdbqt files (one ligand, obey rotational bonds)
// smina/gnina binary formats
class MolGetter {
  model initm;
  tee *log;
  CovInfo cinfo;
  enum Type { RDKIT_SDF, OB, PDBQT, SMINA, GNINA, NONE }; // different inputs

  Type type;
  path lpath;
  bool add_hydrogens;   // add hydrogens before calculating atom types
  bool strip_hydrogens; // strip them after (more efficient)

  // RDKit data structs for SDF/MOL2 reading
  std::unique_ptr<RDKit::SDMolSupplier> rdkit_supplier;

  // openbabel data structs (for covalent docking and receptors)
  OpenBabel::OBConversion conv;
  obmol_opener infileopener;

  // smina data structs
  izfile infile;

  // pdbqt data
  bool pdbqtdone;

  // covalent data (still uses OpenBabel)
  OpenBabel::OBMol covres;              // covalently bonding residue on receptor
  OpenBabel::OBAtom *covatom = nullptr; // covalently bonding atom within this residue
  vec covpos;                           // position for covalently bonding ligand atom
  bool covres_isflex = false; //true if covalently bonded residue should be flexible

  OpenBabel::OBMol covmol;                  // current ligand being docked
  OpenBabel::OBMol origcovmol; //original ligand before covalent modifications
  std::vector<std::vector<int>> match_list; // smarts matches
  unsigned matchpos = UINT_MAX;             // current position in match_list

public:
  MolGetter(bool addH = true, bool stripH = true)
      : add_hydrogens(addH), strip_hydrogens(stripH), type(NONE), pdbqtdone(false) {}

  MolGetter(const std::string &rigid_name, const std::string &flex_name, FlexInfo &finfo, CovInfo &ci, bool addH,
            bool stripH, tee &l)
      : log(&l), cinfo(ci), add_hydrogens(addH), strip_hydrogens(stripH), type(NONE), pdbqtdone(false) {
    create_init_model(rigid_name, flex_name, finfo, l);
  }

  // create the initial model from the specified receptor files
  void create_init_model(const std::string &rigid_name, const std::string &flex_name, FlexInfo &finfo, tee &log);

  // setup for reading from fname
  void setInputFile(const std::string &fname);

  // initialize model to initm and add next molecule
  // return false if no molecule available;
  bool readMoleculeIntoModel(model &m);

  // return model without ligand
  const model &getInitModel() const { return initm; }

private:
  bool createCovalentMoleculeInModel(model &m);
  bool readRDKitMoleculeIntoModel(model &m);
};

// Standalone function to convert OBMol to model (shared by SDF loading and RDKit SMILES paths)
bool convertOBMolToModel(OpenBabel::OBMol& mol, model& m, bool add_hydrogens, bool strip_hydrogens);

// Standalone function to convert RDKit mol to model (for RDKit-native loading)
bool convertRDKitMolToModel(const RDKit::ROMol& mol, model& m, bool add_hydrogens, bool strip_hydrogens);

#endif /* MOLGETTER_H_ */
