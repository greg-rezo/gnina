/*
 * GninaConverter.cpp
 *
 *  Created on: Jun 4, 2014
 *      Author: dkoes
 *
 *  Convert internal molecular data (ie OBMol) into gnina parse tree.
 */

#include "GninaConverter.h"
#include "parsing.h"
#include "PDBQTUtilities.h"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <map>
#include <openbabel/obconversion.h>
#include <openbabel/bond.h>
#include <openbabel/oberror.h>

// RDKit includes for RDKit->OpenBabel conversion
#include <GraphMol/GraphMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/PeriodicTable.h>

#include <sstream>

namespace GninaConverter {

using namespace OpenBabel;
using namespace std;

MCMolConverter::MCMolConverter(OpenBabel::OBMol& m)
    : mol(m) {
  //precompute fragments and tree
  int nc = mol.NumConformers();
  mol.SetConformer(0);
  mol.AddHydrogens();

  mol.SetAutomaticFormalCharge(false);
  DeleteHydrogens(mol); //leaves just polars
  if (mol.NumAtoms() == 0) return;

  //we kind of assume a connected molecule
  unsigned best_root_atom = FindFragments(mol, rigid_fragments);
  torsdof = rigid_fragments.size() - 1;

  unsigned int root_piece = 0;
  for (unsigned j = 0; j < rigid_fragments.size(); j++) {
    if (IsIn((rigid_fragments[j]), best_root_atom)) {
      root_piece = j;
      break;
    } //this is the root rigid molecule fragment
  }

  ConstructTree(tree, rigid_fragments, root_piece, mol, true);

  if (nc != mol.NumConformers())  //didn't lose any in analysis, did we?
      {
    abort(); //there was a bug in openbabel where addhydrogens would eliminate conformers
  }
}

//output data for this conformer
void MCMolConverter::convertConformer(unsigned conf, std::ostream& out) {
  parsing_struct p;
  context c;

  mol.SetConformer(conf);

  std::map<unsigned int, obbranch> tmptree(tree); //tree gets modified by outputtree
  OutputTree(mol, c, p, tmptree, torsdof);

  boost::iostreams::filtering_stream<boost::iostreams::output> strm;
  strm.push(boost::iostreams::gzip_compressor());
  strm.push(out);

  boost::archive::binary_oarchive serialout(strm,
      boost::archive::no_header | boost::archive::no_tracking);

  serialout << torsdof;
  serialout << p;
  serialout << c;
}

// Helper function to perceive bond orders with kekulization error checking
static void perceiveBondOrdersChecked(OBMol& mol, const std::string& mol_name) {
  // Redirect stderr to capture OpenBabel warnings
  std::stringstream captured;
  std::streambuf* old_cerr = std::cerr.rdbuf(captured.rdbuf());

  mol.PerceiveBondOrders();

  // Restore stderr
  std::cerr.rdbuf(old_cerr);

  // Check for kekulization warnings
  std::string output = captured.str();
  if (output.find("kekulize") != std::string::npos ||
      output.find("Kekulize") != std::string::npos) {
    throw std::runtime_error("Kekulization failed for molecule '" + mol_name +
        "': " + output +
        "\nPlease check the input molecule has valid aromatic bond assignments.");
  }
}

//sets up data structures used by both text and binary
//we link with gnina to ensure compatibility
//rootatom, an obatom index (starting at 1) can be specified, if not
//the "best" root is chosen
unsigned convertParsing(OBMol& mol, parsing_struct& p, context& c, int rootatom,
    const vector<int>& norot, bool addH) {
  if (addH) mol.AddHydrogens();

  perceiveBondOrdersChecked(mol, mol.GetTitle());
  mol.SetAromaticPerceived();
  mol.SetAutomaticFormalCharge(false);

  vector<OBAtom *> norotate_atoms;
  if(norot.size() > 0) {
    //need to save atoms before modifying molecule to make sure norotate ids are correct
    for(auto i : norot) {
      OBAtom *a = mol.GetAtom(i);
      if(a && !a->IsElement(1)) {
        norotate_atoms.push_back(a);
      }
    }
  }
  DeleteHydrogens(mol); //leaves just polars

  vector<int> norotate; norotate.reserve(norot.size());
  if(norot.size() > 0) {
    for(auto a: norotate_atoms) {
      norotate.push_back(a->GetIdx());
    }
  }

  vector<vector<int> > rigid_fragments; //the vector of all the rigid molecule fragments, using atom indexes
  map<unsigned int, obbranch> tree;

  //we kind of assume a connected molecule
  unsigned best_root_atom = FindFragments(mol, rigid_fragments, rootatom, norotate);
  unsigned torsdof = rigid_fragments.size() - 1;

  if (rootatom > 0) {
    //user user supplied root
    best_root_atom = rootatom;
  }

  unsigned int root_piece = 0;
  for (unsigned j = 0; j < rigid_fragments.size(); j++) {
    if (IsIn((rigid_fragments[j]), best_root_atom)) {
      root_piece = j;
      break;
    } //this is the root rigid molecule fragment
  }

  ConstructTree(tree, rigid_fragments, root_piece, mol, true);

  OutputTree(mol, c, p, tree, torsdof);

  return torsdof;
}

unsigned convertParsing(OpenBabel::OBMol& mol, parsing_struct& p, context& c,
    bool addH) {
  std::vector<int> nr;
  return convertParsing(mol, p, c, 0, nr, addH);
}

template<class T>
static void convert(OBMol& mol, T& serialout, ostream& out, int rootatom,
    const vector<int>& norotate) {
  parsing_struct p;
  context c;
  unsigned torsdof = convertParsing(mol, p, c, rootatom, norotate);
  serialout << torsdof;
  serialout << p;
  serialout << c;
}

//text output
void convertText(OBMol& mol, ostream& out, int rootatom,
    const vector<int>& norotate) {
  boost::archive::text_oarchive serialout(out,
      boost::archive::no_header | boost::archive::no_tracking);
  convert(mol, serialout, out, rootatom, norotate);
}

void convertText(OpenBabel::OBMol& mol, std::ostream& out) {
  std::vector<int> nr;
  convertText(mol, out, 0, nr);
}

//binary output
void convertBinary(OBMol& mol, ostream& out, int rootatom,
    const vector<int>& norotate) {
  //by definition, gnina format is gzipped
  boost::iostreams::filtering_stream<boost::iostreams::output> strm;
  strm.push(boost::iostreams::gzip_compressor());
  strm.push(out);

  boost::archive::binary_oarchive serialout(strm,
      boost::archive::no_header | boost::archive::no_tracking);
  convert(mol, serialout, strm, rootatom, norotate);
}

void convertBinary(OpenBabel::OBMol& mol, std::ostream& out) {
  std::vector<int> nr;
  convertBinary(mol, out, 0, nr);
}

// Convert RDKit ROMol to OpenBabel OBMol
void convertRDKitToOBMol(const RDKit::ROMol& rdmol, OpenBabel::OBMol& obmol) {
  obmol.Clear();
  obmol.BeginModify();

  // Reserve space
  obmol.ReserveAtoms(rdmol.getNumAtoms());

  // Copy atoms
  const RDKit::Conformer& conf = rdmol.getConformer();
  for (unsigned int i = 0; i < rdmol.getNumAtoms(); i++) {
    const RDKit::Atom* rdatom = rdmol.getAtomWithIdx(i);
    OBAtom* obatom = obmol.NewAtom();

    // Set atomic number
    obatom->SetAtomicNum(rdatom->getAtomicNum());

    // Set coordinates
    const RDGeom::Point3D& pos = conf.getAtomPos(i);
    obatom->SetVector(pos.x, pos.y, pos.z);

    // Set formal charge
    obatom->SetFormalCharge(rdatom->getFormalCharge());

    // Set aromaticity
    if (rdatom->getIsAromatic()) {
      obatom->SetAromatic();
    }
  }

  // Copy bonds
  for (unsigned int i = 0; i < rdmol.getNumBonds(); i++) {
    const RDKit::Bond* rdbond = rdmol.getBondWithIdx(i);
    unsigned int beginIdx = rdbond->getBeginAtomIdx();
    unsigned int endIdx = rdbond->getEndAtomIdx();

    // OpenBabel uses 1-based indexing
    int order = 1;
    switch (rdbond->getBondType()) {
      case RDKit::Bond::SINGLE: order = 1; break;
      case RDKit::Bond::DOUBLE: order = 2; break;
      case RDKit::Bond::TRIPLE: order = 3; break;
      case RDKit::Bond::AROMATIC: order = 5; break; // OB aromatic
      default: order = 1; break;
    }

    obmol.AddBond(beginIdx + 1, endIdx + 1, order);

    // Set aromaticity on bond
    if (rdbond->getIsAromatic()) {
      OBBond* obbond = obmol.GetBond(beginIdx + 1, endIdx + 1);
      if (obbond) {
        obbond->SetAromatic();
      }
    }
  }

  obmol.EndModify();

  // Set molecule name
  if (rdmol.hasProp("_Name")) {
    std::string name;
    rdmol.getProp("_Name", name);
    obmol.SetTitle(name);
  }

  // Perceive properties needed for atom typing
  perceiveBondOrdersChecked(obmol, obmol.GetTitle());
  obmol.SetAromaticPerceived();
}

// Convert RDKit mol to smina parsing struct and context
unsigned convertParsing(const RDKit::ROMol& rdmol, parsing_struct& p, context& c,
    bool addH) {
  // Convert RDKit to OpenBabel
  OBMol obmol;
  convertRDKitToOBMol(rdmol, obmol);

  // Use existing OpenBabel conversion
  return convertParsing(obmol, p, c, addH);
}

} //namespace GninaConverter
