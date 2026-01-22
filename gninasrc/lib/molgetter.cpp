/*
 * molgetter.cpp
 *
 *  Created on: Jun 5, 2014
 *      Author: dkoes
 */
#include "molgetter.h"
#include "GninaConverter.h"
#include "RDKitConverter.h"
#include "parse_pdbqt.h"
#include "parsing.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/timer/timer.hpp>
#include <openbabel/bond.h>
#include <openbabel/builder.h>
#include <openbabel/forcefield.h>
#include <openbabel/generic.h>
#include <openbabel/mol.h>
#include <openbabel/obconversion.h>
#include <sstream>

// RDKit includes for ligand reading
#include <GraphMol/GraphMol.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/MolOps.h>

using namespace OpenBabel;

// Convert an OpenBabel OBMol (with 3D coordinates) to a gnina model
// This is for covalent docking and formats not yet supported by RDKit path
// Returns true on success, false on failure
bool convertOBMolToModel(OpenBabel::OBMol& mol, model& m, bool add_hydrogens, bool strip_hydrogens) {
    try {
        parsing_struct p;
        context c;
        unsigned torsdof = GninaConverter::convertParsing(mol, p, c, add_hydrogens);
        non_rigid_parsed nr;
        postprocess_ligand(nr, p, c, torsdof);
        VINA_CHECK(nr.atoms_atoms_bonds.dim() == nr.atoms.size());

        pdbqt_initializer tmp;
        tmp.initialize_from_nrp(nr, c, true);
        tmp.initialize(nr.mobility_matrix());
        if (strip_hydrogens)
            tmp.m.strip_hydrogens();

        m.append(tmp.m);
        return true;
    } catch (parse_error &e) {
        std::cerr << "\n\nParse error with molecule " << mol.GetTitle() << " in file \"" << e.file.string()
                  << "\": " << e.reason << '\n';
        return false;
    }
}

// Convert an RDKit mol (with 3D coordinates) to a gnina model
// This is the preferred path for ligands - uses pure RDKit with no OpenBabel
// Returns true on success, false on failure
bool convertRDKitMolToModel(const RDKit::ROMol& mol, model& m, bool add_hydrogens, bool strip_hydrogens) {
    return RDKitConverter::convertRDKitToModel(mol, m, add_hydrogens, strip_hydrogens);
}

// remove a hydrogen from a
static void decrement_hydrogen(OBMol &mol, OBAtom *a) {

  if (a->GetImplicitHCount() > 0) {
    a->SetImplicitHCount(a->GetImplicitHCount() - 1);
  } else if (a->ExplicitHydrogenCount() > 0) {
    int hcnt = a->ExplicitHydrogenCount();

    std::vector<OBAtom *>::iterator i;
    std::vector<OBBond *>::iterator k;
    std::vector<OBAtom *> delatoms;
    OBAtom *nbr;

    for (nbr = a->BeginNbrAtom(k); nbr; nbr = a->NextNbrAtom(k))
      if (nbr->GetAtomicNum() == OBElements::Hydrogen)
        delatoms.push_back(nbr);

    mol.IncrementMod();
    for (i = delatoms.begin(); i != delatoms.end(); ++i)
      mol.DeleteHydrogen((OBAtom *)*i);
    mol.DecrementMod();

    // mol.DeleteHydrogens(a);
    a->SetImplicitHCount(hcnt - 1);
  }
}

// create the initial model from the specified receptor files
// mostly because Matt kept complaining about it, this will automatically create
// pdbqts if necessary using open babel
void MolGetter::create_init_model(const std::string &rigid_name, const std::string &flex_name, FlexInfo &finfo,
                                  tee &log) {
  if (rigid_name.size() > 0) {
    // support specifying flexible residues explicitly as pdbqt, but only
    // in compatibility mode where receptor is pdbqt as well
    if (flex_name.size() > 0) {
      if (finfo.has_content()) {
        throw usage_error("Cannot mix -flex option with -flexres or -flexdist options.");
      }
      if (cinfo.has_content()) {
        throw usage_error("Cannot mix -flex option with covalent docking.  Use "
                          "-flexres or -flexdist options.");
      }
      if (extension(rigid_name) != ".pdbqt") {
        throw usage_error("Cannot use -flex option with non-PDBQT receptor.");
      }
      ifile rigidin(rigid_name);
      ifile flexin(flex_name);
      initm = parse_receptor_pdbqt(rigid_name, rigidin, flex_name, flexin);
    } else if (!finfo.has_content() && !cinfo.has_content() && extension(rigid_name) == ".pdbqt") {
      // compatibility mode - read pdbqt directly with no openbabel shenanigans
      ifile rigidin(rigid_name);
      initm = parse_receptor_pdbqt(rigid_name, rigidin);
    } else {
      // default, openbabel mode
      obmol_opener fileopener;
      OBConversion conv;
      conv.SetOutFormat("PDBQT");
      conv.AddOption("r",
                     OBConversion::OUTOPTIONS);      // rigid molecule, otherwise really slow
                                                     // and useless analysis is triggered
      conv.AddOption("c", OBConversion::OUTOPTIONS); // single combined molecule
      fileopener.openForInput(conv, rigid_name);
      OBMol rec;
      if (!conv.Read(&rec))
        throw file_error(rigid_name, true);

      // this is an obnoxious fix for problems in released openbabel (3_1_1 at
      // least) where arginine is not correctly protonated if NH1 is charged
      // also, find covalent rec atom if needed

      FOR_ATOMS_OF_MOL(a, rec) {
        OBResidue *residue = a->GetResidue();
        if (residue && a->GetFormalCharge() == 1) {
          std::string aname = residue->GetAtomID(&*a);
          boost::trim(aname);
          if (aname == "NH1") {
            a->SetFormalCharge(0);
            a->SetImplicitHCount(2);
          }
        }
      }

      // covalent processing
      if (cinfo.has_content()) {
        OBAtom *covrec = cinfo.find_rec_atom(rec);
        if (!covrec) {
          throw usage_error("Could not find receptor atom " + cinfo.rec_atom_string());
        }
        // pull out residue being bonded to
        OBResidue *covr = covrec->GetResidue();
        if (!covr) {
          throw usage_error("Could not get residue of covalent receptor atom.");
        }
        char ch = covr->GetChain();
        int resid = covr->GetNum();
        char icode = covr->GetInsertionCode();

        covres_isflex = finfo.omit_residue(std::tuple<char, int, char>(ch, resid, icode));
        finfo.extract_residue(rec, covr, covres, true);
        covatom = cinfo.find_rec_atom(covres);

        if (!covatom) {
          throw usage_error("Lost receptor atom " + cinfo.rec_atom_string() +
                            " when constructing flexible residue (this should not happen).");
        }
        VINA_CHECK(covatom);
        if (covatom->GetHvyDegree() == 0) {
          throw usage_error("Invalid solitary receptor atom " + cinfo.rec_atom_string() + ". Check bond lengths.");
        }

        for (int i = 0, n = cinfo.get_bond_order(); i < n; i++) {
          decrement_hydrogen(covres, covatom);
        }
      }

      rec.SetChainsPerceived(true);
      rec.AddHydrogens(true);
      FOR_ATOMS_OF_MOL(a, rec) { a->GetPartialCharge(); }

      OBMol rigid;
      std::string flexstr;

      if (rec.NumResidues() > 0) {
        try {
          // Can fail with std::runtime_error if `--flex_limit` is set
          finfo.extractFlex(rec, rigid, flexstr);
        } catch (std::runtime_error &e) {
          // --flex_limit exceeded; print error and quit
          log << e.what() << "\n";
          std::exit(-1);
        }
      } else {
        // No information about residues, whole receptor treated as rigid
        rigid = rec;
      }

      std::string recstr = conv.WriteString(&rigid);
      std::stringstream recstream(recstr);

      if (flexstr.size() > 0) // have flexible component
      {
        std::stringstream flexstream(flexstr);
        initm = parse_receptor_pdbqt(rigid_name, recstream, flex_name, flexstream);
      } else { // rigid only
        initm = parse_receptor_pdbqt(rigid_name, recstream);
      }

      if (finfo.full_output()) {
        rigid.DeleteHydrogens();
        initm.set_rigid(rigid);
      }
    }
  }

  FOR_ATOMS_OF_MOL(a, covres) {
    vector3 c = a->GetVector();
    initm.extra_box_coords.push_back(vec(c.x(), c.y(), c.z()));
  }

  if (strip_hydrogens)
    initm.strip_hydrogens();
}

// setup for reading from fname
void MolGetter::setInputFile(const std::string &fname) {
  if (fname.size() > 0) { // zero if no_lig
    lpath = path(fname);
    std::string ext = lpath.extension().string();

    // Reset previous state
    rdkit_supplier.reset();
    pdbqtdone = false;

    if (ext == ".pdbqt") {
      // built-in pdbqt parsing that respects rotatable bonds in pdbqt
      type = PDBQT;
    } else if (infile.open(lpath, ".smina", true)) { // smina always gzipped
      type = SMINA;
    } else if (infile.open(lpath, ".gnina", true)) { // gnina always gzipped
      type = GNINA;
    } else if (ext == ".sdf" || ext == ".mol" || ext == ".sd") {
      // Use RDKit for SDF/MOL files - better geometry handling
      type = RDKIT_SDF;
      rdkit_supplier = std::make_unique<RDKit::SDMolSupplier>(fname, true, false, false);
    } else if (fname.length() > 0) {
      // Fall back to OpenBabel for other formats (MOL2, PDB ligands, etc.)
      // Covalent docking also uses OpenBabel path
      type = OB;
      infileopener.clear();
      infileopener.openForInput(conv, fname);
      VINA_CHECK(conv.SetOutFormat("PDBQT"));
    }

    // Covalent docking requires OpenBabel for the covalent bond manipulation
    if (cinfo.has_content() && type != OB) {
      // For covalent docking with SDF files, switch to OpenBabel
      if (type == RDKIT_SDF) {
        rdkit_supplier.reset();
        type = OB;
        infileopener.clear();
        infileopener.openForInput(conv, fname);
        VINA_CHECK(conv.SetOutFormat("PDBQT"));
      } else {
        throw usage_error("Provided ligand file format not supported with covalent docking.");
      }
    }
  }
}

// Read next molecule using RDKit and convert to model
bool MolGetter::readRDKitMoleculeIntoModel(model &m) {
  while (!rdkit_supplier->atEnd()) {
    std::unique_ptr<RDKit::ROMol> mol(rdkit_supplier->next());
    if (!mol) {
      continue;  // Failed to parse, skip to next
    }

    // Get molecule name
    std::string name;
    if (mol->hasProp("_Name")) {
      mol->getProp("_Name", name);
    }
    m.set_name(name);

    // Check for 3D coordinates
    if (mol->getNumConformers() == 0) {
      std::cerr << "\nMolecule '" << name << "' has no conformer. Skipping.\n";
      continue;
    }

    // Initialize ring info if not already done (required for rotatable bond detection)
    // SDMolSupplier may not initialize this automatically
    try {
      RDKit::RWMol rwmol(*mol);
      if (!rwmol.getRingInfo()->isInitialized()) {
        RDKit::MolOps::findSSSR(rwmol);
      }
      // Convert RDKit mol to gnina model
      if (convertRDKitMolToModel(rwmol, m, add_hydrogens, strip_hydrogens)) {
        return true;
      }
    } catch (const std::exception& e) {
      std::cerr << "\nError processing molecule '" << name << "': " << e.what() << ". Skipping.\n";
      continue;
    }
    // On failure, continue to next molecule
  }
  return false;
}

// identify a reasonable position for adding to atom a in molecule m
// if we get here, a is probably a metal and we probably need to consider
// atoms outside of the covalent residue to get the right bond vector
// The heuristic is to identify all close by atoms and negate their average
// to get a direction
static vector3 heuristic_position(model &m, OBAtom *a, double bondLength) {
  vector3 newbond;
  const double DISTSQ = 2.5 * 2.5;
  const atomv &atoms = m.get_fixed_atoms();
  vector3 avec = a->GetVector();
  vec v(avec.GetX(), avec.GetY(), avec.GetZ());
  vec sum(0, 0, 0);
  for (unsigned i = 0, n = atoms.size(); i < n; i++) {
    const atom &a = atoms[i];
    if (vec_distance_sqr(v, a.coords) < DISTSQ) {
      sum += v - a.coords;
    }
  }

  double len = sum.norm();
  if (len < 0.01) {
    // giveup and return random vector, which is no worse than what OB would do
    newbond.randomUnitVector();
  } else {
    sum /= sum.norm(); // normalize
    newbond = vector3(sum[0], sum[1], sum[2]);
  }
  newbond *= bondLength;
  newbond += a->GetVector();

  return newbond;
}

/* Covalently attach read molecule to receptor as a flexible residue. */
bool MolGetter::createCovalentMoleculeInModel(model &m) {
  // ligand is going to be made part of the receptor
  if (matchpos >= match_list.size()) { // need a new mol
    bool success = false;
    while (conv.Read(&origcovmol)) {
      std::string name = origcovmol.GetTitle();
      origcovmol.StripSalts();
      m.set_name(name);

      // apply smarts match
      match_list = cinfo.get_matches(origcovmol);
      if (match_list.size() > 0) {
        matchpos = 0;
        success = true;
        break;
      } else {
        *log << "WARNING: Ligand " << name << " did not match covalent_lig_atom_pattern. Skipping\n";
      }
    }
    if (!success) {
      return false; // done reading
    }
  }
  covmol = origcovmol;
  // covmol should be initialized and matchpos set at this point

  FOR_ATOMS_OF_MOL(a, covmol) {
    // mark atoms of covalent ligand
    OBPairData *dp = new OBPairData;
    dp->SetAttribute("CovLig");
    a->SetData(dp);
  }

  OBAtom *latom = covmol.GetAtom(match_list[matchpos][0]);
  matchpos++;

  for (int i = 0, n = cinfo.get_bond_order(); i < n; i++) {
    decrement_hydrogen(covmol, latom);
  }

  // combine residue and ligand into one molecule
  OBMol flex(covres);
  flex += covmol;
  flex.SetChainsPerceived(true);

  int ratom_index = covatom->GetIdx();
  int latom_index = covres.NumAtoms() + latom->GetIdx();

  // position ligand and add bond
  bool success = false;
  if (cinfo.keep_ligand_position()) {
    // create a bond in-place (when minimizing/scoring)
    flex.AddBond(ratom_index, latom_index, cinfo.get_bond_order());
    matchpos = match_list.size(); // single orientation
    success = true;
  } else if (cinfo.has_user_lig_atom_pos()) {
    vec pos = cinfo.lig_atom_pos(covatom, latom);
    vector3 obpos(pos[0], pos[1], pos[2]);
    success = OBBuilder::Connect(flex, ratom_index, latom_index, obpos, cinfo.get_bond_order());
  } else {
    // work around openbabel bug where it is willing to return nan for the bond vector
    auto a = flex.GetAtom(ratom_index);
    vector3 newpos = OBBuilder::GetNewBondVector(flex.GetAtom(ratom_index));
    if (!isfinite(newpos.x())) {
      a->SetHyb(4); // hacky workaround - most common offender is a metal ion
    }

    if (a->GetHyb() < a->GetExplicitDegree()) {
      // in thise case a random vector is returned, so apply another hacky workaround
      auto la = flex.GetAtom(latom_index);
      double bondLength =
          OBElements::GetCovalentRad(a->GetAtomicNum()) + OBElements::GetCovalentRad(la->GetAtomicNum());
      vector3 obpos = heuristic_position(m, a, bondLength);
      success = OBBuilder::Connect(flex, ratom_index, latom_index, obpos, cinfo.get_bond_order());
    } else { // normal case
      success = OBBuilder::Connect(flex, ratom_index, latom_index, cinfo.get_bond_order());
    }
  }
  if (!success)
    throw internal_error("Failed to connect.", __LINE__);

  flex.AddHydrogens();
  unsigned resatoms = covres.NumAtoms();

  if (cinfo.optimize_ligand()) {
    // optimize molecule a little bit, but not the residue
    OBForceField *ff = OBForceField::FindForceField("UFF");

    // we want to keep the conformation of the ligand, but the internal constraints
    // prevent the whole flex optimization from getting reasonable results, so apply
    // them in a second step
    OBFFConstraints constraints;
    for (unsigned i = 0; i < resatoms; i++) {
      constraints.AddAtomConstraint(i + 1);
    }

    if (cinfo.has_user_lig_atom_pos()) {
      // trust the user's positioning
      constraints.AddAtomConstraint(latom_index + 1);
    }

    // full ligand optimization
    ff->Setup(flex, constraints);
    ff->SteepestDescent(1000, 1e-3);
    ff->GetCoordinates(flex);
  }

  // parse with appropriate amount of rigidity
  non_rigid_parsed nrp;
  parsing_struct p;
  context c;
  pdbqt_initializer tmp;

  std::vector<int> norotate;
  norotate.reserve(resatoms);

  // if the cov res is suppose to be flexible, don't do the following
  if (!covres_isflex) {
    // at a minimum, do not fix ratom - maybe need to consider neighbors?
    std::vector<bool> fixres(resatoms, true);
    if (!cinfo.has_fixed_lig_atom())
      fixres[ratom_index - 1] = false;

    for (unsigned i = 0; i < resatoms; i++) {
      if (fixres[i])
        norotate.push_back(i + 1); // indexed by one for dumb reasons
    }
  }

  c.has_cov_lig = true;
  GninaConverter::convertParsing(flex, p, c, 1, norotate, add_hydrogens);
  // conv.WriteFile(&flex,"flex.pdbqt");
  //  create model
  postprocess_residue(nrp, p, c);
  tmp.initialize_from_nrp(nrp, c, false);
  tmp.initialize(nrp.mobility_matrix());

  if (strip_hydrogens)
    tmp.m.strip_hydrogens();

  m.append(tmp.m);

  return true;
}

// initialize model to initm and add next molecule
// return false if no molecule available;
bool MolGetter::readMoleculeIntoModel(model &m) {
  // reinit the model
  m = initm;
  switch (type) {
  case SMINA:
  case GNINA: {
    parsing_struct p;
    context c;
    unsigned torsdof = 0;
    if (!infile)
      return false;
    try {
      boost::archive::binary_iarchive serialin(infile, boost::archive::no_header | boost::archive::no_tracking);
      serialin >> torsdof;
      serialin >> p;
      serialin >> c;

      non_rigid_parsed nr;
      postprocess_ligand(nr, p, c, torsdof);
      VINA_CHECK(nr.atoms_atoms_bonds.dim() == nr.atoms.size());

      pdbqt_initializer tmp;
      tmp.initialize_from_nrp(nr, c, true);
      tmp.initialize(nr.mobility_matrix());

      if (c.sdftext.valid()) {
        // set name
        m.set_name(c.sdftext.name);
      }

      if (strip_hydrogens)
        tmp.m.strip_hydrogens();
      m.append(tmp.m);

      return true;
    } catch (boost::archive::archive_exception &e) {
      return false;
    }
  } break;

  case PDBQT: {
    if (pdbqtdone)
      return false; // can only read one
    model lig = parse_ligand_pdbqt(lpath);
    if (strip_hydrogens)
      lig.strip_hydrogens();
    m.append(lig);
    pdbqtdone = true;
    return true;
  } break;

  case RDKIT_SDF: {
    // Use RDKit for SDF files - better geometry handling, no kekulization issues
    return readRDKitMoleculeIntoModel(m);
  } break;

  case OB: {
    // OpenBabel path for covalent docking and formats not yet supported by RDKit
    if (cinfo.has_content()) {
      return createCovalentMoleculeInModel(m);
    } else { // not covalent
      OpenBabel::OBMol mol;
      while (conv.Read(&mol)) // will return after first success
      {
        std::string name = mol.GetTitle();
        mol.StripSalts();
        m.set_name(name);

        if (mol.NumAtoms() == 0) {
          std::cerr << "\nEmpty molecule " << mol.GetTitle() << ". No output will be generated for the molecule.\n";
          continue;
        }

        // Reject 2D/0D molecules (e.g., raw SMILES strings) - require 3D input or .smi files
        // The batch manager (ligand_batch_manager.cpp) uses RDKit with proper MMFF minimization
        // for .smi files, which produces much better geometry than OpenBabel's UFF
        if (mol.GetDimension() < 3) {
          std::cerr << "\nError: Molecule '" << mol.GetTitle() << "' has no 3D coordinates.\n"
                    << "Raw SMILES input is no longer supported due to poor geometry quality.\n"
                    << "Please use one of these alternatives:\n"
                    << "  1. Save SMILES to a .smi file and use: -l ligands.smi\n"
                    << "  2. Use a 3D structure file (SDF, MOL2, PDB)\n"
                    << "The .smi file path uses RDKit with MMFF minimization for better geometry.\n";
          continue;
        }

        if (convertOBMolToModel(mol, m, add_hydrogens, strip_hydrogens)) {
          return true;
        }
        // On failure, continue to next molecule
      }
    }

    return false; // no valid molecules read
  } break;

  case NONE:
    return true; // nolig
    break;
  }
#ifndef __NVCC__
  return false; // shouldn't get here
#endif
}
