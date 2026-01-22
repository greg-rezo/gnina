#include <boost/program_options.hpp>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/assign.hpp>
#include <boost/bind/bind.hpp>
#include <boost/filesystem/exception.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/iostreams/device/null.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp> // hardware_concurrency // FIXME rm ?
#include <boost/thread/thread.hpp>
#include <boost/timer/timer.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <cmath> // for ceila
#include <exception>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <openbabel/babelconfig.h>
#include <openbabel/mol.h>
#include <openbabel/obconversion.h>
#include <openbabel/parsmart.h>
#include <string>
#include <torch/torch.h>
#include <vector> // ligand paths
#include <set>

// RDKit for symmetry-aware RMSD calculation
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/MolAlign/AlignMolecules.h>
#include <GraphMol/FileParsers/MolSupplier.h>

#include "array3d.h"
#include "box.h"
#include "builtinscoring.h"
#include "cache.h"
#include "cache_gpu.h"
#include "cnn_torch_scorer.h"
#include "coords.h"
#include "covinfo.h"
#include "custom_terms.h"
#include "dl_scorer.h"
#include "everything.h"
#include "file.h"
#include "flexinfo.h"
#include "gpucode.h"
#include "bfgs_parallel.h"
#include "ligand_batch_manager.h"
#include "grid.h"
#include "molgetter.h"
#include "naive_non_cache.h"
#include "non_cache.h"
#include "non_cache_cnn.h"
#include "obmolopener.h"
#include "parallel_mc.h"
#include "parse_error.h"
#include "parse_pdbqt.h"
#include "precalculate_gpu.h"
#include "quasi_newton.h"
#include "result_info.h"
#include "sem.h"
#include "tee.h"
#include "torch_models.h"
#include "user_opts.h"
#include "version.h"
#include "weighted_terms.h"

#include <cuda_profiler_api.h>

// Note: NOT using namespace boost::iostreams to avoid conflict with local tee class
using boost::filesystem::path;

void doing(int verbosity, const std::string &str, tee &log) {
  if (verbosity > 1) {
    log << str << std::string(" ... ");
    log.flush();
  }
}

void done(int verbosity, tee &log) {
  if (verbosity > 1) {
    log << "done.";
    log.endl();
  }
}
std::string default_output(const std::string &input_name) {
  std::string tmp = input_name;
  if (tmp.size() >= 6 && tmp.substr(tmp.size() - 6, 6) == ".pdbqt")
    tmp.resize(tmp.size() - 6); // FIXME?
  return tmp + "_out.pdbqt";
}

void write_all_output(model &m, const output_container &out, sz how_many, std::ostream &outstream) {
  if (out.size() < how_many)
    how_many = out.size();
  VINA_FOR(i, how_many) {
    m.set(out[i].c);
    m.write_model(outstream, i + 1); // so that model numbers start with 1
  }
}

// set m to a random conformer
fl do_randomization(model &m, const vec &corner1, const vec &corner2, int seed, int verbosity, tee &log) {
  conf init_conf = m.get_initial_conf(false);
  rng generator(static_cast<rng::result_type>(seed));
  if (verbosity > 1) {
    log << "Using random seed: " << seed;
    log.endl();
  }
  const sz attempts = 100;
  conf best_conf = init_conf;
  fl best_clash_penalty = 0;
  VINA_FOR(i, attempts) {
    conf c = init_conf;
    c.randomize(corner1, corner2, generator);
    m.set(c);
    fl penalty = m.clash_penalty();
    if (i == 0 || penalty < best_clash_penalty) {
      best_conf = c;
      best_clash_penalty = penalty;
      if (penalty == 0)
        break;
    }
  }
  m.set(best_conf);
  if (verbosity > 1) {
    log << "Clash penalty: " << best_clash_penalty; // FIXME rm?
    log.endl();
  }
  return best_clash_penalty;
}

void refine_structure(model &m, const precalculate &prec, igrid &ig, output_type &out, const vec &cap,
                      const minimization_params &minparm, grid &user_grid, int verbosity, tee &log, igrid &ig_new) {
  // Debug: check within status
  std::cerr << "DEBUG refine_structure: within before adjust=" << ig.within(m)
            << " coords[0]=(" << m.coords[0][0] << "," << m.coords[0][1] << "," << m.coords[0][2] << ")\n";

  change g(m.get_size(), ig.move_receptor());

  ig.adjust_center(m); // for cnn, set cnn box

  std::cerr << "DEBUG refine_structure: within after adjust=" << ig.within(m) << "\n";

  if (!ig.within(m)) {
    std::cout << m.get_name() << " | pose " << m.get_pose_num() << " | initial pose not within box\n";
  }
  quasi_newton quasi_newton_par(minparm);
  const fl slope_orig = ig.getSlope();
  // try 5 times to get ligand into box
  // dkoes - you don't need a very strong constraint to keep ligands in the box,
  // but this factor can really bias the energy landscape
  fl slope = 10;
  VINA_FOR(p, 5) {
    ig.setSlope(slope);
    quasi_newton_par(m, prec, ig, out, g, cap, user_grid); // quasi_newton operator
    m.set(out.c);                                          // just to be sure
    if (ig.within(m)) {
      break;
    }
    std::cout << m.get_name() << " | pose " << m.get_pose_num() << " | ligand outside box\n";
    slope *= 10;
  }
  out.coords = m.get_heavy_atom_movable_coords();
  if (!ig.within(m))
    out.e = max_fl;
  ig.setSlope(slope_orig);
  if (verbosity > 1) {
    // log total and empirical energy, useful for testing CNN + empirical merge
    log.endl();
    fl final_e = ig.eval_deriv(m, cap[1], user_grid);
    log << "Total energy after refinement: " << std::fixed << std::setprecision(5) << final_e;
    log.endl();
    // non_cache::eval for empirical energy
    fl final_emp_e = ig_new.eval(m, cap[1]);
    log << "Empirical energy after refinement: " << std::fixed << std::setprecision(5) << final_emp_e;
    log.endl();
  }
}

std::string vina_remark(fl e, fl lb, fl ub) {
  std::ostringstream remark;
  remark.setf(std::ios::fixed, std::ios::floatfield);
  remark.setf(std::ios::showpoint);
  remark << "REMARK VINA RESULT: " << std::setw(9) << std::setprecision(1) << e << "  " << std::setw(9)
         << std::setprecision(3) << lb << "  " << std::setw(9) << std::setprecision(3) << ub << '\n';
  return remark.str();
}

// Compute symmetry-aware RMSD between two coordinate sets using RDKit
// Returns max_fl if calculation fails
static fl compute_symmetry_rmsd(const vecv& coords_a, const vecv& coords_b,
                                 const std::shared_ptr<RDKit::ROMol>& rdkit_mol) {
  if (!rdkit_mol || coords_a.size() != coords_b.size() ||
      coords_a.size() != rdkit_mol->getNumAtoms()) {
    // Fall back to naive RMSD
    return rmsd_upper_bound(coords_a, coords_b);
  }

  try {
    // Create probe molecule with coords_a
    RDKit::RWMol probe_mol(*rdkit_mol);
    RDKit::Conformer& probe_conf = probe_mol.getConformer();
    for (size_t i = 0; i < coords_a.size(); i++) {
      probe_conf.setAtomPos(i, RDGeom::Point3D(coords_a[i][0], coords_a[i][1], coords_a[i][2]));
    }

    // Create reference molecule with coords_b
    RDKit::RWMol ref_mol(*rdkit_mol);
    RDKit::Conformer& ref_conf = ref_mol.getConformer();
    for (size_t i = 0; i < coords_b.size(); i++) {
      ref_conf.setAtomPos(i, RDGeom::Point3D(coords_b[i][0], coords_b[i][1], coords_b[i][2]));
    }

    // Use getBestRMS for symmetry-aware comparison
    double rmsd = RDKit::MolAlign::getBestRMS(probe_mol, ref_mol);
    return static_cast<fl>(rmsd);
  } catch (...) {
    // Fall back to naive RMSD
    return rmsd_upper_bound(coords_a, coords_b);
  }
}

// Find closest pose in container using symmetry-aware RMSD
static std::pair<sz, fl> find_closest_symmetry(const vecv& coords, const output_container& out,
                                                const std::shared_ptr<RDKit::ROMol>& rdkit_mol) {
  std::pair<sz, fl> result(out.size(), max_fl);
  VINA_FOR_IN(i, out) {
    fl rmsd = compute_symmetry_rmsd(coords, out[i].coords, rdkit_mol);
    if (i == 0 || rmsd < result.second) {
      result = std::make_pair(i, rmsd);
    }
  }
  return result;
}

// Remove redundant poses using symmetry-aware RMSD
output_container remove_redundant(const output_container& in, fl min_rmsd,
                                   const std::shared_ptr<RDKit::ROMol>& rdkit_mol) {
  output_container tmp;

  VINA_FOR_IN(i, in) {
    std::pair<sz, fl> closest_rmsd = find_closest_symmetry(in[i].coords, tmp, rdkit_mol);
    if (closest_rmsd.first >= tmp.size() || closest_rmsd.second > min_rmsd) {
      tmp.push_back(new output_type(in[i])); // not redundant
    }
  }
  return tmp;
}

// Legacy version without RDKit (for compatibility)
output_container remove_redundant(const output_container &in, fl min_rmsd) {
  output_container tmp;

  VINA_FOR_IN(i, in) {
    std::pair<sz, fl> closest_rmsd = find_closest(in[i].coords, tmp);
    if (closest_rmsd.first >= tmp.size() || closest_rmsd.second > min_rmsd) {
      tmp.push_back(new output_type(in[i])); // not redundant
    }
  }
  return tmp;
}

// print info to log about cnn scoring
static void get_cnn_info(model &m, DLScorer &cnn, tee &log, float &cnnscore, float &cnnaffinity, float &cnnvariance) {
  float loss = 0;
  cnnscore = 0;
  cnnaffinity = 0;
  cnnscore = cnn.score(m, false, cnnaffinity, loss, cnnvariance);

  if (cnn.options().verbose) {
    log << "CNNscore: " << std::fixed << std::setprecision(10) << cnnscore;
    log.endl();
    log << "CNNaffinity: " << std::fixed << std::setprecision(10) << cnnaffinity;
    log.endl();
  }
}

// Reference data for symmetry-aware RMSD calculation using RDKit
struct reference_data {
  vecv coords;
  std::shared_ptr<RDKit::ROMol> rdkit_mol;
};

// Compute symmetry-aware RMSD between current model pose and reference coordinates
// Uses RDKit's getBestRMS which handles molecular symmetry via graph automorphisms
// Returns -1 if reference is not available or calculation fails
static fl compute_reference_rmsd(const model &m, const boost::optional<reference_data> &ref) {
  if (!ref || !ref->rdkit_mol || ref->coords.empty()) return -1;

  try {
    // Get current heavy atom coordinates from model
    vecv current_coords;
    const atomv& atoms = m.get_movable_atoms();
    const vecv& coords = m.coordinates();
    for (size_t i = 0; i < m.num_movable_atoms(); i++) {
      if (!atoms[i].is_hydrogen()) {
        current_coords.push_back(coords[i]);
      }
    }

    if (current_coords.size() != ref->coords.size()) return -1;
    if (current_coords.size() != ref->rdkit_mol->getNumAtoms()) return -1;

    // Create two molecules: one with reference coords, one with current coords
    RDKit::RWMol ref_mol(*ref->rdkit_mol);
    RDKit::RWMol probe_mol(*ref->rdkit_mol);

    // Set reference molecule coordinates from stored ref->coords
    RDKit::Conformer &ref_conf = ref_mol.getConformer();
    for (size_t i = 0; i < ref->coords.size(); i++) {
      ref_conf.setAtomPos(i, RDGeom::Point3D(ref->coords[i][0],
                                              ref->coords[i][1],
                                              ref->coords[i][2]));
    }

    // Set probe molecule coordinates with current pose
    RDKit::Conformer &probe_conf = probe_mol.getConformer();
    for (size_t i = 0; i < current_coords.size(); i++) {
      probe_conf.setAtomPos(i, RDGeom::Point3D(current_coords[i][0],
                                                current_coords[i][1],
                                                current_coords[i][2]));
    }

    // Use getBestRMS which accounts for molecular symmetry
    // It finds the optimal atom mapping that minimizes RMSD
    double rmsd = RDKit::MolAlign::getBestRMS(probe_mol, ref_mol);
    return static_cast<fl>(rmsd);
  } catch (...) {
    return -1;
  }
}

// dkoes - return all energies and rmsds to original conf with result
void do_search(model &m, const boost::optional<model> &ref, const boost::optional<reference_data> &ref_data,
               const weighted_terms &sf, const precalculate &prec,
               igrid &ig,
               non_cache &nc, // nc.slope is changed
               const vec &corner1, const vec &corner2, const parallel_mc &par, const user_settings &settings,
               bool compute_atominfo, tee &log, const terms *t, grid &user_grid, DLScorer &cnn,
               std::vector<result_info> &results, szv_grid_cache &grid_cache, const grid_dims &gd, fl slope) {
  boost::timer::cpu_timer time;
  try {

    precalculate_exact exact_prec(sf); // use exact computations for final score
    conf_size s = m.get_size();
    conf c = m.get_initial_conf(nc.move_receptor());
    fl e = max_fl;
    fl intramolecular_energy = max_fl;
    fl cnnscore = 0, cnnaffinity = 0, cnnvariance = 0;
    fl rmsd = 0;
    if (settings.gpu)
      m.initialize_gpu();
    const vec authentic_v(settings.forcecap, settings.forcecap,
                          settings.forcecap); // small cap restricts initial movement from clash

    cnn.set_center_from_model(m);
    non_cache nc_new = non_cache(grid_cache, gd, &prec, slope);
    if (settings.gpu && settings.score_only) {
      // GPU score-only mode
      cache_gpu* cgpu = dynamic_cast<cache_gpu*>(&ig);
      if (!cgpu) {
        log << "ERROR: --gpu --score_only requires grid caching.\n";
        log.endl();
        throw std::runtime_error("--gpu --score_only requires grid caching");
      }

      doing(settings.verbosity, "Performing GPU scoring", log);

      const GPUCacheInfo& cacheInfo = cgpu->get_info();

      // Convert conf to flat array format
      unsigned nlig_roots = 1;
      unsigned n_torsions = 0;
      if (!c.ligands.empty()) {
        n_torsions = c.ligands[0].torsions.size();
      }
      int conf_size = 7 * nlig_roots + n_torsions;

      std::vector<float> input_conf(conf_size);
      conf_to_flat(c, nlig_roots, n_torsions, input_conf.data(), m.gdata);

      // Run GPU score-only
      float gpu_inter_energy, gpu_intra_energy;
      run_gpu_score_only(
          m.gdata, cacheInfo,
          input_conf,
          gpu_inter_energy, gpu_intra_energy
      );

      done(settings.verbosity, log);

      // Compute adjusted energy like CPU version
      intramolecular_energy = gpu_intra_energy;
      e = gpu_inter_energy + gpu_intra_energy;

      // Apply scoring function adjustment
      e = sf.conf_independent(m, e);

      get_cnn_info(m, cnn, log, cnnscore, cnnaffinity, cnnvariance);

      log << "Affinity: " << std::fixed << std::setprecision(5) << e << " (kcal/mol)\n";

      log << "CNNscore: " << std::fixed << std::setprecision(5) << cnnscore << " "
          << "\nCNNaffinity: " << cnnaffinity;
      if (cnnvariance > 0) {
        log << "\nCNNvariance: " << std::fixed << std::setprecision(5) << cnnvariance;
      }
      log.endl();

      log << "Intramolecular energy: " << std::fixed << std::setprecision(5) << intramolecular_energy << "\n";

      results.push_back(result_info(e, cnnscore, cnnaffinity, cnnvariance, -1, compute_reference_rmsd(m, ref_data), m));

      if (compute_atominfo)
        results.back().setAtomValues(m, &sf);
    } else if (settings.score_only) {
      intramolecular_energy = m.eval_intramolecular(exact_prec, authentic_v, c);
      naive_non_cache nnc(&exact_prec); // for out of grid issues
      e = m.eval_adjusted(sf, exact_prec, nnc, authentic_v, c, intramolecular_energy, user_grid);
      get_cnn_info(m, cnn, log, cnnscore, cnnaffinity, cnnvariance);

      log << "Affinity: " << std::fixed << std::setprecision(5) << e << " (kcal/mol)\n";

      log << "CNNscore: " << std::fixed << std::setprecision(5) << cnnscore << " "
          << "\nCNNaffinity: " << cnnaffinity;
      if (cnnvariance > 0) {
        log << "\nCNNvariance: " << std::fixed << std::setprecision(5) << cnnvariance;
      }
      log.endl();

      std::vector<flv> atominfo;
      flv term_values = t->evale_robust(m);
      log << "Intramolecular energy: " << std::fixed << std::setprecision(5) << intramolecular_energy << "\n";

      log << "Term values, before weighting:\n";
      log << std::setprecision(5);
      log << "## " << boost::replace_all_copy(m.get_name(), " ", "_");

      VINA_FOR_IN(i, term_values) { log << ' ' << term_values[i]; }

      conf_independent_inputs in(m);
      const flv nonweight(1, 1.0);
      for (unsigned i = 0, n = t->conf_independent_terms.size(); i < n; i++) {
        flv::const_iterator pos = nonweight.begin();
        log << " " << t->conf_independent_terms[i].eval(in, (fl)0.0, pos);
      }
      log << '\n';

      results.push_back(result_info(e, cnnscore, cnnaffinity, cnnvariance, -1, compute_reference_rmsd(m, ref_data), m));

      if (compute_atominfo)
        results.back().setAtomValues(m, &sf);
    } else if (settings.gpu && settings.local_only) {
      // Parallel BFGS minimize from input pose
      cache_gpu* cgpu = dynamic_cast<cache_gpu*>(&ig);
      if (!cgpu) {
        log << "ERROR: --gpu --local_only requires grid caching.\n";
        log.endl();
        throw std::runtime_error("--gpu --local_only requires grid caching");
      }

      vecv origcoords = m.get_heavy_atom_movable_coords();

      // Set slope=10 to match CPU refine_structure behavior
      // (CPU uses slope=10 initially, increasing if ligand outside box)
      cgpu->set_slope(10);

      log << "Running parallel BFGS minimize (iterations=" << settings.bfgs_iterations << ")\n";
      log.endl();

      doing(settings.verbosity, "Performing parallel BFGS local search", log);

      const GPUCacheInfo& cacheInfo = cgpu->get_info();

      // Convert conf to flat array format
      unsigned nlig_roots = 1;  // Typically 1 for single ligand
      unsigned n_torsions = 0;
      if (!c.ligands.empty()) {
        n_torsions = c.ligands[0].torsions.size();
      }
      int conf_size = 7 * nlig_roots + n_torsions;

      std::vector<float> input_conf(conf_size);
      conf_to_flat(c, nlig_roots, n_torsions, input_conf.data(), m.gdata);

      // Run parallel BFGS minimize
      float out_energy, out_intramolecular;
      std::vector<float> out_conf;
      std::vector<float> out_gradient;

      run_parallel_bfgs_minimize(
          m.gdata, cacheInfo,
          input_conf,
          settings.bfgs_iterations,
          out_energy, out_intramolecular,
          out_conf,
          settings.verbose_grad ? &out_gradient : nullptr
      );

      done(settings.verbosity, log);

      // Output gradient if verbose_grad is set
      // Convert GPU gradient from BFS to DFS order for comparison with CPU
      if (settings.verbose_grad && !out_gradient.empty()) {
        log << "\nGPU Gradient (" << out_gradient.size() << " DOF) - converted to DFS order:\n";
        log << std::fixed << std::setprecision(8);

        // First output rigid body gradients (position + orientation)
        sz idx = 0;
        for (unsigned i = 0; i < nlig_roots; i++) {
          for (unsigned j = 0; j < 3; j++) {
            log << "  g[" << idx++ << "] = " << out_gradient[i * 6 + j] << "  (pos" << j << ")\n";
          }
          for (unsigned j = 0; j < 3; j++) {
            log << "  g[" << idx++ << "] = " << out_gradient[i * 6 + 3 + j] << "  (ori" << j << ")\n";
          }
        }

        // Convert torsion gradients from BFS to DFS order
        for (unsigned dfs_torsion_idx = 0; dfs_torsion_idx < n_torsions; dfs_torsion_idx++) {
          unsigned dfs_node_idx = dfs_torsion_idx + nlig_roots;
          unsigned bfs_node_idx = m.gdata.dfs_order_bfs_indices[dfs_node_idx];
          unsigned bfs_torsion_idx = bfs_node_idx - nlig_roots;
          log << "  g[" << idx++ << "] = " << out_gradient[6 * nlig_roots + bfs_torsion_idx] << "  (tor" << dfs_torsion_idx << ")\n";
        }
        log.endl();
      }

      // Convert result back to conf
      flat_to_conf(out_conf.data(), nlig_roots, n_torsions, c, m.gdata);
      m.set(c);

      // Compute proper energies using model for accurate reporting
      naive_non_cache nnc(&exact_prec);
      fl intramolecular_energy = m.eval_intramolecular(exact_prec, authentic_v, c);
      e = m.eval_adjusted(sf, exact_prec, nnc, authentic_v, c, intramolecular_energy, user_grid);

      get_cnn_info(m, cnn, log, cnnscore, cnnaffinity, cnnvariance);

      vecv newcoords = m.get_heavy_atom_movable_coords();
      assert(newcoords.size() == origcoords.size());
      for (unsigned i = 0, n = newcoords.size(); i < n; i++) {
        rmsd += (newcoords[i] - origcoords[i]).norm_sqr();
      }
      rmsd /= newcoords.size();
      rmsd = sqrt(rmsd);

      log << "Affinity: " << std::fixed << std::setprecision(5) << e << "  " << intramolecular_energy
          << " (kcal/mol)\nRMSD: " << rmsd << "\n";
      log << "CNNscore: " << std::fixed << std::setprecision(5) << cnnscore << " "
          << "\nCNNaffinity: " << cnnaffinity;
      if (cnnvariance > 0) {
        log << "\nCNNvariance: " << std::fixed << std::setprecision(5) << cnnvariance;
      }
      log.endl();

      if (!nc.within(m))
        log << "WARNING: not all movable atoms are within the search space\n";

      done(settings.verbosity, log);
      results.push_back(result_info(e, cnnscore, cnnaffinity, cnnvariance, rmsd, compute_reference_rmsd(m, ref_data), m));

      if (compute_atominfo)
        results.back().setAtomValues(m, &sf);
    } else if (settings.local_only) {
      standard_local_only:
      vecv origcoords = m.get_heavy_atom_movable_coords();
      output_type out(c, e);

      // Output INITIAL gradient if verbose_grad is set (before optimization)
      if (settings.verbose_grad) {
        change g(m.get_size(), false);
        fl grad_e;
        if (settings.cpu_grid) {
          grad_e = m.eval_deriv(prec, ig, authentic_v, c, g, user_grid);
        } else {
          grad_e = m.eval_deriv(prec, nc, authentic_v, c, g, user_grid);
        }
        log << "\nCPU Initial Gradient (energy=" << std::fixed << std::setprecision(5) << grad_e << "):\n";
        log << std::fixed << std::setprecision(8);
        sz idx = 0;
        for (sz i = 0; i < g.ligands.size(); i++) {
          const ligand_change& lig = g.ligands[i];
          for (sz j = 0; j < 3; j++) {
            log << "  g[" << idx++ << "] = " << lig.rigid.position[j] << "  (pos" << j << ")\n";
          }
          for (sz j = 0; j < 3; j++) {
            log << "  g[" << idx++ << "] = " << lig.rigid.orientation[j] << "  (ori" << j << ")\n";
          }
          for (sz j = 0; j < lig.torsions.size(); j++) {
            log << "  g[" << idx++ << "] = " << lig.torsions[j] << "  (tor" << j << ")\n";
          }
        }
        log.endl();
      }

      doing(settings.verbosity, "Performing local search", log);
      // Use grid cache (ig) when cpu_grid is set, otherwise use pairwise (nc)
      if (settings.cpu_grid) {
        refine_structure(m, prec, ig, out, authentic_v, par.mc.ssd_par.minparm, user_grid, settings.verbosity, log, ig);
      } else {
        refine_structure(m, prec, nc, out, authentic_v, par.mc.ssd_par.minparm, user_grid, settings.verbosity, log, nc_new);
      }
      done(settings.verbosity, log);
      m.set(out.c);

      // Output FINAL gradient if verbose_grad is set (after optimization)
      if (settings.verbose_grad) {
        change g(m.get_size(), false);
        fl grad_e;
        if (settings.cpu_grid) {
          grad_e = m.eval_deriv(prec, ig, authentic_v, out.c, g, user_grid);
        } else {
          grad_e = m.eval_deriv(prec, nc, authentic_v, out.c, g, user_grid);
        }
        log << "\nCPU Final Gradient (energy=" << std::fixed << std::setprecision(5) << grad_e << "):\n";
        log << std::fixed << std::setprecision(8);
        // Output gradient values - ligand has position (3) + orientation (3) + torsions
        sz idx = 0;
        for (sz i = 0; i < g.ligands.size(); i++) {
          const ligand_change& lig = g.ligands[i];
          for (sz j = 0; j < 3; j++) {
            log << "  g[" << idx++ << "] = " << lig.rigid.position[j] << "  (pos" << j << ")\n";
          }
          for (sz j = 0; j < 3; j++) {
            log << "  g[" << idx++ << "] = " << lig.rigid.orientation[j] << "  (ori" << j << ")\n";
          }
          for (sz j = 0; j < lig.torsions.size(); j++) {
            log << "  g[" << idx++ << "] = " << lig.torsions[j] << "  (tor" << j << ")\n";
          }
        }
        log.endl();
      }

      // be as exact as possible for final score
      naive_non_cache nnc(&exact_prec); // for out of grid issues

      fl intramolecular_energy = m.eval_intramolecular(exact_prec, authentic_v, out.c);
      e = m.eval_adjusted(sf, exact_prec, nnc, authentic_v, out.c, intramolecular_energy, user_grid);

      // reset the center after last call to set
      get_cnn_info(m, cnn, log, cnnscore, cnnaffinity, cnnvariance);

      vecv newcoords = m.get_heavy_atom_movable_coords();
      assert(newcoords.size() == origcoords.size());
      for (unsigned i = 0, n = newcoords.size(); i < n; i++) {
        rmsd += (newcoords[i] - origcoords[i]).norm_sqr();
      }
      rmsd /= newcoords.size();
      rmsd = sqrt(rmsd);
      log << "Affinity: " << std::fixed << std::setprecision(5) << e << "  " << intramolecular_energy
          << " (kcal/mol)\nRMSD: " << rmsd << "\n";
      log << "CNNscore: " << std::fixed << std::setprecision(5) << cnnscore << " "
          << "\nCNNaffinity: " << cnnaffinity;
      if (cnnvariance > 0) {
        log << "\nCNNvariance: " << std::fixed << std::setprecision(5) << cnnvariance;
      }
      log.endl();

      if (!nc.within(m))
        log << "WARNING: not all movable atoms are within the search space\n";

      done(settings.verbosity, log);
      results.push_back(result_info(e, cnnscore, cnnaffinity, cnnvariance, rmsd, compute_reference_rmsd(m, ref_data), m));

      if (compute_atominfo)
        results.back().setAtomValues(m, &sf);
    } else if (settings.gpu) {
      // Parallel BFGS without Monte Carlo
      // Try to get cache_gpu info
      cache_gpu* cgpu = dynamic_cast<cache_gpu*>(&ig);
      if (!cgpu) {
        log << "ERROR: --gpu requires grid caching. Cannot use with no_cache mode.\n";
        log.endl();
        throw std::runtime_error("--gpu requires grid caching");
      }

      log << "Running parallel BFGS (exhaustiveness=" << settings.exhaustiveness
          << ", bfgs_iterations=" << settings.bfgs_iterations << ")\n";
      log << "Using random seed: " << settings.seed;
      log.endl();

      doing(settings.verbosity, "Running parallel BFGS optimization", log);

      // Get cache info and gpu data
      const GPUCacheInfo& cacheInfo = cgpu->get_info();

      // Compute box bounds
      gfloat3 box_min(corner1[0], corner1[1], corner1[2]);
      gfloat3 box_max(corner2[0], corner2[1], corner2[2]);

      // Extract receptor atoms for direct pairwise mode
      std::vector<float> receptor_coords;
      std::vector<uint8_t> receptor_types;
      if (settings.direct_pairwise) {
          const atomv& rec_atoms = m.get_fixed_atoms();
          receptor_coords.reserve(rec_atoms.size() * 3);
          receptor_types.reserve(rec_atoms.size());
          for (const auto& atom : rec_atoms) {
              receptor_coords.push_back(atom.coords[0]);
              receptor_coords.push_back(atom.coords[1]);
              receptor_coords.push_back(atom.coords[2]);
              receptor_types.push_back(static_cast<uint8_t>(atom.sm));
          }
      }

      // Run parallel BFGS
      std::vector<float> energies;
      std::vector<std::vector<float>> conformations;
      run_parallel_bfgs_docking(
              m.gdata, cacheInfo,
              settings.exhaustiveness,
              settings.bfgs_iterations,
              box_min, box_max,
              settings.seed,
              energies, conformations,
              settings.verbosity,
              settings.direct_pairwise,
              settings.direct_pairwise ? &receptor_coords : nullptr,
              settings.direct_pairwise ? &receptor_types : nullptr
          );

      done(settings.verbosity, log);

      // Convert results to output_container
      output_container out_cont;
      for (size_t i = 0; i < energies.size(); i++) {
        if (energies[i] < 1e10 && std::isfinite(energies[i])) {  // Filter out failed optimizations
          // Validate conformation data
          if (conformations[i].empty() || conformations[i].size() < 7) {
            continue;  // Skip invalid conformations
          }

          // Check for NaN/Inf in conformation
          bool valid = true;
          for (size_t j = 0; j < conformations[i].size() && valid; j++) {
            if (!std::isfinite(conformations[i][j])) {
              valid = false;
            }
          }
          if (!valid) {
            continue;  // Skip conformations with NaN/Inf
          }

          // Check if position is within reasonable bounds (10x box size)
          float margin = 100.0f;  // Allow some margin outside box
          if (conformations[i][0] < box_min.x - margin || conformations[i][0] > box_max.x + margin ||
              conformations[i][1] < box_min.y - margin || conformations[i][1] > box_max.y + margin ||
              conformations[i][2] < box_min.z - margin || conformations[i][2] > box_max.z + margin) {
            continue;  // Skip poses far outside the box
          }

          conf c_result = m.get_initial_conf(nc.move_receptor());
          // Copy conformation data back
          // Note: conformations[i] contains [position, quaternion, torsions]
          if (c_result.ligands.empty()) {
            continue;  // Skip if no ligands
          }

          // Set position
          c_result.ligands[0].rigid.position[0] = conformations[i][0];
          c_result.ligands[0].rigid.position[1] = conformations[i][1];
          c_result.ligands[0].rigid.position[2] = conformations[i][2];
          // Set orientation quaternion
          c_result.ligands[0].rigid.orientation = qt(
              conformations[i][3], conformations[i][4],
              conformations[i][5], conformations[i][6]);
          // Set torsions
          for (size_t t = 0; t < c_result.ligands[0].torsions.size() && (t + 7) < conformations[i].size(); t++) {
            c_result.ligands[0].torsions[t] = conformations[i][t + 7];
          }

          output_type out_result(c_result, energies[i]);
          m.set(out_result.c);
          out_result.coords = m.get_heavy_atom_movable_coords();
          // Don't cluster by RMSD yet - just collect all valid poses
          // Clustering should happen after refinement/scoring when we know final scores
          out_cont.push_back(new output_type(out_result));
        }
      }

      // Debug: count poses by energy range
      if (settings.verbosity >= 1) {
        int n_invalid = 0, n_positive = 0, n_negative = 0;
        float min_e = 1e10, max_e = -1e10;
        for (size_t i = 0; i < energies.size(); i++) {
          if (energies[i] >= 1e10 || !std::isfinite(energies[i])) {
            n_invalid++;
          } else {
            if (energies[i] < min_e) min_e = energies[i];
            if (energies[i] > max_e) max_e = energies[i];
            if (energies[i] > 0) n_positive++;
            else n_negative++;
          }
        }
        log << "GPU BFGS pose stats: " << energies.size() << " total, "
            << n_invalid << " invalid (>=1e10), "
            << n_positive << " positive, " << n_negative << " negative\n";
        log << "  Energy range: [" << min_e << ", " << max_e << "]\n";
        log << "  Valid poses after filtering: " << out_cont.size() << "\n";
      }

      // Sort by energy first to limit refinement to best initial poses
      out_cont.sort();  // Default sort is by energy

      // Limit to reasonable number for refinement
      sz max_to_refine = settings.num_modes * settings.cnn_refine_mult;  // Refine more than we need, then cluster
      while (out_cont.size() > max_to_refine) {
        out_cont.pop_back();
      }

      // Refine and score results
      doing(settings.verbosity, "Refining results", log);
      VINA_FOR_IN(i, out_cont) {
        refine_structure(m, prec, nc, out_cont[i], authentic_v, par.mc.ssd_par.minparm, user_grid, settings.verbosity,
                         log, nc_new);
        get_cnn_info(m, cnn, log, cnnscore, cnnaffinity, cnnvariance);
        out_cont[i].cnnscore = cnnscore;
        out_cont[i].cnnaffinity = cnnaffinity;
        out_cont[i].cnnvariance = cnnvariance;

        if (not_max(out_cont[i].e)) {
          intramolecular_energy = m.eval_intramolecular(exact_prec, authentic_v, out_cont[i].c);
          out_cont[i].e =
              m.eval_adjusted(sf, exact_prec, nc_new, authentic_v, out_cont[i].c, intramolecular_energy, user_grid);
          out_cont[i].intramol = intramolecular_energy;
        }
      }

      auto sorter = [settings](const output_type &lhs, const output_type &rhs) {
        switch (settings.sort_order) {
        case Energy:
          return lhs.e < rhs.e;
        case CNNaffinity:
          return lhs.cnnaffinity > rhs.cnnaffinity;
        case CNNscore:
        default:
          return lhs.cnnscore > rhs.cnnscore;
        }
      };

      // Sort by user-specified metric BEFORE clustering
      out_cont.sort(sorter);

      // Now cluster by RMSD - this keeps the best-scoring pose from each cluster
      // Use symmetry-aware RMSD for clustering if RDKit mol is available
      sz before_cluster = out_cont.size();
      if (ref_data && ref_data->rdkit_mol) {
        out_cont = remove_redundant(out_cont, settings.out_min_rmsd, ref_data->rdkit_mol);
      } else {
        out_cont = remove_redundant(out_cont, settings.out_min_rmsd);
      }
      if (settings.verbosity >= 1) {
        log << "Clustering: " << before_cluster << " -> " << out_cont.size() << " unique poses (min_rmsd=" << settings.out_min_rmsd << ")\n";
      }
      done(settings.verbosity, log);

      log.setf(std::ios::fixed, std::ios::floatfield);
      log.setf(std::ios::showpoint);
      log << '\n';
      log << "mode |  affinity  |  intramol  |    CNN     |   CNN\n";
      log << "     | (kcal/mol) | (kcal/mol) | pose score | affinity\n";
      log << "-----+------------+------------+------------+----------\n";

      model best_mode_model = m;
      if (!out_cont.empty())
        best_mode_model.set(out_cont.front().c);

      sz how_many = 0;
      VINA_FOR_IN(i, out_cont) {
        if (!not_max(out_cont[i].e))
          continue;
        if (how_many >= settings.num_modes)
          break;
        ++how_many;
        m.set(out_cont[i].c);
        log << std::setw(5) << how_many << std::setw(12) << std::setprecision(2) << out_cont[i].e << std::setw(12)
            << std::setprecision(2) << out_cont[i].intramol;
        log << " " << std::setw(12) << std::setprecision(4) << out_cont[i].cnnscore << "  " << std::setw(9)
            << std::setprecision(3) << out_cont[i].cnnaffinity;
        log.endl();

        results.push_back(
            result_info(out_cont[i].e, out_cont[i].cnnscore, out_cont[i].cnnaffinity, out_cont[i].cnnvariance, -1, compute_reference_rmsd(m, ref_data), m));

        if (compute_atominfo)
          results.back().setAtomValues(m, &sf);
      }
      done(settings.verbosity, log);

      if (how_many < 1) {
        log << "WARNING: Could not find any conformations completely within the search space.\n";
        log.endl();
      }
    } else { // docking
    standard_docking:
      rng generator(static_cast<rng::result_type>(settings.seed));
      log << "Using random seed: " << settings.seed;
      log.endl();

      output_container out_cont;
      doing(settings.verbosity, "Performing search", log);
      par(m, out_cont, prec, ig, corner1, corner2, generator, user_grid, nc);
      done(settings.verbosity, log);
      doing(settings.verbosity, "Refining results", log);

      VINA_FOR_IN(i, out_cont) {
        if (settings.cnnopts.cnn_scoring == CNNmetropolisrescore) // don't refine with cnn if rescoring
        {
          refine_structure(m, prec, nc_new, out_cont[i], authentic_v, par.mc.ssd_par.minparm, user_grid,
                           settings.verbosity, log, nc_new);
        } else
          refine_structure(m, prec, nc, out_cont[i], authentic_v, par.mc.ssd_par.minparm, user_grid, settings.verbosity,
                           log, nc_new);

        get_cnn_info(m, cnn, log, cnnscore, cnnaffinity, cnnvariance);

        out_cont[i].cnnscore = cnnscore;
        out_cont[i].cnnaffinity = cnnaffinity;
        out_cont[i].cnnvariance = cnnvariance;

        if (not_max(out_cont[i].e)) {
          intramolecular_energy = m.eval_intramolecular(exact_prec, authentic_v, out_cont[i].c);
          // we want vina energies not CNN
          out_cont[i].e =
              m.eval_adjusted(sf, exact_prec, nc_new, authentic_v, out_cont[i].c, intramolecular_energy, user_grid);
          out_cont[i].intramol = intramolecular_energy;
        }
      }

      auto sorter = [settings](const output_type &lhs, const output_type &rhs) {
        switch (settings.sort_order) {
        case Energy:
          return lhs.e < rhs.e;
        case CNNaffinity:
          return lhs.cnnaffinity > rhs.cnnaffinity; // reverse
        case CNNscore:
        default:
          return lhs.cnnscore > rhs.cnnscore; // reverse
        }
      };

      out_cont.sort(sorter);
      // Use symmetry-aware RMSD for clustering if RDKit mol is available
      if (ref_data && ref_data->rdkit_mol) {
        out_cont = remove_redundant(out_cont, settings.out_min_rmsd, ref_data->rdkit_mol);
      } else {
        out_cont = remove_redundant(out_cont, settings.out_min_rmsd);
      }

      done(settings.verbosity, log);

      log.setf(std::ios::fixed, std::ios::floatfield);
      log.setf(std::ios::showpoint);
      log << '\n';
      log << "mode |  affinity  |  intramol  |    CNN     |   CNN\n";
      log << "     | (kcal/mol) | (kcal/mol) | pose score | affinity\n";
      log << "-----+------------+------------+------------+----------\n";

      model best_mode_model = m;
      if (!out_cont.empty())
        best_mode_model.set(out_cont.front().c);

      sz how_many = 0;
      VINA_FOR_IN(i, out_cont) {
        if (!not_max(out_cont[i].e))
          continue; // may sort by something other than energy, so do not break
        if (how_many >= settings.num_modes)
          break; // check energy_range sanity FIXME
        ++how_many;
        m.set(out_cont[i].c);
        log << std::setw(5) << how_many << std::setw(12) << std::setprecision(2) << out_cont[i].e << std::setw(12)
            << std::setprecision(2) << out_cont[i].intramol;
        log << " " << std::setw(12) << std::setprecision(4) << out_cont[i].cnnscore << "  " << std::setw(9)
            << std::setprecision(3) << out_cont[i].cnnaffinity;
        log.endl();

        // dkoes - setup result_info
        results.push_back(
            result_info(out_cont[i].e, out_cont[i].cnnscore, out_cont[i].cnnaffinity, out_cont[i].cnnvariance, -1, compute_reference_rmsd(m, ref_data), m));

        if (compute_atominfo)
          results.back().setAtomValues(m, &sf);
      }
      done(settings.verbosity, log);

      if (how_many < 1) {
        log << "WARNING: Could not find any conformations completely within the search space.\n"
            << "WARNING: Check that it is large enough for all movable atoms, including those in the flexible side "
               "chains.";
        log.endl();
      }
    }
  } catch (numerical_error &ne) {
    log << "ERROR processing " << m.get_name() << "\n";
    log << ne.what() << "\n";
  }
  // std::cout << "Refine time " << time.elapsed().wall / 1000000000.0 << "\n";
}

void load_ent_values(const grid_dims &gd, std::istream &user_in, array3d<fl> &user_data) {
  std::string line;
  user_data = array3d<fl>(gd[0].n + 1, gd[1].n + 1, gd[2].n + 1);

  for (sz z = 0; z < gd[2].n + 1; z++) {
    for (sz y = 0; y < gd[1].n + 1; y++) {
      for (sz x = 0; x < gd[0].n + 1; x++) {
        std::getline(user_in, line);
        user_data(x, y, z) = ::atof(line.c_str());
      }
    }
  }
  std::cout << user_data(gd[0].n - 3, gd[1].n, gd[2].n) << "\n";
}

void main_procedure(model &m, precalculate &prec,
                    const boost::optional<model> &ref, // m is non-const (FIXME?)
                    const user_settings &settings, bool no_cache, bool compute_atominfo, const grid_dims &gd,
                    minimization_params minparm, const weighted_terms &wt, tee &log, std::vector<result_info> &results,
                    grid &user_grid, DLScorer &cnn) {
  // Store reference data from initial ligand pose for symmetry-aware RMSD calculation
  boost::optional<reference_data> ref_data;
  {
    const atomv& atoms = m.get_movable_atoms();
    const vecv& coords = m.coordinates();
    reference_data data;
    for (size_t i = 0; i < m.num_movable_atoms(); i++) {
      if (!atoms[i].is_hydrogen()) {
        data.coords.push_back(coords[i]);
      }
    }
    if (!data.coords.empty()) {
      // Create RDKit molecule from model for symmetry-aware RMSD
      try {
        // Write model to SDF string then parse with RDKit
        std::stringstream sdf_ss;
        bool sdfvalid = false;
        m.write_ligand(sdf_ss, sdfvalid);
        std::string sdf_str = sdf_ss.str();
        // Add SDF terminator if not present
        if (sdf_str.find("$$$$") == std::string::npos) {
          sdf_str += "$$$$\n";
        }
        // Parse SDF with RDKit, removing hydrogens for heavy atom RMSD
        RDKit::SDMolSupplier supplier;
        supplier.setData(sdf_str);
        if (!supplier.atEnd()) {
          RDKit::ROMol *mol = supplier.next();
          if (mol) {
            // Remove hydrogens to match heavy atom RMSD
            data.rdkit_mol = std::shared_ptr<RDKit::ROMol>(
                RDKit::MolOps::removeHs(*mol));
            delete mol;
          }
        }
      } catch (...) {
        // Failed to create RDKit mol, referenceRMSD will be -1
      }
      if (data.rdkit_mol) {
        ref_data = data;
      }
    }
  }

  doing(settings.verbosity, "Setting up the scoring function", log);

  done(settings.verbosity, log);
  log << std::fixed << std::setprecision(10);

  vec corner1(gd[0].begin, gd[1].begin, gd[2].begin);
  vec corner2(gd[0].end, gd[1].end, gd[2].end);

  parallel_mc par;
  sz heuristic = m.num_movable_atoms() + 10 * m.get_size().num_degrees_of_freedom();
  par.mc.num_steps = unsigned(70 * 3 * (50 + heuristic) / 2); // 2 * 70 -> 8 * 20 // FIXME
  if (settings.num_mc_steps > 0) {
    par.mc.num_steps = settings.num_mc_steps;
  }
  if (settings.max_mc_steps > 0 && par.mc.num_steps > settings.max_mc_steps) {
    par.mc.num_steps = settings.max_mc_steps;
  }
  if (settings.temperature > 0) {
    par.mc.temperature = settings.temperature; // expose temperature to user for cnn metropolis
  }

  par.mc.ssd_par.evals = unsigned((25 + m.num_movable_atoms()) / 3);
  // For --local_only (but not --minimize), use bfgs_iterations to match GPU behavior
  if (settings.local_only && !settings.dominimize) {
    minparm.maxiters = settings.bfgs_iterations;
  } else if (minparm.maxiters == 0) {
    minparm.maxiters = par.mc.ssd_par.evals;
  }
  par.mc.ssd_par.minparm = minparm;
  par.mc.min_rmsd = 1.0;
  par.mc.num_saved_mins = settings.num_modes > settings.num_mc_saved ? settings.num_modes : settings.num_mc_saved;
  par.mc.hunt_cap = vec(10, 10, 10);
  par.num_tasks = settings.exhaustiveness;
  par.num_threads = settings.cpu;
  par.display_progress = true;

  szv_grid_cache gridcache(m, prec.cutoff_sqr());
  const fl slope = 1e3; // FIXME: too large? used to be 100
  if (settings.randomize_only) {
    for (unsigned i = 0; i < settings.num_modes; i++) {
      fl e = do_randomization(m, corner1, corner2, settings.seed + i, settings.verbosity, log);
      results.push_back(result_info(e, -1, 0, 0, -1, compute_reference_rmsd(m, ref_data), m));
    }
    return;
  } else {
    non_cache *nc = NULL;
    if (settings.cnnopts.cnn_scoring >= CNNrefinement) {
      nc = new non_cache_cnn(gridcache, gd, &prec, slope, cnn);
    } else {
      nc = new non_cache(gridcache, gd, &prec, slope);
    }

    if (no_cache || settings.cnnopts.cnn_scoring == CNNall) {
      do_search(m, ref, ref_data, wt, prec, *nc, *nc, corner1, corner2, par, settings, compute_atominfo, log,
                wt.unweighted_terms(), user_grid, cnn, results, gridcache, gd, slope);
    } else {
      // Cache is needed for --gpu mode even with local_only or score_only
      // CPU score_only uses naive_non_cache, but GPU score_only needs grids
      // With --cpu_grid, CPU local_only also uses grid cache
      bool cache_needed = !(settings.randomize_only ||
                           (settings.score_only && !settings.gpu) ||
                           (settings.local_only && !settings.gpu && !settings.cpu_grid));

      // For --gpu mode, use a static cache to avoid recreating for each molecule
      // This significantly speeds up multi-ligand docking with the same receptor
      static std::unique_ptr<cache> bfgs_cache;
      static grid_dims bfgs_cache_gd;
      static bool bfgs_cache_initialized = false;

      std::unique_ptr<cache> local_cache;
      cache* c = nullptr;

      if (settings.gpu && settings.gpu) {
        // Check if we can reuse the static cache (same grid dimensions)
        bool can_reuse = bfgs_cache_initialized &&
                         bfgs_cache_gd[0].begin == gd[0].begin && bfgs_cache_gd[0].end == gd[0].end &&
                         bfgs_cache_gd[1].begin == gd[1].begin && bfgs_cache_gd[1].end == gd[1].end &&
                         bfgs_cache_gd[2].begin == gd[2].begin && bfgs_cache_gd[2].end == gd[2].end;

        if (!can_reuse) {
          // Create new cache for --gpu mode
          if (cache_needed)
            doing(settings.verbosity, "Analyzing the binding site (creating GPU cache)", log);
          bfgs_cache.reset(new cache_gpu("scoring_function_version001", gd, slope,
                                         dynamic_cast<precalculate_gpu *>(&prec)));
          bfgs_cache_gd = gd;
          bfgs_cache_initialized = true;
        }
        c = bfgs_cache.get();

        // Populate with any new atom types needed by this ligand
        if (cache_needed) {
          std::vector<smt> atom_types_needed;
          m.get_movable_atom_types(atom_types_needed);
          c->populate(m, prec, atom_types_needed, user_grid);
          if (!can_reuse)
            done(settings.verbosity, log);
        }
      } else {
        // Standard path: create a new cache for each molecule
        if (cache_needed)
          doing(settings.verbosity, "Analyzing the binding site", log);
        local_cache.reset((settings.gpu) ? new cache_gpu("scoring_function_version001", gd, slope,
                                                                 dynamic_cast<precalculate_gpu *>(&prec))
                                                 : new cache("scoring_function_version001", gd, slope));
        c = local_cache.get();
        if (cache_needed) {
          std::vector<smt> atom_types_needed;
          m.get_movable_atom_types(atom_types_needed);
          c->populate(m, prec, atom_types_needed, user_grid);
          done(settings.verbosity, log);
        }
      }

      // Set forcecap for GPU cache (for curl parameter in BFGS energy evaluation)
      c->set_forcecap(settings.forcecap);

      do_search(m, ref, ref_data, wt, prec, *c, *nc, corner1, corner2, par, settings, compute_atominfo, log,
                wt.unweighted_terms(), user_grid, cnn, results, gridcache, gd, slope);
    }

    delete nc;
  }
}

struct options_occurrence {
  bool some;
  bool all;
  options_occurrence() : some(false), all(true) {} // convenience
  options_occurrence &operator+=(const options_occurrence &x) {
    some = some || x.some;
    all = all && x.all;
    return *this;
  }
};

options_occurrence get_occurrence(boost::program_options::variables_map &vm,
                                  boost::program_options::options_description &d) {
  options_occurrence tmp;
  VINA_FOR_IN(i, d.options()) {
    const std::string &str = (*d.options()[i]).long_name();
    if ((str.substr(0, 4) == "size" || str.substr(0, 6) == "center")) {
      if (vm.count(str))
        tmp.some = true;
      else
        tmp.all = false;
    }
  }
  return tmp;
}

void check_occurrence(boost::program_options::variables_map &vm, boost::program_options::options_description &d) {
  VINA_FOR_IN(i, d.options()) {
    const std::string &str = (*d.options()[i]).long_name();
    if ((str.substr(0, 4) == "size" || str.substr(0, 6) == "center") && !vm.count(str))
      std::cerr << "Required parameter --" << str << " is missing!\n";
  }
}

template <class T>
inline void read_atomconstants_field(smina_atom_type::info &info, T(smina_atom_type::info::*field), unsigned int line,
                                     const std::string &field_name, std::istream &in) {
  if (!(in >> (info.*field))) {
    throw usage_error("Error at line " + boost::lexical_cast<std::string>(line) + " while reading field '" +
                      field_name + "' from the atom constants file.");
  }
}

void setup_atomconstants_from_file(const std::string &atomconstants_file) {
  std::ifstream file(atomconstants_file.c_str());
  if (file) {
    // create map from atom type names to indices
    boost::unordered_map<std::string, unsigned> atomindex;
    for (size_t i = 0u; i < smina_atom_type::NumTypes; ++i) {
      atomindex[smina_atom_type::default_data[i].smina_name] = i;
    }

    // parse each line of the file
    std::string line;
    unsigned lineno = 1;
    while (std::getline(file, line)) {
      std::string name;
      std::stringstream ss(line);

      if (line.length() == 0 || line[0] == '#')
        continue;

      ss >> name;

      if (atomindex.count(name)) {
        unsigned i = atomindex[name];
        smina_atom_type::info &info = smina_atom_type::data[i];

        // change this atom's parameters
#define read_field(field) read_atomconstants_field(info, &smina_atom_type::info::field, lineno, #field, ss)
        read_field(ad_radius);
        read_field(ad_depth);
        read_field(ad_solvation);
        read_field(ad_volume);
        read_field(covalent_radius);
        read_field(xs_radius);
        read_field(xs_hydrophobe);
        read_field(xs_donor);
        read_field(xs_acceptor);
        read_field(ad_heteroatom);
#undef read_field
      } else {
        std::cerr << "Line " << lineno << ": ommitting atom type name " << name << "\n";
      }
      lineno++;
    }
  } else
    throw usage_error("Error opening atom constants file:  " + atomconstants_file);
}

void print_atom_info(std::ostream &out) {
  out << "#Name radius depth solvation volume covalent_radius xs_radius xs_hydrophobe xs_donor xs_acceptr "
         "ad_heteroatom\n";
  VINA_FOR(i, smina_atom_type::NumTypes) {
    smina_atom_type::info &info = smina_atom_type::data[i];
    out << info.smina_name;
    out << " " << info.ad_radius;
    out << " " << info.ad_depth;
    out << " " << info.ad_solvation;
    out << " " << info.ad_volume;
    out << " " << info.covalent_radius;
    out << " " << info.xs_radius;
    out << " " << info.xs_hydrophobe;
    out << " " << info.xs_donor;
    out << " " << info.xs_acceptor;
    out << " " << info.ad_heteroatom;
    out << "\n";
  }
}

const fl box_granularity = 0.375;

// set grid dims to match center/size
static void setup_grid_dims(fl center_x, fl center_y, fl center_z, fl size_x, fl size_y, fl size_z, grid_dims &gd) {
  vec span(size_x, size_y, size_z);
  vec center(center_x, center_y, center_z);
  VINA_FOR_IN(i, gd) {
    gd[i].n = sz(std::ceil(span[i] / box_granularity));
    fl real_span = box_granularity * gd[i].n;
    gd[i].begin = center[i] - real_span / 2;
    gd[i].end = gd[i].begin + real_span;
  }
}
void setup_user_gd(grid_dims &gd, std::ifstream &user_in) {
  std::string line;
  size_t pLines = 3;
  std::vector<std::string> temp;
  fl center_x = 0, center_y = 0, center_z = 0, size_x = 0, size_y = 0, size_z = 0;

  for (; pLines > 0; --pLines) // Eat first 3 lines
    std::getline(user_in, line);
  pLines = 3;

  // Read in SPACING
  std::getline(user_in, line);
  boost::algorithm::split(temp, line, boost::algorithm::is_space());
  const fl granularity = ::atof(temp[1].c_str());
  // Read in NELEMENTS
  std::getline(user_in, line);
  boost::algorithm::split(temp, line, boost::algorithm::is_space());
  size_x = (::atof(temp[1].c_str()) + 1) * granularity; // + 1 here?
  size_y = (::atof(temp[2].c_str()) + 1) * granularity;
  size_z = (::atof(temp[3].c_str()) + 1) * granularity;
  // Read in CENTER
  std::getline(user_in, line);
  boost::algorithm::split(temp, line, boost::algorithm::is_space());
  center_x = ::atof(temp[1].c_str()) + 0.5 * granularity;
  center_y = ::atof(temp[2].c_str()) + 0.5 * granularity;
  center_z = ::atof(temp[3].c_str()) + 0.5 * granularity;

  vec span(size_x, size_y, size_z);
  vec center(center_x, center_y, center_z);
  VINA_FOR_IN(i, gd) {
    gd[i].n = sz(std::ceil(span[i] / granularity));
    fl real_span = granularity * gd[i].n;
    gd[i].begin = center[i] - real_span / 2;
    gd[i].end = gd[i].begin + real_span;
  }
}

// work queue job format
struct worker_job {
  unsigned int molid;
  model *m;
  std::vector<result_info> *results;
  grid_dims gd;

  worker_job(unsigned int molid, model *m, std::vector<result_info> *results, grid_dims gd)
      : molid(molid), m(m), results(results), gd(gd){};

  worker_job() : molid(0), m(NULL), results(NULL) {
    for (int i = 0; i < 3; i++) {
      gd[i] = grid_dim();
    }
  };
};

// writer queue job format
struct writer_job {
  unsigned int molid;
  std::vector<result_info> *results;

  writer_job(unsigned int molid, std::vector<result_info> *results) : molid(molid), results(results){};

  writer_job() : molid(0), results(NULL){};
};

template <typename T> struct job_queue {
  job_queue(unsigned ms=0) : max_size(ms), jobs(0){};

  void push(T &job) {
    int cnt = has_work.value();
    while(max_size > 0 && cnt > max_size) {
      boost::thread::yield();
      cnt = has_work.value();
    }
    jobs.push(job);
    has_work.signal();
  }

  // Returns false and doesn't modify job iff the queue has been
  // closed. Should not be called again in the same thread afterwards.
  bool wait_and_pop(T &job) {
    has_work.wait();
    return !jobs.pop(job);
  }

  // Signal that all jobs are done. num_possible_waiters will be waiting
  // on the sem, so arrange for them to wake. They'll find an empty
  // queue that distinguishes these special signals.
  void close(size_t num_possible_waiters) {
    for (size_t i = 0; i < num_possible_waiters; i++)
      has_work.signal();
  }

  unsigned max_size = 0;
  sem has_work;
  boost::lockfree::queue<T> jobs;
};

// A struct of parameters that define the current run. These are packed together
// because of boost's restriction on the number of arguments you can
// give to bind (max args is 9, but I need 10+ for the following thread
// functions) so I can reduce the number of args I pass.
struct global_state {
  user_settings *settings;
  boost::shared_ptr<precalculate> prec;
  minimization_params *minparms;
  weighted_terms *wt;
  grid *user_grid;
  tee *log;
  std::ofstream *atomoutfile;
  cnn_options cnnopts;

  global_state(user_settings *settings, boost::shared_ptr<precalculate> prec, minimization_params *minparms,
               weighted_terms *wt, grid *user_grid, tee *log, std::ofstream *atomoutfile, const cnn_options &co)
      : settings(settings), prec(prec), minparms(minparms), wt(wt), user_grid(user_grid), log(log),
        atomoutfile(atomoutfile), cnnopts(co){};
};

// function to occupy the worker threads with individual ligands from the work queue
// TODO: see if implementing weight sharing between CNNScorer instances results
// in enough memory efficiency to avoid using a single one
void threads_at_work(job_queue<worker_job> *wrkq, job_queue<writer_job> *writerq, global_state *gs, MolGetter *mols,
                     int *nligs, std::shared_ptr<DLScorer> dl_scorer) // copy dl_scorer so it can maintain state
{
  if (!gs->settings->cnn_cpu)
    initializeCUDA(gs->settings->device);

  if (gs->settings->gpu)
    thread_buffer.init(available_mem(gs->settings->cpu));

  worker_job j;
  while (!wrkq->wait_and_pop(j)) {
    __sync_fetch_and_add(nligs, 1);

    main_procedure(*(j.m), *gs->prec, boost::optional<model>(), *gs->settings,
                   false, // no_cache == false
                   gs->atomoutfile->is_open() || gs->settings->include_atom_info, j.gd, *gs->minparms, *gs->wt,
                   *gs->log, *(j.results), *gs->user_grid, *dl_scorer);

    writer_job k(j.molid, j.results);
    writerq->push(k);
    delete j.m;
  }
}

void write_out(std::vector<result_info> &results, ozfile &outfile, std::string &outext, user_settings &settings,
               const weighted_terms &wt, ozfile &outflex, std::string &outfext, std::ofstream &atomoutfile) {
  if (outfile) {
    // write out molecular data
    for (unsigned j = 0, nr = results.size(); j < nr; j++) {
      results[j].write(outfile, outext, settings.include_atom_info, &wt, j + 1);
    }
  }
  if (outflex) {
    // write out flexible residue data data
    for (unsigned j = 0, nr = results.size(); j < nr; j++) {
      results[j].writeFlex(outflex, outfext, j + 1);
    }
  }
  if (atomoutfile) {
    for (unsigned j = 0, m = results.size(); j < m; j++) {
      results[j].writeAtomValues(atomoutfile, &wt);
    }
  }
}

// function for the writing thread to write ligands in order to output file
void thread_a_writing(job_queue<writer_job> *writerq, global_state *gs, ozfile *outfile, std::string *outext,
                      ozfile *outflex, std::string *outfext, int *nligs) {
  try {
    int nwritten = 0;
    boost::unordered_map<int, std::vector<result_info> *> proc_out;
    writer_job j;
    while (!writerq->wait_and_pop(j)) {
      if (j.molid == nwritten) {
        write_out(*j.results, *outfile, *outext, *gs->settings, *gs->wt, *outflex, *outfext, *gs->atomoutfile);
        nwritten++;
        delete j.results;
        for (boost::unordered_map<int, std::vector<result_info> *>::iterator i;
             (i = proc_out.find(nwritten)) != proc_out.end();) {
          write_out(*i->second, *outfile, *outext, *gs->settings, *gs->wt, *outflex, *outfext, *gs->atomoutfile);
          nwritten++;
          delete i->second;
        }
      } else {
        proc_out[j.molid] = j.results;
      }
    }
  } catch (file_error &e) {
    std::cerr << "\n\nError: could not open \"" << e.name.string() << "\" for " << (e.in ? "reading" : "writing")
              << ".\n";
  } catch (boost::filesystem::filesystem_error &e) {
    std::cerr << "\n\nFile system error: " << e.what() << '\n';
  } catch (usage_error &e) {
    std::cerr << "\n\nUsage error: " << e.what() << "\n";
  }
}

int main(int argc, char *argv[]) {
  using namespace boost::program_options;
  const std::string version_string =
      std::string("gnina ") + GIT_TAG + " " + GIT_BRANCH + ":" + GIT_REV + "   Built " __DATE__ ".";
  const std::string error_message = "\n\n\
Please report this error at https://github.com/gnina/gnina/issues\n"
                                    "Please remember to include the following in your problem report:\n\
    * the EXACT error message,\n\
    * your version of the program,\n\
    * the type of computer system you are running it on,\n\
	* all command line options,\n\
	* configuration file (if used),\n\
    * ligand file as provided to gnina,\n\
    * receptor file as provided to gnina,\n\
	* output file (if any),\n\
	* random seed the program used (this is printed when the program starts).\n\
\n\
Thank you!\n";

  const std::string cite_message = "              _             \n"
                                   "             (_)            \n"
                                   "   __ _ _ __  _ _ __   __ _ \n"
                                   "  / _` | '_ \\| | '_ \\ / _` |\n"
                                   " | (_| | | | | | | | | (_| |\n"
                                   "  \\__, |_| |_|_|_| |_|\\__,_|\n"
                                   "   __/ |                    \n"
                                   "  |___/                     \n"
                                   "\n" +
                                   version_string +
                                   "\ngnina is based on smina and AutoDock Vina.\nPlease cite appropriately.\n";

  try {
    std::string rigid_name, flex_name, config_name, log_name, atom_name;
    std::vector<std::string> ligand_names;
    std::string out_name;
    std::string outf_name;
    std::string ligand_names_file;
    std::string atomconstants_file;
    std::string custom_file_name;
    std::string usergrid_file_name;
    std::string flex_res;
    double flex_dist = -1.0;
    fl center_x = 0, center_y = 0, center_z = 0, size_x = 0, size_y = 0, size_z = 0;
    fl autobox_add = 4;
    bool autobox_extend = true;
    std::string autobox_ligand;
    std::string flexdist_ligand;
    std::string builtin_scoring;
    int flex_limit = -1;
    int flex_max = -1;
    int nflex = -1;
    bool nflex_hard_limit = true; // TODO@RMeli: Use for defining "soft" flexmax

    // -0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 0.05846
    fl weight_gauss1 = -0.035579;
    fl weight_gauss2 = -0.005156;
    fl weight_repulsion = 0.840245;
    fl weight_hydrophobic = -0.035069;
    fl weight_hydrogen = -0.587439;
    fl weight_rot = 0.05846;
    fl user_grid_lambda;
    bool help = false, help_hidden = false, version = false;
    bool quiet = false;
    bool accurate_line = false;
    bool simple_ascent = false;
    bool flex_hydrogens = false;
    bool print_terms = false;
    bool print_atom_types = false;
    bool add_hydrogens = true;
    bool strip_hydrogens = false;
    bool full_flex_output = false;

    CovOptions copt;

    user_settings settings;
    cnn_options &cnnopts = settings.cnnopts;

    minimization_params minparms;
    ApproxType approx = LinearApprox;
    fl approx_factor = 32;

    positional_options_description positional; // remains empty

    options_description inputs("Input");
    inputs.add_options()("receptor,r", value<std::string>(&rigid_name), "rigid part of the receptor")(
        "flex", value<std::string>(&flex_name), "flexible side chains, if any (PDBQT)")(
        "ligand,l", value<std::vector<std::string>>(&ligand_names),
        "ligand(s)")("flexres", value<std::string>(&flex_res),
                     "flexible side chains specified by comma separated list of chain:resid")(
        "flexdist_ligand", value<std::string>(&flexdist_ligand),
        "Ligand to use for flexdist")("flexdist", value<double>(&flex_dist),
                                      "set all side chains within specified distance to flexdist_ligand to flexible")(
        "flex_limit", value<int>(&flex_limit), "Hard limit for the number of flexible residues")(
        "flex_max", value<int>(&flex_max), "Retain at at most the closest flex_max flexible residues");

    // options_description search_area("Search area (required, except with --score_only)");
    options_description search_area("Search space (required)");
    search_area.add_options()("center_x", value<fl>(&center_x), "X coordinate of the center")(
        "center_y", value<fl>(&center_y), "Y coordinate of the center")(
        "center_z", value<fl>(&center_z), "Z coordinate of the center")("size_x", value<fl>(&size_x),
                                                                        "size in the X dimension (Angstroms)")(
        "size_y", value<fl>(&size_y), "size in the Y dimension (Angstroms)")("size_z", value<fl>(&size_z),
                                                                             "size in the Z dimension (Angstroms)")(
        "autobox_ligand", value<std::string>(&autobox_ligand), "Ligand to use for autobox. A multi-ligand file still only defines a single box.")(
        "autobox_add", value<fl>(&autobox_add),
        "Amount of buffer space to add to auto-generated box (default +4 on all six sides)")(
        "autobox_extend", value<bool>(&autobox_extend)->default_value(true),
        "Expand the autobox if needed to ensure the input conformation of the ligand being docked can freely rotate "
        "within the box.")("no_lig", bool_switch(&settings.no_lig)->default_value(false),
                           "no ligand; for sampling/minimizing flexible residues");

    options_description covalent("Covalent docking");
    covalent.add_options()("covalent_rec_atom", value<std::string>(&copt.covalent_rec_atom),
                           "Receptor atom ligand is covalently bound to.  Can be specified as chain:resnum:atom_name "
                           "or as x,y,z Cartesian coordinates.")(
        "covalent_lig_atom_pattern", value<std::string>(&copt.covalent_lig_atom_pattern),
        "SMARTS expression for ligand atom that will covalently bind protein.")(
        "covalent_lig_atom_position", value<std::string>(&copt.covalent_lig_atom_position),
        "Optional.  Initial placement of covalently bonding ligand atom in x,y,z Cartesian coordinates.  If not "
        "specified, OpenBabel's GetNewBondVector function will be used to position ligand.")(
        "covalent_fix_lig_atom_position", bool_switch(&copt.covalent_fix_lig_atom_position),
        "If covalent_lig_atom_position is specified, fix the ligand atom to this position as opposed to using this "
        "position to define the initial structure.")(
        "covalent_bond_order", value<int>(&copt.bond_order)->default_value(1),
        "Bond order of covalent bond. Default 1.")("covalent_optimize_lig", bool_switch(&copt.covalent_optimize_lig),
                                                   "Optimize the covalent complex of ligand and residue using UFF. "
                                                   "This will change bond angles and lengths of the ligand.");

    options_description outputs("Output");
    outputs.add_options()("out,o", value<std::string>(&out_name), "output file name, format taken from file extension")(
        "out_flex", value<std::string>(&outf_name), "output file for flexible receptor residues")(
        "log", value<std::string>(&log_name), "optionally, write log file")(
        "atom_terms", value<std::string>(&atom_name), "optionally write per-atom interaction term values")(
        "atom_term_data", bool_switch(&settings.include_atom_info)->default_value(false),
        "embedded per-atom interaction terms in output sd data")(
        "pose_sort_order", value<pose_sort_order>(&settings.sort_order)->default_value(CNNscore),
        "How to sort docking results: CNNscore (default), CNNaffinity, Energy")(
        "full_flex_output", bool_switch(&full_flex_output)->default_value(false),
        "Output entire structure for out_flex, not just flexible residues.");

    options_description scoremin("Scoring and minimization options");
    scoremin.add_options()(
        "scoring", value<std::string>(&builtin_scoring),
        ("specify alternative built-in scoring function: " + builtin_scoring_functions.names(" ")).c_str())(
        "custom_scoring", value<std::string>(&custom_file_name), "custom scoring function file")(
        "custom_atoms", value<std::string>(&atomconstants_file), "custom atom type parameters file")(
        "score_only", bool_switch(&settings.score_only)->default_value(false),
        "score provided ligand pose")("local_only", bool_switch(&settings.local_only)->default_value(false),
                                      "local search only using autobox (you probably want to use --minimize)")(
        "gpu", bool_switch(&settings.gpu)->default_value(false),
        "use GPU for docking (parallel BFGS instead of Monte Carlo)")(
        "direct_pairwise", bool_switch(&settings.direct_pairwise)->default_value(false),
        "use direct pairwise scoring with LUT instead of grid interpolation (GPU only, reduces L2 cache pressure)")(
        "batch_size", value<int>(&settings.batch_size)->default_value(50000),
        "target total poses per GPU batch for multi-ligand batch docking (default: 50000)")(
        "no_batch", bool_switch(&settings.no_batch)->default_value(false),
        "disable batch docking (process one ligand at a time, for debugging)")(
        "fast_embed", bool_switch(&settings.fast_embed)->default_value(false),
        "use fast template-based 3D coordinate generation for SMILES (skips distance geometry)")(
        "parallel_embed", value<int>(&settings.parallel_embed)->default_value(1),
        "generate N conformers per SMILES with different ring puckerings, each processed as separate ligand (default: 1)")(
        "prune_rms_thresh", value<fl>(&settings.prune_rms_thresh)->default_value(0),
        "RMSD threshold for pruning similar conformers during parallel_embed (default: 0 = disabled, try 0.5-1.0 for diversity)")(
        "skip_torsion_randomize", bool_switch(&settings.skip_torsion_randomize)->default_value(false),
        "skip torsion randomization (for debugging, not recommended for production)")(
        "no_rdkit_smiles", bool_switch(&settings.no_rdkit_smiles)->default_value(false),
        "use OpenBabel instead of RDKit for SMILES 3D generation (slower but may be more reliable)")(
        "cpu_grid", bool_switch(&settings.cpu_grid)->default_value(false),
        "use grid-based scoring for CPU local_only (to match GPU behavior)")(
        "bfgs_iterations", value<int>(&settings.bfgs_iterations)->default_value(50),
        "max BFGS iterations for --gpu and --local_only modes")(
        "verbose_grad", bool_switch(&settings.verbose_grad)->default_value(false),
        "output gradient values for debugging")(
        "minimize", bool_switch(&settings.dominimize)->default_value(false), "energy minimization")(
        "randomize_only", bool_switch(&settings.randomize_only), "generate random poses, attempting to avoid clashes")(
        "num_mc_steps", value<int>(&settings.num_mc_steps), "fixed number of monte carlo steps to take in each chain")(
        "max_mc_steps", value<int>(&settings.max_mc_steps), "cap on number of monte carlo steps to take in each chain")(
        "num_mc_saved", value<int>(&settings.num_mc_saved), "number of top poses saved in each monte carlo chain")(
        "temperature", value<fl>(&settings.temperature), "temperature for metropolis accept criterion")(
        "minimize_iters", value<unsigned>(&minparms.maxiters)->default_value(0),
        "number iterations of steepest descent; default scales with rotors and usually isn't sufficient for "
        "convergence")("accurate_line", bool_switch(&accurate_line), "use accurate line search")(
        "simple_ascent", bool_switch(&simple_ascent),
        "use simple gradient ascent")("minimize_early_term", bool_switch(&minparms.early_term),
                                      "Stop minimization before convergence conditions are fully met.")(
        "minimize_single_full", bool_switch(&minparms.single_min),
        "During docking perform a single full minimization instead of a truncated pre-evaluate followed by a full.")(
        "approximation", value<ApproxType>(&approx), "approximation (linear, spline, or exact) to use")(
        "factor", value<fl>(&approx_factor), "approximation factor: higher results in a finer-grained approximation")(
        "force_cap", value<fl>(&settings.forcecap),
        "max allowed force; lower values more gently minimize clashing structures")(
        "user_grid", value<std::string>(&usergrid_file_name),
        "Autodock map file for user grid data based calculations")("user_grid_lambda",
                                                                   value<fl>(&user_grid_lambda)->default_value(-1.0),
                                                                   "Scales user_grid and functional scoring")(
        "print_terms", bool_switch(&print_terms), "Print all available terms with default parameterizations")(
        "print_atom_types", bool_switch(&print_atom_types), "Print all available atom types");

    options_description hidden("Hidden options for internal testing");
    hidden.add_options()("verbosity", value<int>(&settings.verbosity)->default_value(1),
                         "Adjust the verbosity of the output, default: 1")(
        "flex_hydrogens", bool_switch(&flex_hydrogens),
        "Enable torsions affecting only hydrogens (e.g. OH groups). This is stupid but provides compatibility with "
        "Vina.")("outputmin", value<int>(&minparms.outputframes),
                 "output minout.sdf of minimization with provided amount of interpolation")(
        "cnn_gradient_check", bool_switch(&cnnopts.gradient_check)->default_value(false),
        "Perform internal checks on gradient.");

    options_description cnn("Convolutional neural net (CNN) scoring");
    cnn.add_options() //
        ("cnn_scoring", value<cnn_scoring_level>(&cnnopts.cnn_scoring)->default_value(CNNrescore),
         "Amount of CNN scoring: none, rescore (default), refinement, metrorescore (metropolis+rescore), "
         "metrorefine (metropolis+refine), all") //
        ("cnn", value<std::vector<std::string>>(&cnnopts.cnn_model_names)->multitoken(),
         ("built-in model to use, specify PREFIX_ensemble to evaluate an ensemble of models starting with PREFIX: " +
          builtin_torch_models())
             .c_str()) //
        ("cnn_model", value<std::vector<std::string>>(&cnnopts.cnn_models)->multitoken(),
         "torch cnn model file; if not specified a default model ensemble will be used") //
        ("cnn_rotation", value<unsigned>(&cnnopts.cnn_rotations)->default_value(0),      //
         "evaluate multiple rotations of pose (max 24)")("cnn_mix_emp_force",
                                                         bool_switch(&cnnopts.mix_emp_force)->default_value(false),
                                                         "Merge CNN and empirical minus forces") //
        ("cnn_mix_emp_energy", bool_switch(&cnnopts.mix_emp_energy)->default_value(false),
         "Merge CNN and empirical energy") //
        ("cnn_empirical_weight", value<fl>(&cnnopts.empirical_weight)->default_value(1.0),
         "Weight for scaling and merging empirical force and energy ")                            //
        ("cnn_center_x", value<fl>(&cnnopts.cnn_center[0]),
         "X coordinate of the CNN center")                                                    //
        ("cnn_center_y", value<fl>(&cnnopts.cnn_center[1]), "Y coordinate of the CNN center") //
        ("cnn_center_z", value<fl>(&cnnopts.cnn_center[2]), "Z coordinate of the CNN center") //
        ("cnn_verbose", bool_switch(&cnnopts.verbose), "Enable verbose output for CNN debugging");

    options_description misc("Misc (optional)");
    misc.add_options()(
        "cpu", value<int>(&settings.cpu),
        "the number of CPUs to use (the default is to try to detect the number of CPUs or, failing that, use 1)")(
        "seed", value<int>(&settings.seed),
        "explicit random seed")("exhaustiveness", value<int>(&settings.exhaustiveness)->default_value(8),
                                "exhaustiveness of the global search (roughly proportional to time)")(
        "num_modes", value<sz>(&settings.num_modes)->default_value(9), "maximum number of binding modes to generate")(
        "cnn_scoring_mult", value<sz>(&settings.cnn_refine_mult)->default_value(20),
        "multiplier for num_modes to determine poses sent to CNN scoring (max_poses = num_modes * cnn_scoring_mult)")(
        "min_rmsd_filter", value<fl>(&settings.out_min_rmsd)->default_value(1.0),
        "rmsd value used to filter final poses to remove redundancy")("quiet,q", bool_switch(&quiet),
                                                                      "Suppress output messages")(
        "addH", value<bool>(&add_hydrogens), "automatically add hydrogens in ligands (on by default)")(
        "stripH", value<bool>(&strip_hydrogens),
        "remove polar hydrogens from molecule _after_ performing atom typing for efficiency (off by default - nonpolar are always removed)")(
        "device", value<int>(&settings.device)->default_value(0),
        "GPU device to use")("cnn_cpu", bool_switch(&settings.cnn_cpu), "Run CNN scoring on CPU instead of GPU");

    options_description config("Configuration file (optional)");
    config.add_options()("config", value<std::string>(&config_name), "the above options can be put here");
    options_description info("Information (optional)");
    info.add_options()("help", bool_switch(&help), "display usage summary")(
        "help_hidden", bool_switch(&help_hidden),
        "display usage summary with hidden options")("version", bool_switch(&version), "display program version");

    options_description desc, desc_simple;
    desc.add(inputs)
        .add(search_area)
        .add(covalent)
        .add(outputs)
        .add(scoremin)
        .add(cnn)
        .add(hidden)
        .add(misc)
        .add(config)
        .add(info);
    desc_simple.add(inputs)
        .add(search_area)
        .add(covalent)
        .add(scoremin)
        .add(cnn)
        .add(outputs)
        .add(misc)
        .add(config)
        .add(info);

    variables_map vm;
    try {
      store(command_line_parser(argc, argv)
                .options(desc)
                .style(command_line_style::default_style ^ command_line_style::allow_guessing)
                .positional(positional)
                .run(),
            vm);
      notify(vm);
    } catch (boost::program_options::error &e) {
      std::cerr << "Command line parse error: " << e.what() << '\n' << "\nCorrect usage:\n" << desc_simple << '\n';
      return 1;
    }
    if (vm.count("config")) {
      try {
        ifile config_stream(config_name);
        store(parse_config_file(config_stream, desc), vm);
        notify(vm);
      } catch (boost::program_options::error &e) {
        std::cerr << "Configuration file parse error: " << e.what() << '\n'
                  << "\nCorrect usage:\n"
                  << desc_simple << '\n';
        return 1;
      }
    }
    if (help) {
      std::cout << desc_simple << '\n';
      return 0;
    }
    if (help_hidden) {
      std::cout << desc << '\n';
      return 0;
    }
    if (version) {
      std::cout << version_string << '\n';
      return 0;
    }

    tee log(quiet);
    if (vm.count("log") > 0)
      log.init(log_name);

    if (!atomconstants_file.empty())
      setup_atomconstants_from_file(atomconstants_file);

    if (print_terms) {
      custom_terms t;
      t.print_available_terms(std::cout);
      return 0;
    }

    if (print_atom_types) {
      print_atom_info(std::cout);
      return 0;
    }

#if (OB_VERSION > OB_VERSION_CHECK(2, 3, 2))
    OpenBabel::OBPlugin::LoadAllPlugins(); // for some reason loading on demand can be slow
#endif
    cnnopts.seed = settings.seed;

    OpenBabel::vector3 dummy; // openbabel uses system rand initialized with time seed
    dummy.randomUnitVector(); // this setups up the obrandom object with time
    srand(settings.seed);     // so now it is safe(?) to set the system seed

    set_fixed_rotable_hydrogens(!flex_hydrogens);

    if (settings.dominimize) // set default settings for minimization
    {
      if (!vm.count("force_cap"))
        settings.forcecap = 10; // nice and soft

      if (minparms.maxiters == 0)
        minparms.maxiters = 10000; // will presumably converge
      settings.local_only = true;
      minparms.type = minimization_params::BFGSAccurateLineSearch;

      if (!vm.count("approximation"))
        approx = SplineApprox; // use high accuracy approximation for --minimize
      if (!vm.count("factor"))
        approx_factor = 10;
    }

    // Set soft forcecap for --local_only (CPU and GPU) to match --minimize behavior
    if (settings.local_only && !settings.dominimize && !vm.count("force_cap")) {
      settings.forcecap = 10;
    }

    // Use accurate line search for --local_only to match GPU behavior
    if (settings.local_only && !settings.dominimize) {
      minparms.type = minimization_params::BFGSAccurateLineSearch;
    }

     // output banner
    log << cite_message << '\n';
    
   // check for GPU
    bool torchgpu = false;
    if (torch::cuda::is_available() && !settings.cnn_cpu) {
      torchgpu = true;
      if (settings.device > 0) {
        log << "WARNING: Torch backend ignores device argument.  Use CUDA_VISIBLE_DEVICES environment to control CUDA "
               "device used.\n";
      }
    } else if (!settings.cnn_cpu) {
      log << "WARNING: No GPU detected. CNN scoring will be slow.\n"
             "Recommend running with single model (--cnn fast)\n"
             "or without cnn scoring (--cnn_scoring=none).\n\n";    
    }

    if (accurate_line) {
      minparms.type = minimization_params::BFGSAccurateLineSearch;
    }

    if (simple_ascent) {
      minparms.type = minimization_params::Simple;
    }

    bool search_box_needed =
        !(settings.score_only || settings.local_only); // randomize_only and local_only still need the search space;
                                                       // dkoes - for local get box from ligand
    bool output_produced = !settings.score_only;
    bool receptor_needed = !settings.randomize_only;
    
    if (cnnopts.cnn_scoring == CNNnone) {
      settings.sort_order = Energy;
    }

    if(copt.covalent_rec_atom != "" && cnnopts.cnn_scoring != CNNnone) {
      log << "WARNING: CNN scoring not yet calibrated for covalent docking.  Recommend running with --cnn_scoring none\n";
    }

    if (receptor_needed) {
      if (vm.count("receptor") <= 0) {
        std::cerr << "Missing receptor.\n"
                  << "\nCorrect usage:\n"
                  << desc_simple << '\n';
        return 1;
      }
    }

    if (ligand_names.size() == 0) {
      if (!settings.no_lig) {
        std::cerr << "Missing ligand.\n"
                  << "\nCorrect usage:\n"
                  << desc_simple << '\n';
        return 1;
      } else // put in "fake" ligand
      {
        ligand_names.push_back("");
      }
    } else if (settings.no_lig) // ligand specified with no_lig
    {
      std::cerr << "Ligand specified with --no_lig.\n"
                << "\nCorrect usage:\n"
                << desc_simple << '\n';
      return 1;
    } else {
      for (const auto &lname : ligand_names) {
        if (!boost::filesystem::exists(lname)) {
          throw file_error(lname, true);
        }
      }
    }

    if (settings.exhaustiveness < 1)
      throw usage_error("exhaustiveness must be 1 or greater");
    if (settings.num_modes < 1)
      throw usage_error("num_modes must be 1 or greater");

    boost::optional<std::string> flex_name_opt;
    if (vm.count("flex"))
      flex_name_opt = flex_name;

    if (vm.count("flex") && !vm.count("receptor"))
      throw usage_error(
          "Flexible side chains are not allowed without the rest of the receptor"); // that's the only way parsing
                                                                                    // works, actually

    if (flex_limit > -1 && flex_max > -1) {
      throw usage_error("--flex_lim and --flex_max can't be used together.");
    } else if (flex_limit > -1) {
      nflex = flex_limit;
      nflex_hard_limit = true;
    } else if (flex_max > -1) {
      nflex = flex_max;
      nflex_hard_limit = false;
    }

    std::ofstream atomoutfile;
    if (vm.count("atom_terms") > 0)
      atomoutfile.open(atom_name.c_str());

    log << "Commandline:";
    for (unsigned i = 0; i < argc; i++) {
      log << " " << argv[i];
    }
    log << "\n";

    FlexInfo finfo(flex_res, flex_dist, flexdist_ligand, nflex, nflex_hard_limit, full_flex_output, log);
    copt.dont_move_ligand = !search_box_needed; 
    CovInfo cinfo(copt, log);
    // dkoes - parse in receptor once
    MolGetter mols(rigid_name, flex_name, finfo, cinfo, add_hydrogens, strip_hydrogens, log);

    if (autobox_ligand.length() > 0) {
      setup_autobox(mols.getInitModel(), autobox_ligand, autobox_add, center_x, center_y, center_z, size_x, size_y,
                    size_z);
    }

    if (search_box_needed && autobox_ligand.length() == 0) {
      options_occurrence oo = get_occurrence(vm, search_area);
      if (!oo.all) {
        check_occurrence(vm, search_area);
        std::cerr << "\nCorrect usage:\n" << desc_simple << std::endl;
        return 1;
      }
      if (size_x <= 0 || size_y <= 0 || size_z <= 0)
        throw usage_error("Search space dimensions should be positive");
    }

    if (flex_dist > 0 && flexdist_ligand.size() == 0) {
      throw usage_error("Must specify flexdist_ligand with flex_dist");
    }

    if (nflex > 0 && flex_res.size() > 0) {
      log << "WARNING: --flex_limit and --flexmax ignored with --flexres\n\n";
    }

    grid_dims gd; // n's = 0 via default c'tor
    grid_dims user_gd;
    grid user_grid;

    flv weights;

    // dkoes, set the scoring function
    custom_terms t;
    if (user_grid_lambda != -1.0) {
      t.set_scaling_factor(user_grid_lambda);
    }
    if (custom_file_name.size() > 0) {
      ifile custom_file(custom_file_name);
      t.add_terms_from_file(custom_file);
    } else if (builtin_scoring.size() > 0) {
      if (!builtin_scoring_functions.set(t, builtin_scoring)) {
        throw usage_error("Invalid built-in scoring function: " + builtin_scoring + ". Options are:\n" +
                          builtin_scoring_functions.names("\n"));
      }
    } else {
      t.add_vina();
    }

    if (settings.verbosity > 1)
      log << std::setw(12) << std::left << "Weights" << " Terms\n" << t << "\n";

    // Print out flexible residues
    if (finfo.has_content()) {
      finfo.print_flex();
    }

    if (usergrid_file_name.size() > 0) {
      ifile user_in(usergrid_file_name);
      fl ug_scaling_factor = 1.0;
      if (user_grid_lambda != -1.0) {
        ug_scaling_factor = 1 - user_grid_lambda;
      }
      setup_user_gd(user_gd, user_in);
      user_grid.init(user_gd, user_in, ug_scaling_factor); // initialize user grid
    }

    if (search_box_needed) {
      setup_grid_dims(center_x, center_y, center_z, size_x, size_y, size_z, gd);
    }

    if (settings.verbosity > 1) {
      log << "Using search box with center " << center_x << "," << center_y << "," << center_z << " and size " << size_x
          << "," << size_y << "," << size_z << "\n";
    }

    if (vm.count("cpu") == 0) {
      settings.cpu = boost::thread::hardware_concurrency();
      if (settings.verbosity > 1) {
        if (settings.cpu > 0)
          log << "Detected " << settings.cpu << " CPU" << ((settings.cpu > 1) ? "s" : "") << '\n';
        else
          log << "Could not detect the number of CPUs, using 1\n";
      }
    }
    if (settings.cpu < 1)
      settings.cpu = 1;
    if (settings.verbosity > 1 && settings.exhaustiveness < settings.cpu)
      log << "WARNING: at low exhaustiveness, it may be impossible to utilize all CPUs\n";
    torch::set_num_threads(settings.cpu);

    if (settings.verbosity <= 1) {
      OpenBabel::obErrorLog.SetOutputLevel(OpenBabel::obError);
    }
    // dkoes, hoist precalculation outside of loop
    weighted_terms wt(&t, t.weights());

    boost::shared_ptr<precalculate> prec;

    if (settings.gpu || approx == GPU) { // don't get a choice
      prec = boost::shared_ptr<precalculate>(new precalculate_gpu(wt, approx_factor));
    } else if (approx == SplineApprox)
      prec = boost::shared_ptr<precalculate>(new precalculate_splines(wt, approx_factor));
    else if (approx == LinearApprox)
      prec = boost::shared_ptr<precalculate>(new precalculate_linear(wt, approx_factor));
    else if (approx == Exact)
      prec = boost::shared_ptr<precalculate>(new precalculate_exact(wt));

    // setup single outfile
    using namespace OpenBabel;
    ozfile outfile;
    std::string outext;
    if (out_name.length() > 0) {
      outext = outfile.open(out_name);
    }

    ozfile outflex;
    std::string outfext;
    if (outf_name.length() > 0) {
      outfext = outflex.open(outf_name);
    }

    if (settings.score_only) // output header
    {
      std::vector<std::string> enabled_names = t.get_names(true);
      log << "## Name";
      VINA_FOR_IN(i, enabled_names) { log << " " << enabled_names[i]; }
      for (unsigned i = 0, n = t.conf_independent_terms.size(); i < n; i++) {
        log << " " << t.conf_independent_terms[i].name;
      }
      log << "\n";
    }

    // ============================================================================
    // GPU Batch Docking Mode (default when --gpu is passed)
    // ============================================================================
    // When multiple ligands are provided with --gpu, batch them together for
    // efficient GPU processing. This groups similar-sized ligands and processes
    // them in batches of ~50k poses.
    if (settings.gpu && !settings.no_batch && !settings.score_only && !settings.local_only) {
      boost::timer::cpu_timer time;
      initializeCUDA(settings.device);
      thread_buffer.init(available_mem(settings.cpu));

      log << "Using GPU batch docking mode\n";
      log << "  Exhaustiveness: " << settings.exhaustiveness << "\n";
      log << "  Target batch size: " << settings.batch_size << " poses\n";
      log << "  BFGS iterations: " << settings.bfgs_iterations << "\n";
      log.endl();

      // Load all ligands
      LigandBatchManager batch_mgr;
      batch_mgr.target_total_poses = settings.batch_size;
      batch_mgr.exhaustiveness = settings.exhaustiveness;
      batch_mgr.max_gpu_memory = (size_t)(LigandBatchManager::get_available_gpu_memory() * 0.8);
      batch_mgr.fast_embed = settings.fast_embed;
      batch_mgr.parallel_embed = settings.parallel_embed;
      batch_mgr.skip_torsion_randomize = settings.skip_torsion_randomize;
      batch_mgr.prune_rms_thresh = settings.prune_rms_thresh;

      // Check if all inputs are SMILES files - use parallel RDKit loader if so
      bool all_smiles = true;
      for (const auto& fname : ligand_names) {
        std::string ext = fname.substr(fname.find_last_of(".") + 1);
        if (ext != "smi" && ext != "smiles") {
          all_smiles = false;
          break;
        }
      }

      size_t num_loaded;
      // Use parallel RDKit 3D generation for SMILES-only input (faster)
      // unless --no_rdkit_smiles is set (for debugging)
      if (all_smiles && !settings.no_rdkit_smiles) {
        log << "Using parallel RDKit 3D generation for SMILES input\n";
        num_loaded = batch_mgr.load_smiles_parallel(mols, ligand_names, log, 0, settings.verbosity);
      } else {
        num_loaded = batch_mgr.load_all_ligands(mols, ligand_names, log, settings.verbosity);
      }
      if (num_loaded == 0) {
        log << "No ligands loaded. Exiting.\n";
        return 0;
      }

      // Sort and group into batches
      batch_mgr.sort_and_group_ligands(settings.verbosity);

      // Create reference data for each ligand (for computing referenceRMSD in output)
      // Must be done before any processing modifies the coordinates
      std::map<unsigned int, reference_data> ligand_ref_data;
      for (const auto& lig : batch_mgr.all_ligands) {
        const model& m = *lig.m;
        const atomv& atoms = m.get_movable_atoms();
        const vecv& coords = m.coordinates();
        reference_data data;
        for (size_t i = 0; i < m.num_movable_atoms(); i++) {
          if (!atoms[i].is_hydrogen()) {
            data.coords.push_back(coords[i]);
          }
        }
        if (!data.coords.empty()) {
          // Create RDKit molecule from model for symmetry-aware RMSD
          try {
            std::stringstream sdf_ss;
            bool sdfvalid = false;
            m.write_ligand(sdf_ss, sdfvalid);
            std::string sdf_str = sdf_ss.str();
            if (sdf_str.find("$$$$") == std::string::npos) {
              sdf_str += "$$$$\n";
            }
            RDKit::SDMolSupplier supplier;
            supplier.setData(sdf_str);
            if (!supplier.atEnd()) {
              RDKit::ROMol *mol = supplier.next();
              if (mol) {
                data.rdkit_mol = std::shared_ptr<RDKit::ROMol>(
                    RDKit::MolOps::removeHs(*mol));
                delete mol;
              }
            }
          } catch (...) {
            // Failed to create RDKit mol, referenceRMSD will be -1
          }
          if (data.rdkit_mol) {
            ligand_ref_data[lig.ligand_id] = data;
          }
        }
      }

      // Setup precalculate for GPU
      precalculate_gpu* prec_gpu = dynamic_cast<precalculate_gpu*>(prec.get());
      if (!prec_gpu) {
        std::cerr << "ERROR: GPU batch docking requires precalculate_gpu\n";
        return 1;
      }

      // Box bounds
      vec corner1(gd[0].begin, gd[1].begin, gd[2].begin);
      vec corner2(gd[0].end, gd[1].end, gd[2].end);
      gfloat3 box_min(corner1[0], corner1[1], corner1[2]);
      gfloat3 box_max(corner2[0], corner2[1], corner2[2]);

      // Collect all atom types from all ligands (for grid population)
      std::set<smt> all_atom_types_set;
      for (const auto& lig : batch_mgr.all_ligands) {
        std::vector<smt> lig_types;
        lig.m->get_movable_atom_types(lig_types);
        all_atom_types_set.insert(lig_types.begin(), lig_types.end());
      }
      std::vector<smt> all_atom_types(all_atom_types_set.begin(), all_atom_types_set.end());

      // Determine if CNN scoring is needed
      bool use_cnn = (cnnopts.cnn_scoring != CNNnone);

      // Load CNN model
      std::shared_ptr<DLScorer> dl_scorer;
      if (use_cnn) {
        if (torchgpu) {
          dl_scorer = std::make_shared<CNNTorchScorer<true>>(cnnopts, &log);
        } else {
          dl_scorer = std::make_shared<CNNTorchScorer<false>>(cnnopts, &log);
        }
      }

      // BFGS optimization
      {
        // Create cache_gpu ONCE and populate with all atom types
        cache_gpu cgpu("scoring_function_version001", gd, 1000 /*slope*/, prec_gpu);
        cgpu.set_forcecap(settings.forcecap);
        cgpu.populate(*(batch_mgr.all_ligands[0].m), *prec, all_atom_types, user_grid);
        log << "Grid cache populated with " << all_atom_types.size() << " atom types\n";
        log.endl();

        // Process each batch
        for (size_t batch_idx = 0; batch_idx < batch_mgr.num_batches(); batch_idx++) {
          LigandBatchGroup& group = batch_mgr.batch_groups[batch_idx];

          log << "Processing batch " << (batch_idx + 1) << "/" << batch_mgr.num_batches()
              << " (" << group.ligand_indices.size() << " ligands, "
              << group.total_optimizers << " poses)\n";
          log.endl();

          // Process the batch (reuse same grid cache)
          batch_mgr.process_batch(group, cgpu, box_min, box_max,
                                 settings.bfgs_iterations, settings.seed + batch_idx, settings.verbosity);
        }
        // cgpu destructor called here, freeing grid memory
      }

      // Free BFGS GPU memory before CNN scoring
      cudaDeviceSynchronize();
      batch_mgr.clear_gpu_memory();
      c10::cuda::CUDACachingAllocator::emptyCache();

      // Refine and output results
      std::cerr << "\nRefining and writing results...\n" << std::flush;

      // Timing accumulators for post-processing phases
      double time_pose_validation = 0;
      double time_model_set = 0;
      double time_cnn_scoring = 0;
      double time_rmsd_clustering = 0;
      double time_result_creation = 0;
      double time_file_writing = 0;
      boost::timer::cpu_timer phase_timer;

      // Sort ligands back to original order
      std::sort(batch_mgr.all_ligands.begin(), batch_mgr.all_ligands.end(),
                [](const LigandDescriptor& a, const LigandDescriptor& b) {
                  return a.ligand_id < b.ligand_id;
                });

      // ========================================================================
      // Phase 1: Validate poses and compute coordinates for all ligands
      // ========================================================================
      struct LigandData {
        output_container out_cont;
        size_t ligand_idx;
      };
      std::vector<LigandData> all_ligand_data;
      all_ligand_data.reserve(batch_mgr.all_ligands.size());

      for (size_t lig_idx = 0; lig_idx < batch_mgr.all_ligands.size(); lig_idx++) {
        LigandDescriptor& lig = batch_mgr.all_ligands[lig_idx];
        model& m = *lig.m;

        // Skip if no results
        if (lig.energies.empty()) continue;

        // Convert GPU results to output_container
        output_container out_cont;
        conf init_conf = m.get_initial_conf(false);

        phase_timer.start();
        for (size_t i = 0; i < lig.energies.size(); i++) {
          float e = lig.energies[i];
          const std::vector<float>& conformation = lig.conformations[i];

          if (e >= 1e10 || !std::isfinite(e) || conformation.size() < 7) continue;

          // Validate conformation
          bool valid = true;
          for (size_t j = 0; j < conformation.size() && valid; j++) {
            if (!std::isfinite(conformation[j])) valid = false;
          }
          if (!valid) continue;

          // Check bounds
          float margin = 100.0f;
          if (conformation[0] < box_min.x - margin || conformation[0] > box_max.x + margin ||
              conformation[1] < box_min.y - margin || conformation[1] > box_max.y + margin ||
              conformation[2] < box_min.z - margin || conformation[2] > box_max.z + margin) {
            continue;
          }

          // Create conf from GPU result
          conf c = init_conf;
          if (c.ligands.empty()) continue;

          c.ligands[0].rigid.position[0] = conformation[0];
          c.ligands[0].rigid.position[1] = conformation[1];
          c.ligands[0].rigid.position[2] = conformation[2];
          c.ligands[0].rigid.orientation = qt(conformation[3], conformation[4],
                                               conformation[5], conformation[6]);
          for (size_t t = 0; t < c.ligands[0].torsions.size() && (t + 7) < conformation.size(); t++) {
            c.ligands[0].torsions[t] = conformation[t + 7];
          }

          output_type out(c, e);
          out_cont.push_back(new output_type(out));
        }
        time_pose_validation += phase_timer.elapsed().wall / 1e9;

        // Sort by energy and limit to top poses BEFORE expensive model.set() calls
        out_cont.sort();
        sz max_to_refine = settings.num_modes * settings.cnn_refine_mult;
        while (out_cont.size() > max_to_refine) {
          out_cont.pop_back();
        }

        // Now compute coordinates only for the top poses (needed for RMSD clustering)
        phase_timer.start();
        VINA_FOR_IN(i, out_cont) {
          m.set(out_cont[i].c);
          out_cont[i].coords = m.get_heavy_atom_movable_coords();
        }
        time_model_set += phase_timer.elapsed().wall / 1e9;

        if (!out_cont.empty()) {
          all_ligand_data.push_back({std::move(out_cont), lig_idx});
        }
      }

      // ========================================================================
      // Phase 2: Multi-ligand batched CNN scoring
      // ========================================================================
      phase_timer.start();
      if (use_cnn && dl_scorer && !all_ligand_data.empty()) {
        // Extract receptor coords once (same for all ligands)
        std::vector<float3> receptor_coords;
        std::vector<smt> receptor_types;

        // Get receptor from first ligand
        {
          model& m = *batch_mgr.all_ligands[all_ligand_data[0].ligand_idx].m;
          const atomv& atoms = m.get_movable_atoms();
          const vecv& coords = m.coordinates();

          // Get flex/inflex boundary
          sz num_flex_atoms = m.m_num_movable_atoms;
          if (m.ligands.size() > 0) {
            num_flex_atoms = m.ligands[0].node.begin;
          }

          // Add flex atoms (excluding covalent)
          for (sz i = 0; i < num_flex_atoms; i++) {
            if (!atoms[i].iscov) {
              const vec& c = coords[i];
              receptor_coords.push_back(float3({c[0], c[1], c[2]}));
              receptor_types.push_back(atoms[i].sm);
            }
          }

          // Add inflex atoms (excluding covalent)
          for (sz i = m.m_num_movable_atoms, n = coords.size(); i < n; i++) {
            if (!atoms[i].iscov) {
              const vec& c = coords[i];
              receptor_coords.push_back(float3({c[0], c[1], c[2]}));
              receptor_types.push_back(atoms[i].sm);
            }
          }

          // Add fixed receptor atoms
          for (const auto& a : m.get_fixed_atoms()) {
            receptor_coords.push_back(float3({a.coords[0], a.coords[1], a.coords[2]}));
            receptor_types.push_back(a.sm);
          }
        }

        // Collect ALL poses from ALL ligands
        std::vector<LigandPoseData> all_poses;
        std::vector<std::pair<size_t, size_t>> pose_to_ligand;  // Maps pose index to (ligand_data_idx, pose_idx)

        for (size_t ld_idx = 0; ld_idx < all_ligand_data.size(); ld_idx++) {
          LigandData& ld = all_ligand_data[ld_idx];
          model& m = *batch_mgr.all_ligands[ld.ligand_idx].m;
          const atomv& atoms = m.get_movable_atoms();

          // Get ligand atom offset
          sz lig_offset = 0;
          if (m.ligands.size() > 0) {
            lig_offset = m.ligands[0].node.begin;
          }
          sz num_lig_atoms = m.m_num_movable_atoms - lig_offset;

          VINA_FOR_IN(pose_idx, ld.out_cont) {
            // Set model to this conformation to get ligand coords
            m.set(ld.out_cont[pose_idx].c);
            const vecv& coords = m.coordinates();

            LigandPoseData pose_data;
            pose_data.ligand_id = ld.ligand_idx;
            pose_data.coords.reserve(num_lig_atoms);
            pose_data.types.reserve(num_lig_atoms);

            for (sz i = 0; i < num_lig_atoms; i++) {
              const vec& c = coords[i + lig_offset];
              pose_data.coords.push_back(float3({c[0], c[1], c[2]}));
              pose_data.types.push_back(atoms[i + lig_offset].sm);
            }

            // Compute center from ligand coords
            float cx = 0, cy = 0, cz = 0;
            for (const auto& coord : pose_data.coords) {
              cx += coord.x;
              cy += coord.y;
              cz += coord.z;
            }
            if (!pose_data.coords.empty()) {
              cx /= pose_data.coords.size();
              cy /= pose_data.coords.size();
              cz /= pose_data.coords.size();
            }
            pose_data.center = vec(cx, cy, cz);

            all_poses.push_back(std::move(pose_data));
            pose_to_ligand.push_back({ld_idx, pose_idx});
          }
        }

        std::cerr << "Multi-ligand CNN scoring: " << all_poses.size() << " poses from "
                  << all_ligand_data.size() << " ligands\n" << std::flush;

        // Single batched CNN scoring call for ALL poses from ALL ligands
        auto all_results = dl_scorer->score_multi_ligand_batch(
            receptor_coords, receptor_types, all_poses);

        // Distribute results back to ligands
        for (size_t i = 0; i < all_results.size(); i++) {
          size_t ld_idx = pose_to_ligand[i].first;
          size_t pose_idx = pose_to_ligand[i].second;

          all_ligand_data[ld_idx].out_cont[pose_idx].cnnscore = std::get<0>(all_results[i]);
          all_ligand_data[ld_idx].out_cont[pose_idx].cnnaffinity = std::get<1>(all_results[i]);
          all_ligand_data[ld_idx].out_cont[pose_idx].cnnvariance = std::get<2>(all_results[i]);
        }

        // Clear GPU cache after multi-ligand scoring
        c10::cuda::CUDACachingAllocator::emptyCache();
      }
      time_cnn_scoring = phase_timer.elapsed().wall / 1e9;

      // ========================================================================
      // Phase 3: Sort, cluster, and write results for each ligand
      // ========================================================================
      // For multi-conformer mode: group poses by parent_ligand_id and aggregate
      auto sorter = [&settings](const output_type& lhs, const output_type& rhs) {
        switch (settings.sort_order) {
        case Energy:
          return lhs.e < rhs.e;
        case CNNaffinity:
          return lhs.cnnaffinity > rhs.cnnaffinity;
        case CNNscore:
        default:
          return lhs.cnnscore > rhs.cnnscore;
        }
      };

      // Check if we have multi-conformer ligands to aggregate
      bool has_multi_conformer = false;
      for (const LigandData& ld : all_ligand_data) {
        if (batch_mgr.all_ligands[ld.ligand_idx].parent_ligand_id >= 0) {
          has_multi_conformer = true;
          break;
        }
      }

      if (has_multi_conformer) {
        // Group poses by parent_ligand_id
        // Pose with source model reference
        struct PoseWithModel {
          output_type* pose;
          model* source_model;
          std::string parent_name;
          unsigned int ligand_id;  // For ref_data lookup
        };
        std::map<int, std::vector<PoseWithModel>> parent_poses;

        for (LigandData& ld : all_ligand_data) {
          LigandDescriptor& lig = batch_mgr.all_ligands[ld.ligand_idx];
          int parent_id = lig.parent_ligand_id;
          if (parent_id < 0) parent_id = lig.ligand_id;  // No parent, use self

          // Get parent name (strip _confN suffix if present)
          std::string parent_name = lig.name;
          size_t conf_pos = parent_name.rfind("_conf");
          if (conf_pos != std::string::npos) {
            parent_name = parent_name.substr(0, conf_pos);
          }

          for (sz i = 0; i < ld.out_cont.size(); i++) {
            parent_poses[parent_id].push_back({&ld.out_cont[i], lig.m.get(), parent_name, lig.ligand_id});
          }
        }

        // Process each parent ligand
        for (auto& [parent_id, poses] : parent_poses) {
          if (poses.empty()) continue;

          // Sort all poses from this parent by score
          std::sort(poses.begin(), poses.end(),
            [&sorter](const PoseWithModel& a, const PoseWithModel& b) {
              return sorter(*a.pose, *b.pose);
            });

          // Trim to max_to_refine (same as single-conformer path)
          sz max_to_refine = settings.num_modes * settings.cnn_refine_mult;
          if (poses.size() > max_to_refine) {
            poses.resize(max_to_refine);
          }

          // Cluster by RMSD (coords already computed in Phase 1)
          phase_timer.start();
          std::vector<PoseWithModel> clustered;
          for (const PoseWithModel& p : poses) {
            bool dominated = false;
            for (const PoseWithModel& existing : clustered) {
              fl this_rmsd = 0;
              if (p.pose->coords.size() == existing.pose->coords.size()) {
                for (sz i = 0; i < p.pose->coords.size(); i++) {
                  fl dx = p.pose->coords[i][0] - existing.pose->coords[i][0];
                  fl dy = p.pose->coords[i][1] - existing.pose->coords[i][1];
                  fl dz = p.pose->coords[i][2] - existing.pose->coords[i][2];
                  this_rmsd += dx*dx + dy*dy + dz*dz;
                }
                this_rmsd = std::sqrt(this_rmsd / p.pose->coords.size());
                if (this_rmsd < settings.out_min_rmsd) {
                  dominated = true;
                  break;
                }
              }
            }
            if (!dominated) {
              clustered.push_back(p);
            }
          }
          time_rmsd_clustering += phase_timer.elapsed().wall / 1e9;

          // Convert to result_info and write
          phase_timer.start();
          std::vector<result_info> results;
          sz how_many = 0;
          for (const PoseWithModel& p : clustered) {
            if (!not_max(p.pose->e)) continue;
            if (how_many >= settings.num_modes) break;

            p.source_model->set(p.pose->c);
            // Look up reference data for this ligand to compute referenceRMSD
            fl refrmsd = -1;
            auto ref_it = ligand_ref_data.find(p.ligand_id);
            if (ref_it != ligand_ref_data.end()) {
              boost::optional<reference_data> ref_opt(ref_it->second);
              refrmsd = compute_reference_rmsd(*p.source_model, ref_opt);
            }
            results.push_back(result_info(p.pose->e, p.pose->cnnscore,
                                          p.pose->cnnaffinity, p.pose->cnnvariance,
                                          -1, refrmsd, *p.source_model));
            // Override name with parent name (strip _confN suffix)
            results.back().setName(p.parent_name);
            if (atomoutfile.is_open() || settings.include_atom_info) {
              results.back().setAtomValues(*p.source_model, &wt);
            }
            how_many++;
          }
          time_result_creation += phase_timer.elapsed().wall / 1e9;

          // Write results
          phase_timer.start();
          if (outfile) {
            for (unsigned j = 0; j < results.size(); j++) {
              results[j].write(outfile, outext, settings.include_atom_info, &wt, j + 1);
            }
          }
          if (outflex) {
            for (unsigned j = 0; j < results.size(); j++) {
              results[j].writeFlex(outflex, outfext, j + 1);
            }
          }
          if (atomoutfile) {
            for (unsigned j = 0; j < results.size(); j++) {
              results[j].writeAtomValues(atomoutfile, &wt);
            }
          }
          time_file_writing += phase_timer.elapsed().wall / 1e9;
        }
      } else {
        // Original behavior for non-multi-conformer mode
        for (LigandData& ld : all_ligand_data) {
          LigandDescriptor& lig = batch_mgr.all_ligands[ld.ligand_idx];
          model& m = *lig.m;
          output_container& out_cont = ld.out_cont;

          out_cont.sort(sorter);

          // Cluster by RMSD
          phase_timer.start();
          out_cont = remove_redundant(out_cont, settings.out_min_rmsd);
          time_rmsd_clustering += phase_timer.elapsed().wall / 1e9;

          // Convert to result_info and write
          phase_timer.start();
          std::vector<result_info> results;
          // Look up reference data for this ligand once
          boost::optional<reference_data> ref_opt;
          auto ref_it = ligand_ref_data.find(lig.ligand_id);
          if (ref_it != ligand_ref_data.end()) {
            ref_opt = ref_it->second;
          }
          sz how_many = 0;
          VINA_FOR_IN(i, out_cont) {
            if (!not_max(out_cont[i].e)) continue;
            if (how_many >= settings.num_modes) break;

            m.set(out_cont[i].c);
            fl refrmsd = compute_reference_rmsd(m, ref_opt);
            results.push_back(result_info(out_cont[i].e, out_cont[i].cnnscore,
                                          out_cont[i].cnnaffinity, out_cont[i].cnnvariance,
                                          -1, refrmsd, m));
            if (atomoutfile.is_open() || settings.include_atom_info) {
              results.back().setAtomValues(m, &wt);
            }
            how_many++;
          }
          time_result_creation += phase_timer.elapsed().wall / 1e9;

          // Write results
          phase_timer.start();
          if (outfile) {
            for (unsigned j = 0; j < results.size(); j++) {
              results[j].write(outfile, outext, settings.include_atom_info, &wt, j + 1);
            }
          }
          if (outflex) {
            for (unsigned j = 0; j < results.size(); j++) {
              results[j].writeFlex(outflex, outfext, j + 1);
            }
          }
          if (atomoutfile) {
            for (unsigned j = 0; j < results.size(); j++) {
              results[j].writeAtomValues(atomoutfile, &wt);
            }
          }
          time_file_writing += phase_timer.elapsed().wall / 1e9;
        }
      }

      // Print post-processing timing breakdown
      log << "\nPost-processing timing breakdown:\n";
      log << "  Pose validation (incl. model.set): " << std::fixed << std::setprecision(2) << time_pose_validation << "s\n";
      log << "    - model.set() calls:             " << time_model_set << "s\n";
      log << "  CNN scoring:                       " << time_cnn_scoring << "s\n";
      log << "  RMSD clustering:                   " << time_rmsd_clustering << "s\n";
      log << "  Result creation:                   " << time_result_creation << "s\n";
      log << "  File writing:                      " << time_file_writing << "s\n";
      double total_pp = time_pose_validation + time_cnn_scoring + time_rmsd_clustering + time_result_creation + time_file_writing;
      log << "  Total post-processing:             " << total_pp << "s\n";
      log.endl();

      // Print timing summary
      log << "\nBatch docking completed.\n";
      log << "  Total ligands: " << batch_mgr.num_ligands() << "\n";
      log << "  Total batches: " << batch_mgr.num_batches() << "\n";
      log << "  Total time: " << time.elapsed().wall / 1000000000.0 << " seconds\n";
      log << "  Time per ligand: " << (time.elapsed().wall / 1000000.0) / batch_mgr.num_ligands() << " ms\n";
      log.endl();

      cudaDeviceSynchronize();
      return 0;  // Exit after batch docking
    }
    // ============================================================================
    // End GPU Batch Docking Mode
    // ============================================================================

    int nligs = 0;
    size_t nthreads = settings.cpu;
    global_state gs(&settings, prec, &minparms, &wt, &user_grid, &log, &atomoutfile, cnnopts);
    boost::thread_group worker_threads;
    boost::timer::cpu_timer time;
    std::shared_ptr<DLScorer> dl_scorer;
    job_queue<worker_job> wrkq(nthreads*2);
    job_queue<writer_job> writerq;

    if (torchgpu)
      dl_scorer = std::make_shared<CNNTorchScorer<true>>(cnnopts, &log);
    else
      dl_scorer = std::make_shared<CNNTorchScorer<false>>(cnnopts, &log);

    if (!settings.local_only)
      nthreads = 1; // docking is multithreaded already, don't add additional parallelism other than pipeline

    // launch worker threads to process ligands in the work queue
    for (int i = 0; i < nthreads; i++) {
      worker_threads.create_thread(
          boost::bind(threads_at_work, &wrkq, &writerq, &gs, &mols, &nligs, dl_scorer->fresh_copy()));
    }

    // launch writer thread to write results wherever they go
    boost::thread writer_thread(thread_a_writing, &writerq, &gs, &outfile, &outext, &outflex, &outfext, &nligs);

    try {
      // loop over input ligands, adding them to the work queue
      for (unsigned l = 0, nl = ligand_names.size(); l < nl; l++) {
        doing(settings.verbosity, "Reading input", log);
        const std::string ligand_name = ligand_names[l];
        mols.setInputFile(ligand_name);

        unsigned i = 0;

        for (;;) {
          model *m = new model;

          if (!mols.readMoleculeIntoModel(*m)) {
            delete m;
            break;
          }
          m->set_pose_num(i);
          m->gdata.device_id = settings.device;

          grid_dims gdbox(gd);
          if (settings.local_only) {
            gdbox = m->movable_atoms_box(autobox_add, box_granularity);
            bool skip = false;
            for (unsigned pos = 0; pos < 3; pos++) {
              if (gdbox.elems[pos].n * box_granularity > 100) {
                // we will run out of memory if the grid is too large
                log << "WARNING: Ligand " << i << " in " << ligand_name
                    << " has an extent greater than 100A. Skipping.\n";
                skip = true;
                break;
              }
            }
            if (skip)
              continue;
          } else if (autobox_extend && !settings.no_lig) {
            // make sure every dimension is large enough for the ligand to fit
            fl maxdim = m->max_span(0);
            setup_grid_dims(center_x, center_y, center_z, size_x > maxdim ? size_x : maxdim,
                            size_y > maxdim ? size_y : maxdim, size_z > maxdim ? size_z : maxdim, gdbox);
          }

          done(settings.verbosity, log);
          std::vector<result_info> *results = new std::vector<result_info>();
          worker_job j(i, m, results, gdbox);
          wrkq.push(j);

          i++;
          if (settings.no_lig)
            break;
        }
      }
    } catch (...) {
      // clean up threads before passing along exception
      wrkq.close(nthreads);
      worker_threads.join_all();
      writerq.close(1);
      writer_thread.join();
      cudaDeviceSynchronize();
      throw;
    }

    // join all the threads when their work is done
    wrkq.close(nthreads);
    worker_threads.join_all();
    writerq.close(1);
    writer_thread.join();

    sz free_byte = 0, total_byte = 0;
    if (settings.verbosity > 1 && cudaMemGetInfo(&free_byte, &total_byte) == cudaSuccess) {
      double free_db = (double)free_byte;
      double total_db = (double)total_byte;
      double used_db = total_db - free_db;
      log << "GPU memory usage: " << int(used_db / 1024.0 / 1024.0) << " MB"
          << "\n";
    }

    cudaDeviceSynchronize();

    // std::cout << "Loop time " << time.elapsed().wall / 1000000000.0 << "\n";
  } catch (file_error &e) {
    std::cerr << "\n\nError: could not open \"" << e.name.string() << "\" for " << (e.in ? "reading" : "writing")
              << ".\n";
    return 1;
  } catch (boost::filesystem::filesystem_error &e) {
    std::cerr << "\n\nFile system error: " << e.what() << '\n';
    return 1;
  } catch (usage_error &e) {
    std::cerr << "\n\nUsage error: " << e.what() << "\n";
    return 1;
  } catch (parse_error &e) {
    std::cerr << "\n\nParse error on line " << e.line << " in file \"" << e.file.string() << "\": " << e.reason << '\n';
    return 1;
  } catch (std::bad_alloc &) {
    std::cerr << "\n\nError: insufficient memory!\n";
    return 1;
  } catch (scoring_function_error e) {
    std::cerr << "\n\nError with scoring function specification.\n";
    std::cerr << e.msg << "[" << e.name << "]\n";
    return 1;
  } catch (std::runtime_error e) {
    std::cerr << "\nRuntime Error\n";
    std::cerr << e.what() << "\n";
    sz free_byte = 0, total_byte = 0;
    cudaMemGetInfo(&free_byte, &total_byte);

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0,
           free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);

    return 1;
  }

  // Errors that shouldn't happen:

  catch (std::exception &e) {
    std::cerr << "\n\nAn error occurred: " << e.what() << ". " << error_message;
    return 1;
  } catch (internal_error &e) {
    std::cerr << "\n\nAn internal error occurred in " << e.file << "(" << e.line << "). " << error_message;
    return 1;
  } catch (...) {
    std::cerr << "\n\nAn unknown error occurred. " << error_message;
    return 1;
  }
}
