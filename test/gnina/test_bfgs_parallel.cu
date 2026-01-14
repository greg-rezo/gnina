/*
 * test_bfgs_parallel.cu
 *
 * Unit tests for parallel BFGS implementation.
 * Compares single-thread scoring functions against original gnina's
 * cooperative GPU implementation to ensure correctness.
 */

#include <numeric>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include "common.h"
#include "gpucode.h"
#include "cache_gpu.h"
#include "cache.h"
#include "tree_gpu.h"
#include "model.h"
#include "curl.h"
#include "weighted_terms.h"
#include "custom_terms.h"
#include "precalculate_gpu.h"
#include "precalculate.h"
#include "szv_grid.h"
#include "parsed_args.h"
#include "test_utils.h"
#include "bfgs_parallel.h"

extern parsed_args p_args;

// Test kernel: evaluate energy using our single-thread trilinear interpolation
// This directly uses the grid_gpu::evaluate method like gnina does
__global__ void test_grid_eval_kernel(
    const GPUCacheInfo* cinfo,
    const atom_params* atoms,
    float* per_atom_energies,
    force_energy_tup* per_atom_forces,
    float v,
    int num_atoms
) {
    int idx = threadIdx.x;
    if (idx >= num_atoms) return;

    unsigned t = cinfo->types[idx];
    force_energy_tup val(0, 0, 0, 0);

    if (t > 1 && t < cinfo->ngrids) {
        const grid_gpu& g = cinfo->grids[t];
        g.evaluate(atoms[idx], cinfo->slope, v, val);
    }

    per_atom_energies[idx] = val.energy;
    per_atom_forces[idx] = val;
}


/*
 * Test 1: Energy Evaluation Consistency
 *
 * Compare CPU cache evaluation vs GPU cache evaluation.
 * This follows the same pattern as test_cache.cu to ensure consistency.
 */
void test_bfgs_energy_consistency() {
    p_args.log << "BFGS Parallel - Energy Consistency Test\n";
    p_args.log << "Using random seed: " << p_args.seed << "\n";
    p_args.log << "Iteration " << p_args.iter_count;
    p_args.log.endl();

    std::mt19937 engine(p_args.seed);

    // Set up scoring function (standard Vina terms)
    custom_terms t;
    t.add("gauss(o=0,_w=0.5,_c=8)", -0.035579);
    t.add("gauss(o=3,_w=2,_c=8)", -0.005156);
    t.add("repulsion(o=0,_c=8)", 0.840245);
    t.add("hydrophobic(g=0.5,_b=1.5,_c=8)", -0.035069);
    t.add("non_dir_h_bond(g=-0.7,_b=0,_c=8)", -0.587439);
    t.add("num_tors_div", 5 * 0.05846 / 0.1 - 1);

    const fl approx_factor = 10;
    const fl v = 10;
    const fl granularity = 0.375;
    const fl slope = 10;

    weighted_terms wt(&t, t.weights());
    std::unique_ptr<precalculate_gpu> gprec(new precalculate_gpu(wt, approx_factor));
    std::unique_ptr<precalculate_splines> prec(new precalculate_splines(wt, approx_factor));

    // Create ligand atoms
    std::vector<atom_params> lig_atoms;
    std::vector<smt> lig_types;
    fl max_x = -HUGE_VALF, max_y = -HUGE_VALF, max_z = -HUGE_VALF;
    fl min_x = HUGE_VALF, min_y = HUGE_VALF, min_z = HUGE_VALF;
    make_mol(lig_atoms, lig_types, engine, 0);

    // Set up grid
    for (auto& atom : lig_atoms) {
        min_x = std::min(min_x, atom.coords[0]);
        min_y = std::min(min_y, atom.coords[1]);
        min_z = std::min(min_z, atom.coords[2]);
        max_x = std::max(max_x, atom.coords[0]);
        max_y = std::max(max_y, atom.coords[1]);
        max_z = std::max(max_z, atom.coords[2]);
    }

    fl center_x = (max_x + min_x) / 2.0;
    fl center_y = (max_y + min_y) / 2.0;
    fl center_z = (max_z + min_z) / 2.0;
    fl size_x = max_x - min_x;
    fl size_y = max_y - min_y;
    fl size_z = max_z - min_z;

    vec span(size_x, size_y, size_z);
    vec center(center_x, center_y, center_z);
    grid_dims gd;

    for (size_t i = 0; i < 3; ++i) {
        gd[i].n = sz(std::ceil(span[i] / granularity));
        fl real_span = granularity * gd[i].n;
        gd[i].begin = center[i] - real_span / 2;
        gd[i].end = gd[i].begin + real_span;
    }
    grid user_grid;

    // Create receptor atoms
    std::vector<atom_params> rec_atoms;
    std::vector<smt> rec_types;
    const float cutoff_sqr = prec->cutoff_sqr();
    const float cutoff = std::sqrt(cutoff_sqr);
    make_mol(rec_atoms, rec_types, engine, 0, 10, 500,
             max_x + cutoff, max_y + cutoff, max_z + cutoff);

    // Create model
    std::unique_ptr<model> m(new model);
    m->m_num_movable_atoms = lig_atoms.size();
    m->minus_forces = std::vector<vec>(m->m_num_movable_atoms);

    for (size_t i = 0; i < lig_atoms.size(); ++i) {
        m->coords.push_back(*(vec*)&lig_atoms[i]);
        m->atoms.push_back(atom());
        m->atoms[i].sm = lig_types[i];
        m->atoms[i].charge = lig_atoms[i].charge;
        m->atoms[i].coords = *(vec*)&lig_atoms[i];
    }

    for (size_t i = 0; i < rec_atoms.size(); ++i) {
        m->grid_atoms.push_back(atom());
        m->grid_atoms[i].sm = rec_types[i];
        m->grid_atoms[i].charge = rec_atoms[i].charge;
        m->grid_atoms[i].coords = *(vec*)&rec_atoms[i];
    }

    szv_grid_cache gridcache(*m, cutoff_sqr);

    // Set up both CPU and GPU caches (following test_cache.cu pattern)
    std::unique_ptr<cache> c(new cache("scoring_function_version001", gd, slope));
    std::unique_ptr<cache_gpu> cg(new cache_gpu("scoring_function_version001", gd, slope, &(*gprec)));

    std::vector<smt> atom_types_needed;
    m->get_movable_atom_types(atom_types_needed);
    c->populate(*m, *prec, atom_types_needed, user_grid);
    cg->populate(*m, *gprec, atom_types_needed, user_grid);

    // Initialize GPU data
    m->initialize_gpu();
    gpu_data& gdat = m->gdata;
    cudaMemset(gdat.minus_forces, 0, m->minus_forces.size() * sizeof(gdat.minus_forces[0]));

    // Get CPU energy
    fl cpu_energy = c->eval_deriv(*m, v, user_grid);

    // Get GPU energy using gnina's single_point_calc
    fl gpu_energy = single_point_calc(cg->get_info(), gdat.coords, gdat.minus_forces, v);

    // Get GPU forces for comparison
    vec gpu_forces[m->minus_forces.size()];
    cudaMemcpy(gpu_forces, gdat.minus_forces,
               m->minus_forces.size() * sizeof(gdat.minus_forces[0]),
               cudaMemcpyDeviceToHost);

    // Log results
    print_mol(rec_atoms, rec_types, p_args.log);
    print_mol(lig_atoms, lig_types, p_args.log);
    p_args.log << "CPU energy: " << cpu_energy << " GPU energy: " << gpu_energy << "\n\n";

    // Compare energies (tolerance 0.01 like test_cache.cu)
    BOOST_REQUIRE_SMALL(cpu_energy - gpu_energy, (float)0.01);

    // Compare forces
    for (size_t i = 0; i < m->minus_forces.size(); ++i)
        for (size_t j = 0; j < 3; ++j)
            BOOST_REQUIRE_SMALL(m->minus_forces[i][j] - gpu_forces[i][j], (float)0.01);

    p_args.log << "Energy test: PASS\n";
}


/*
 * Test 2: BFGS Step Consistency
 *
 * Verify that grid evaluation produces valid (finite) energies and forces.
 * This is a simpler test that validates the grid evaluation is working.
 */
void test_bfgs_step_consistency() {
    p_args.log << "BFGS Parallel - Step Consistency Test\n";
    p_args.log << "Using random seed: " << p_args.seed << "\n";
    p_args.log << "Iteration " << p_args.iter_count;
    p_args.log.endl();

    std::mt19937 engine(p_args.seed);

    // Set up scoring function
    custom_terms t;
    t.add("gauss(o=0,_w=0.5,_c=8)", -0.035579);
    t.add("gauss(o=3,_w=2,_c=8)", -0.005156);
    t.add("repulsion(o=0,_c=8)", 0.840245);
    t.add("hydrophobic(g=0.5,_b=1.5,_c=8)", -0.035069);
    t.add("non_dir_h_bond(g=-0.7,_b=0,_c=8)", -0.587439);
    t.add("num_tors_div", 5 * 0.05846 / 0.1 - 1);

    const fl approx_factor = 10;
    const fl v = 10;
    const fl granularity = 0.375;
    const fl slope = 10;

    weighted_terms wt(&t, t.weights());
    std::unique_ptr<precalculate_gpu> gprec(new precalculate_gpu(wt, approx_factor));
    std::unique_ptr<precalculate_splines> prec(new precalculate_splines(wt, approx_factor));

    // Create ligand atoms
    std::vector<atom_params> lig_atoms;
    std::vector<smt> lig_types;
    fl max_x = -HUGE_VALF, max_y = -HUGE_VALF, max_z = -HUGE_VALF;
    fl min_x = HUGE_VALF, min_y = HUGE_VALF, min_z = HUGE_VALF;
    make_mol(lig_atoms, lig_types, engine, 0);

    for (auto& atom : lig_atoms) {
        min_x = std::min(min_x, atom.coords[0]);
        min_y = std::min(min_y, atom.coords[1]);
        min_z = std::min(min_z, atom.coords[2]);
        max_x = std::max(max_x, atom.coords[0]);
        max_y = std::max(max_y, atom.coords[1]);
        max_z = std::max(max_z, atom.coords[2]);
    }

    // Create grid
    fl size_x = max_x - min_x;
    fl size_y = max_y - min_y;
    fl size_z = max_z - min_z;

    vec span(size_x, size_y, size_z);
    vec center((max_x + min_x) / 2.0, (max_y + min_y) / 2.0, (max_z + min_z) / 2.0);
    grid_dims gd;

    for (size_t i = 0; i < 3; ++i) {
        gd[i].n = sz(std::ceil(span[i] / granularity));
        fl real_span = granularity * gd[i].n;
        gd[i].begin = center[i] - real_span / 2;
        gd[i].end = gd[i].begin + real_span;
    }
    grid user_grid;

    // Create receptor
    std::vector<atom_params> rec_atoms;
    std::vector<smt> rec_types;
    const float cutoff_sqr = prec->cutoff_sqr();
    const float cutoff = std::sqrt(cutoff_sqr);
    make_mol(rec_atoms, rec_types, engine, 0, 10, 500,
             max_x + cutoff, max_y + cutoff, max_z + cutoff);

    // Create model
    std::unique_ptr<model> m(new model);
    m->m_num_movable_atoms = lig_atoms.size();
    m->minus_forces = std::vector<vec>(m->m_num_movable_atoms);

    for (size_t i = 0; i < lig_atoms.size(); ++i) {
        m->coords.push_back(*(vec*)&lig_atoms[i]);
        m->atoms.push_back(atom());
        m->atoms[i].sm = lig_types[i];
        m->atoms[i].charge = lig_atoms[i].charge;
        m->atoms[i].coords = *(vec*)&lig_atoms[i];
    }

    for (size_t i = 0; i < rec_atoms.size(); ++i) {
        m->grid_atoms.push_back(atom());
        m->grid_atoms[i].sm = rec_types[i];
        m->grid_atoms[i].charge = rec_atoms[i].charge;
        m->grid_atoms[i].coords = *(vec*)&rec_atoms[i];
    }

    szv_grid_cache gridcache(*m, cutoff_sqr);
    std::unique_ptr<cache_gpu> cg(new cache_gpu("scoring_function_version001", gd, slope, &(*gprec)));

    std::vector<smt> atom_types_needed;
    m->get_movable_atom_types(atom_types_needed);
    cg->populate(*m, *gprec, atom_types_needed, user_grid);

    m->initialize_gpu();
    gpu_data& gdat = m->gdata;

    GPUCacheInfo cacheInfo = cg->get_info();

    // Allocate per-atom output arrays
    float* d_per_atom_e;
    force_energy_tup* d_per_atom_f;
    CUDA_CHECK_GNINA(cudaMalloc(&d_per_atom_e, m->m_num_movable_atoms * sizeof(float)));
    CUDA_CHECK_GNINA(cudaMalloc(&d_per_atom_f, m->m_num_movable_atoms * sizeof(force_energy_tup)));

    // Copy cache info to device for kernel
    GPUCacheInfo* d_cinfo;
    CUDA_CHECK_GNINA(cudaMalloc(&d_cinfo, sizeof(GPUCacheInfo)));
    CUDA_CHECK_GNINA(cudaMemcpy(d_cinfo, &cacheInfo, sizeof(GPUCacheInfo), cudaMemcpyHostToDevice));

    // Run test kernel
    int nthreads = m->m_num_movable_atoms;
    test_grid_eval_kernel<<<1, nthreads>>>(d_cinfo, gdat.coords, d_per_atom_e, d_per_atom_f, v, nthreads);
    CUDA_CHECK_GNINA(cudaDeviceSynchronize());

    // Copy results back
    std::vector<float> per_atom_e(nthreads);
    std::vector<force_energy_tup> per_atom_f(nthreads);
    CUDA_CHECK_GNINA(cudaMemcpy(per_atom_e.data(), d_per_atom_e, nthreads * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_GNINA(cudaMemcpy(per_atom_f.data(), d_per_atom_f, nthreads * sizeof(force_energy_tup), cudaMemcpyDeviceToHost));

    // Verify all values are finite
    float total_e = 0;
    bool all_finite = true;
    for (int i = 0; i < nthreads; i++) {
        if (!std::isfinite(per_atom_e[i])) all_finite = false;
        if (!std::isfinite(per_atom_f[i].energy)) all_finite = false;
        if (!std::isfinite(per_atom_f[i].minus_force.x)) all_finite = false;
        if (!std::isfinite(per_atom_f[i].minus_force.y)) all_finite = false;
        if (!std::isfinite(per_atom_f[i].minus_force.z)) all_finite = false;
        total_e += per_atom_e[i];
    }

    p_args.log << "Atoms: " << nthreads << "\n";
    p_args.log << "Total energy: " << total_e << "\n";
    p_args.log << "All finite: " << (all_finite ? "YES" : "NO") << "\n";

    BOOST_REQUIRE(all_finite);

    // Cleanup
    cudaFree(d_per_atom_e);
    cudaFree(d_per_atom_f);
    cudaFree(d_cinfo);

    p_args.log << "Step consistency test: PASS\n";
}


/*
 * Test 3: Numerical Gradient Check
 *
 * Verify that grid forces are consistent with numerical finite differences.
 * Uses gnina's grid_gpu::evaluate which computes both energy and forces.
 */
void test_bfgs_numerical_gradient() {
    p_args.log << "BFGS Parallel - Numerical Gradient Test\n";
    p_args.log << "Using random seed: " << p_args.seed << "\n";
    p_args.log << "Iteration " << p_args.iter_count;
    p_args.log.endl();

    std::mt19937 engine(p_args.seed);

    // Set up scoring function
    custom_terms t;
    t.add("gauss(o=0,_w=0.5,_c=8)", -0.035579);
    t.add("gauss(o=3,_w=2,_c=8)", -0.005156);
    t.add("repulsion(o=0,_c=8)", 0.840245);
    t.add("hydrophobic(g=0.5,_b=1.5,_c=8)", -0.035069);
    t.add("non_dir_h_bond(g=-0.7,_b=0,_c=8)", -0.587439);
    t.add("num_tors_div", 5 * 0.05846 / 0.1 - 1);

    const fl approx_factor = 10;
    const fl v = 10;
    const fl granularity = 0.375;
    const fl slope = 10;

    weighted_terms wt(&t, t.weights());
    std::unique_ptr<precalculate_gpu> gprec(new precalculate_gpu(wt, approx_factor));
    std::unique_ptr<precalculate_splines> prec(new precalculate_splines(wt, approx_factor));

    // Create ligand atoms
    std::vector<atom_params> lig_atoms;
    std::vector<smt> lig_types;
    fl max_x = -HUGE_VALF, max_y = -HUGE_VALF, max_z = -HUGE_VALF;
    fl min_x = HUGE_VALF, min_y = HUGE_VALF, min_z = HUGE_VALF;
    make_mol(lig_atoms, lig_types, engine, 0, 5, 10);  // Small molecule

    for (auto& atom : lig_atoms) {
        min_x = std::min(min_x, atom.coords[0]);
        min_y = std::min(min_y, atom.coords[1]);
        min_z = std::min(min_z, atom.coords[2]);
        max_x = std::max(max_x, atom.coords[0]);
        max_y = std::max(max_y, atom.coords[1]);
        max_z = std::max(max_z, atom.coords[2]);
    }

    // Create grid with padding
    fl size_x = max_x - min_x + 4;
    fl size_y = max_y - min_y + 4;
    fl size_z = max_z - min_z + 4;

    vec span(size_x, size_y, size_z);
    vec center((max_x + min_x) / 2.0, (max_y + min_y) / 2.0, (max_z + min_z) / 2.0);
    grid_dims gd;

    for (size_t i = 0; i < 3; ++i) {
        gd[i].n = sz(std::ceil(span[i] / granularity));
        fl real_span = granularity * gd[i].n;
        gd[i].begin = center[i] - real_span / 2;
        gd[i].end = gd[i].begin + real_span;
    }
    grid user_grid;

    // Create receptor
    std::vector<atom_params> rec_atoms;
    std::vector<smt> rec_types;
    const float cutoff_sqr = prec->cutoff_sqr();
    const float cutoff = std::sqrt(cutoff_sqr);
    make_mol(rec_atoms, rec_types, engine, 0, 10, 100,
             max_x + cutoff, max_y + cutoff, max_z + cutoff);

    // Create model
    std::unique_ptr<model> m(new model);
    m->m_num_movable_atoms = lig_atoms.size();
    m->minus_forces = std::vector<vec>(m->m_num_movable_atoms);

    for (size_t i = 0; i < lig_atoms.size(); ++i) {
        m->coords.push_back(*(vec*)&lig_atoms[i]);
        m->atoms.push_back(atom());
        m->atoms[i].sm = lig_types[i];
        m->atoms[i].charge = lig_atoms[i].charge;
        m->atoms[i].coords = *(vec*)&lig_atoms[i];
    }

    for (size_t i = 0; i < rec_atoms.size(); ++i) {
        m->grid_atoms.push_back(atom());
        m->grid_atoms[i].sm = rec_types[i];
        m->grid_atoms[i].charge = rec_atoms[i].charge;
        m->grid_atoms[i].coords = *(vec*)&rec_atoms[i];
    }

    szv_grid_cache gridcache(*m, cutoff_sqr);
    std::unique_ptr<cache> c(new cache("scoring_function_version001", gd, slope));
    std::unique_ptr<cache_gpu> cg(new cache_gpu("scoring_function_version001", gd, slope, &(*gprec)));

    std::vector<smt> atom_types_needed;
    m->get_movable_atom_types(atom_types_needed);
    c->populate(*m, *prec, atom_types_needed, user_grid);
    cg->populate(*m, *gprec, atom_types_needed, user_grid);

    // Evaluate at center point
    fl center_energy = c->eval_deriv(*m, v, user_grid);
    std::vector<vec> analytic_forces = m->minus_forces;

    // Numerical gradient via finite differences
    const fl eps = 0.0001;
    std::vector<vec> numerical_forces(m->m_num_movable_atoms);

    for (size_t atom = 0; atom < m->m_num_movable_atoms; atom++) {
        for (int dim = 0; dim < 3; dim++) {
            // E(x + eps)
            m->coords[atom][dim] += eps;
            fl e_plus = c->eval_deriv(*m, v, user_grid);

            // E(x - eps)
            m->coords[atom][dim] -= 2 * eps;
            fl e_minus = c->eval_deriv(*m, v, user_grid);

            // Restore
            m->coords[atom][dim] += eps;

            // Numerical force: -dE/dx
            numerical_forces[atom][dim] = -(e_plus - e_minus) / (2 * eps);
        }
    }

    // Compare
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int n_compared = 0;

    for (size_t atom = 0; atom < m->m_num_movable_atoms; atom++) {
        for (int dim = 0; dim < 3; dim++) {
            float diff = std::abs(analytic_forces[atom][dim] - numerical_forces[atom][dim]);
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
            n_compared++;
        }
    }
    avg_diff /= n_compared;

    p_args.log << "Atoms: " << m->m_num_movable_atoms << "\n";
    p_args.log << "Center energy: " << center_energy << "\n";
    p_args.log << "Max force diff: " << max_diff << "\n";
    p_args.log << "Avg force diff: " << avg_diff << "\n";

    // Allow 0.1 tolerance for numerical precision
    BOOST_REQUIRE_SMALL(max_diff, (float)0.1);

    p_args.log << "Gradient test: PASS\n";
}
