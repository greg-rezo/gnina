/*
 * test_bfgs_parallel.h
 *
 * Header for parallel BFGS unit tests.
 * Tests comparing CPU (bfgs.h) and GPU (bfgs_parallel.cu) implementations.
 */

#ifndef TEST_BFGS_PARALLEL_H
#define TEST_BFGS_PARALLEL_H

// Test energy evaluation consistency between single-thread and cooperative GPU
void test_bfgs_energy_consistency();

// Test that BFGS steps produce valid results
void test_bfgs_step_consistency();

// Test gradient computation against numerical finite differences
void test_bfgs_numerical_gradient();

// ============================================================================
// CPU vs GPU BFGS Component Comparison Tests (tight tolerances)
// ============================================================================

// Test search direction computation: p = -H * g
// Compares CPU minus_mat_vec_product() with GPU kernel
// Tolerance: 1e-5
void test_bfgs_search_direction_cpu_vs_gpu();

// Test BFGS Hessian update
// Compares CPU bfgs_update() with GPU bfgs_hessian_update_single_thread()
// Tolerance: 1e-5
void test_bfgs_hessian_update_cpu_vs_gpu();

// Test line search foundation (energy consistency for line search)
// Verifies CPU and GPU produce consistent energies for identical configurations
// Tolerance: 0.01
void test_bfgs_line_search_cpu_vs_gpu();

// Test full BFGS single step (search direction + Hessian update)
// Verifies one complete iteration produces identical results
// Tolerance: 1e-5
void test_bfgs_single_step_cpu_vs_gpu();

// Test multi-step BFGS trajectory
// Verifies CPU and GPU follow same optimization trajectory over 5 iterations
// Tolerance: 1e-4 (accumulated numerical error)
void test_bfgs_trajectory_cpu_vs_gpu();

#endif // TEST_BFGS_PARALLEL_H
