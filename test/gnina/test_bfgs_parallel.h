/*
 * test_bfgs_parallel.h
 *
 * Header for parallel BFGS unit tests
 */

#ifndef TEST_BFGS_PARALLEL_H
#define TEST_BFGS_PARALLEL_H

// Test energy evaluation consistency between single-thread and cooperative GPU
void test_bfgs_energy_consistency();

// Test that BFGS steps produce valid results
void test_bfgs_step_consistency();

// Test gradient computation against numerical finite differences
void test_bfgs_numerical_gradient();

#endif // TEST_BFGS_PARALLEL_H
