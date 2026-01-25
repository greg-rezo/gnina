/*
 * test_multi_ligand_batch.h
 *
 * Unit tests for multi-ligand CNN batch scoring.
 * Verifies that forward_multi_ligand_batch produces same results as forward_batch.
 */

#ifndef TEST_MULTI_LIGAND_BATCH_H_
#define TEST_MULTI_LIGAND_BATCH_H_

// Test that forward_multi_ligand_batch produces same scores as forward_batch
// within tolerance of 0.01
void test_multi_ligand_batch_vs_single();

// Test with multiple ligands of different sizes
void test_multi_ligand_batch_variable_sizes();

// Test that chunking (batches > MAX_CHUNK_SIZE) works correctly
void test_multi_ligand_batch_large_batch();

#endif /* TEST_MULTI_LIGAND_BATCH_H_ */
