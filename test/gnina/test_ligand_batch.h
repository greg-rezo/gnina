/*
 * test_ligand_batch.h
 *
 * Unit tests for ligand batch manager and batching logic.
 */

#ifndef TEST_LIGAND_BATCH_H
#define TEST_LIGAND_BATCH_H

// Test LigandBatchGroup memory estimation
void test_batch_memory_estimation();

// Test LigandBatchGroup compatibility checking (size mismatch detection)
void test_batch_compatibility();

// Test LigandBatchManager sorting and grouping algorithm
void test_batch_grouping();

// Test that ligands are properly distributed to batches based on target pose count
void test_batch_pose_distribution();

// Test that batch groups respect max GPU memory constraint
void test_batch_memory_constraint();

#endif // TEST_LIGAND_BATCH_H
