/*
 * test_ligand_batch.cpp
 *
 * Unit tests for ligand batch manager and batching logic.
 */

#include "test_ligand_batch.h"
#include "ligand_batch_manager.h"
#include <iostream>
#include <cassert>
#include <cmath>

// Helper to create mock LigandDescriptor with specified dimensions
static LigandDescriptor create_mock_ligand(
    unsigned int id,
    unsigned int num_atoms,
    unsigned int num_torsions
) {
    LigandDescriptor desc;
    desc.ligand_id = id;
    desc.name = "mock_ligand_" + std::to_string(id);
    desc.num_atoms = num_atoms;
    desc.num_torsions = num_torsions;
    desc.num_nodes = 1 + num_torsions;  // 1 root + torsions
    desc.n_conf = 7 + num_torsions;     // 7 for rigid body + torsions
    desc.n_change = 6 + num_torsions;   // 6 for rigid body + torsions
    desc.gpu_initialized = false;
    return desc;
}

void test_batch_memory_estimation() {
    std::cout << "Testing batch memory estimation..." << std::flush;

    LigandBatchGroup group;
    group.exhaustiveness = 1024;

    // Create some mock ligands
    std::vector<LigandDescriptor> ligands;
    ligands.push_back(create_mock_ligand(0, 30, 5));
    ligands.push_back(create_mock_ligand(1, 40, 8));
    ligands.push_back(create_mock_ligand(2, 35, 6));

    for (auto& lig : ligands) {
        group.ligands.push_back(&lig);
    }
    group.compute_max_dimensions();

    // Verify max dimensions
    assert(group.max_atoms == 40);
    assert(group.max_torsions == 8);
    assert(group.max_nodes == 9);  // 1 + 8
    assert(group.max_conf_size == 15);  // 7 + 8
    assert(group.max_change_size == 14);  // 6 + 8
    assert(group.total_optimizers == 3 * 1024);

    // Verify memory estimation is reasonable (should be > 0 and not huge)
    size_t mem = group.estimate_memory();
    assert(mem > 0);
    assert(mem < 1e9);  // Less than 1GB for 3 small ligands

    // Memory should scale linearly with total_optimizers
    LigandBatchGroup group2 = group;
    group2.total_optimizers = group.total_optimizers * 2;
    size_t mem2 = group2.estimate_memory();
    // Should be roughly double (within 10% due to fixed overhead)
    assert(mem2 > mem * 1.8 && mem2 < mem * 2.2);

    std::cout << " PASSED" << std::endl;
    std::cout << "  Memory per batch (3 ligands, 3072 poses): "
              << mem / (1024.0 * 1024.0) << " MB" << std::endl;
}

void test_batch_compatibility() {
    std::cout << "Testing batch compatibility checking..." << std::flush;

    LigandBatchGroup group;
    group.exhaustiveness = 1024;

    // Add a small ligand
    std::vector<LigandDescriptor> ligands;
    ligands.push_back(create_mock_ligand(0, 20, 3));
    group.ligands.push_back(&ligands[0]);
    group.compute_max_dimensions();

    // Similar-sized ligand should be compatible
    LigandDescriptor similar = create_mock_ligand(1, 25, 4);
    assert(group.is_compatible(similar, 2.0f) == true);

    // Much larger ligand should NOT be compatible (2x ratio check)
    LigandDescriptor large = create_mock_ligand(2, 50, 10);  // 2.5x atoms
    assert(group.is_compatible(large, 2.0f) == false);

    // Empty group should accept any ligand
    LigandBatchGroup empty_group;
    empty_group.exhaustiveness = 1024;
    assert(empty_group.is_compatible(large, 2.0f) == true);

    std::cout << " PASSED" << std::endl;
}

void test_batch_grouping() {
    std::cout << "Testing batch sorting and grouping..." << std::flush;

    // Create a batch manager with small target for testing
    LigandBatchManager mgr;
    mgr.target_total_poses = 5000;  // Small batch for testing
    mgr.exhaustiveness = 1000;
    mgr.max_gpu_memory = 0;  // No memory limit

    // Create ligands of varying sizes
    mgr.all_ligands.push_back(create_mock_ligand(0, 50, 10));  // Large
    mgr.all_ligands.push_back(create_mock_ligand(1, 20, 3));   // Small
    mgr.all_ligands.push_back(create_mock_ligand(2, 45, 9));   // Large
    mgr.all_ligands.push_back(create_mock_ligand(3, 22, 4));   // Small
    mgr.all_ligands.push_back(create_mock_ligand(4, 48, 11));  // Large
    mgr.all_ligands.push_back(create_mock_ligand(5, 25, 5));   // Small

    // Run grouping algorithm
    mgr.sort_and_group_ligands(0);  // Silent

    // With target 5000 poses and 1000 exhaustiveness, we can fit 5 ligands per batch
    // But size compatibility may cause additional splits
    assert(mgr.num_batches() > 0);

    // Verify that similar-sized ligands are grouped together
    // (small ligands should be in different batches than large ones)
    for (const LigandBatchGroup& group : mgr.batch_groups) {
        if (group.ligands.size() > 1) {
            unsigned min_atoms = UINT_MAX, max_atoms = 0;
            for (const LigandDescriptor* lig : group.ligands) {
                if (lig->num_atoms < min_atoms) min_atoms = lig->num_atoms;
                if (lig->num_atoms > max_atoms) max_atoms = lig->num_atoms;
            }
            // Verify ratio constraint is respected
            assert(max_atoms <= min_atoms * 2);
        }
    }

    std::cout << " PASSED" << std::endl;
    std::cout << "  Created " << mgr.num_batches() << " batches from 6 ligands" << std::endl;
}

void test_batch_pose_distribution() {
    std::cout << "Testing batch pose distribution..." << std::flush;

    LigandBatchManager mgr;
    mgr.target_total_poses = 3000;
    mgr.exhaustiveness = 1000;
    mgr.max_gpu_memory = 0;

    // Add 10 similar-sized ligands (should be compatible)
    for (int i = 0; i < 10; i++) {
        mgr.all_ligands.push_back(create_mock_ligand(i, 30 + i % 3, 5));
    }

    mgr.sort_and_group_ligands(0);

    // Total poses across all batches should equal 10 * 1000 = 10000
    int total_poses = 0;
    for (const LigandBatchGroup& group : mgr.batch_groups) {
        total_poses += group.total_optimizers;
        // Each batch should not exceed target (or be close to it)
        assert(group.total_optimizers <= mgr.target_total_poses + mgr.exhaustiveness);
    }
    assert(total_poses == 10 * mgr.exhaustiveness);

    // With target 3000 and exhaustiveness 1000, expect ~4 batches
    // (3 ligands per batch, with 10 ligands total)
    assert(mgr.num_batches() >= 3 && mgr.num_batches() <= 5);

    std::cout << " PASSED" << std::endl;
    std::cout << "  " << mgr.num_batches() << " batches for 10 ligands, "
              << total_poses << " total poses" << std::endl;
}

void test_batch_memory_constraint() {
    std::cout << "Testing batch memory constraint..." << std::flush;

    LigandBatchManager mgr;
    mgr.target_total_poses = 100000;  // Large target
    mgr.exhaustiveness = 1000;
    mgr.max_gpu_memory = 10 * 1024 * 1024;  // 10MB limit (very small for testing)

    // Add many similar-sized ligands
    for (int i = 0; i < 50; i++) {
        mgr.all_ligands.push_back(create_mock_ligand(i, 40, 8));
    }

    mgr.sort_and_group_ligands(0);

    // Each batch should respect memory constraint
    for (const LigandBatchGroup& group : mgr.batch_groups) {
        size_t estimated_mem = group.estimate_memory();
        // Allow some tolerance since we check before adding a ligand
        assert(estimated_mem <= mgr.max_gpu_memory * 1.1);
    }

    std::cout << " PASSED" << std::endl;
    std::cout << "  Created " << mgr.num_batches() << " batches with 10MB limit" << std::endl;
}
