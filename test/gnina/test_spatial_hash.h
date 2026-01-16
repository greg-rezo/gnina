/*
 * test_spatial_hash.h
 *
 * Unit tests for spatial hash neighbor finding.
 */

#ifndef TEST_SPATIAL_HASH_H
#define TEST_SPATIAL_HASH_H

#include "spatial_hash.h"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <random>

// ============================================================================
// Test 1: Basic Cell Indexing
// ============================================================================

inline void test_spatial_hash_cell_indexing() {
    printf("Testing spatial hash cell indexing...\n");

    SpatialHashBuilder builder;

    // Create a simple grid
    std::vector<float> coords = {0, 0, 0};  // Single atom at origin
    std::vector<uint8_t> types = {5};  // Carbon type

    float3 box_min = make_float3(-10, -10, -10);
    float3 box_max = make_float3(10, 10, 10);

    builder.build(coords, types, box_min, box_max, 8.0f);

    // Verify grid dimensions
    // Box is 20x20x20 + 2*8 margin = 36x36x36, cell size 8 -> 5x5x5 cells
    printf("  Grid dim: %d x %d x %d\n", builder.grid_dim.x, builder.grid_dim.y, builder.grid_dim.z);
    assert(builder.grid_dim.x >= 3);
    assert(builder.grid_dim.y >= 3);
    assert(builder.grid_dim.z >= 3);

    // Test cell index calculation
    int cell_idx = builder.get_cell_index(0, 0, 0);
    printf("  Origin cell index: %d\n", cell_idx);
    assert(cell_idx >= 0 && cell_idx < builder.num_cells);

    // Test cell coordinate round-trip
    int3 cell_coords = builder.get_cell_coords(cell_idx);
    int cell_idx2 = cell_coords.x + cell_coords.y * builder.grid_dim.x +
                    cell_coords.z * builder.grid_dim.x * builder.grid_dim.y;
    assert(cell_idx == cell_idx2);

    printf("  Cell indexing tests passed\n");
}

// ============================================================================
// Test 2: Neighbor Finding Accuracy
// ============================================================================

inline void test_spatial_hash_neighbor_accuracy() {
    printf("Testing spatial hash neighbor finding accuracy...\n");

    SpatialHashBuilder builder;

    // Create a cluster of atoms
    std::vector<float> coords;
    std::vector<uint8_t> types;

    // Place atoms in a 3x3x3 grid with 4 Angstrom spacing
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                coords.push_back(x * 4.0f);
                coords.push_back(y * 4.0f);
                coords.push_back(z * 4.0f);
                types.push_back(5);  // Carbon
            }
        }
    }

    int num_atoms = coords.size() / 3;
    printf("  Created %d atoms in 3x3x3 grid\n", num_atoms);

    float3 box_min = make_float3(-10, -10, -10);
    float3 box_max = make_float3(10, 10, 10);

    builder.build(coords, types, box_min, box_max, 8.0f);
    builder.print_stats();

    // Test neighbor finding for center atom (should find all 26 neighbors)
    std::vector<int> neighbors;
    builder.find_neighbors_host(0, 0, 0, 8.0f, neighbors);

    // Center atom at (0,0,0) should find neighbors at distance:
    // - 4.0 (face neighbors): 6 atoms
    // - 5.66 (edge neighbors): 12 atoms
    // - 6.93 (corner neighbors): 8 atoms
    // Total within 8 A: 26 neighbors (excluding self)

    printf("  Center atom has %zu neighbors within 8 A\n", neighbors.size());

    // The center atom itself might be included, let's check
    int expected_neighbors = 26;  // All surrounding atoms
    // Allow for the center atom itself to be included
    assert(neighbors.size() >= (size_t)expected_neighbors);

    printf("  Neighbor finding accuracy test passed\n");
}

// ============================================================================
// Test 3: Edge Cases
// ============================================================================

inline void test_spatial_hash_edge_cases() {
    printf("Testing spatial hash edge cases...\n");

    SpatialHashBuilder builder;

    // Test 1: Empty receptor
    {
        std::vector<float> coords;
        std::vector<uint8_t> types;

        float3 box_min = make_float3(-10, -10, -10);
        float3 box_max = make_float3(10, 10, 10);

        builder.build(coords, types, box_min, box_max, 8.0f);

        std::vector<int> neighbors;
        builder.find_neighbors_host(0, 0, 0, 8.0f, neighbors);
        assert(neighbors.size() == 0);
        printf("  Empty receptor test passed\n");
    }

    // Test 2: Single atom
    {
        std::vector<float> coords = {5.0f, 5.0f, 5.0f};
        std::vector<uint8_t> types = {5};

        float3 box_min = make_float3(0, 0, 0);
        float3 box_max = make_float3(10, 10, 10);

        builder.build(coords, types, box_min, box_max, 8.0f);

        // Query near the atom
        std::vector<int> neighbors;
        builder.find_neighbors_host(5.0f, 5.0f, 5.0f, 8.0f, neighbors);
        assert(neighbors.size() == 1);  // Should find the single atom

        // Query far from the atom
        builder.find_neighbors_host(-5.0f, -5.0f, -5.0f, 8.0f, neighbors);
        assert(neighbors.size() == 0);  // Should find nothing

        printf("  Single atom test passed\n");
    }

    // Test 3: Atoms at cell boundaries
    {
        std::vector<float> coords;
        std::vector<uint8_t> types;

        // Place atoms exactly on cell boundaries (at multiples of cell_size)
        for (float x = 0; x <= 16; x += 8) {
            for (float y = 0; y <= 16; y += 8) {
                for (float z = 0; z <= 16; z += 8) {
                    coords.push_back(x);
                    coords.push_back(y);
                    coords.push_back(z);
                    types.push_back(5);
                }
            }
        }

        float3 box_min = make_float3(-1, -1, -1);
        float3 box_max = make_float3(17, 17, 17);

        builder.build(coords, types, box_min, box_max, 8.0f);

        // Validate that all neighbors are found correctly
        bool valid = builder.validate(8.0f);
        assert(valid);

        printf("  Cell boundary test passed\n");
    }

    printf("  All edge case tests passed\n");
}

// ============================================================================
// Test 4: Validation (Brute Force Comparison)
// ============================================================================

inline void test_spatial_hash_validation() {
    printf("Testing spatial hash validation (brute force comparison)...\n");

    SpatialHashBuilder builder;

    // Create random atoms
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-15.0f, 15.0f);

    std::vector<float> coords;
    std::vector<uint8_t> types;

    int num_atoms = 100;
    for (int i = 0; i < num_atoms; i++) {
        coords.push_back(dist(rng));
        coords.push_back(dist(rng));
        coords.push_back(dist(rng));
        types.push_back(i % 28);  // Cycle through atom types
    }

    float3 box_min = make_float3(-20, -20, -20);
    float3 box_max = make_float3(20, 20, 20);

    builder.build(coords, types, box_min, box_max, 8.0f);
    builder.print_stats();

    // Run validation
    bool valid = builder.validate(8.0f);
    assert(valid);

    printf("  Validation test passed with %d atoms\n", num_atoms);
}

// ============================================================================
// Test 5: Atom Type and Flags Preservation
// ============================================================================

inline void test_spatial_hash_atom_types() {
    printf("Testing spatial hash atom type preservation...\n");

    SpatialHashBuilder builder;

    // Create atoms of each type
    std::vector<float> coords;
    std::vector<uint8_t> types;

    for (int t = 0; t < 28; t++) {
        coords.push_back(t * 2.0f);
        coords.push_back(0);
        coords.push_back(0);
        types.push_back(t);
    }

    float3 box_min = make_float3(-10, -10, -10);
    float3 box_max = make_float3(60, 10, 10);

    builder.build(coords, types, box_min, box_max, 8.0f);

    // Verify that sorted atoms preserve type and flags
    for (const auto& atom : builder.sorted_atoms) {
        // Verify radius group matches type
        uint8_t expected_group = SMINA_TYPE_TO_RADIUS_GROUP[atom.smina_type];
        assert(atom.radius_group == expected_group);

        // Verify flags match type
        uint8_t expected_flags = SMINA_TYPE_FLAGS[atom.smina_type];
        assert(atom.flags == expected_flags);
    }

    printf("  Atom type preservation test passed for all 28 types\n");
}

// ============================================================================
// Run All Spatial Hash Tests
// ============================================================================

inline void run_all_spatial_hash_tests() {
    printf("\n=== Spatial Hash Unit Tests ===\n\n");

    test_spatial_hash_cell_indexing();
    test_spatial_hash_neighbor_accuracy();
    test_spatial_hash_edge_cases();
    test_spatial_hash_validation();
    test_spatial_hash_atom_types();

    printf("\n=== All Spatial Hash Tests Passed ===\n\n");
}

#endif // TEST_SPATIAL_HASH_H
