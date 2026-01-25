/*
 * test_multi_ligand_batch.cpp
 *
 * Unit tests for multi-ligand CNN batch scoring.
 * Verifies that forward_multi_ligand_batch produces same results as forward (single-pose).
 */

#include "test_multi_ligand_batch.h"
#include "torch_model.h"
#include "torch_models.h"
#include "atom_type.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

static const float TOLERANCE = 0.01f;

// Helper to create random ligand coordinates
static void generate_random_ligand(
    std::vector<float3>& coords,
    std::vector<smt>& types,
    size_t num_atoms,
    float3 center,
    std::mt19937& rng
) {
    std::uniform_real_distribution<float> pos_dist(-5.0f, 5.0f);
    std::uniform_int_distribution<int> type_dist(0, 10);  // Use first 10 atom types

    coords.resize(num_atoms);
    types.resize(num_atoms);

    for (size_t i = 0; i < num_atoms; i++) {
        coords[i] = {
            center.x + pos_dist(rng),
            center.y + pos_dist(rng),
            center.z + pos_dist(rng)
        };
        types[i] = static_cast<smt>(type_dist(rng));
    }
}

// Helper to create a TorchModel using built-in "fast" model
template <bool isCUDA>
static std::shared_ptr<TorchModel<isCUDA>> create_test_model() {
    using namespace boost::iostreams;

    // Use the "all_default_to_default_1_3_1" model (the "fast" model)
    const char* modelstart = torch_models["all_default_to_default_1_3_1"].first;
    const char* modelend = torch_models["all_default_to_default_1_3_1"].second;

    basic_array_source<char> model_source(modelstart, modelend - modelstart);
    stream<basic_array_source<char>> model_stream(model_source);

    return std::make_shared<TorchModel<isCUDA>>(model_stream, "fast", nullptr);
}

void test_multi_ligand_batch_vs_single() {
    std::cout << "Testing multi-ligand batch vs single ligand scoring..." << std::flush;

    auto model = create_test_model<true>();

    std::mt19937 rng(12345);

    // Create a mock receptor
    std::vector<float3> rec_coords;
    std::vector<smt> rec_types;
    generate_random_ligand(rec_coords, rec_types, 100, {0, 0, 0}, rng);

    // Create multiple ligand poses
    const size_t NUM_POSES = 10;
    std::vector<std::vector<float3>> lig_coords_batch(NUM_POSES);
    std::vector<std::vector<smt>> lig_types_batch(NUM_POSES);
    std::vector<vec> centers_batch(NUM_POSES);

    for (size_t i = 0; i < NUM_POSES; i++) {
        float3 center = {static_cast<float>(i * 2), 0, 0};
        generate_random_ligand(lig_coords_batch[i], lig_types_batch[i], 30, center, rng);
        centers_batch[i] = vec(center.x, center.y, center.z);
    }

    // Score using forward_multi_ligand_batch
    auto batch_results = model->forward_multi_ligand_batch(
        rec_coords, rec_types,
        lig_coords_batch, lig_types_batch,
        centers_batch, false);

    // Score each pose individually using forward (single-pose)
    size_t mismatches = 0;
    for (size_t i = 0; i < NUM_POSES; i++) {
        auto single_result = model->forward(
            rec_coords, rec_types,
            lig_coords_batch[i], lig_types_batch[i],
            centers_batch[i], false, false);

        float batch_score = batch_results[i][0];
        float batch_aff = batch_results[i][1];
        float single_score = single_result[0];
        float single_aff = single_result[1];

        float score_diff = std::abs(batch_score - single_score);
        float aff_diff = std::abs(batch_aff - single_aff);

        if (score_diff > TOLERANCE || aff_diff > TOLERANCE) {
            std::cerr << "\n  MISMATCH pose " << i << ": "
                      << "batch=(score=" << batch_score << ", aff=" << batch_aff << ") "
                      << "vs single=(score=" << single_score << ", aff=" << single_aff << ") "
                      << "diff=(score=" << score_diff << ", aff=" << aff_diff << ")";
            mismatches++;
        }
    }

    assert(mismatches == 0);
    std::cout << " PASSED" << std::endl;
}

void test_multi_ligand_batch_variable_sizes() {
    std::cout << "Testing multi-ligand batch with variable ligand sizes..." << std::flush;

    auto model = create_test_model<true>();

    std::mt19937 rng(54321);

    // Create a mock receptor
    std::vector<float3> rec_coords;
    std::vector<smt> rec_types;
    generate_random_ligand(rec_coords, rec_types, 100, {0, 0, 0}, rng);

    // Create ligand poses of varying sizes
    const size_t NUM_POSES = 5;
    std::vector<size_t> ligand_sizes = {20, 35, 25, 40, 15};
    std::vector<std::vector<float3>> lig_coords_batch(NUM_POSES);
    std::vector<std::vector<smt>> lig_types_batch(NUM_POSES);
    std::vector<vec> centers_batch(NUM_POSES);

    for (size_t i = 0; i < NUM_POSES; i++) {
        float3 center = {static_cast<float>(i * 3), 0, 0};
        generate_random_ligand(lig_coords_batch[i], lig_types_batch[i], ligand_sizes[i], center, rng);
        centers_batch[i] = vec(center.x, center.y, center.z);
    }

    // Score using forward_multi_ligand_batch
    auto batch_results = model->forward_multi_ligand_batch(
        rec_coords, rec_types,
        lig_coords_batch, lig_types_batch,
        centers_batch, false);

    // Verify we got results for all poses
    assert(batch_results.size() == NUM_POSES);

    // Score each pose individually and compare
    size_t mismatches = 0;
    for (size_t i = 0; i < NUM_POSES; i++) {
        auto single_result = model->forward(
            rec_coords, rec_types,
            lig_coords_batch[i], lig_types_batch[i],
            centers_batch[i], false, false);

        float score_diff = std::abs(batch_results[i][0] - single_result[0]);
        float aff_diff = std::abs(batch_results[i][1] - single_result[1]);

        if (score_diff > TOLERANCE || aff_diff > TOLERANCE) {
            std::cerr << "\n  MISMATCH pose " << i << " (size=" << ligand_sizes[i] << "): "
                      << "score_diff=" << score_diff << ", aff_diff=" << aff_diff;
            mismatches++;
        }
    }

    assert(mismatches == 0);
    std::cout << " PASSED" << std::endl;
}

void test_multi_ligand_batch_large_batch() {
    std::cout << "Testing multi-ligand batch with large batch (tests chunking)..." << std::flush;

    auto model = create_test_model<true>();

    std::mt19937 rng(99999);

    // Create a mock receptor
    std::vector<float3> rec_coords;
    std::vector<smt> rec_types;
    generate_random_ligand(rec_coords, rec_types, 80, {0, 0, 0}, rng);

    // Create more poses than MAX_CHUNK_SIZE (128) to test chunking
    const size_t NUM_POSES = 200;
    std::vector<std::vector<float3>> lig_coords_batch(NUM_POSES);
    std::vector<std::vector<smt>> lig_types_batch(NUM_POSES);
    std::vector<vec> centers_batch(NUM_POSES);

    for (size_t i = 0; i < NUM_POSES; i++) {
        float3 center = {static_cast<float>((i % 10) * 2), static_cast<float>((i / 10) * 2), 0};
        generate_random_ligand(lig_coords_batch[i], lig_types_batch[i], 25, center, rng);
        centers_batch[i] = vec(center.x, center.y, center.z);
    }

    // Score using forward_multi_ligand_batch
    auto batch_results = model->forward_multi_ligand_batch(
        rec_coords, rec_types,
        lig_coords_batch, lig_types_batch,
        centers_batch, false);

    assert(batch_results.size() == NUM_POSES);

    // Verify a sample of poses (checking all 200 would be slow)
    std::vector<size_t> sample_indices = {0, 50, 127, 128, 150, 199};
    size_t mismatches = 0;

    for (size_t i : sample_indices) {
        auto single_result = model->forward(
            rec_coords, rec_types,
            lig_coords_batch[i], lig_types_batch[i],
            centers_batch[i], false, false);

        float score_diff = std::abs(batch_results[i][0] - single_result[0]);
        float aff_diff = std::abs(batch_results[i][1] - single_result[1]);

        if (score_diff > TOLERANCE || aff_diff > TOLERANCE) {
            std::cerr << "\n  MISMATCH pose " << i << ": "
                      << "score_diff=" << score_diff << ", aff_diff=" << aff_diff;
            mismatches++;
        }
    }

    assert(mismatches == 0);
    std::cout << " PASSED (" << NUM_POSES << " poses, verified " << sample_indices.size() << " samples)" << std::endl;
}
