/*
 * test_scoring_lut.h
 *
 * Unit tests for scoring LUT generation and accuracy.
 */

#ifndef TEST_SCORING_LUT_H
#define TEST_SCORING_LUT_H

#include "scoring_lut.h"
#include "atom_constants.h"
#include <cmath>
#include <cstdio>
#include <cassert>

// ============================================================================
// Test 1: VdW Group Mapping Correctness
// ============================================================================

inline void test_vdw_group_mapping() {
    printf("Testing VdW group mapping...\n");

    // Verify that SMINA_TYPE_TO_RADIUS_GROUP maps correctly to xs_radius values
    for (int t = 0; t < 28; t++) {
        smt smina_type = static_cast<smt>(t);
        float expected_radius = xs_radius(smina_type);
        int group = SMINA_TYPE_TO_RADIUS_GROUP[t];
        float mapped_radius = RADIUS_VALUES[group];

        if (fabsf(expected_radius - mapped_radius) > 0.001f) {
            printf("FAIL: Type %d (%s): expected radius %.3f, got group %d (radius %.3f)\n",
                   t, smina_type_to_string(smina_type), expected_radius, group, mapped_radius);
            assert(false);
        }
    }

    printf("  All 28 SMINA types map correctly to radius groups\n");
}

// ============================================================================
// Test 2: VdW Sum Index Symmetry
// ============================================================================

inline void test_vdw_sum_symmetry() {
    printf("Testing VdW sum index symmetry...\n");

    for (int a = 0; a < N_RADIUS_GROUPS; a++) {
        for (int b = 0; b < N_RADIUS_GROUPS; b++) {
            int idx_ab = get_vdw_sum_idx(a, b);
            int idx_ba = get_vdw_sum_idx(b, a);

            if (idx_ab != idx_ba) {
                printf("FAIL: vdw_sum_idx(%d, %d) = %d, but vdw_sum_idx(%d, %d) = %d\n",
                       a, b, idx_ab, b, a, idx_ba);
                assert(false);
            }

            // Also verify the index is in valid range
            assert(idx_ab >= 0 && idx_ab < N_VDW_GROUPS);
        }
    }

    printf("  All vdw_sum indices are symmetric and in valid range [0, %d)\n", N_VDW_GROUPS);
}

// ============================================================================
// Test 3: VdW Sum Value Round-trip
// ============================================================================

inline void test_vdw_sum_roundtrip() {
    printf("Testing VdW sum value round-trip...\n");

    for (int a = 0; a < N_RADIUS_GROUPS; a++) {
        for (int b = 0; b < N_RADIUS_GROUPS; b++) {
            float expected_sum = RADIUS_VALUES[a] + RADIUS_VALUES[b];
            int idx = get_vdw_sum_idx(a, b);
            float retrieved_sum = get_vdw_sum_value(idx);

            if (fabsf(expected_sum - retrieved_sum) > 0.001f) {
                printf("FAIL: groups (%d, %d) -> idx %d: expected sum %.3f, got %.3f\n",
                       a, b, idx, expected_sum, retrieved_sum);
                assert(false);
            }
        }
    }

    printf("  All vdw_sum values round-trip correctly\n");
}

// ============================================================================
// Test 4: LUT vs Analytical Computation
// ============================================================================

inline void test_lut_vs_analytical() {
    printf("Testing LUT vs analytical computation...\n");

    ScoringLUTGenerator gen(DEFAULT_CUTOFF, N_DIST_BINS_COMPACT);
    ScoringLUTMinimal lut;
    gen.generate(lut);

    float max_base_error = 0;
    float max_hydro_error = 0;
    float max_hbond_error = 0;

    // Test at various distances for each vdw_sum
    for (int vdw_idx = 0; vdw_idx < N_VDW_GROUPS; vdw_idx++) {
        float vdw_sum = gen.vdw_sums[vdw_idx];

        for (float r = 0.5f; r < DEFAULT_CUTOFF; r += 0.1f) {
            // Direct computation
            float direct_base_e, direct_base_de;
            float direct_hydro_e, direct_hydro_de;
            float direct_hbond_e, direct_hbond_de;

            gen.compute_base_terms(vdw_sum, r, direct_base_e, direct_base_de);
            gen.compute_hydro_term(vdw_sum, r, direct_hydro_e, direct_hydro_de);
            gen.compute_hbond_term(vdw_sum, r, direct_hbond_e, direct_hbond_de);

            // LUT lookup with interpolation
            float d_sq = r * r;
            float bin_scale = (float)N_DIST_BINS_COMPACT / (DEFAULT_CUTOFF * DEFAULT_CUTOFF);

            float lut_base_e, lut_base_dor;
            float lut_hydro_e, lut_hydro_dor;
            float lut_hbond_e, lut_hbond_dor;

            lut_lookup((half*)lut.base_energy, (half*)lut.base_deriv,
                       vdw_idx, d_sq, bin_scale, N_DIST_BINS_COMPACT,
                       lut_base_e, lut_base_dor);
            lut_lookup((half*)lut.hydro_energy, (half*)lut.hydro_deriv,
                       vdw_idx, d_sq, bin_scale, N_DIST_BINS_COMPACT,
                       lut_hydro_e, lut_hydro_dor);
            lut_lookup((half*)lut.hbond_energy, (half*)lut.hbond_deriv,
                       vdw_idx, d_sq, bin_scale, N_DIST_BINS_COMPACT,
                       lut_hbond_e, lut_hbond_dor);

            // Track errors
            float base_error = fabsf(lut_base_e - direct_base_e);
            float hydro_error = fabsf(lut_hydro_e - direct_hydro_e);
            float hbond_error = fabsf(lut_hbond_e - direct_hbond_e);

            if (base_error > max_base_error) max_base_error = base_error;
            if (hydro_error > max_hydro_error) max_hydro_error = hydro_error;
            if (hbond_error > max_hbond_error) max_hbond_error = hbond_error;
        }
    }

    printf("  Max base term error: %.6f kcal/mol\n", max_base_error);
    printf("  Max hydro term error: %.6f kcal/mol\n", max_hydro_error);
    printf("  Max hbond term error: %.6f kcal/mol\n", max_hbond_error);

    // Assert errors are within acceptable tolerance
    // With 64 bins and interpolation, we expect < 0.01 kcal/mol error
    const float TOLERANCE = 0.02f;  // 0.02 kcal/mol
    assert(max_base_error < TOLERANCE);
    assert(max_hydro_error < TOLERANCE);
    assert(max_hbond_error < TOLERANCE);

    printf("  All errors within tolerance (%.3f kcal/mol)\n", TOLERANCE);
}

// ============================================================================
// Test 5: Edge Cases
// ============================================================================

inline void test_lut_edge_cases() {
    printf("Testing LUT edge cases...\n");

    ScoringLUTGenerator gen(DEFAULT_CUTOFF, N_DIST_BINS_COMPACT);
    ScoringLUTMinimal lut;
    gen.generate(lut);

    float bin_scale = (float)N_DIST_BINS_COMPACT / (DEFAULT_CUTOFF * DEFAULT_CUTOFF);

    // Test at very small distance (should have large repulsion)
    {
        float r = 1.0f;
        float d_sq = r * r;
        int vdw_idx = get_vdw_sum_idx(5, 5);  // C-C pair, vdw_sum = 3.8

        float lut_e, lut_dor;
        lut_lookup((half*)lut.base_energy, (half*)lut.base_deriv,
                   vdw_idx, d_sq, bin_scale, N_DIST_BINS_COMPACT,
                   lut_e, lut_dor);

        // d_surf = 1.0 - 3.8 = -2.8, so repulsion should be large
        float expected_repulsion = WEIGHT_REPULSION * (-2.8f) * (-2.8f);
        printf("  r=1.0 C-C: energy=%.3f (repulsion component ~%.3f)\n", lut_e, expected_repulsion);
        assert(lut_e > 1.0f);  // Should be positive (repulsive)
    }

    // Test at cutoff boundary
    {
        float r = DEFAULT_CUTOFF - 0.01f;
        float d_sq = r * r;
        int vdw_idx = get_vdw_sum_idx(5, 5);  // C-C pair

        float lut_e, lut_dor;
        lut_lookup((half*)lut.base_energy, (half*)lut.base_deriv,
                   vdw_idx, d_sq, bin_scale, N_DIST_BINS_COMPACT,
                   lut_e, lut_dor);

        printf("  r=%.2f C-C: energy=%.6f (should be near zero)\n", r, lut_e);
        assert(fabsf(lut_e) < 0.01f);  // Should be very small at cutoff
    }

    // Test optimal distance (d_surf = 0)
    {
        float vdw_sum = RADIUS_VALUES[5] + RADIUS_VALUES[3];  // C + O = 3.6
        float r = vdw_sum;  // At optimal distance
        float d_sq = r * r;
        int vdw_idx = get_vdw_sum_idx(5, 3);

        float lut_e, lut_dor;
        lut_lookup((half*)lut.base_energy, (half*)lut.base_deriv,
                   vdw_idx, d_sq, bin_scale, N_DIST_BINS_COMPACT,
                   lut_e, lut_dor);

        // At d_surf=0: gauss1 = 1, gauss2 = exp(-9/4), repulsion = 0
        float expected = WEIGHT_GAUSS1 * 1.0f + WEIGHT_GAUSS2 * expf(-9.0f/4.0f);
        printf("  r=%.2f C-O (optimal): energy=%.6f (expected ~%.6f)\n", r, lut_e, expected);
        assert(fabsf(lut_e - expected) < 0.01f);
    }

    printf("  All edge cases pass\n");
}

// ============================================================================
// Test 6: Atom Type Flags
// ============================================================================

inline void test_atom_type_flags() {
    printf("Testing atom type flags...\n");

    // Check hydrophobic flags match xs_is_hydrophobic
    for (int t = 0; t < 28; t++) {
        smt smina_type = static_cast<smt>(t);
        bool expected_hydro = xs_is_hydrophobic(smina_type);
        bool flag_hydro = (SMINA_TYPE_FLAGS[t] & FLAG_HYDROPHOBIC) != 0;

        if (expected_hydro != flag_hydro) {
            printf("FAIL: Type %d (%s): expected hydrophobic=%d, got flag=%d\n",
                   t, smina_type_to_string(smina_type), expected_hydro, flag_hydro);
            assert(false);
        }
    }

    // Check donor flags match xs_is_donor
    for (int t = 0; t < 28; t++) {
        smt smina_type = static_cast<smt>(t);
        bool expected_donor = xs_is_donor(smina_type);
        bool flag_donor = (SMINA_TYPE_FLAGS[t] & FLAG_DONOR) != 0;

        if (expected_donor != flag_donor) {
            printf("FAIL: Type %d (%s): expected donor=%d, got flag=%d\n",
                   t, smina_type_to_string(smina_type), expected_donor, flag_donor);
            assert(false);
        }
    }

    // Check acceptor flags match xs_is_acceptor
    for (int t = 0; t < 28; t++) {
        smt smina_type = static_cast<smt>(t);
        bool expected_acceptor = xs_is_acceptor(smina_type);
        bool flag_acceptor = (SMINA_TYPE_FLAGS[t] & FLAG_ACCEPTOR) != 0;

        if (expected_acceptor != flag_acceptor) {
            printf("FAIL: Type %d (%s): expected acceptor=%d, got flag=%d\n",
                   t, smina_type_to_string(smina_type), expected_acceptor, flag_acceptor);
            assert(false);
        }
    }

    printf("  All hydrophobic, donor, and acceptor flags correct\n");
}

// ============================================================================
// Test 7: Gradient Numerical Check
// ============================================================================

inline void test_gradient_numerical() {
    printf("Testing gradient numerical accuracy...\n");

    ScoringLUTGenerator gen(DEFAULT_CUTOFF, N_DIST_BINS_COMPACT);

    float max_error = 0;
    const float eps = 0.0001f;

    for (int vdw_idx = 0; vdw_idx < N_VDW_GROUPS; vdw_idx++) {
        float vdw_sum = gen.vdw_sums[vdw_idx];

        for (float r = 1.0f; r < DEFAULT_CUTOFF - 0.1f; r += 0.5f) {
            // Analytical derivative
            float e, de_dr;
            gen.compute_base_terms(vdw_sum, r, e, de_dr);

            // Numerical derivative
            float e_plus, e_minus, de_dr_unused;
            gen.compute_base_terms(vdw_sum, r + eps, e_plus, de_dr_unused);
            gen.compute_base_terms(vdw_sum, r - eps, e_minus, de_dr_unused);
            float numerical_de_dr = (e_plus - e_minus) / (2 * eps);

            float error = fabsf(de_dr - numerical_de_dr);
            if (error > max_error) max_error = error;

            // Check relative error for non-zero derivatives
            if (fabsf(de_dr) > 0.001f) {
                float rel_error = error / fabsf(de_dr);
                if (rel_error > 0.01f) {  // 1% relative error threshold
                    printf("  Warning: vdw_idx=%d r=%.2f: analytical=%.6f numerical=%.6f (rel_err=%.2f%%)\n",
                           vdw_idx, r, de_dr, numerical_de_dr, rel_error * 100);
                }
            }
        }
    }

    printf("  Max absolute gradient error: %.6f\n", max_error);
    assert(max_error < 0.01f);
    printf("  Gradient numerical check passed\n");
}

// ============================================================================
// Run All Tests
// ============================================================================

inline void run_all_scoring_lut_tests() {
    printf("\n=== Scoring LUT Unit Tests ===\n\n");

    test_vdw_group_mapping();
    test_vdw_sum_symmetry();
    test_vdw_sum_roundtrip();
    test_lut_vs_analytical();
    test_lut_edge_cases();
    test_atom_type_flags();
    test_gradient_numerical();

    printf("\n=== All Scoring LUT Tests Passed ===\n\n");
}

#endif // TEST_SCORING_LUT_H
