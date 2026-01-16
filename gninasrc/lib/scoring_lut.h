/*
 * scoring_lut.h
 *
 * Lookup table (LUT) based scoring for direct pairwise computation.
 * Replaces grid-based scoring to avoid L2 cache thrashing at high exhaustiveness.
 *
 * Key insight: All Vina scoring terms depend on surface distance d_surf = r - vdw_sum
 * where vdw_sum = xs_radius(atom1) + xs_radius(atom2).
 *
 * With only 10 unique xs_radius values, there are 55 unique vdw_sum values,
 * allowing compact LUTs that fit in GPU shared memory.
 */

#ifndef SCORING_LUT_H
#define SCORING_LUT_H

#include <vector>
#include <cmath>
#include <cstdint>

// CUDA headers only needed when compiling device code
#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#else
// Provide a half type stub for non-CUDA compilation (host-only)
typedef unsigned short half;
inline float __half2float(half h) {
    // This is just a stub for host compilation - actual LUT generation uses float
    union { unsigned short u; half h; } conv;
    conv.h = h;
    // Simplified conversion (good enough for testing)
    unsigned short bits = conv.u;
    unsigned int sign = (bits >> 15) & 1;
    unsigned int exp = (bits >> 10) & 0x1f;
    unsigned int mant = bits & 0x3ff;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Denormalized
        float val = mant / 1024.0f * powf(2.0f, -14);
        return sign ? -val : val;
    } else if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    } else {
        float val = (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15);
        return sign ? -val : val;
    }
}
inline half __float2half(float f) {
    // Simplified conversion stub
    union { float f; unsigned int u; } conv;
    conv.f = f;
    unsigned int bits = conv.u;
    unsigned int sign = (bits >> 31) & 1;
    int exp = ((bits >> 23) & 0xff) - 127 + 15;
    unsigned int mant = (bits >> 13) & 0x3ff;

    if (exp <= 0) return (half)(sign << 15);  // Flush to zero
    if (exp >= 31) return (half)((sign << 15) | (31 << 10));  // Infinity
    return (half)((sign << 15) | (exp << 10) | mant);
}
#endif

// ============================================================================
// Constants
// ============================================================================

// Number of unique xs_radius values across all 28 SMINA atom types
constexpr int N_RADIUS_GROUPS = 10;

// Number of unique vdw_sum values: N_RADIUS_GROUPS * (N_RADIUS_GROUPS + 1) / 2 = 55
constexpr int N_VDW_GROUPS = 55;

// Number of distance bins for LUT (d² from 0 to cutoff²)
constexpr int N_DIST_BINS = 100;

// Default cutoff distance (Angstroms)
constexpr float DEFAULT_CUTOFF = 8.0f;

// Vina scoring weights (from custom_terms.h add_vina())
constexpr float WEIGHT_GAUSS1 = -0.035579f;
constexpr float WEIGHT_GAUSS2 = -0.005156f;
constexpr float WEIGHT_REPULSION = 0.840245f;
constexpr float WEIGHT_HYDROPHOBIC = -0.035069f;
constexpr float WEIGHT_HBOND = -0.587439f;

// Gauss parameters
constexpr float GAUSS1_OFFSET = 0.0f;
constexpr float GAUSS1_WIDTH = 0.5f;
constexpr float GAUSS2_OFFSET = 3.0f;
constexpr float GAUSS2_WIDTH = 2.0f;

// Hydrophobic parameters
constexpr float HYDRO_GOOD = 0.5f;
constexpr float HYDRO_BAD = 1.5f;

// H-bond parameters
constexpr float HBOND_GOOD = -0.7f;
constexpr float HBOND_BAD = 0.0f;

// ============================================================================
// VdW Radius Group Mapping
// ============================================================================

// The 10 unique xs_radius values (in Angstroms)
// Index 0-9 corresponds to radius_group_id
// Host version for CPU code
constexpr float RADIUS_VALUES_HOST[N_RADIUS_GROUPS] = {
    0.37f,  // Group 0: H, PolarH
    1.20f,  // Group 1: Metals (Mg, Mn, Zn, Ca, Fe, GenericMetal)
    1.50f,  // Group 2: F
    1.70f,  // Group 3: O (all variants)
    1.80f,  // Group 4: N (all variants), Cl
    1.90f,  // Group 5: C (all variants)
    1.92f,  // Group 6: B
    2.00f,  // Group 7: S (all variants), Br
    2.10f,  // Group 8: P
    2.20f,  // Group 9: I
};

#ifdef __CUDACC__
// Device constant version for GPU code (visible when compiling with nvcc)
__device__ __constant__ float RADIUS_VALUES_DEVICE[N_RADIUS_GROUPS] = {
    0.37f, 1.20f, 1.50f, 1.70f, 1.80f, 1.90f, 1.92f, 2.00f, 2.10f, 2.20f
};
#endif

// Use RADIUS_VALUES_HOST for host code
#define RADIUS_VALUES RADIUS_VALUES_HOST

// Map from SMINA atom type (0-27) to radius group (0-9)
// Based on xs_radius values from atom_constants.h
// Note: Host version for CPU code, device __constant__ version for GPU code
constexpr uint8_t SMINA_TYPE_TO_RADIUS_GROUP_HOST[28] = {
    0,  // 0: Hydrogen (0.37)
    0,  // 1: PolarHydrogen (0.37)
    5,  // 2: AliphaticCarbonXSHydrophobe (1.90)
    5,  // 3: AliphaticCarbonXSNonHydrophobe (1.90)
    5,  // 4: AromaticCarbonXSHydrophobe (1.90)
    5,  // 5: AromaticCarbonXSNonHydrophobe (1.90)
    4,  // 6: Nitrogen (1.80)
    4,  // 7: NitrogenXSDonor (1.80)
    4,  // 8: NitrogenXSDonorAcceptor (1.80)
    4,  // 9: NitrogenXSAcceptor (1.80)
    3,  // 10: Oxygen (1.70)
    3,  // 11: OxygenXSDonor (1.70)
    3,  // 12: OxygenXSDonorAcceptor (1.70)
    3,  // 13: OxygenXSAcceptor (1.70)
    7,  // 14: Sulfur (2.00)
    7,  // 15: SulfurAcceptor (2.00)
    8,  // 16: Phosphorus (2.10)
    2,  // 17: Fluorine (1.50)
    4,  // 18: Chlorine (1.80)
    7,  // 19: Bromine (2.00)
    9,  // 20: Iodine (2.20)
    1,  // 21: Magnesium (1.20)
    1,  // 22: Manganese (1.20)
    1,  // 23: Zinc (1.20)
    1,  // 24: Calcium (1.20)
    1,  // 25: Iron (1.20)
    1,  // 26: GenericMetal (1.20)
    6,  // 27: Boron (1.92)
};

// Device version in constant memory (only defined when compiling with nvcc)
#ifdef __CUDACC__
__device__ __constant__ uint8_t SMINA_TYPE_TO_RADIUS_GROUP[28] = {
    0, 0, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 7, 7, 8, 2, 4, 7, 9, 1, 1, 1, 1, 1, 1, 6
};
#else
// For host-only compilation, alias to host version
#define SMINA_TYPE_TO_RADIUS_GROUP SMINA_TYPE_TO_RADIUS_GROUP_HOST
#endif

// Atom type flags (packed into uint8_t)
constexpr uint8_t FLAG_HYDROPHOBIC = 0x01;
constexpr uint8_t FLAG_DONOR = 0x02;
constexpr uint8_t FLAG_ACCEPTOR = 0x04;

// Map from SMINA atom type to flags (host version)
constexpr uint8_t SMINA_TYPE_FLAGS_HOST[28] = {
    0,                                      // 0: Hydrogen
    0,                                      // 1: PolarHydrogen
    FLAG_HYDROPHOBIC,                       // 2: AliphaticCarbonXSHydrophobe
    0,                                      // 3: AliphaticCarbonXSNonHydrophobe
    FLAG_HYDROPHOBIC,                       // 4: AromaticCarbonXSHydrophobe
    0,                                      // 5: AromaticCarbonXSNonHydrophobe
    0,                                      // 6: Nitrogen
    FLAG_DONOR,                             // 7: NitrogenXSDonor
    FLAG_DONOR | FLAG_ACCEPTOR,             // 8: NitrogenXSDonorAcceptor
    FLAG_ACCEPTOR,                          // 9: NitrogenXSAcceptor
    0,                                      // 10: Oxygen
    FLAG_DONOR,                             // 11: OxygenXSDonor
    FLAG_DONOR | FLAG_ACCEPTOR,             // 12: OxygenXSDonorAcceptor
    FLAG_ACCEPTOR,                          // 13: OxygenXSAcceptor
    0,                                      // 14: Sulfur
    0,                                      // 15: SulfurAcceptor (XS doesn't do sulfur acceptors)
    0,                                      // 16: Phosphorus
    FLAG_HYDROPHOBIC,                       // 17: Fluorine
    FLAG_HYDROPHOBIC,                       // 18: Chlorine
    FLAG_HYDROPHOBIC,                       // 19: Bromine
    FLAG_HYDROPHOBIC,                       // 20: Iodine
    FLAG_DONOR,                             // 21: Magnesium
    FLAG_DONOR,                             // 22: Manganese
    FLAG_DONOR,                             // 23: Zinc
    FLAG_DONOR,                             // 24: Calcium
    FLAG_DONOR,                             // 25: Iron
    FLAG_DONOR,                             // 26: GenericMetal
    FLAG_HYDROPHOBIC,                       // 27: Boron
};

// Device version in constant memory
#ifdef __CUDACC__
__device__ __constant__ uint8_t SMINA_TYPE_FLAGS[28] = {
    0, 0, 1, 0, 1, 0, 0, 2, 6, 4, 0, 2, 6, 4, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1
};
#else
#define SMINA_TYPE_FLAGS SMINA_TYPE_FLAGS_HOST
#endif

// ============================================================================
// VdW Sum Index Mapping
// ============================================================================

// Host/Device function qualifier macro
#ifdef __CUDACC__
#define LUT_HOSTDEVICE __host__ __device__
#else
#define LUT_HOSTDEVICE
#endif

// Convert (radius_group_a, radius_group_b) to vdw_sum_idx (0-54)
// Uses triangular indexing: idx = min + (max * (max + 1)) / 2
LUT_HOSTDEVICE inline int get_vdw_sum_idx(int group_a, int group_b) {
    int min_g = (group_a < group_b) ? group_a : group_b;
    int max_g = (group_a < group_b) ? group_b : group_a;
    return min_g + (max_g * (max_g + 1)) / 2;
}

// Get vdw_sum value for a given index
LUT_HOSTDEVICE inline float get_vdw_sum_value(int vdw_idx) {
    // Reverse the triangular indexing to get (min_g, max_g)
    // max_g is the largest k such that k*(k+1)/2 <= vdw_idx
    int max_g = 0;
    while ((max_g + 1) * (max_g + 2) / 2 <= vdw_idx) {
        max_g++;
    }
    int min_g = vdw_idx - (max_g * (max_g + 1)) / 2;
#ifdef __CUDA_ARCH__
    return RADIUS_VALUES_DEVICE[min_g] + RADIUS_VALUES_DEVICE[max_g];
#else
    return RADIUS_VALUES_HOST[min_g] + RADIUS_VALUES_HOST[max_g];
#endif
}

// Precomputed vdw_sum values for all 55 indices
inline void compute_vdw_sum_table(float* table) {
    for (int i = 0; i < N_VDW_GROUPS; i++) {
        table[i] = get_vdw_sum_value(i);
    }
}

// Precomputed (group_a, group_b) -> vdw_idx lookup table
inline void compute_vdw_pair_table(uint8_t table[N_RADIUS_GROUPS][N_RADIUS_GROUPS]) {
    for (int a = 0; a < N_RADIUS_GROUPS; a++) {
        for (int b = 0; b < N_RADIUS_GROUPS; b++) {
            table[a][b] = (uint8_t)get_vdw_sum_idx(a, b);
        }
    }
}

// ============================================================================
// Scoring Function Helpers
// ============================================================================

// Gaussian function: exp(-(x/width)^2)
inline float gaussian(float x, float width) {
    return expf(-(x / width) * (x / width));
}

// Gaussian derivative: d/dx exp(-(x/w)^2) = -2x/w^2 * exp(-(x/w)^2)
inline float gaussian_deriv(float x, float width) {
    float g = gaussian(x, width);
    return -2.0f * x / (width * width) * g;
}

// Slope step function (linear interpolation between bad and good)
inline float slope_step(float x_bad, float x_good, float x) {
    if (x_bad < x_good) {
        if (x <= x_bad) return 0.0f;
        if (x >= x_good) return 1.0f;
    } else {
        if (x >= x_bad) return 0.0f;
        if (x <= x_good) return 1.0f;
    }
    return (x - x_bad) / (x_good - x_bad);
}

// Slope step derivative
inline float slope_step_deriv(float x_bad, float x_good, float x) {
    if (x_bad < x_good) {
        if (x <= x_bad || x >= x_good) return 0.0f;
    } else {
        if (x >= x_bad || x <= x_good) return 0.0f;
    }
    return 1.0f / (x_good - x_bad);
}

// ============================================================================
// LUT Data Structure
// ============================================================================

// LUT entry: stores energy and derivative (dE/dr divided by r for direct force computation)
struct LUTEntry {
    half energy;
    half deriv_over_r;  // dE/dr / r, so force = -deriv_over_r * displacement
};

// Complete scoring LUT for GPU shared memory
// Total size: 3 components * 55 groups * 100 bins * 4 bytes = 66 KB
// This is too big for shared memory (48KB), so we use a more compact format
struct ScoringLUTCompact {
    // Base terms (gauss1 + gauss2 + repulsion): always applied
    half base_energy[N_VDW_GROUPS][N_DIST_BINS];     // 11 KB
    half base_deriv[N_VDW_GROUPS][N_DIST_BINS];      // 11 KB

    // Hydrophobic term: applied when both atoms are hydrophobic
    half hydro_energy[N_VDW_GROUPS][N_DIST_BINS];    // 11 KB
    half hydro_deriv[N_VDW_GROUPS][N_DIST_BINS];     // 11 KB

    // H-bond term: applied for donor-acceptor pairs
    half hbond_energy[N_VDW_GROUPS][N_DIST_BINS];    // 11 KB
    half hbond_deriv[N_VDW_GROUPS][N_DIST_BINS];     // 11 KB
};
// Total: 66 KB - too big, need to reduce

// Minimal LUT: fewer bins with interpolation
constexpr int N_DIST_BINS_COMPACT = 64;

struct ScoringLUTMinimal {
    // Base terms
    half base_energy[N_VDW_GROUPS][N_DIST_BINS_COMPACT];   // 7 KB
    half base_deriv[N_VDW_GROUPS][N_DIST_BINS_COMPACT];    // 7 KB

    // Hydrophobic term
    half hydro_energy[N_VDW_GROUPS][N_DIST_BINS_COMPACT];  // 7 KB
    half hydro_deriv[N_VDW_GROUPS][N_DIST_BINS_COMPACT];   // 7 KB

    // H-bond term
    half hbond_energy[N_VDW_GROUPS][N_DIST_BINS_COMPACT];  // 7 KB
    half hbond_deriv[N_VDW_GROUPS][N_DIST_BINS_COMPACT];   // 7 KB
};
// Total: 42 KB - fits in 48 KB shared memory with 6 KB to spare

// ============================================================================
// LUT Generation (CPU-side)
// ============================================================================

class ScoringLUTGenerator {
public:
    float cutoff;
    float cutoff_sq;
    float bin_size;  // d² per bin
    int n_bins;

    // Precomputed vdw_sum values
    float vdw_sums[N_VDW_GROUPS];

    ScoringLUTGenerator(float cutoff_ = DEFAULT_CUTOFF, int n_bins_ = N_DIST_BINS_COMPACT)
        : cutoff(cutoff_), n_bins(n_bins_) {
        cutoff_sq = cutoff * cutoff;
        bin_size = cutoff_sq / n_bins;
        compute_vdw_sum_table(vdw_sums);
    }

    // Get distance from bin index (center of bin)
    float bin_to_distance(int bin) const {
        float d_sq = (bin + 0.5f) * bin_size;
        return sqrtf(d_sq);
    }

    // Get bin index from distance squared
    int distance_sq_to_bin(float d_sq) const {
        int bin = (int)(d_sq / bin_size);
        if (bin >= n_bins) bin = n_bins - 1;
        if (bin < 0) bin = 0;
        return bin;
    }

    // Compute base terms (gauss1 + gauss2 + repulsion) for given vdw_sum and distance
    void compute_base_terms(float vdw_sum, float r, float& energy, float& deriv) const {
        float d_surf = r - vdw_sum;

        energy = 0;
        deriv = 0;

        // Gauss1: weight * exp(-((d_surf - offset) / width)^2)
        float x1 = d_surf - GAUSS1_OFFSET;
        float g1 = gaussian(x1, GAUSS1_WIDTH);
        energy += WEIGHT_GAUSS1 * g1;
        deriv += WEIGHT_GAUSS1 * gaussian_deriv(x1, GAUSS1_WIDTH);

        // Gauss2
        float x2 = d_surf - GAUSS2_OFFSET;
        float g2 = gaussian(x2, GAUSS2_WIDTH);
        energy += WEIGHT_GAUSS2 * g2;
        deriv += WEIGHT_GAUSS2 * gaussian_deriv(x2, GAUSS2_WIDTH);

        // Repulsion: weight * d_surf^2 if d_surf < 0
        if (d_surf < 0) {
            energy += WEIGHT_REPULSION * d_surf * d_surf;
            deriv += WEIGHT_REPULSION * 2.0f * d_surf;
        }
    }

    // Compute hydrophobic term
    void compute_hydro_term(float vdw_sum, float r, float& energy, float& deriv) const {
        float d_surf = r - vdw_sum;
        energy = WEIGHT_HYDROPHOBIC * slope_step(HYDRO_BAD, HYDRO_GOOD, d_surf);
        deriv = WEIGHT_HYDROPHOBIC * slope_step_deriv(HYDRO_BAD, HYDRO_GOOD, d_surf);
    }

    // Compute h-bond term
    void compute_hbond_term(float vdw_sum, float r, float& energy, float& deriv) const {
        float d_surf = r - vdw_sum;
        energy = WEIGHT_HBOND * slope_step(HBOND_BAD, HBOND_GOOD, d_surf);
        deriv = WEIGHT_HBOND * slope_step_deriv(HBOND_BAD, HBOND_GOOD, d_surf);
    }

    // Generate complete LUT
    void generate(ScoringLUTMinimal& lut) const {
        for (int vdw_idx = 0; vdw_idx < N_VDW_GROUPS; vdw_idx++) {
            float vdw_sum = vdw_sums[vdw_idx];

            for (int bin = 0; bin < n_bins; bin++) {
                float r = bin_to_distance(bin);

                // Handle r = 0 case
                if (r < 0.01f) r = 0.01f;

                float e, de_dr;

                // Base terms
                compute_base_terms(vdw_sum, r, e, de_dr);
                lut.base_energy[vdw_idx][bin] = __float2half(e);
                lut.base_deriv[vdw_idx][bin] = __float2half(de_dr / r);

                // Hydrophobic term
                compute_hydro_term(vdw_sum, r, e, de_dr);
                lut.hydro_energy[vdw_idx][bin] = __float2half(e);
                lut.hydro_deriv[vdw_idx][bin] = __float2half(de_dr / r);

                // H-bond term
                compute_hbond_term(vdw_sum, r, e, de_dr);
                lut.hbond_energy[vdw_idx][bin] = __float2half(e);
                lut.hbond_deriv[vdw_idx][bin] = __float2half(de_dr / r);
            }
        }
    }

    // Validate LUT against direct computation
    float validate(const ScoringLUTMinimal& lut, int n_test_points = 1000) const {
        float max_error = 0;

        for (int vdw_idx = 0; vdw_idx < N_VDW_GROUPS; vdw_idx++) {
            float vdw_sum = vdw_sums[vdw_idx];

            for (int i = 0; i < n_test_points; i++) {
                // Random distance within cutoff
                float r = 0.5f + (cutoff - 0.5f) * ((float)i / n_test_points);

                // Direct computation
                float direct_e, direct_de;
                compute_base_terms(vdw_sum, r, direct_e, direct_de);

                // LUT lookup with interpolation
                float d_sq = r * r;
                float bin_f = d_sq / bin_size;
                int bin = (int)bin_f;
                float frac = bin_f - bin;
                if (bin >= n_bins - 1) { bin = n_bins - 2; frac = 1.0f; }

                float lut_e = __half2float(lut.base_energy[vdw_idx][bin]) * (1 - frac)
                            + __half2float(lut.base_energy[vdw_idx][bin + 1]) * frac;

                float error = fabsf(lut_e - direct_e);
                if (error > max_error) max_error = error;
            }
        }

        return max_error;
    }
};

// ============================================================================
// LUT Access (works on both CPU and GPU)
// ============================================================================

// Make functions work on both CPU and GPU
#ifdef __CUDA_ARCH__
#define LUT_DEVICE __device__ __forceinline__
#else
#define LUT_DEVICE inline
#endif

// Interpolate LUT value at given d² (distance squared)
LUT_DEVICE void lut_lookup(
    const half* energy_table,
    const half* deriv_table,
    int vdw_idx,
    float d_sq,
    float bin_scale,  // n_bins / cutoff_sq
    int n_bins,
    float& energy,
    float& deriv_over_r
) {
    float bin_f = d_sq * bin_scale;
    int bin = (int)bin_f;
    float frac = bin_f - bin;

    // Clamp to valid range
    if (bin >= n_bins - 1) {
        bin = n_bins - 2;
        frac = 1.0f;
    }
    if (bin < 0) {
        bin = 0;
        frac = 0.0f;
    }

    int idx0 = vdw_idx * n_bins + bin;
    int idx1 = idx0 + 1;

    // Linear interpolation
    energy = __half2float(energy_table[idx0]) * (1.0f - frac)
           + __half2float(energy_table[idx1]) * frac;
    deriv_over_r = __half2float(deriv_table[idx0]) * (1.0f - frac)
                 + __half2float(deriv_table[idx1]) * frac;
}

#endif // SCORING_LUT_H
