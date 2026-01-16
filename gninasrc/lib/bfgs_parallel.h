/*
 * bfgs_parallel.h
 *
 * Massively parallel BFGS optimizer where each CUDA thread runs
 * an independent optimizer. No sync points between threads.
 *
 * Design: MAX_THREADS independent optimizers (e.g., 7424 for L4 GPU)
 * - Each thread = 1 complete BFGS optimizer
 * - Zero sync points between threads
 * - Run for fixed max_iterations (no early stopping)
 * - Multiple ligands can share receptor grid data
 */

#ifndef BFGS_PARALLEL_H
#define BFGS_PARALLEL_H

#include <cuda_runtime.h>
#include <vector>
#include "gpu_math.h"
#include "tree_gpu.h"
#include "grid_gpu.h"
#include "interacting_pairs.h"
#include "gpucode.h"
#include "bfgs_diagnostics.h"

// Forward declarations
struct gpu_data;
struct ScoringLUTMinimal;  // From scoring_lut.h
struct SpatialHashGPU;     // From spatial_hash.h

// Maximum supported values (can be overridden at runtime based on batch)
#define PARALLEL_MAX_ATOMS 100
#define PARALLEL_MAX_TORSIONS 64
#define PARALLEL_MAX_CONF_SIZE (7 + PARALLEL_MAX_TORSIONS)     // 71 floats
#define PARALLEL_MAX_CHANGE_SIZE (6 + PARALLEL_MAX_TORSIONS)   // 70 floats
#define PARALLEL_MAX_HESSIAN_SIZE ((PARALLEL_MAX_CHANGE_SIZE * (PARALLEL_MAX_CHANGE_SIZE + 1)) / 2)

// Default thread count for L4 GPU (58 SMs × 128 threads)
#define DEFAULT_MAX_THREADS 7424

// Scoring context for a single ligand type
// This is shared (read-only) across all poses of the same ligand
struct ScoringContext {
    // Grid-based scoring (from GPUCacheInfo)
    const grid_gpu* grids;        // Pre-computed energy grids per atom type
    unsigned ngrids;              // Number of grids

    gfloat3 gridbegins;           // Grid bounding box min
    gfloat3 gridends;             // Grid bounding box max
    float slope;                  // Out-of-bounds penalty slope
    float cutoff_sq;              // Interaction cutoff squared

    // Ligand atom properties (per-atom, fixed for ligand type)
    const unsigned* atom_types;   // Atom type per atom [num_atoms]
    unsigned num_atoms;           // Total movable atoms

    // Reference (local) coordinates for each atom (marked_coord = gfloat3 + owner_idx)
    const marked_coord* atom_local_data;  // [num_atoms] - coords in node's local frame

    // Atom params (coords + charge) for charge-dependent scoring
    const atom_params* atom_params_data;  // [num_atoms] - for accessing charges

    // Tree structure for coordinate transforms (BFS order)
    const segment_node* tree_nodes;  // Tree nodes on device [num_nodes]
    unsigned num_nodes;              // Total tree nodes
    unsigned num_layers;             // Tree depth
    unsigned nlig_roots;             // Number of ligand roots (rigid bodies)

    // Atom ownership mapping
    const unsigned* atom_owners;     // Node index that owns each atom [num_atoms]

    // Intramolecular pairs for internal energy
    const interacting_pair* pairs;   // Pairs for intramolecular interactions
    unsigned num_pairs;

    // Spline data for pair interactions
    const GPUSplineInfo* splineInfo;

    // Curl parameter for energy capping (from forcecap)
    float forcecap;

    // Dimensions
    int n_conf;    // 7 * nlig_roots + n_torsions
    int n_change;  // 6 * nlig_roots + n_torsions
    int n_torsions;

    // Direct pairwise scoring (alternative to grid-based)
    // When enabled, uses LUT + spatial hash instead of grid interpolation
    bool use_direct_pairwise;         // Flag to enable direct pairwise scoring
    const struct ScoringLUTMinimal* lut;   // Scoring lookup table (in global memory)
    const struct SpatialHashGPU* spatial_hash;  // Spatial hash for receptor neighbor finding
    const uint8_t* ligand_smina_types;     // SMINA atom types for ligand atoms [num_atoms]
};

// Per-optimizer state
// Each CUDA thread has its own BFGSState
struct BFGSState {
    // Pointers into batch memory (assigned at kernel start)
    float* x;           // Current conf [n_conf]
    float* x_new;       // Trial conf [n_conf]
    float* g;           // Current gradient [n_change]
    float* g_new;       // Trial gradient [n_change]
    float* p;           // Search direction [n_change]
    float* y;           // Gradient difference [n_change]
    float* h;           // Hessian triangular [n_change*(n_change+1)/2]

    // Scratch space for Cartesian coords and forces
    float* coords;      // [num_atoms × 3]
    float* forces;      // [num_atoms × 3]

    // Per-node scratch for derivative computation
    float* node_forces;  // [num_nodes × 3]
    float* node_torques; // [num_nodes × 3]

    // Scalars
    float energy;
    float best_energy;

    // Which ligand this optimizer is for
    int ligand_id;
};

// Batch memory allocation
// Pre-allocated contiguous memory for all optimizers
struct BFGSBatchMemory {
    // Per-optimizer arrays (strided by n_optimizers)
    float* all_x;           // [n_optimizers × max_conf_size]
    float* all_x_new;       // [n_optimizers × max_conf_size]
    float* all_g;           // [n_optimizers × max_change_size]
    float* all_g_new;       // [n_optimizers × max_change_size]
    float* all_p;           // [n_optimizers × max_change_size]
    float* all_y;           // [n_optimizers × max_change_size]
    float* all_h;           // [n_optimizers × max_hessian_size]
    float* all_coords;      // [n_optimizers × max_atoms × 3]
    float* all_forces;      // [n_optimizers × max_atoms × 3]
    float* all_node_forces; // [n_optimizers × max_nodes × 3]
    float* all_node_torques;// [n_optimizers × max_nodes × 3]

    // Output energies
    float* all_energies;    // [n_optimizers]
    float* all_best_energies; // [n_optimizers]

    // Final conformations (best found)
    float* all_best_confs;  // [n_optimizers × max_conf_size]

    // Dimensions
    int max_conf_size;
    int max_change_size;
    int max_hessian_size;
    int max_atoms;
    int max_nodes;
    int n_optimizers;

    // Total memory allocated (for debugging/stats)
    size_t total_bytes;
};

// Multi-ligand batch structure
struct LigandBatch {
    // Per-ligand scoring contexts (device pointers)
    ScoringContext* ligand_contexts;  // [num_ligands]

    // Mapping: optimizer_id → ligand_id
    int* optimizer_to_ligand;         // [total_optimizers]

    // Per-ligand optimizer counts
    int* optimizers_per_ligand;       // [num_ligands]

    int num_ligands;
    int total_optimizers;
};

// Host-side functions

// Allocate batch memory on GPU
void allocate_batch_memory(
    BFGSBatchMemory& mem,
    int n_optimizers,
    int max_conf_size,
    int max_change_size,
    int max_atoms,
    int max_nodes
);

// Free batch memory
void free_batch_memory(BFGSBatchMemory& mem);

// Create scoring context from gpu_data/cache
void create_scoring_context(
    ScoringContext& ctx,
    const struct gpu_data& gdata,
    const struct GPUCacheInfo& cacheInfo
);

// High-level interface for running parallel BFGS docking
// verbosity: 0=silent, 1=warnings/info, 2=detailed
// direct_pairwise: use direct pairwise scoring with LUT instead of grid interpolation
// receptor_coords/types: receptor atom data for direct pairwise mode (ignored if direct_pairwise=false)
void run_parallel_bfgs_docking(
    const struct gpu_data& gdata,
    const struct GPUCacheInfo& cacheInfo,
    int n_poses,
    int max_iterations,
    const gfloat3& box_min,
    const gfloat3& box_max,
    unsigned int seed,
    std::vector<float>& out_energies,
    std::vector<std::vector<float>>& out_conformations,
    int verbosity = 0,
    bool direct_pairwise = false,
    const std::vector<float>* receptor_coords = nullptr,
    const std::vector<uint8_t>* receptor_types = nullptr
);

// High-level interface for local minimization from input pose
// Returns: final energy (inter-adjusted), intramolecular energy
// out_conf: optimized conformation in flat format
// out_gradient: optional output for final gradient (pass nullptr to skip)
// verbosity: 0=silent, 1=warnings/info, 2=detailed
void run_parallel_bfgs_minimize(
    const struct gpu_data& gdata,
    const struct GPUCacheInfo& cacheInfo,
    const std::vector<float>& input_conf,  // Initial conformation in flat format
    int max_iterations,
    float& out_energy,
    float& out_intramolecular,
    std::vector<float>& out_conf,
    std::vector<float>* out_gradient = nullptr,  // Optional: final gradient
    int verbosity = 0
);

// High-level interface for score-only mode (no optimization)
// Evaluates energy at the given conformation using GPU
// Returns: inter-molecular energy (grid-based), intramolecular energy
void run_gpu_score_only(
    const struct gpu_data& gdata,
    const struct GPUCacheInfo& cacheInfo,
    const std::vector<float>& input_conf,  // Conformation in flat format
    float& out_inter_energy,
    float& out_intramolecular_energy
);

// Helper: convert conf struct to flat array format for GPU
// flat_conf must be pre-allocated with size >= 7 * nlig_roots + n_torsions
// Uses gpu_data for DFS-to-BFS torsion index conversion
void conf_to_flat(
    const struct conf& c,
    unsigned nlig_roots,
    unsigned n_torsions,
    float* flat_conf,
    const struct gpu_data& gdata
);

// Helper: convert flat array format back to conf struct
// Uses gpu_data for BFS-to-DFS torsion index conversion
void flat_to_conf(
    const float* flat_conf,
    unsigned nlig_roots,
    unsigned n_torsions,
    struct conf& c,
    const struct gpu_data& gdata
);

// Launch parallel BFGS kernel
// verbosity: 0=silent, 1=warnings/info, 2=detailed
void launch_parallel_bfgs(
    const LigandBatch& batch,
    BFGSBatchMemory& mem,
    int max_iterations,
    const gfloat3& box_min,
    const gfloat3& box_max,
    unsigned int random_seed,
    int verbosity = 0
);

// Collect results from GPU
void collect_bfgs_results(
    const BFGSBatchMemory& mem,
    int n_optimizers,
    std::vector<float>& energies,
    std::vector<std::vector<float>>& conformations
);

// Device functions (defined in bfgs_parallel.cu)

// Single-thread tree transformation: internal coords → Cartesian
__device__ void set_conf_single_thread(
    const ScoringContext& ctx,
    const float* conf,
    float* coords,
    float* node_origins,
    float* node_orientations,
    float* node_axes          // [num_nodes × 3] output: transformed axes in lab frame
);

// Single-thread intramolecular energy
__device__ float eval_intramolecular_single_thread(
    const ScoringContext& ctx,
    const float* coords,
    float* forces
);

// Single-thread gradient computation (Cartesian forces → internal gradient)
__device__ void compute_gradient_single_thread(
    const ScoringContext& ctx,
    const float* coords,
    const float* forces,
    const float* node_origins,  // Node origins from set_conf [num_nodes × 3]
    const float* node_axes,     // Transformed axes from set_conf [num_nodes × 3]
    float* node_forces,
    float* node_torques,
    float* gradient
);

// Single-thread line search
__device__ float line_search_single_thread(
    const ScoringContext& ctx,
    float* x,
    float* x_new,
    const float* g,
    const float* p,
    float f0,
    float* coords,
    float* forces,
    float* node_forces,
    float* node_torques,
    float* g_new,
    float& f_new
);

// BFGS Hessian update (single thread)
__device__ void bfgs_hessian_update_single_thread(
    float* h,
    const float* p,
    const float* y,
    float alpha,
    int n
);

// ============================================================================
// Inline Device Functions (shared between bfgs_parallel.cu and warp_coop_bfgs.cu)
// ============================================================================

// Helper: evaluate a single spline at distance r
// Returns energy value and sets deriv to derivative
__device__ inline float evaluate_spline_device(
    float* spline,
    float r,
    float fraction,
    float cutoff,
    float& deriv
) {
    if (r >= cutoff || r < 0) {
        deriv = 0;
        return 0;
    }

    unsigned index = (unsigned)(r / fraction);
    unsigned base = 5 * index;

    float x = spline[base];
    float a = spline[base + 1];
    float b = spline[base + 2];
    float c = spline[base + 3];
    float d = spline[base + 4];

    float lx = r - x;
    float val = ((a * lx + b) * lx + c) * lx + d;
    deriv = (3 * a * lx + 2 * b) * lx + c;

    return val;
}

// Evaluate intramolecular pair interaction with charge-dependent components
// Matches gpucode.cu eval_deriv_gpu()
__device__ inline float eval_pair_deriv_gpu(
    const GPUSplineInfo* splineInfo,
    unsigned t,       // atom A type
    float charge,     // atom A charge
    unsigned rt,      // atom B type
    float rcharge,    // atom B charge
    float r2,         // distance squared
    float& dor        // output: derivative/r for force computation
) {
    float r = sqrtf(r2);

    // Sort types so t1 <= t2, and sort charges accordingly
    unsigned t1, t2;
    float charge1, charge2;
    if (t < rt) {
        t1 = t;
        t2 = rt;
        charge1 = fabsf(charge);
        charge2 = fabsf(rcharge);
    } else {
        t1 = rt;
        t2 = t;
        charge1 = fabsf(rcharge);
        charge2 = fabsf(charge);
    }

    // Symmetric indexing for spline lookup
    unsigned tindex = t1 + t2 * (t2 + 1) / 2;
    const GPUSplineInfo& spInfo = splineInfo[tindex];
    unsigned n = spInfo.n;  // number of charge-dependent components

    float ret = 0, d = 0;

    // Evaluate up to 4 charge-dependent spline components
    if (n > 0) {
        float fraction = spInfo.fraction;
        float cutoff = spInfo.cutoff;
        float val, deriv;

        // Component 0: TypeDependentOnly (no charge adjustment)
        val = evaluate_spline_device(spInfo.splines[0], r, fraction, cutoff, deriv);
        ret += val;
        d += deriv;

        // Component 1: AbsAChargeDependent (multiply by abs(chargeA))
        if (n > 1) {
            val = evaluate_spline_device(spInfo.splines[1], r, fraction, cutoff, deriv);
            ret += val * charge1;
            d += deriv * charge1;

            // Component 2: AbsBChargeDependent (multiply by abs(chargeB))
            if (n > 2) {
                val = evaluate_spline_device(spInfo.splines[2], r, fraction, cutoff, deriv);
                ret += val * charge2;
                d += deriv * charge2;

                // Component 3: ABChargeDependent (multiply by chargeA * chargeB)
                if (n > 3) {
                    val = evaluate_spline_device(spInfo.splines[3], r, fraction, cutoff, deriv);
                    ret += val * charge2 * charge1;
                    d += deriv * charge2 * charge1;
                }
            }
        }
    }

    dor = d / r;  // Divide by distance to normalize for force computation
    return ret;
}

#endif // BFGS_PARALLEL_H
