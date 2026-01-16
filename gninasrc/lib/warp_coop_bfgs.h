/*
 * warp_coop_bfgs.h
 *
 * Warp-cooperative BFGS optimizer where multiple threads collaborate
 * on each optimizer using warp shuffles for ~5 cycle register-to-register
 * communication instead of ~32-273 cycle L1/L2 access.
 *
 * Configuration: Template-based with OPTS_PER_WARP parameter
 * - WarpCoop4: 8 threads per optimizer, 4 optimizers per warp (default)
 * - WarpCoop8: 4 threads per optimizer, 8 optimizers per warp
 */

#ifndef WARP_COOP_BFGS_H
#define WARP_COOP_BFGS_H

#include <cuda_runtime.h>
#include <vector>
#include "gpu_math.h"
#include "tree_gpu.h"
#include "grid_gpu.h"
#include "bfgs_parallel.h"  // For ScoringContext

// ============================================================================
// Configuration Template
// ============================================================================

// NOTE: OPTS_PER_WARP must be <= 4 to avoid register spills.
// With OPTS_PER_WARP=4 (8 threads/opt), each thread needs ~100+ registers for
// the distributed state (coords, forces, Hessian, gradients). Higher values
// like OPTS_PER_WARP=8 (4 threads/opt) cause excessive register pressure and
// performance degradation from spills to local memory.

template<
    int OPTS_PER_WARP_,
    int MAX_ATOMS_ = 50,
    int MAX_TORSIONS_ = 10
>
struct WarpCoopConfig {
    // Primary parameters
    // IMPORTANT: OPTS_PER_WARP should be <= 4 to avoid register spills
    static constexpr int OPTS_PER_WARP = OPTS_PER_WARP_;
    static constexpr int MAX_ATOMS = MAX_ATOMS_;
    static constexpr int MAX_TORSIONS = MAX_TORSIONS_;

    // Derived constants
    static constexpr int THREADS_PER_OPT = 32 / OPTS_PER_WARP;
    static constexpr int MAX_NODES = MAX_TORSIONS + 1;
    static constexpr int N_CONF = 7 + MAX_TORSIONS;
    static constexpr int N_CHANGE = 6 + MAX_TORSIONS;
    static constexpr int N_HESSIAN = N_CHANGE * (N_CHANGE + 1) / 2;

    // Per-thread distribution (ceiling division)
    static constexpr int ATOMS_PER_THREAD = (MAX_ATOMS + THREADS_PER_OPT - 1) / THREADS_PER_OPT;
    static constexpr int CONF_PER_THREAD = (N_CONF + THREADS_PER_OPT - 1) / THREADS_PER_OPT;
    static constexpr int CHANGE_PER_THREAD = (N_CHANGE + THREADS_PER_OPT - 1) / THREADS_PER_OPT;
    static constexpr int HESS_PER_THREAD = (N_HESSIAN + THREADS_PER_OPT - 1) / THREADS_PER_OPT;
    static constexpr int NODES_PER_THREAD = (MAX_NODES + THREADS_PER_OPT - 1) / THREADS_PER_OPT;

    // Memory per optimizer (floats)
    static constexpr int FLOATS_PER_OPT =
        3 * N_CONF +           // x, x_new, best_conf
        4 * N_CHANGE +         // g, g_new, p, y
        N_HESSIAN +            // Hessian
        6 * MAX_ATOMS +        // coords, forces
        15 * MAX_NODES +       // origins(3) + orientations(9) + axes(3)
        6 * MAX_NODES +        // node_forces, node_torques
        2;                     // energy, best_energy

    // Registers per thread (data + working)
    static constexpr int DATA_REGS_PER_THREAD =
        (FLOATS_PER_OPT + THREADS_PER_OPT - 1) / THREADS_PER_OPT;
    static constexpr int WORKING_REGS = 50;
    static constexpr int REGS_PER_THREAD = DATA_REGS_PER_THREAD + WORKING_REGS;

    // Validation
    static_assert(32 % OPTS_PER_WARP == 0, "OPTS_PER_WARP must divide 32");
    static_assert(THREADS_PER_OPT >= 1 && THREADS_PER_OPT <= 32, "Invalid threads per optimizer");
    static_assert(REGS_PER_THREAD <= 255, "Exceeds max 255 registers per thread");
};

// Predefined configurations
using WarpCoop1  = WarpCoopConfig<1,  50, 10>;  // 32 threads/opt, 1 opt/warp
using WarpCoop2  = WarpCoopConfig<2,  50, 10>;  // 16 threads/opt, 2 opts/warp
using WarpCoop4  = WarpCoopConfig<4,  50, 10>;  // 8 threads/opt,  4 opts/warp (default)
using WarpCoop8  = WarpCoopConfig<8,  50, 10>;  // 4 threads/opt,  8 opts/warp

// Default configuration
using DefaultWarpCoopConfig = WarpCoop4;

// ============================================================================
// Thread Indexing (device-only, constructor in .cu file)
// ============================================================================

template<typename Config>
struct ThreadIndex {
    int warp_lane;          // 0-31: position within warp
    int opt_in_warp;        // 0 to OPTS_PER_WARP-1: which optimizer in this warp
    int local_lane;         // 0 to THREADS_PER_OPT-1: position within optimizer
    int opt_base_lane;      // Base warp lane for this optimizer
    unsigned int opt_mask;  // Shuffle mask for this optimizer's threads

    __device__ __forceinline__ ThreadIndex();

    // Get global optimizer ID
    __device__ __forceinline__ int global_opt_id(int warps_per_block) const;
};

// ============================================================================
// Shuffle Operations (device-only, defined in .cu file)
// ============================================================================

template<typename Config>
struct ShuffleOps {
    const ThreadIndex<Config>& idx;

    __device__ __forceinline__ ShuffleOps(const ThreadIndex<Config>& i) : idx(i) {}

    // Shuffle within optimizer group
    __device__ __forceinline__ float shfl(float val, int src_local_lane) const;

    // Broadcast from local lane 0
    __device__ __forceinline__ float broadcast(float val) const;

    // Reduce sum across optimizer's threads
    __device__ __forceinline__ float reduce_sum(float val) const;

    // Reduce max across optimizer's threads
    __device__ __forceinline__ float reduce_max(float val) const;
};

// ============================================================================
// Distributed Array Access
// ============================================================================

template<typename Config>
struct DistributedArray {
    // Get element from distributed array (striped across threads)
    __device__ __forceinline__ static float get(
        const float* local_data,
        int global_idx,
        const ShuffleOps<Config>& shfl
    ) {
        int owner_lane = global_idx % Config::THREADS_PER_OPT;
        int local_slot = global_idx / Config::THREADS_PER_OPT;
        float val = local_data[local_slot];
        return shfl.shfl(val, owner_lane);
    }

    // Set element in distributed array (only owner thread writes)
    __device__ __forceinline__ static void set(
        float* local_data,
        int global_idx,
        float value,
        const ThreadIndex<Config>& idx
    ) {
        int owner_lane = global_idx % Config::THREADS_PER_OPT;
        int local_slot = global_idx / Config::THREADS_PER_OPT;
        if (idx.local_lane == owner_lane) {
            local_data[local_slot] = value;
        }
    }

    // Set element with broadcast (all threads have the value)
    __device__ __forceinline__ static void set_broadcast(
        float* local_data,
        int global_idx,
        float value,
        const ThreadIndex<Config>& idx
    ) {
        int owner_lane = global_idx % Config::THREADS_PER_OPT;
        int local_slot = global_idx / Config::THREADS_PER_OPT;
        if (idx.local_lane == owner_lane) {
            local_data[local_slot] = value;
        }
    }
};

// ============================================================================
// Atom Partition
// ============================================================================

template<typename Config>
struct AtomPartition {
    int start;
    int end;
    int count;

    __device__ __forceinline__ AtomPartition(int local_lane, int num_atoms) {
        // Evenly distribute atoms across threads
        int base = num_atoms / Config::THREADS_PER_OPT;
        int remainder = num_atoms % Config::THREADS_PER_OPT;

        int min_val = (local_lane < remainder) ? local_lane : remainder;
        start = local_lane * base + min_val;
        end = start + base + (local_lane < remainder ? 1 : 0);
        count = end - start;
    }
};

// ============================================================================
// Per-Optimizer State (Distributed in Registers)
// ============================================================================

template<typename Config>
struct WarpCoopState {
    // Conformations (distributed across threads)
    float x[Config::CONF_PER_THREAD];           // Current conf
    float x_new[Config::CONF_PER_THREAD];       // Trial conf
    float best_conf[Config::CONF_PER_THREAD];   // Best found

    // Gradients and search vectors
    float g[Config::CHANGE_PER_THREAD];         // Current gradient
    float g_new[Config::CHANGE_PER_THREAD];     // Trial gradient
    float p[Config::CHANGE_PER_THREAD];         // Search direction
    float y[Config::CHANGE_PER_THREAD];         // Gradient difference

    // Hessian (triangular, distributed)
    float h[Config::HESS_PER_THREAD];

    // Node transforms (distributed)
    float node_origins[Config::NODES_PER_THREAD * 3];
    float node_orientations[Config::NODES_PER_THREAD * 9];
    float node_axes[Config::NODES_PER_THREAD * 3];
    float node_forces[Config::NODES_PER_THREAD * 3];
    float node_torques[Config::NODES_PER_THREAD * 3];

    // Atom data (each thread owns its atoms' data directly)
    float my_coords[Config::ATOMS_PER_THREAD * 3];
    float my_forces[Config::ATOMS_PER_THREAD * 3];

    // Scalars (used by all threads, but only lane 0 writes)
    float energy;
    float best_energy;
};

// ============================================================================
// Host Interface
// ============================================================================

// Run warp-cooperative BFGS docking
// Template parameter selects configuration
template<typename Config = DefaultWarpCoopConfig>
void run_warp_coop_bfgs_docking(
    const struct gpu_data& gdata,
    const struct GPUCacheInfo& cacheInfo,
    int n_poses,
    int max_iterations,
    const gfloat3& box_min,
    const gfloat3& box_max,
    unsigned int seed,
    std::vector<float>& out_energies,
    std::vector<std::vector<float>>& out_conformations,
    int verbosity = 0
);

// Non-template wrapper for CLI (uses default config)
void run_warp_coop_bfgs_docking_default(
    const struct gpu_data& gdata,
    const struct GPUCacheInfo& cacheInfo,
    int n_poses,
    int max_iterations,
    const gfloat3& box_min,
    const gfloat3& box_max,
    unsigned int seed,
    std::vector<float>& out_energies,
    std::vector<std::vector<float>>& out_conformations,
    int verbosity = 0
);

#endif // WARP_COOP_BFGS_H
