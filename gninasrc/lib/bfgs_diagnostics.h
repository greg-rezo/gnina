/*
 * bfgs_diagnostics.h
 *
 * Runtime diagnostics for GPU BFGS optimization kernel.
 * Provides warnings for:
 * - Grid memory exceeding L2 cache
 * - Per-optimizer memory requirements
 * - Register pressure
 */

#ifndef BFGS_DIAGNOSTICS_H
#define BFGS_DIAGNOSTICS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>

// Resource information structure
struct BFGSResourceInfo {
    // GPU properties
    size_t l2_cache_size;
    int sm_count;
    int registers_per_sm;

    // Grid analysis
    size_t grid_memory_bytes;
    float grid_l2_ratio;  // grid_memory / l2_cache
    bool grid_fits_l2;

    // Kernel analysis
    int kernel_registers;
    int max_threads_per_sm;
    float occupancy;

    // Ligand analysis
    int num_torsions;
    int hessian_size;
    size_t per_thread_memory;
};

// Check GPU properties (no-op, logging removed)
inline void check_gpu_properties(int verbosity = 1) {
    (void)verbosity;  // unused
}

// Estimate grid memory from explicit dimensions
inline size_t estimate_grid_memory_explicit(
    int nx, int ny, int nz,
    unsigned ngrids,
    bool has_chargedata = true
) {
    size_t points_per_grid = (size_t)nx * ny * nz;
    size_t grids_count = ngrids * (has_chargedata ? 2 : 1);
    return points_per_grid * grids_count * sizeof(float);
}

// Warn if grid memory exceeds L2 cache
// WARNING/ERROR messages always appear regardless of verbosity
inline void warn_grid_l2_fit(
    size_t grid_bytes,
    int verbosity = 0
) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t l2_size = prop.l2CacheSize;
    float ratio = (float)grid_bytes / l2_size;

    if (verbosity >= 1) {
        fprintf(stderr, "INFO: Grid memory: %.1f MB (%.0f%% of L2 cache)\n",
                grid_bytes / (1024.0 * 1024.0), ratio * 100);
    }

    // Warnings always appear regardless of verbosity
    if (ratio > 1.0) {
        fprintf(stderr, "ERROR: Grid size (%.1f MB) exceeds L2 cache (%.1f MB).\n",
                grid_bytes / (1024.0 * 1024.0), l2_size / (1024.0 * 1024.0));
        fprintf(stderr, "       Performance will be severely degraded.\n");
        fprintf(stderr, "       Consider: increasing grid spacing, reducing box size.\n");
    } else if (ratio > 0.85) {
        fprintf(stderr, "WARNING: Grid size (%.1f MB) approaches L2 cache limit.\n",
                grid_bytes / (1024.0 * 1024.0));
        fprintf(stderr, "         Performance may be impacted. Headroom: %.1f MB\n",
                (l2_size - grid_bytes) / (1024.0 * 1024.0));
    }
}

// Check per-optimizer memory requirements
// WARNING messages always appear regardless of verbosity
inline void check_optimizer_resources(
    int n_conf,
    int n_change,
    int num_atoms,
    int num_nodes,
    int verbosity = 0
) {
    // Per-thread memory calculation (matches BFGSBatchMemory layout)
    int hessian_size = n_change * (n_change + 1) / 2;
    size_t per_thread_bytes =
        (n_conf * 2 +           // x, x_new
         n_change * 4 +         // g, g_new, p, y
         hessian_size +         // h (triangular Hessian)
         num_atoms * 3 * 2 +    // coords, forces
         num_nodes * 3 * 2      // node_forces, node_torques
        ) * sizeof(float);

    // Thresholds based on typical GPU memory constraints
    const size_t WARN_THRESHOLD = 20 * 1024;   // 20 KB
    const size_t ERROR_THRESHOLD = 50 * 1024;  // 50 KB

    // Warnings always appear regardless of verbosity
    if (per_thread_bytes > ERROR_THRESHOLD) {
        fprintf(stderr, "WARNING: Per-optimizer memory (%.1f KB) is very high.\n",
                per_thread_bytes / 1024.0);
        fprintf(stderr, "         Consider ligands with fewer torsions for optimal GPU performance.\n");
    } else if (per_thread_bytes > WARN_THRESHOLD && verbosity >= 2) {
        fprintf(stderr, "INFO: Per-optimizer memory (%.1f KB) approaching limit.\n",
                per_thread_bytes / 1024.0);
    }
}

// Check kernel register usage (must be called from .cu file with function pointer)
// WARNING messages always appear regardless of verbosity
inline void check_kernel_registers(
    const char* kernel_name,
    int num_regs,
    int verbosity = 0
) {
    // Target: 64 registers/thread for 1024 threads/SM on L4
    const int TARGET_REGS = 64;
    const int WARN_REGS = 96;   // Drops to 512 threads/SM
    const int ERROR_REGS = 128; // Drops to 256 threads/SM
    (void)kernel_name;  // unused now

    // Warnings always appear regardless of verbosity
    if (num_regs > ERROR_REGS) {
        fprintf(stderr, "WARNING: Kernel register usage (%d) exceeds %d.\n",
                num_regs, ERROR_REGS);
        fprintf(stderr, "         SM occupancy limited to ~256 threads.\n");
    } else if (num_regs > WARN_REGS) {
        fprintf(stderr, "WARNING: Kernel register usage (%d) exceeds %d.\n",
                num_regs, WARN_REGS);
        fprintf(stderr, "         SM occupancy limited to ~512 threads (target: 1024).\n");
    } else if (num_regs > TARGET_REGS && verbosity >= 2) {
        fprintf(stderr, "INFO: Kernel register usage (%d) slightly above target (%d).\n",
                num_regs, TARGET_REGS);
    }
}

// Get kernel register count via CUDA API
template<typename KernelFunc>
inline int get_kernel_registers(KernelFunc kernel) {
    cudaFuncAttributes attrs;
    cudaError_t err = cudaFuncGetAttributes(&attrs, kernel);
    if (err != cudaSuccess) {
        fprintf(stderr, "WARNING: Could not query kernel attributes: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return attrs.numRegs;
}

// Combined diagnostic check for BFGS launch
// Always runs checks - warnings appear regardless of verbosity
// INFO messages only appear when verbosity >= 1
inline void check_bfgs_resources(
    int n_conf,
    int n_change,
    int num_atoms,
    int num_nodes,
    size_t grid_memory_bytes,
    int verbosity = 0
) {
    if (verbosity >= 1) {
        check_gpu_properties(verbosity);
    }
    check_optimizer_resources(n_conf, n_change, num_atoms, num_nodes, verbosity);
    if (grid_memory_bytes > 0) {
        warn_grid_l2_fit(grid_memory_bytes, verbosity);
    }
}

#endif // BFGS_DIAGNOSTICS_H
