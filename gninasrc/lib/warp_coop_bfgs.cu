/*
 * warp_coop_bfgs.cu
 *
 * Warp-cooperative BFGS implementation where multiple threads
 * collaborate on each optimizer using warp shuffles.
 */

#include "warp_coop_bfgs.h"
#include "gpu_math.h"
#include "tree_gpu.h"
#include "grid_gpu.h"
#include "gpucode.h"
#include "bfgs_parallel.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <cstdio>
#include <stdexcept>
#include <sstream>

// Thrust for spatial sorting (DISABLED - warp_coop not actively developed)
// #include <thrust/device_ptr.h>
// #include <thrust/sort.h>
// #include <thrust/sequence.h>

// Debug mode - uncomment to enable verbose kernel debug output
// #define WARP_COOP_DEBUG

// Error checking macro (defined early for spatial sorting functions)
#define CUDA_CHECK_WARP(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
    } \
} while(0)

// ============================================================================
// Spatial Sorting for Cache Locality (DISABLED - warp_coop not actively developed)
// ============================================================================
// See bfgs_parallel.cu for active spatial sorting implementation
#if 0
// Sort poses by 3D position using Morton codes (Z-order curve) so that
// adjacent threads/warps process spatially-nearby poses, improving L1 cache hit rate.

// Morton encoding: spread bits of a 10-bit integer for 3D interleaving
__device__ __forceinline__ unsigned int morton_spread_bits(unsigned int v) {
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

// Encode 3D grid cell coordinates into Morton code
__device__ __forceinline__ unsigned int morton_encode_3d(unsigned int x, unsigned int y, unsigned int z) {
    return morton_spread_bits(x) | (morton_spread_bits(y) << 1) | (morton_spread_bits(z) << 2);
}

// Kernel to compute Morton-encoded cell IDs for each pose based on position
__global__ void compute_pose_cell_ids_kernel(
    const float* __restrict__ confs,      // [n_poses * n_conf]
    unsigned int* __restrict__ cell_ids,  // [n_poses] output
    int* __restrict__ pose_indices,       // [n_poses] initialized to 0,1,2,...
    int n_poses,
    int n_conf,
    gfloat3 box_min,
    gfloat3 box_size,
    int grid_divisions  // e.g., 8 -> 8x8x8 = 512 cells
) {
    int pose_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pose_id >= n_poses) return;

    // Pose position is first 3 floats (translation x, y, z)
    float px = confs[pose_id * n_conf + 0];
    float py = confs[pose_id * n_conf + 1];
    float pz = confs[pose_id * n_conf + 2];

    // Normalize to [0, grid_divisions) range
    float inv_size_x = (box_size.x > 0) ? grid_divisions / box_size.x : 0;
    float inv_size_y = (box_size.y > 0) ? grid_divisions / box_size.y : 0;
    float inv_size_z = (box_size.z > 0) ? grid_divisions / box_size.z : 0;

    int cx = min(grid_divisions - 1, max(0, (int)((px - box_min.x) * inv_size_x)));
    int cy = min(grid_divisions - 1, max(0, (int)((py - box_min.y) * inv_size_y)));
    int cz = min(grid_divisions - 1, max(0, (int)((pz - box_min.z) * inv_size_z)));

    // Morton code for Z-order curve spatial locality
    cell_ids[pose_id] = morton_encode_3d(cx, cy, cz);
    pose_indices[pose_id] = pose_id;
}

// Kernel to reorder conformations according to sorted indices
__global__ void reorder_confs_kernel(
    const float* __restrict__ confs_in,       // [n_poses * n_conf]
    float* __restrict__ confs_out,            // [n_poses * n_conf]
    const int* __restrict__ sorted_indices,   // [n_poses]
    int* __restrict__ original_indices,       // [n_poses] output: maps new pose index -> original
    int n_poses,
    int n_conf
) {
    int new_pose_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_pose_id >= n_poses) return;

    int old_pose_id = sorted_indices[new_pose_id];

    // Store mapping from new index to original index
    original_indices[new_pose_id] = old_pose_id;

    // Copy all n_conf floats for this pose
    const float* src = confs_in + old_pose_id * n_conf;
    float* dst = confs_out + new_pose_id * n_conf;
    for (int i = 0; i < n_conf; i++) {
        dst[i] = src[i];
    }
}

// Host function to sort poses spatially for better cache locality
// Returns device pointer to original_indices array (caller must free)
int* sort_poses_spatially(
    float* d_confs,           // Device pointer to conformations, will be reordered
    int n_poses,
    int n_conf,
    gfloat3 box_min,
    gfloat3 box_max,
    int verbosity
) {
    const int GRID_DIVISIONS = 8;  // 8x8x8 = 512 cells
    gfloat3 box_size = {box_max.x - box_min.x, box_max.y - box_min.y, box_max.z - box_min.z};

    // Allocate temporary buffers
    unsigned int* d_cell_ids;
    int* d_pose_indices;
    int* d_original_indices;
    float* d_confs_sorted;

    CUDA_CHECK_WARP(cudaMalloc(&d_cell_ids, n_poses * sizeof(unsigned int)));
    CUDA_CHECK_WARP(cudaMalloc(&d_pose_indices, n_poses * sizeof(int)));
    CUDA_CHECK_WARP(cudaMalloc(&d_original_indices, n_poses * sizeof(int)));
    CUDA_CHECK_WARP(cudaMalloc(&d_confs_sorted, n_poses * n_conf * sizeof(float)));

    // Step 1: Compute cell IDs for each pose
    int block_size = 256;
    int grid_size = (n_poses + block_size - 1) / block_size;
    compute_pose_cell_ids_kernel<<<grid_size, block_size>>>(
        d_confs, d_cell_ids, d_pose_indices,
        n_poses, n_conf, box_min, box_size, GRID_DIVISIONS
    );
    CUDA_CHECK_WARP(cudaDeviceSynchronize());

    // Step 2: Sort pose indices by cell ID using thrust
    thrust::device_ptr<unsigned int> keys(d_cell_ids);
    thrust::device_ptr<int> values(d_pose_indices);
    thrust::sort_by_key(keys, keys + n_poses, values);

    // Step 3: Reorder confs according to sorted indices
    reorder_confs_kernel<<<grid_size, block_size>>>(
        d_confs, d_confs_sorted, d_pose_indices, d_original_indices,
        n_poses, n_conf
    );
    CUDA_CHECK_WARP(cudaDeviceSynchronize());

    // Step 4: Copy sorted confs back to original buffer
    CUDA_CHECK_WARP(cudaMemcpy(d_confs, d_confs_sorted, n_poses * n_conf * sizeof(float), cudaMemcpyDeviceToDevice));

    // Cleanup temporary buffers (but keep original_indices)
    cudaFree(d_cell_ids);
    cudaFree(d_pose_indices);
    cudaFree(d_confs_sorted);

    if (verbosity >= 1) {
        fprintf(stderr, "  Spatial sorting: %d poses into %d^3 grid cells\n",
                n_poses, GRID_DIVISIONS);
    }

    return d_original_indices;  // Caller must free this
}
#endif

// Use inline single-threaded computation for debugging
// #define USE_INLINE_DEBUG

#ifdef WARP_COOP_DEBUG
#define WARP_DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define WARP_DEBUG_PRINT(fmt, ...) ((void)0)
#endif

// Normalize angle to [-pi, pi] (matches reference)
__device__ inline float normalize_angle_warp(float a) {
    const float pi = 3.14159265358979323846f;
    if (!isfinite(a)) return 0.0f;
    a = fmodf(a, 2*pi);
    if (a > pi) a -= 2*pi;
    if (a < -pi) a += 2*pi;
    return a;
}

// Convert angle-axis rotation to quaternion (matches reference angle_to_quaternion_device)
__device__ inline void angle_to_quaternion_warp(float rx, float ry, float rz, float* q) {
    const float epsilon_fl = 1e-7f;
    float angle = sqrtf(rx*rx + ry*ry + rz*rz);
    if (angle > epsilon_fl) {
        float s = sinf(angle * 0.5f) / angle;
        q[0] = cosf(angle * 0.5f);  // w
        q[1] = s * rx;               // x
        q[2] = s * ry;               // y
        q[3] = s * rz;               // z
    } else {
        // Identity quaternion for small/zero angles (matches reference)
        q[0] = 1.0f;
        q[1] = 0.0f;
        q[2] = 0.0f;
        q[3] = 0.0f;
    }
}

// Quaternion multiplication: result = q1 * q2
__device__ inline void quat_mult_warp(const float* q1, const float* q2, float* result) {
    result[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
    result[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2];
    result[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1];
    result[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0];
}

// Increment conformation: x_new = x + alpha * p (with proper quaternion handling)
// This matches reference increment_conf_single_thread
// x has 7 * nlig_roots + n_torsions components (3 pos + 4 quat per root + torsions)
// p has 6 * nlig_roots + n_torsions components (3 pos + 3 angle-axis per root + torsions)
template<typename Config>
__device__ void increment_conf_warp(
    const float* all_x,    // Current conformation (n_conf values)
    const float* all_p,    // Search direction (n_change values)
    float alpha,
    int nlig_roots,
    int n_conf,
    int n_change,
    WarpCoopState<Config>& state,
    const ThreadIndex<Config>& idx
) {
    using DA = DistributedArray<Config>;

    // Process rigid body transforms for each ligand root
    for (int r = 0; r < nlig_roots; r++) {
        // Position increment (simple addition)
        for (int j = 0; j < 3; j++) {
            int x_idx = r * 7 + j;
            int p_idx = r * 6 + j;
            if ((x_idx % Config::THREADS_PER_OPT) == idx.local_lane) {
                float x_new = all_x[x_idx] + alpha * all_p[p_idx];
                DA::set(state.x_new, x_idx, x_new, idx);
            }
        }

        // Orientation increment via quaternion multiplication
        // p contains angle-axis rotation, x contains quaternion
        float rot[3] = {alpha * all_p[r * 6 + 3],
                       alpha * all_p[r * 6 + 4],
                       alpha * all_p[r * 6 + 5]};
        float dq[4], q[4], qnew[4];
        angle_to_quaternion_warp(rot[0], rot[1], rot[2], dq);
        q[0] = all_x[r * 7 + 3];
        q[1] = all_x[r * 7 + 4];
        q[2] = all_x[r * 7 + 5];
        q[3] = all_x[r * 7 + 6];
        quat_mult_warp(dq, q, qnew);

        // Normalize quaternion (matching reference epsilon)
        const float epsilon_fl = 1e-7f;
        float qn = sqrtf(qnew[0]*qnew[0] + qnew[1]*qnew[1] + qnew[2]*qnew[2] + qnew[3]*qnew[3]);
        if (qn > epsilon_fl) {
            qnew[0] /= qn; qnew[1] /= qn; qnew[2] /= qn; qnew[3] /= qn;
        }

        // Store quaternion (each thread stores its lane's portion)
        for (int j = 0; j < 4; j++) {
            int x_idx = r * 7 + 3 + j;
            if ((x_idx % Config::THREADS_PER_OPT) == idx.local_lane) {
                DA::set(state.x_new, x_idx, qnew[j], idx);
            }
        }
    }

    // Torsion increments (with angle normalization)
    int torsion_start_x = 7 * nlig_roots;
    int torsion_start_p = 6 * nlig_roots;
    int n_torsions = n_conf - torsion_start_x;

    for (int t = idx.local_lane; t < n_torsions; t += Config::THREADS_PER_OPT) {
        int x_idx = torsion_start_x + t;
        int p_idx = torsion_start_p + t;
        // Match reference: add raw increment and normalize only the final result
        float x_new = all_x[x_idx] + alpha * all_p[p_idx];
        x_new = normalize_angle_warp(x_new);
        DA::set(state.x_new, x_idx, x_new, idx);
    }
}

// ============================================================================
// ThreadIndex Method Implementations
// ============================================================================

template<typename Config>
__device__ __forceinline__ ThreadIndex<Config>::ThreadIndex() {
    warp_lane = threadIdx.x % 32;
    opt_in_warp = warp_lane / Config::THREADS_PER_OPT;
    local_lane = warp_lane % Config::THREADS_PER_OPT;
    opt_base_lane = opt_in_warp * Config::THREADS_PER_OPT;
    opt_mask = ((1u << Config::THREADS_PER_OPT) - 1u) << opt_base_lane;
}

template<typename Config>
__device__ __forceinline__ int ThreadIndex<Config>::global_opt_id(int warps_per_block) const {
    int warp_id = threadIdx.x / 32;
    return blockIdx.x * warps_per_block * Config::OPTS_PER_WARP
         + warp_id * Config::OPTS_PER_WARP
         + opt_in_warp;
}

// Explicit instantiation for used configurations
template struct ThreadIndex<WarpCoop4>;
template struct ThreadIndex<WarpCoop8>;

// ============================================================================
// ShuffleOps Method Implementations
// ============================================================================

template<typename Config>
__device__ __forceinline__ float ShuffleOps<Config>::shfl(float val, int src_local_lane) const {
    int src_warp_lane = idx.opt_base_lane + src_local_lane;
    // Use optimizer mask so only threads in this optimizer participate
    // This allows independent progress when threads process different data
    return __shfl_sync(idx.opt_mask, val, src_warp_lane);
}

template<typename Config>
__device__ __forceinline__ float ShuffleOps<Config>::broadcast(float val) const {
    // Use optimizer mask so only threads in this optimizer participate
    return __shfl_sync(idx.opt_mask, val, idx.opt_base_lane);
}

template<typename Config>
__device__ __forceinline__ float ShuffleOps<Config>::reduce_sum(float val) const {
    // Use XOR-based butterfly reduction to stay within optimizer group
    // XOR preserves group boundaries since groups are power-of-2 aligned
    if constexpr (Config::THREADS_PER_OPT >= 16) {
        val += __shfl_xor_sync(idx.opt_mask, val, 8);
    }
    if constexpr (Config::THREADS_PER_OPT >= 8) {
        val += __shfl_xor_sync(idx.opt_mask, val, 4);
    }
    if constexpr (Config::THREADS_PER_OPT >= 4) {
        val += __shfl_xor_sync(idx.opt_mask, val, 2);
    }
    if constexpr (Config::THREADS_PER_OPT >= 2) {
        val += __shfl_xor_sync(idx.opt_mask, val, 1);
    }
    // After XOR reduction, all threads in the group have the sum
    return val;
}

template<typename Config>
__device__ __forceinline__ float ShuffleOps<Config>::reduce_max(float val) const {
    // Use XOR-based butterfly reduction to stay within optimizer group
    if constexpr (Config::THREADS_PER_OPT >= 16) {
        val = fmaxf(val, __shfl_xor_sync(idx.opt_mask, val, 8));
    }
    if constexpr (Config::THREADS_PER_OPT >= 8) {
        val = fmaxf(val, __shfl_xor_sync(idx.opt_mask, val, 4));
    }
    if constexpr (Config::THREADS_PER_OPT >= 4) {
        val = fmaxf(val, __shfl_xor_sync(idx.opt_mask, val, 2));
    }
    if constexpr (Config::THREADS_PER_OPT >= 2) {
        val = fmaxf(val, __shfl_xor_sync(idx.opt_mask, val, 1));
    }
    // After XOR reduction, all threads in the group have the max
    return val;
}

// Explicit instantiation for used configurations
template struct ShuffleOps<WarpCoop4>;
template struct ShuffleOps<WarpCoop8>;

// ============================================================================
// Helper Device Functions
// ============================================================================

__device__ __forceinline__ void normalize_quat_warp(float* q) {
    float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm > 1e-10f) {
        float inv_norm = 1.0f / norm;
        q[0] *= inv_norm;
        q[1] *= inv_norm;
        q[2] *= inv_norm;
        q[3] *= inv_norm;
    }
}

__device__ __forceinline__ void quat_to_matrix_warp(const float* q, float* m) {
    // q = [w, x, y, z]
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;

    // Column-major for consistency with existing code
    m[0] = 1.0f - 2.0f*(yy + zz);  m[3] = 2.0f*(xy - wz);         m[6] = 2.0f*(xz + wy);
    m[1] = 2.0f*(xy + wz);         m[4] = 1.0f - 2.0f*(xx + zz);  m[7] = 2.0f*(yz - wx);
    m[2] = 2.0f*(xz - wy);         m[5] = 2.0f*(yz + wx);         m[8] = 1.0f - 2.0f*(xx + yy);
}

__device__ __forceinline__ void mat_vec_mult_warp(const float* m, const float* v, float* out) {
    out[0] = m[0]*v[0] + m[3]*v[1] + m[6]*v[2];
    out[1] = m[1]*v[0] + m[4]*v[1] + m[7]*v[2];
    out[2] = m[2]*v[0] + m[5]*v[1] + m[8]*v[2];
}

// Rodrigues rotation: rotate v around axis by angle theta
__device__ __forceinline__ void rodrigues_rotate(
    const float* axis,  // normalized
    float theta,
    const float* v,
    float* out
) {
    float c = cosf(theta);
    float s = sinf(theta);
    float dot = axis[0]*v[0] + axis[1]*v[1] + axis[2]*v[2];
    float cross[3] = {
        axis[1]*v[2] - axis[2]*v[1],
        axis[2]*v[0] - axis[0]*v[2],
        axis[0]*v[1] - axis[1]*v[0]
    };
    out[0] = v[0]*c + cross[0]*s + axis[0]*dot*(1.0f - c);
    out[1] = v[1]*c + cross[1]*s + axis[1]*dot*(1.0f - c);
    out[2] = v[2]*c + cross[2]*s + axis[2]*dot*(1.0f - c);
}

// ============================================================================
// Trilinear Interpolation (from bfgs_parallel.cu)
// ============================================================================

__device__ __forceinline__ void trilinear_interp_warp(
    const grid_gpu& grid,
    const array3d_gpu<fl, fl>& data,
    float x, float y, float z,
    float slope,
    float v,  // curl/forcecap
    float* energy,
    float* fx, float* fy, float* fz
) {
    // Convert to grid coordinates
    float sx = (x - grid.m_init.x) * grid.m_factor.x;
    float sy = (y - grid.m_init.y) * grid.m_factor.y;
    float sz = (z - grid.m_init.z) * grid.m_factor.z;

    // Handle out-of-bounds (use pre-computed m_dim_fl_minus_1 for exact match)
    float miss_x = 0, miss_y = 0, miss_z = 0;
    int region_x = 0, region_y = 0, region_z = 0;

    if (sx < 0) { miss_x = -sx; region_x = -1; sx = 0; }
    else if (sx >= grid.m_dim_fl_minus_1.x) { miss_x = sx - grid.m_dim_fl_minus_1.x; region_x = 1; sx = grid.m_dim_fl_minus_1.x - 0.001f; }

    if (sy < 0) { miss_y = -sy; region_y = -1; sy = 0; }
    else if (sy >= grid.m_dim_fl_minus_1.y) { miss_y = sy - grid.m_dim_fl_minus_1.y; region_y = 1; sy = grid.m_dim_fl_minus_1.y - 0.001f; }

    if (sz < 0) { miss_z = -sz; region_z = -1; sz = 0; }
    else if (sz >= grid.m_dim_fl_minus_1.z) { miss_z = sz - grid.m_dim_fl_minus_1.z; region_z = 1; sz = grid.m_dim_fl_minus_1.z - 0.001f; }

    int ix = (int)sx;
    int iy = (int)sy;
    int iz = (int)sz;

    float fx_ = sx - ix;
    float fy_ = sy - iy;
    float fz_ = sz - iz;

    // Get corner values using __ldg for texture cache
    float c000 = __ldg(&data(ix, iy, iz));
    float c001 = __ldg(&data(ix, iy, iz+1));
    float c010 = __ldg(&data(ix, iy+1, iz));
    float c011 = __ldg(&data(ix, iy+1, iz+1));
    float c100 = __ldg(&data(ix+1, iy, iz));
    float c101 = __ldg(&data(ix+1, iy, iz+1));
    float c110 = __ldg(&data(ix+1, iy+1, iz));
    float c111 = __ldg(&data(ix+1, iy+1, iz+1));

    // Trilinear interpolation
    float c00 = c000 * (1.0f - fz_) + c001 * fz_;
    float c01 = c010 * (1.0f - fz_) + c011 * fz_;
    float c10 = c100 * (1.0f - fz_) + c101 * fz_;
    float c11 = c110 * (1.0f - fz_) + c111 * fz_;

    float c0 = c00 * (1.0f - fy_) + c01 * fy_;
    float c1 = c10 * (1.0f - fy_) + c11 * fy_;

    float e = c0 * (1.0f - fx_) + c1 * fx_;

    // Gradients (only used if in-bounds for that dimension)
    float gx = (c1 - c0) * grid.m_factor.x;
    float gy = ((c01 - c00) * (1.0f - fx_) + (c11 - c10) * fx_) * grid.m_factor.y;
    float gz = ((c001 - c000) * (1.0f - fy_) * (1.0f - fx_) +
                (c011 - c010) * fy_ * (1.0f - fx_) +
                (c101 - c100) * (1.0f - fy_) * fx_ +
                (c111 - c110) * fy_ * fx_) * grid.m_factor.z;

    // Apply curl/forcecap
    if (v > 0 && e > 0) {
        float tmp = e / v;
        e = v * tmp / (1.0f + tmp);
        float factor = 1.0f / ((1.0f + tmp) * (1.0f + tmp));
        gx *= factor;
        gy *= factor;
        gz *= factor;
    }

    // Out-of-bounds penalty (convert from grid units to Angstroms using m_factor_inv)
    float penalty = slope * (miss_x * grid.m_factor_inv.x +
                             miss_y * grid.m_factor_inv.y +
                             miss_z * grid.m_factor_inv.z);
    e += penalty;

    // Out-of-bounds gradient: zero the interpolated gradient, use only penalty gradient
    // (matches bfgs_parallel.cu behavior - returns gradient, NOT force)
    *fx = (region_x == 0 ? gx : 0) + slope * region_x;
    *fy = (region_y == 0 ? gy : 0) + slope * region_y;
    *fz = (region_z == 0 ? gz : 0) + slope * region_z;
    *energy = e;
}

// ============================================================================
// Set Conf (Internal Coords -> Cartesian)
// ============================================================================

template<typename Config>
__device__ void set_conf_warp_coop(
    const ScoringContext& ctx,
    WarpCoopState<Config>& state,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    using DA = DistributedArray<Config>;
#ifdef WARP_COOP_DEBUG
    bool debug_thread = (threadIdx.x == 0 && blockIdx.x == 0);
    (void)debug_thread;  // Suppress unused warning
#endif

    WARP_DEBUG_PRINT("set_conf: num_layers=%d, num_nodes=%d\n", ctx.num_layers, ctx.num_nodes);

    // === Step 1: Root node transform ===
    float root_pos[3], root_quat[4], root_mat[9];

    // Gather root position and quaternion from distributed storage
    root_pos[0] = DA::get(state.x, 0, shfl);
    root_pos[1] = DA::get(state.x, 1, shfl);
    root_pos[2] = DA::get(state.x, 2, shfl);
    root_quat[0] = DA::get(state.x, 3, shfl);
    root_quat[1] = DA::get(state.x, 4, shfl);
    root_quat[2] = DA::get(state.x, 5, shfl);
    root_quat[3] = DA::get(state.x, 6, shfl);

    WARP_DEBUG_PRINT("set_conf: root_pos=(%.2f,%.2f,%.2f)\n", root_pos[0], root_pos[1], root_pos[2]);

    // Normalize quaternion and convert to matrix
    normalize_quat_warp(root_quat);
    quat_to_matrix_warp(root_quat, root_mat);

    WARP_DEBUG_PRINT("set_conf: root transform done\n");

    // Store root transform (distributed)
    for (int i = 0; i < 3; i++) {
        DA::set(state.node_origins, i, root_pos[i], idx);
    }
    for (int i = 0; i < 9; i++) {
        DA::set(state.node_orientations, i, root_mat[i], idx);
    }

#ifdef USE_INLINE_DEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0) printf("set_conf: CHECKPOINT 1 - root done\n");
#endif

    // === Step 2: Torsion nodes (sequential by layer) ===
    // IMPORTANT: All threads must participate in DA::get shuffles, so no early continues!
    for (int layer = 1; layer < (int)ctx.num_layers; layer++) {
        for (int nid = 1; nid < (int)ctx.num_nodes; nid++) {
            // Check if this node is in current layer (all threads evaluate this uniformly)
            bool process_node = ((int)ctx.tree_nodes[nid].layer == layer);
            int parent = ctx.tree_nodes[nid].parent;
            if (parent < 0) process_node = false;

            // ALL threads participate in shuffles (required for correctness)
            // Use parent=0 as dummy if not processing (safe since node 0 exists)
            int safe_parent = process_node ? parent : 0;

            float parent_origin[3], parent_mat[9];
            for (int i = 0; i < 3; i++) {
                parent_origin[i] = DA::get(state.node_origins, safe_parent * 3 + i, shfl);
            }
            for (int i = 0; i < 9; i++) {
                parent_mat[i] = DA::get(state.node_orientations, safe_parent * 9 + i, shfl);
            }
            // Torsion index: nid + 6 * nlig_roots (general formula, not hardcoded for nlig_roots=1)
            float torsion = DA::get(state.x, nid + 6 * ctx.nlig_roots, shfl);

            // Only process if this node is in current layer
            if (process_node) {
                float rel_axis[3] = {
                    ctx.tree_nodes[nid].relative_axis.x,
                    ctx.tree_nodes[nid].relative_axis.y,
                    ctx.tree_nodes[nid].relative_axis.z
                };
                float rel_origin[3] = {
                    ctx.tree_nodes[nid].relative_origin.x,
                    ctx.tree_nodes[nid].relative_origin.y,
                    ctx.tree_nodes[nid].relative_origin.z
                };

                // Transform axis to world frame
                float axis[3];
                mat_vec_mult_warp(parent_mat, rel_axis, axis);

                // Normalize axis
                float axis_norm = sqrtf(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
                if (axis_norm > 1e-10f) {
                    axis[0] /= axis_norm;
                    axis[1] /= axis_norm;
                    axis[2] /= axis_norm;
                }

                // Store axis for gradient computation
                for (int i = 0; i < 3; i++) {
                    DA::set(state.node_axes, nid * 3 + i, axis[i], idx);
                }

                // Transform relative origin to world frame
                float local_origin[3];
                mat_vec_mult_warp(parent_mat, rel_origin, local_origin);

                // Node origin = parent_origin + transformed relative origin
                float origin[3];
                origin[0] = parent_origin[0] + local_origin[0];
                origin[1] = parent_origin[1] + local_origin[1];
                origin[2] = parent_origin[2] + local_origin[2];

                // Compute rotation matrix for this torsion using Rodrigues
                float c = cosf(torsion);
                float s = sinf(torsion);
                float t = 1.0f - c;

                // Rodrigues rotation matrix
                float rot[9];
                rot[0] = c + axis[0]*axis[0]*t;
                rot[1] = axis[1]*axis[0]*t + axis[2]*s;
                rot[2] = axis[2]*axis[0]*t - axis[1]*s;
                rot[3] = axis[0]*axis[1]*t - axis[2]*s;
                rot[4] = c + axis[1]*axis[1]*t;
                rot[5] = axis[2]*axis[1]*t + axis[0]*s;
                rot[6] = axis[0]*axis[2]*t + axis[1]*s;
                rot[7] = axis[1]*axis[2]*t - axis[0]*s;
                rot[8] = c + axis[2]*axis[2]*t;

                // Node orientation = rot * parent_mat
                float mat[9];
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        mat[i + j*3] = 0;
                        for (int k = 0; k < 3; k++) {
                            mat[i + j*3] += rot[i + k*3] * parent_mat[k + j*3];
                        }
                    }
                }

                // Store node transform
                for (int i = 0; i < 3; i++) {
                    DA::set(state.node_origins, nid * 3 + i, origin[i], idx);
                }
                for (int i = 0; i < 9; i++) {
                    DA::set(state.node_orientations, nid * 9 + i, mat[i], idx);
                }
            }
        }
    }

#ifdef USE_INLINE_DEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0) printf("set_conf: CHECKPOINT 2 - torsion loop done\n");
#endif

    // === Step 3: Gather ALL node transforms locally (lockstep) ===
    // This is needed because the atom loop is divergent (different threads process different atoms)
    // We must gather in lockstep before the divergent loop
    float all_origins[Config::MAX_NODES * 3];
    float all_orientations[Config::MAX_NODES * 9];

    for (int nid = 0; nid < (int)ctx.num_nodes; nid++) {
        for (int i = 0; i < 3; i++) {
            all_origins[nid * 3 + i] = DA::get(state.node_origins, nid * 3 + i, shfl);
        }
        for (int i = 0; i < 9; i++) {
            all_orientations[nid * 9 + i] = DA::get(state.node_orientations, nid * 9 + i, shfl);
        }
    }

#ifdef USE_INLINE_DEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0) printf("set_conf: CHECKPOINT 3 - gather done\n");
#endif

    // === Step 4: Atom coordinates (PARALLEL over atoms - no shuffles!) ===
    AtomPartition<Config> atoms(idx.local_lane, ctx.num_atoms);

    WARP_DEBUG_PRINT("set_conf: atoms.start=%d, atoms.end=%d\n", atoms.start, atoms.end);

    for (int a = atoms.start; a < atoms.end; a++) {
        int local_slot = a - atoms.start;
        int owner = ctx.atom_owners[a];

        // Use pre-gathered node transform (no shuffle needed!)
        float origin[3], mat[9];
        for (int i = 0; i < 3; i++) {
            origin[i] = all_origins[owner * 3 + i];
        }
        for (int i = 0; i < 9; i++) {
            mat[i] = all_orientations[owner * 9 + i];
        }

        // Get local coords from atom data
        float local[3] = {
            ctx.atom_local_data[a].coords.x,
            ctx.atom_local_data[a].coords.y,
            ctx.atom_local_data[a].coords.z
        };

        // Transform: world = origin + mat * local
        float rotated[3];
        mat_vec_mult_warp(mat, local, rotated);

        state.my_coords[local_slot * 3 + 0] = origin[0] + rotated[0];
        state.my_coords[local_slot * 3 + 1] = origin[1] + rotated[1];
        state.my_coords[local_slot * 3 + 2] = origin[2] + rotated[2];
    }

    WARP_DEBUG_PRINT("set_conf: DONE\n");

#ifdef WARP_COOP_DEBUG
    // Debug: print atom 0 coords (only from thread that owns it)
    AtomPartition<Config> dbg_atoms(idx.local_lane, ctx.num_atoms);
    if (dbg_atoms.start == 0 && threadIdx.x < 8) {  // Thread 0 of first optimizer owns atom 0
        printf("set_conf VERIFY: atom[0] coords=(%.4f,%.4f,%.4f) local=(%.4f,%.4f,%.4f) owner=%d\n",
               state.my_coords[0], state.my_coords[1], state.my_coords[2],
               ctx.atom_local_data[0].coords.x, ctx.atom_local_data[0].coords.y, ctx.atom_local_data[0].coords.z,
               ctx.atom_owners[0]);
        printf("set_conf VERIFY: root origin=(%.4f,%.4f,%.4f)\n",
               all_origins[0], all_origins[1], all_origins[2]);
    }
#endif
}

// ============================================================================
// Energy Evaluation
// ============================================================================

// Forward declaration for intramolecular energy
template<typename Config>
__device__ float eval_intramolecular_warp_coop(
    const ScoringContext& ctx,
    WarpCoopState<Config>& state,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
);

template<typename Config>
__device__ float eval_energy_warp_coop(
    const ScoringContext& ctx,
    WarpCoopState<Config>& state,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    AtomPartition<Config> atoms(idx.local_lane, ctx.num_atoms);

    float my_energy = 0.0f;

    for (int a = atoms.start; a < atoms.end; a++) {
        int local_slot = a - atoms.start;

        float x = state.my_coords[local_slot * 3 + 0];
        float y = state.my_coords[local_slot * 3 + 1];
        float z = state.my_coords[local_slot * 3 + 2];

        unsigned atype = ctx.atom_types[a];
        // Skip hydrogens (type <= 1) and out-of-bounds types (matches bfgs_parallel.cu)
        if (atype <= 1 || atype >= ctx.ngrids) continue;

        const grid_gpu& grid = ctx.grids[atype];

        float e, fx, fy, fz;
        trilinear_interp_warp(
            grid,
            grid.data,
            x, y, z,
            ctx.slope,
            ctx.forcecap,
            &e, &fx, &fy, &fz
        );

        my_energy += e;
        state.my_forces[local_slot * 3 + 0] = fx;
        state.my_forces[local_slot * 3 + 1] = fy;
        state.my_forces[local_slot * 3 + 2] = fz;

        // Handle charge-dependent grid if present (matches bfgs_parallel.cu)
        if (ctx.atom_params_data != nullptr && grid.chargedata.dim0() > 0) {
            float charge = ctx.atom_params_data[a].charge;
            if (charge != 0) {
                float ce, cfx, cfy, cfz;
                trilinear_interp_warp(
                    grid,
                    grid.chargedata,
                    x, y, z,
                    ctx.slope,
                    ctx.forcecap,
                    &ce, &cfx, &cfy, &cfz
                );
                my_energy += charge * ce;
                state.my_forces[local_slot * 3 + 0] += charge * cfx;
                state.my_forces[local_slot * 3 + 1] += charge * cfy;
                state.my_forces[local_slot * 3 + 2] += charge * cfz;
            }
        }
    }

    // Reduce intermolecular energy across threads
    float inter_energy = shfl.reduce_sum(my_energy);

    // Add intramolecular energy (also updates forces)
    float intra_energy = eval_intramolecular_warp_coop<Config>(ctx, state, idx, shfl);

#ifdef WARP_COOP_DEBUG
    WARP_DEBUG_PRINT("eval_energy: inter=%f, intra=%f, total=%f\n",
                     inter_energy, intra_energy, inter_energy + intra_energy);
#endif

    return inter_energy + intra_energy;
}

// ============================================================================
// Intramolecular Energy (pairs within ligand)
// ============================================================================

template<typename Config>
__device__ float eval_intramolecular_warp_coop(
    const ScoringContext& ctx,
    WarpCoopState<Config>& state,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    // Early exit if no pairs
    if (ctx.pairs == nullptr || ctx.splineInfo == nullptr || ctx.num_pairs == 0) {
        return 0.0f;
    }

    // First, gather ALL coordinates so every thread can access any atom
    // This is O(num_atoms) communication but enables parallel pair evaluation
    float all_coords[Config::MAX_ATOMS * 3];
    AtomPartition<Config> atoms(idx.local_lane, ctx.num_atoms);

    // Each thread broadcasts its atoms' coordinates
    for (int a = 0; a < (int)ctx.num_atoms; a++) {
        // Determine which thread owns this atom
        int base = ctx.num_atoms / Config::THREADS_PER_OPT;
        int remainder = ctx.num_atoms % Config::THREADS_PER_OPT;
        int owner_lane;
        if (a < (base + 1) * remainder) {
            owner_lane = a / (base + 1);
        } else {
            owner_lane = remainder + (a - (base + 1) * remainder) / base;
        }

        // Get local slot for this atom on the owner thread
        int owner_start = owner_lane * base + ((owner_lane < remainder) ? owner_lane : remainder);
        int local_slot = a - owner_start;

        // Broadcast coordinates from owner
        for (int c = 0; c < 3; c++) {
            float val = state.my_coords[local_slot * 3 + c];
            all_coords[a * 3 + c] = shfl.shfl(val, owner_lane);
        }
    }

    // Curl parameter from forcecap
    const float v = ctx.forcecap;
    float my_energy = 0.0f;

    // Accumulate forces for all atoms (will scatter later)
    float all_forces_delta[Config::MAX_ATOMS * 3];
    for (int i = 0; i < (int)ctx.num_atoms * 3; i++) {
        all_forces_delta[i] = 0.0f;
    }

    // Distribute pairs across threads
    for (unsigned p = idx.local_lane; p < ctx.num_pairs; p += Config::THREADS_PER_OPT) {
        const interacting_pair& ip = ctx.pairs[p];

        // Get atom positions
        float r_x = all_coords[ip.b * 3 + 0] - all_coords[ip.a * 3 + 0];
        float r_y = all_coords[ip.b * 3 + 1] - all_coords[ip.a * 3 + 1];
        float r_z = all_coords[ip.b * 3 + 2] - all_coords[ip.a * 3 + 2];

        float r2 = r_x*r_x + r_y*r_y + r_z*r_z;

        if (r2 >= ctx.cutoff_sq) continue;

        // Get atom types and charges
        unsigned t1 = ip.t1;
        unsigned t2 = ip.t2;
        float charge_a = (ctx.atom_params_data != nullptr) ? ctx.atom_params_data[ip.a].charge : 0;
        float charge_b = (ctx.atom_params_data != nullptr) ? ctx.atom_params_data[ip.b].charge : 0;

        // Evaluate pair interaction
        float dor;
        float energy = eval_pair_deriv_gpu(ctx.splineInfo, t1, charge_a, t2, charge_b, r2, dor);

        // Compute derivative vector
        float deriv_x = r_x * dor;
        float deriv_y = r_y * dor;
        float deriv_z = r_z * dor;

        // Apply soft curl
        if (energy > 0) {
            float tmp = v / (v + energy);
            energy *= tmp;
            float tmp_sq = tmp * tmp;
            deriv_x *= tmp_sq;
            deriv_y *= tmp_sq;
            deriv_z *= tmp_sq;
        }

        my_energy += energy;

        // Accumulate forces (deriv points from a to b)
        all_forces_delta[ip.b * 3 + 0] += deriv_x;
        all_forces_delta[ip.b * 3 + 1] += deriv_y;
        all_forces_delta[ip.b * 3 + 2] += deriv_z;
        all_forces_delta[ip.a * 3 + 0] -= deriv_x;
        all_forces_delta[ip.a * 3 + 1] -= deriv_y;
        all_forces_delta[ip.a * 3 + 2] -= deriv_z;
    }

    // Reduce forces across threads and add to my_forces
    // IMPORTANT: Must iterate over ALL atoms in lockstep for reduce_sum to work!
    // Each thread contributes its force_delta, but only the owner writes the result
    for (int a = 0; a < (int)ctx.num_atoms; a++) {
        for (int c = 0; c < 3; c++) {
            float force_delta = shfl.reduce_sum(all_forces_delta[a * 3 + c]);
            // Only the owner thread writes the result
            if (a >= atoms.start && a < atoms.end) {
                int local_slot = a - atoms.start;
                state.my_forces[local_slot * 3 + c] += force_delta;
            }
        }
    }

    // Reduce energy across threads
    return shfl.reduce_sum(my_energy);
}

// ============================================================================
// Gradient Computation
// ============================================================================

template<typename Config>
__device__ void compute_gradient_warp_coop(
    const ScoringContext& ctx,
    WarpCoopState<Config>& state,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    using DA = DistributedArray<Config>;
    AtomPartition<Config> atoms(idx.local_lane, ctx.num_atoms);

    // === Gather ALL node origins locally (lockstep) before divergent atom loop ===
    float all_origins[Config::MAX_NODES * 3];
    for (int nid = 0; nid < (int)ctx.num_nodes; nid++) {
        for (int i = 0; i < 3; i++) {
            all_origins[nid * 3 + i] = DA::get(state.node_origins, nid * 3 + i, shfl);
        }
    }

    // === Step 1: Accumulate forces to nodes (each thread handles its atoms) ===
    // Use local accumulators, then reduce
    float local_node_f[Config::MAX_NODES * 3];
    float local_node_t[Config::MAX_NODES * 3];

    // Initialize to zero
    for (int i = 0; i < (int)ctx.num_nodes * 3; i++) {
        local_node_f[i] = 0.0f;
        local_node_t[i] = 0.0f;
    }

    for (int a = atoms.start; a < atoms.end; a++) {
        int local_slot = a - atoms.start;
        int owner = ctx.atom_owners[a];

        float fx = state.my_forces[local_slot * 3 + 0];
        float fy = state.my_forces[local_slot * 3 + 1];
        float fz = state.my_forces[local_slot * 3 + 2];

        // Accumulate force to owner node
        local_node_f[owner * 3 + 0] += fx;
        local_node_f[owner * 3 + 1] += fy;
        local_node_f[owner * 3 + 2] += fz;

        // Get atom position and node origin for torque (use pre-gathered origins!)
        float ax = state.my_coords[local_slot * 3 + 0];
        float ay = state.my_coords[local_slot * 3 + 1];
        float az = state.my_coords[local_slot * 3 + 2];

        float ox = all_origins[owner * 3 + 0];
        float oy = all_origins[owner * 3 + 1];
        float oz = all_origins[owner * 3 + 2];

        float rx = ax - ox;
        float ry = ay - oy;
        float rz = az - oz;

        // Torque = r x f
        local_node_t[owner * 3 + 0] += ry * fz - rz * fy;
        local_node_t[owner * 3 + 1] += rz * fx - rx * fz;
        local_node_t[owner * 3 + 2] += rx * fy - ry * fx;
    }

    // === Step 2: Reduce node forces/torques across threads ===
    for (int nid = 0; nid < (int)ctx.num_nodes; nid++) {
        for (int d = 0; d < 3; d++) {
            float f = shfl.reduce_sum(local_node_f[nid * 3 + d]);
            float t = shfl.reduce_sum(local_node_t[nid * 3 + d]);
            DA::set(state.node_forces, nid * 3 + d, f, idx);
            DA::set(state.node_torques, nid * 3 + d, t, idx);
        }
    }

    // === Step 3: Tree reduction (child -> parent) ===
    // Physics: When force F acts at point P, torque about point O is: τ_O = τ_P + (P - O) × F
    for (int layer = ctx.num_layers - 1; layer > 0; layer--) {
        for (int nid = ctx.num_nodes - 1; nid >= 1; nid--) {
            if ((int)ctx.tree_nodes[nid].layer != layer) continue;
            int parent = ctx.tree_nodes[nid].parent;
            if (parent < 0) continue;

            // Gather child force/torque (all 3 components at once for cross product)
            float child_fx = DA::get(state.node_forces, nid * 3 + 0, shfl);
            float child_fy = DA::get(state.node_forces, nid * 3 + 1, shfl);
            float child_fz = DA::get(state.node_forces, nid * 3 + 2, shfl);

            float child_tx = DA::get(state.node_torques, nid * 3 + 0, shfl);
            float child_ty = DA::get(state.node_torques, nid * 3 + 1, shfl);
            float child_tz = DA::get(state.node_torques, nid * 3 + 2, shfl);

            // r = child_origin - parent_origin (using pre-gathered origins)
            float rx = all_origins[nid * 3 + 0] - all_origins[parent * 3 + 0];
            float ry = all_origins[nid * 3 + 1] - all_origins[parent * 3 + 1];
            float rz = all_origins[nid * 3 + 2] - all_origins[parent * 3 + 2];

            // Get parent values
            float parent_fx = DA::get(state.node_forces, parent * 3 + 0, shfl);
            float parent_fy = DA::get(state.node_forces, parent * 3 + 1, shfl);
            float parent_fz = DA::get(state.node_forces, parent * 3 + 2, shfl);

            float parent_tx = DA::get(state.node_torques, parent * 3 + 0, shfl);
            float parent_ty = DA::get(state.node_torques, parent * 3 + 1, shfl);
            float parent_tz = DA::get(state.node_torques, parent * 3 + 2, shfl);

            // parent_force += child_force
            DA::set(state.node_forces, parent * 3 + 0, parent_fx + child_fx, idx);
            DA::set(state.node_forces, parent * 3 + 1, parent_fy + child_fy, idx);
            DA::set(state.node_forces, parent * 3 + 2, parent_fz + child_fz, idx);

            // parent_torque += cross(r, child_force) + child_torque
            // cross(r, f) = (ry*fz - rz*fy, rz*fx - rx*fz, rx*fy - ry*fx)
            DA::set(state.node_torques, parent * 3 + 0, parent_tx + (ry * child_fz - rz * child_fy) + child_tx, idx);
            DA::set(state.node_torques, parent * 3 + 1, parent_ty + (rz * child_fx - rx * child_fz) + child_ty, idx);
            DA::set(state.node_torques, parent * 3 + 2, parent_tz + (rx * child_fy - ry * child_fx) + child_tz, idx);
        }
    }

    // === Step 4: Pre-gather node data (lockstep to avoid divergent shuffles) ===
    float all_node_forces[Config::MAX_NODES * 3];
    float all_node_torques[Config::MAX_NODES * 3];
    float all_node_axes[Config::MAX_NODES * 3];
    for (int nid = 0; nid < (int)ctx.num_nodes; nid++) {
        for (int d = 0; d < 3; d++) {
            all_node_forces[nid * 3 + d] = DA::get(state.node_forces, nid * 3 + d, shfl);
            all_node_torques[nid * 3 + d] = DA::get(state.node_torques, nid * 3 + d, shfl);
            all_node_axes[nid * 3 + d] = DA::get(state.node_axes, nid * 3 + d, shfl);
        }
    }

    // === Step 5: Compute gradient (parallel over DOFs, no shuffles needed) ===
    // Layout: [pos1(3), rot1(3), pos2(3), rot2(3), ..., torsions...]
    // For nlig_roots roots: 6*nlig_roots rigid body DOFs, then n_change - 6*nlig_roots torsion DOFs
    int rigid_dofs = 6 * ctx.nlig_roots;

    for (int i = idx.local_lane; i < ctx.n_change; i += Config::THREADS_PER_OPT) {
        float g_val;
        if (i < rigid_dofs) {
            // Rigid body DOF: determine which root and whether pos or rot
            int root = i / 6;
            int local_dof = i % 6;
            if (local_dof < 3) {
                // Translation: gradient = root force
                g_val = all_node_forces[root * 3 + local_dof];
            } else {
                // Rotation: gradient = root torque
                g_val = all_node_torques[root * 3 + (local_dof - 3)];
            }
        } else {
            // Torsion: gradient = dot(torque, axis)
            // Matches reference: gradient[nid + 5*nlig_roots], so nid = i - 5*nlig_roots
            int nid = i - 5 * ctx.nlig_roots;
            if (nid >= 0 && nid < (int)ctx.num_nodes) {
                float t0 = all_node_torques[nid * 3 + 0];
                float t1 = all_node_torques[nid * 3 + 1];
                float t2 = all_node_torques[nid * 3 + 2];
                float a0 = all_node_axes[nid * 3 + 0];
                float a1 = all_node_axes[nid * 3 + 1];
                float a2 = all_node_axes[nid * 3 + 2];
                g_val = t0 * a0 + t1 * a1 + t2 * a2;
            } else {
                g_val = 0.0f;
            }
        }
        DA::set(state.g, i, g_val, idx);
    }
}

// ============================================================================
// Hessian-Vector Multiply
// ============================================================================

template<typename Config>
__device__ void hessian_vector_multiply(
    WarpCoopState<Config>& state,
    int n_change,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    using DA = DistributedArray<Config>;

    // Pre-gather gradient (lockstep, same indices for all threads)
    float all_g[Config::N_CHANGE];
    for (int j = 0; j < n_change; j++) {
        all_g[j] = DA::get(state.g, j, shfl);
    }

    // Pre-gather Hessian (lockstep)
    int n_hessian = n_change * (n_change + 1) / 2;
    float all_h[Config::N_HESSIAN];
    for (int k = 0; k < n_hessian; k++) {
        all_h[k] = DA::get(state.h, k, shfl);
    }

    // p = -H * g (parallel over rows, no shuffles needed)
    for (int i = idx.local_lane; i < n_change; i += Config::THREADS_PER_OPT) {
        float sum = 0.0f;
        for (int j = 0; j < n_change; j++) {
            // Triangular index
            int h_idx = (i <= j) ? (i + j * (j + 1) / 2) : (j + i * (i + 1) / 2);
            sum += all_h[h_idx] * all_g[j];
        }
        DA::set(state.p, i, -sum, idx);
    }
}

// ============================================================================
// BFGS Hessian Update
// ============================================================================

template<typename Config>
__device__ void bfgs_hessian_update(
    WarpCoopState<Config>& state,
    float alpha,
    int n_change,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    using DA = DistributedArray<Config>;

    // Pre-gather g_new, g, and p (lockstep)
    float all_g_new[Config::N_CHANGE];
    float all_g[Config::N_CHANGE];
    float all_p[Config::N_CHANGE];
    for (int i = 0; i < n_change; i++) {
        all_g_new[i] = DA::get(state.g_new, i, shfl);
        all_g[i] = DA::get(state.g, i, shfl);
        all_p[i] = DA::get(state.p, i, shfl);
    }

    // y = g_new - g
    float all_y[Config::N_CHANGE];
    for (int i = 0; i < n_change; i++) {
        all_y[i] = all_g_new[i] - all_g[i];
    }
    // Store y to state (parallel, no shuffles)
    for (int i = idx.local_lane; i < n_change; i += Config::THREADS_PER_OPT) {
        DA::set(state.y, i, all_y[i], idx);
    }

    // Compute yp = y · p
    float my_yp = 0.0f;
    for (int i = idx.local_lane; i < n_change; i += Config::THREADS_PER_OPT) {
        my_yp += all_y[i] * all_p[i];
    }
    float yp = shfl.reduce_sum(my_yp);

    if (alpha * yp < 1e-10f) return;  // Skip update if alpha*yp too small

    int n_hessian = n_change * (n_change + 1) / 2;

    // Pre-gather Hessian (lockstep) - needed for H*y computation
    float all_h[Config::N_HESSIAN];
    for (int k = 0; k < n_hessian; k++) {
        all_h[k] = DA::get(state.h, k, shfl);
    }

    // Step 1: Compute minus_hy = -H * y (matrix-vector product)
    // Each thread computes a subset of the minus_hy vector
    float minus_hy[Config::N_CHANGE];
    for (int i = 0; i < n_change; i++) {
        minus_hy[i] = 0.0f;
        for (int j = 0; j < n_change; j++) {
            // Triangular indexing: h[i,j] where i <= j
            int h_idx = (i <= j) ? (i + j * (j + 1) / 2) : (j + i * (i + 1) / 2);
            minus_hy[i] -= all_h[h_idx] * all_y[j];
        }
    }

    // Step 2: Compute yhy = y · H · y = -y · minus_hy
    float my_yhy = 0.0f;
    for (int i = idx.local_lane; i < n_change; i += Config::THREADS_PER_OPT) {
        my_yhy -= all_y[i] * minus_hy[i];
    }
    float yhy = shfl.reduce_sum(my_yhy);

    // Step 3: Compute update coefficients
    float r = 1.0f / (alpha * yp);
    float coef = alpha * alpha * (r * r * yhy + r);

    // Step 4: Update Hessian with correct BFGS formula
    // H_new = H + alpha * r * (minus_hy[i]*p[j] + minus_hy[j]*p[i]) + coef * p[i] * p[j]
    for (int k = idx.local_lane; k < n_hessian; k += Config::THREADS_PER_OPT) {
        // Convert triangular index to (i, j) with i <= j
        int j = (int)(sqrtf(2.0f * k + 0.25f) - 0.5f);
        int i = k - j * (j + 1) / 2;

        float h_ij = all_h[k];
        float p_i = all_p[i];
        float p_j = all_p[j];

        // Correct BFGS update formula
        float h_new = h_ij
            + alpha * r * (minus_hy[i] * p_j + minus_hy[j] * p_i)
            + coef * p_i * p_j;

        DA::set(state.h, k, h_new, idx);
    }
}

// ============================================================================
// Line Search
// ============================================================================

// Accurate line search with quadratic/cubic interpolation (matches reference)
template<typename Config>
__device__ float line_search_warp_coop(
    const ScoringContext& ctx,
    WarpCoopState<Config>& state,
    float f0,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    using DA = DistributedArray<Config>;
    const float ALF = 1.0e-4f;
    const float epsilon_fl = 1e-7f;

    // Pre-gather g, p, and x (all threads in lockstep)
    float all_g[Config::N_CHANGE];
    float all_p[Config::N_CHANGE];
    float all_x[Config::N_CONF];
    for (int i = 0; i < ctx.n_change; i++) {
        all_g[i] = DA::get(state.g, i, shfl);
        all_p[i] = DA::get(state.p, i, shfl);
    }
    for (int i = 0; i < ctx.n_conf; i++) {
        all_x[i] = DA::get(state.x, i, shfl);
    }

    // Compute slope = g . p
    float my_slope = 0.0f;
    for (int i = idx.local_lane; i < ctx.n_change; i += Config::THREADS_PER_OPT) {
        my_slope += all_g[i] * all_p[i];
    }
    float slope = shfl.reduce_sum(my_slope);

    // Not a descent direction
    if (slope >= 0.0f) {
        return 0.0f;
    }

    // Compute minimum step size (alamin)
    float my_test = 0.0f;
    for (int i = idx.local_lane; i < ctx.n_change; i += Config::THREADS_PER_OPT) {
        float temp = fabsf(all_p[i]);
        if (temp > my_test) my_test = temp;
    }
    float test = shfl.reduce_max(my_test);
    float alamin = (test > 0) ? (epsilon_fl / test) : epsilon_fl;

    float alpha = 1.0f;
    float alpha2 = 0.0f, f2 = 0.0f;
    bool first_backtrack = true;

    const int MAX_LINE_SEARCH_ITERS = 50;
    for (int ls_iter = 0; ls_iter < MAX_LINE_SEARCH_ITERS; ls_iter++) {
        // Check for too small step
        if (alpha < alamin || !isfinite(alpha)) {
            return 0.0f;
        }

        // x_new = x + alpha * p (with proper quaternion and torsion handling)
        increment_conf_warp<Config>(all_x, all_p, alpha, ctx.nlig_roots,
                                    ctx.n_conf, ctx.n_change, state, idx);

        // Temporarily use x_new as x for evaluation
        float temp_x[Config::CONF_PER_THREAD];
        for (int i = 0; i < Config::CONF_PER_THREAD; i++) {
            temp_x[i] = state.x[i];
            state.x[i] = state.x_new[i];
        }

        // Evaluate energy at new point
        set_conf_warp_coop<Config>(ctx, state, idx, shfl);
        float f_new = eval_energy_warp_coop<Config>(ctx, state, idx, shfl);

        // Restore x
        for (int i = 0; i < Config::CONF_PER_THREAD; i++) {
            state.x[i] = temp_x[i];
        }

        // Handle NaN energy
        if (!isfinite(f_new)) {
            return 0.0f;
        }

        // Check Armijo sufficient decrease condition
        if (f_new <= f0 + ALF * alpha * slope) {
            // Success - swap x and x_new, compute gradient
            for (int i = 0; i < Config::CONF_PER_THREAD; i++) {
                temp_x[i] = state.x[i];
                state.x[i] = state.x_new[i];
            }
            set_conf_warp_coop<Config>(ctx, state, idx, shfl);
            compute_gradient_warp_coop<Config>(ctx, state, idx, shfl);

            // Copy g to g_new
            for (int i = 0; i < ctx.n_change; i++) {
                all_g[i] = DA::get(state.g, i, shfl);
            }
            for (int i = idx.local_lane; i < ctx.n_change; i += Config::THREADS_PER_OPT) {
                DA::set(state.g_new, i, all_g[i], idx);
            }

            // Restore x (keep x_new as accepted)
            for (int i = 0; i < Config::CONF_PER_THREAD; i++) {
                state.x[i] = temp_x[i];
            }

            state.energy = f_new;
            return alpha;
        }

        // Backtrack using interpolation (matches reference)
        float tmplam;
        if (first_backtrack) {
            // First backtrack: quadratic interpolation
            float denom = 2.0f * (f_new - f0 - slope);
            if (fabsf(denom) > epsilon_fl) {
                tmplam = -slope / denom;
            } else {
                tmplam = 0.5f * alpha;
            }
            first_backtrack = false;
        } else {
            // Subsequent backtracks: cubic interpolation
            float rhs1 = f_new - f0 - alpha * slope;
            float rhs2 = f2 - f0 - alpha2 * slope;
            float alpha_diff = alpha - alpha2;

            if (fabsf(alpha_diff) < epsilon_fl) {
                tmplam = 0.5f * alpha;
            } else {
                float a = (rhs1 / (alpha * alpha) - rhs2 / (alpha2 * alpha2)) / alpha_diff;
                float b = (-alpha2 * rhs1 / (alpha * alpha) + alpha * rhs2 / (alpha2 * alpha2)) / alpha_diff;

                if (fabsf(a) < epsilon_fl) {
                    // Linear case
                    tmplam = (fabsf(b) > epsilon_fl) ? (-slope / (2.0f * b)) : (0.5f * alpha);
                } else {
                    float disc = b * b - 3.0f * a * slope;
                    if (disc < 0) {
                        tmplam = 0.5f * alpha;
                    } else if (b <= 0) {
                        tmplam = (-b + sqrtf(disc)) / (3.0f * a);
                    } else {
                        tmplam = -slope / (b + sqrtf(disc));
                    }
                }
                // Clamp to <= 0.5 * alpha
                if (tmplam > 0.5f * alpha) {
                    tmplam = 0.5f * alpha;
                }
            }
        }

        // Save for next cubic interpolation
        alpha2 = alpha;
        f2 = f_new;

        // Update alpha with minimum bound
        alpha = fmaxf(tmplam, 0.1f * alpha);
    }

    return 0.0f;
}

// ============================================================================
// Main Kernel
// ============================================================================

template<typename Config>
__global__ void __launch_bounds__(128, 8)
bfgs_warp_cooperative_kernel(
    const ScoringContext* __restrict__ contexts,
    const float* __restrict__ initial_confs,
    float* __restrict__ out_energies,
    float* __restrict__ out_confs,
    int max_iterations,
    int n_optimizers,
    int n_conf,
    int n_change
) {
    constexpr int WARPS_PER_BLOCK = 4;

    // Thread indexing
    ThreadIndex<Config> idx;
    ShuffleOps<Config> shfl(idx);

    int global_opt_id = idx.global_opt_id(WARPS_PER_BLOCK);
    if (global_opt_id >= n_optimizers) return;

    const ScoringContext& ctx = contexts[0];
    WarpCoopState<Config> state;

    using DA = DistributedArray<Config>;

    // Debug: only thread 0 of first optimizer prints
#ifdef WARP_COOP_DEBUG
    bool debug_thread = (threadIdx.x == 0 && blockIdx.x == 0);
    (void)debug_thread;  // Suppress unused warning
#endif
    WARP_DEBUG_PRINT("KERNEL: Starting opt %d, n_conf=%d, n_change=%d\n", global_opt_id, n_conf, n_change);

    // === Initialize conformation from input ===
    for (int i = idx.local_lane; i < n_conf; i += Config::THREADS_PER_OPT) {
        float val = initial_confs[global_opt_id * n_conf + i];
        DA::set(state.x, i, val, idx);
    }

    WARP_DEBUG_PRINT("KERNEL: Initialized conformation\n");

    // === Initialize Hessian to identity (all threads iterate in lockstep) ===
    int n_hessian = n_change * (n_change + 1) / 2;
    for (int k = 0; k < n_hessian; k++) {
        // Diagonal elements: k = i + i*(i+1)/2 => k = i*(i+3)/2
        int j = (int)(sqrtf(2.0f * k + 0.25f) - 0.5f);
        int i = k - j * (j + 1) / 2;
        float val = (i == j) ? 1.0f : 0.0f;
        DA::set(state.h, k, val, idx);  // Only owner writes
    }

    WARP_DEBUG_PRINT("KERNEL: Initialized Hessian\n");

    state.energy = 1e10f;
    state.best_energy = 1e10f;

    // === Initial energy and gradient ===
#ifdef USE_INLINE_DEBUG
    // DEBUG: Compute atom[0] using simple inline math (matching reference)
    // ALL threads gather conf (shuffles require all threads to participate)
    float dbg_pos[3], dbg_quat[4];
    dbg_pos[0] = DA::get(state.x, 0, shfl);
    dbg_pos[1] = DA::get(state.x, 1, shfl);
    dbg_pos[2] = DA::get(state.x, 2, shfl);
    dbg_quat[0] = DA::get(state.x, 3, shfl);
    dbg_quat[1] = DA::get(state.x, 4, shfl);
    dbg_quat[2] = DA::get(state.x, 5, shfl);
    dbg_quat[3] = DA::get(state.x, 6, shfl);

    // Only one thread prints
    if (global_opt_id == 0 && idx.local_lane == 0) {
        float qnorm = sqrtf(dbg_quat[0]*dbg_quat[0] + dbg_quat[1]*dbg_quat[1] + dbg_quat[2]*dbg_quat[2] + dbg_quat[3]*dbg_quat[3]);
        if (qnorm > 1e-10f) {
            dbg_quat[0] /= qnorm; dbg_quat[1] /= qnorm; dbg_quat[2] /= qnorm; dbg_quat[3] /= qnorm;
        }
        float a = dbg_quat[0], b = dbg_quat[1], c = dbg_quat[2], d = dbg_quat[3];
        float aa = a*a, bb = b*b, cc = c*c, dd = d*d;
        float ab = a*b, ac = a*c, ad = a*d, bc = b*c, bd = b*d, cd = c*d;
        float mat[9];
        mat[0] = aa + bb - cc - dd;  mat[3] = 2*(bc - ad);       mat[6] = 2*(bd + ac);
        mat[1] = 2*(bc + ad);        mat[4] = aa - bb + cc - dd; mat[7] = 2*(cd - ab);
        mat[2] = 2*(bd - ac);        mat[5] = 2*(cd + ab);       mat[8] = aa - bb - cc + dd;
        int owner = ctx.atom_owners[0];
        if (owner == 0) {
            float lx = ctx.atom_local_data[0].coords.x, ly = ctx.atom_local_data[0].coords.y, lz = ctx.atom_local_data[0].coords.z;
            float rx = mat[0]*lx + mat[3]*ly + mat[6]*lz;
            float ry = mat[1]*lx + mat[4]*ly + mat[7]*lz;
            float rz = mat[2]*lx + mat[5]*ly + mat[8]*lz;
            printf("DEBUG: EXPECTED atom[0]=(%.4f,%.4f,%.4f)\n", dbg_pos[0]+rx, dbg_pos[1]+ry, dbg_pos[2]+rz);
        }
    }
#endif

    set_conf_warp_coop<Config>(ctx, state, idx, shfl);

#ifdef USE_INLINE_DEBUG
    // Print warp-coop atom[0] coords for comparison
    if (global_opt_id == 0) {
        AtomPartition<Config> dbg_atoms(idx.local_lane, ctx.num_atoms);
        if (dbg_atoms.start == 0) {
            printf("DEBUG: ACTUAL   atom[0]=(%.4f,%.4f,%.4f)\n",
                   state.my_coords[0], state.my_coords[1], state.my_coords[2]);
        }
    }
    __syncwarp(idx.opt_mask);
#endif

    WARP_DEBUG_PRINT("KERNEL: Calling eval_energy_warp_coop\n");
    state.energy = eval_energy_warp_coop<Config>(ctx, state, idx, shfl);

    WARP_DEBUG_PRINT("KERNEL: Initial energy = %f\n", state.energy);
    WARP_DEBUG_PRINT("KERNEL: Calling compute_gradient_warp_coop\n");
    compute_gradient_warp_coop<Config>(ctx, state, idx, shfl);
    WARP_DEBUG_PRINT("KERNEL: Gradient computed\n");
    state.best_energy = state.energy;

    // Copy current conf to best (gather first to avoid divergent shuffle)
    float temp_conf[Config::N_CONF];
    for (int i = 0; i < n_conf; i++) {
        temp_conf[i] = DA::get(state.x, i, shfl);
    }
    for (int i = idx.local_lane; i < n_conf; i += Config::THREADS_PER_OPT) {
        DA::set(state.best_conf, i, temp_conf[i], idx);
    }

    WARP_DEBUG_PRINT("KERNEL: Starting BFGS iterations (max=%d)\n", max_iterations);
#ifdef WARP_COOP_DEBUG
    // Debug: print initial x values
    float debug_x[6];
    for (int i = 0; i < 6 && i < n_conf; i++) {
        debug_x[i] = DA::get(state.x, i, shfl);
    }
    WARP_DEBUG_PRINT("KERNEL: initial x[0..5]=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f)\n",
                     debug_x[0], debug_x[1], debug_x[2], debug_x[3], debug_x[4], debug_x[5]);
#endif

    // === BFGS iterations ===
    for (int iter = 0; iter < max_iterations; iter++) {
#ifdef WARP_COOP_DEBUG
        // Debug: compute gradient norm (pre-gather in lockstep to avoid divergent shuffles)
        float all_g_debug[Config::N_CHANGE];
        for (int i = 0; i < n_change; i++) {
            all_g_debug[i] = DA::get(state.g, i, shfl);
        }
        float my_g_norm = 0.0f;
        for (int i = idx.local_lane; i < n_change; i += Config::THREADS_PER_OPT) {
            my_g_norm += all_g_debug[i] * all_g_debug[i];
        }
        float g_norm = sqrtf(shfl.reduce_sum(my_g_norm));
        if (debug_thread && iter < 5) printf("KERNEL: iter %d, energy=%f, |g|=%f\n", iter, state.energy, g_norm);
#endif

        // Search direction: p = -H * g
        hessian_vector_multiply<Config>(state, n_change, idx, shfl);

#ifdef WARP_COOP_DEBUG
        // Debug: compute search direction norm (pre-gather in lockstep)
        float all_p_debug[Config::N_CHANGE];
        for (int i = 0; i < n_change; i++) {
            all_p_debug[i] = DA::get(state.p, i, shfl);
        }
        float my_p_norm = 0.0f;
        for (int i = idx.local_lane; i < n_change; i += Config::THREADS_PER_OPT) {
            my_p_norm += all_p_debug[i] * all_p_debug[i];
        }
        float p_norm = sqrtf(shfl.reduce_sum(my_p_norm));
        if (debug_thread && iter < 5) printf("KERNEL: iter %d, |p|=%f\n", iter, p_norm);
#endif

        // Line search
        float alpha = line_search_warp_coop<Config>(ctx, state, state.energy, idx, shfl);

#ifdef WARP_COOP_DEBUG
        if (debug_thread && iter < 5) printf("KERNEL: iter %d, alpha=%f, new_energy=%f\n", iter, alpha, state.energy);
#endif

        if (alpha == 0.0f) break;  // Line search failed

        // Shanno-Phua scaling on first iteration
        // Rescales the Hessian based on the initial step to improve convergence
        if (iter == 0) {
            // Compute y = g_new - g
            float all_g_new[Config::N_CHANGE];
            float all_g[Config::N_CHANGE];
            float all_p[Config::N_CHANGE];
            for (int i = 0; i < n_change; i++) {
                all_g_new[i] = DA::get(state.g_new, i, shfl);
                all_g[i] = DA::get(state.g, i, shfl);
                all_p[i] = DA::get(state.p, i, shfl);
            }

            // Compute yy = y · y and yp = y · p
            float my_yy = 0.0f, my_yp = 0.0f;
            for (int i = idx.local_lane; i < n_change; i += Config::THREADS_PER_OPT) {
                float yi = all_g_new[i] - all_g[i];
                my_yy += yi * yi;
                my_yp += yi * all_p[i];
            }
            float yy = shfl.reduce_sum(my_yy);
            float yp = shfl.reduce_sum(my_yp);

            if (yy > 1e-10f) {
                float scale = alpha * yp / yy;
                // Reset Hessian to scaled identity (all threads iterate all elements in lockstep)
                int n_hessian = n_change * (n_change + 1) / 2;
                for (int k = 0; k < n_hessian; k++) {
                    // Decode triangular index to get (i, j)
                    int j = (int)(sqrtf(2.0f * k + 0.25f) - 0.5f);
                    int i = k - j * (j + 1) / 2;
                    float val = (i == j) ? scale : 0.0f;
                    DA::set(state.h, k, val, idx);  // Only owner writes
                }
            }
        }

        // Hessian update
        bfgs_hessian_update<Config>(state, alpha, n_change, idx, shfl);

        // Accept step: x = x_new (pre-gather to avoid divergent shuffles)
        float all_x_new[Config::N_CONF];
        for (int i = 0; i < n_conf; i++) {
            all_x_new[i] = DA::get(state.x_new, i, shfl);
        }
        for (int i = idx.local_lane; i < n_conf; i += Config::THREADS_PER_OPT) {
            DA::set(state.x, i, all_x_new[i], idx);
        }

        // g = g_new (pre-gather to avoid divergent shuffles)
        float all_g_new2[Config::N_CHANGE];
        for (int i = 0; i < n_change; i++) {
            all_g_new2[i] = DA::get(state.g_new, i, shfl);
        }
        for (int i = idx.local_lane; i < n_change; i += Config::THREADS_PER_OPT) {
            DA::set(state.g, i, all_g_new2[i], idx);
        }

        // Track best
        if (state.energy < state.best_energy) {
            state.best_energy = state.energy;
            float all_x2[Config::N_CONF];
            for (int i = 0; i < n_conf; i++) {
                all_x2[i] = DA::get(state.x, i, shfl);
            }
            for (int i = idx.local_lane; i < n_conf; i += Config::THREADS_PER_OPT) {
                DA::set(state.best_conf, i, all_x2[i], idx);
            }
        }
    }

    // === Output (lane 0 writes) ===
    if (idx.local_lane == 0) {
        out_energies[global_opt_id] = state.best_energy;
    }

    // Pre-gather best_conf (lockstep to avoid divergent shuffles)
    float all_best_conf[Config::N_CONF];
    for (int i = 0; i < n_conf; i++) {
        all_best_conf[i] = DA::get(state.best_conf, i, shfl);
    }

    // All threads write their portion of best_conf
    for (int i = idx.local_lane; i < n_conf; i += Config::THREADS_PER_OPT) {
        out_confs[global_opt_id * n_conf + i] = all_best_conf[i];
    }
}

// ============================================================================
// Random Initialization Kernel
// ============================================================================

template<typename Config>
__global__ void init_random_confs_kernel(
    float* confs,
    int n_confs,
    int n_conf,
    int nlig_roots,  // Number of ligand roots (for multi-root support)
    gfloat3 box_min,
    gfloat3 box_max,
    unsigned int seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_confs) return;

    curandState rng;
    curand_init(seed, tid, 0, &rng);

    float* my_conf = confs + tid * n_conf;

    // Random position and quaternion for each ligand root
    for (int r = 0; r < nlig_roots; r++) {
        // Random position within box
        my_conf[r * 7 + 0] = box_min.x + curand_uniform(&rng) * (box_max.x - box_min.x);
        my_conf[r * 7 + 1] = box_min.y + curand_uniform(&rng) * (box_max.y - box_min.y);
        my_conf[r * 7 + 2] = box_min.z + curand_uniform(&rng) * (box_max.z - box_min.z);

        // Random quaternion (normalized)
        float u1 = curand_uniform(&rng);
        float u2 = curand_uniform(&rng) * 2.0f * 3.14159265f;
        float u3 = curand_uniform(&rng) * 2.0f * 3.14159265f;

        float sqrt1u1 = sqrtf(1.0f - u1);
        float sqrtu1 = sqrtf(u1);

        my_conf[r * 7 + 3] = sqrt1u1 * sinf(u2);  // w
        my_conf[r * 7 + 4] = sqrt1u1 * cosf(u2);  // x
        my_conf[r * 7 + 5] = sqrtu1 * sinf(u3);   // y
        my_conf[r * 7 + 6] = sqrtu1 * cosf(u3);   // z
    }

    // Random torsions in [-pi, pi] (start after rigid body values)
    int torsion_start = 7 * nlig_roots;
    for (int i = torsion_start; i < n_conf; i++) {
        my_conf[i] = (curand_uniform(&rng) * 2.0f - 1.0f) * 3.14159265f;
    }

#ifdef WARP_COOP_DEBUG
    if (tid == 0) {
        printf("INIT_RAND: tid=%d seed=%u box_min=(%.4f,%.4f,%.4f) box_max=(%.4f,%.4f,%.4f)\n",
               tid, seed, box_min.x, box_min.y, box_min.z, box_max.x, box_max.y, box_max.z);
        printf("INIT_RAND: conf[0..6]=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f)\n",
               my_conf[0], my_conf[1], my_conf[2], my_conf[3], my_conf[4], my_conf[5], my_conf[6]);
    }
#endif
}

// ============================================================================
// Host Interface
// ============================================================================

template<typename Config>
void run_warp_coop_bfgs_docking(
    const gpu_data& gdata,
    const GPUCacheInfo& cacheInfo,
    int n_poses,
    int max_iterations,
    const gfloat3& box_min,
    const gfloat3& box_max,
    unsigned int seed,
    std::vector<float>& out_energies,
    std::vector<std::vector<float>>& out_conformations,
    int verbosity
) {
    // Create scoring context
    ScoringContext ctx;
    create_scoring_context(ctx, gdata, cacheInfo);

    int n_conf = ctx.n_conf;
    int n_change = ctx.n_change;

    if (verbosity >= 1) {
        fprintf(stderr, "Warp-cooperative BFGS:\n");
        fprintf(stderr, "  Config: %d opts/warp, %d threads/opt\n",
                Config::OPTS_PER_WARP, Config::THREADS_PER_OPT);
        fprintf(stderr, "  n_conf=%d, n_change=%d, num_atoms=%d, num_nodes=%d\n",
                n_conf, n_change, ctx.num_atoms, ctx.num_nodes);
        fprintf(stderr, "  Poses: %d, Max iterations: %d\n", n_poses, max_iterations);
    }

    // Validate configuration - throw exception if ligand is too large
    if (ctx.num_atoms > Config::MAX_ATOMS) {
        std::ostringstream oss;
        oss << "Warp-coop BFGS: ligand has " << ctx.num_atoms << " atoms but MAX_ATOMS="
            << Config::MAX_ATOMS << ". Use --gpu instead of --warp_coop for large ligands.";
        throw std::runtime_error(oss.str());
    }
    int num_torsions = ctx.num_nodes > 0 ? ctx.num_nodes - 1 : 0;
    if (num_torsions > Config::MAX_TORSIONS) {
        std::ostringstream oss;
        oss << "Warp-coop BFGS: ligand has " << num_torsions << " torsions but MAX_TORSIONS="
            << Config::MAX_TORSIONS << ". Use --gpu instead of --warp_coop for large ligands.";
        throw std::runtime_error(oss.str());
    }
    if (ctx.num_nodes > Config::MAX_NODES) {
        std::ostringstream oss;
        oss << "Warp-coop BFGS: ligand has " << ctx.num_nodes << " nodes but MAX_NODES="
            << Config::MAX_NODES << ". Use --gpu instead of --warp_coop for large ligands.";
        throw std::runtime_error(oss.str());
    }

    // Allocate device memory
    ScoringContext* d_contexts;
    float* d_initial_confs;
    float* d_out_energies;
    float* d_out_confs;

    CUDA_CHECK_WARP(cudaMalloc(&d_contexts, sizeof(ScoringContext)));
    CUDA_CHECK_WARP(cudaMemcpy(d_contexts, &ctx, sizeof(ScoringContext), cudaMemcpyHostToDevice));

    CUDA_CHECK_WARP(cudaMalloc(&d_initial_confs, n_poses * n_conf * sizeof(float)));
    CUDA_CHECK_WARP(cudaMalloc(&d_out_energies, n_poses * sizeof(float)));
    CUDA_CHECK_WARP(cudaMalloc(&d_out_confs, n_poses * n_conf * sizeof(float)));

    // Initialize random conformations
    int init_blocks = (n_poses + 255) / 256;
    fprintf(stderr, "DEBUG: Launching init_random_confs_kernel...\n");
    init_random_confs_kernel<Config><<<init_blocks, 256>>>(
        d_initial_confs, n_poses, n_conf, ctx.nlig_roots, box_min, box_max, seed
    );
    CUDA_CHECK_WARP(cudaDeviceSynchronize());
    fprintf(stderr, "DEBUG: init_random_confs_kernel done\n");

    // Spatial sorting disabled for warp_coop - see bfgs_parallel.cu instead

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_contexts);
        cudaFree(d_initial_confs);
        cudaFree(d_out_energies);
        cudaFree(d_out_confs);
        std::ostringstream oss;
        oss << "CUDA error after init kernel: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }

    // Launch kernel
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 128
    constexpr int OPTS_PER_BLOCK = WARPS_PER_BLOCK * Config::OPTS_PER_WARP;

    int num_blocks = (n_poses + OPTS_PER_BLOCK - 1) / OPTS_PER_BLOCK;
    fprintf(stderr, "DEBUG: Launching bfgs_warp_cooperative_kernel with %d blocks, %d threads/block...\n",
            num_blocks, THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    CUDA_CHECK_WARP(cudaEventCreate(&start));
    CUDA_CHECK_WARP(cudaEventCreate(&stop));
    CUDA_CHECK_WARP(cudaEventRecord(start));

    bfgs_warp_cooperative_kernel<Config><<<num_blocks, THREADS_PER_BLOCK>>>(
        d_contexts,
        d_initial_confs,
        d_out_energies,
        d_out_confs,
        max_iterations,
        n_poses,
        n_conf,
        n_change
    );

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_contexts);
        cudaFree(d_initial_confs);
        cudaFree(d_out_energies);
        cudaFree(d_out_confs);
        std::ostringstream oss;
        oss << "CUDA error after BFGS kernel launch: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }

    fprintf(stderr, "DEBUG: Waiting for kernel...\n");
    CUDA_CHECK_WARP(cudaEventRecord(stop));
    CUDA_CHECK_WARP(cudaEventSynchronize(stop));
    fprintf(stderr, "DEBUG: Kernel complete\n");

    float kernel_ms;
    CUDA_CHECK_WARP(cudaEventElapsedTime(&kernel_ms, start, stop));

    if (verbosity >= 1) {
        fprintf(stderr, "  Kernel time: %.2f ms\n", kernel_ms);
        fprintf(stderr, "  Throughput: %.0f poses/sec\n", 1000.0f * n_poses / kernel_ms);
    }

    // Copy results back
    out_energies.resize(n_poses);
    CUDA_CHECK_WARP(cudaMemcpy(out_energies.data(), d_out_energies,
                               n_poses * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> all_confs(n_poses * n_conf);
    CUDA_CHECK_WARP(cudaMemcpy(all_confs.data(), d_out_confs,
                               n_poses * n_conf * sizeof(float), cudaMemcpyDeviceToHost));

    out_conformations.resize(n_poses);
    for (int i = 0; i < n_poses; i++) {
        out_conformations[i].assign(
            all_confs.begin() + i * n_conf,
            all_confs.begin() + (i + 1) * n_conf
        );
    }

    // Cleanup
    CUDA_CHECK_WARP(cudaFree(d_contexts));
    CUDA_CHECK_WARP(cudaFree(d_initial_confs));
    CUDA_CHECK_WARP(cudaFree(d_out_energies));
    CUDA_CHECK_WARP(cudaFree(d_out_confs));
    CUDA_CHECK_WARP(cudaEventDestroy(start));
    CUDA_CHECK_WARP(cudaEventDestroy(stop));
}

// Explicit template instantiations
template void run_warp_coop_bfgs_docking<WarpCoop4>(
    const gpu_data&, const GPUCacheInfo&, int, int,
    const gfloat3&, const gfloat3&, unsigned int,
    std::vector<float>&, std::vector<std::vector<float>>&, int
);

template void run_warp_coop_bfgs_docking<WarpCoop8>(
    const gpu_data&, const GPUCacheInfo&, int, int,
    const gfloat3&, const gfloat3&, unsigned int,
    std::vector<float>&, std::vector<std::vector<float>>&, int
);

// Non-template wrapper
void run_warp_coop_bfgs_docking_default(
    const gpu_data& gdata,
    const GPUCacheInfo& cacheInfo,
    int n_poses,
    int max_iterations,
    const gfloat3& box_min,
    const gfloat3& box_max,
    unsigned int seed,
    std::vector<float>& out_energies,
    std::vector<std::vector<float>>& out_conformations,
    int verbosity
) {
    run_warp_coop_bfgs_docking<DefaultWarpCoopConfig>(
        gdata, cacheInfo, n_poses, max_iterations,
        box_min, box_max, seed, out_energies, out_conformations, verbosity
    );
}
