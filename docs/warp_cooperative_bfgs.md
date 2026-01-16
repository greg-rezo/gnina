# Warp-Cooperative BFGS Design

## Overview

This document describes a warp-cooperative BFGS implementation where multiple threads collaborate on each optimizer, using warp shuffles for ~5 cycle register-to-register communication instead of ~32-273 cycle L1/L2 access.

## Configuration

### Template Parameters

```cpp
// Primary configuration template
template<
    int OPTS_PER_WARP,      // 1, 2, 4, 8, 16, or 32
    int MAX_ATOMS = 50,
    int MAX_TORSIONS = 10
>
struct WarpCoopConfig {
    // Derived constants
    static constexpr int THREADS_PER_OPT = 32 / OPTS_PER_WARP;
    static constexpr int MAX_NODES = MAX_TORSIONS + 1;
    static constexpr int N_CONF = 7 + MAX_TORSIONS;
    static constexpr int N_CHANGE = 6 + MAX_TORSIONS;
    static constexpr int N_HESSIAN = N_CHANGE * (N_CHANGE + 1) / 2;

    // Per-thread distribution
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

    // Registers per thread
    static constexpr int REGS_PER_THREAD =
        (FLOATS_PER_OPT + THREADS_PER_OPT - 1) / THREADS_PER_OPT + 50;  // +50 working

    // Validation
    static_assert(32 % OPTS_PER_WARP == 0, "OPTS_PER_WARP must divide 32");
    static_assert(THREADS_PER_OPT >= 1 && THREADS_PER_OPT <= 32, "Invalid threads per optimizer");
    static_assert(REGS_PER_THREAD <= 255, "Exceeds max registers per thread");
};
```

### Predefined Configurations

```cpp
// Standard configurations
using WarpCoop1  = WarpCoopConfig<1,  50, 10>;  // 32 threads/opt, 1 opt/warp
using WarpCoop2  = WarpCoopConfig<2,  50, 10>;  // 16 threads/opt, 2 opts/warp
using WarpCoop4  = WarpCoopConfig<4,  50, 10>;  // 8 threads/opt,  4 opts/warp  <- DEFAULT
using WarpCoop8  = WarpCoopConfig<8,  50, 10>;  // 4 threads/opt,  8 opts/warp
using WarpCoop16 = WarpCoopConfig<16, 50, 10>;  // 2 threads/opt,  16 opts/warp
using WarpCoop32 = WarpCoopConfig<32, 50, 10>;  // 1 thread/opt,   32 opts/warp (current design)

// Memory analysis for each configuration (MAX_ATOMS=50, MAX_TORSIONS=10)
//
// | Config | Threads/Opt | Opts/Warp | Floats/Opt | Regs/Thread | Fits? |
// |--------|-------------|-----------|------------|-------------|-------|
// | Coop1  | 32          | 1         | 784        | 75          | Yes   |
// | Coop2  | 16          | 2         | 784        | 99          | Yes   |
// | Coop4  | 8           | 4         | 784        | 148         | Yes   |
// | Coop8  | 4           | 8         | 784        | 246         | Yes   |
// | Coop16 | 2           | 16        | 784        | 442         | NO    |
// | Coop32 | 1           | 32        | 784        | 834         | NO    |
```

### Runtime Configuration Macro

```cpp
// Select configuration at compile time
#ifndef WARP_COOP_OPTS_PER_WARP
#define WARP_COOP_OPTS_PER_WARP 4
#endif

#ifndef WARP_COOP_MAX_ATOMS
#define WARP_COOP_MAX_ATOMS 50
#endif

#ifndef WARP_COOP_MAX_TORSIONS
#define WARP_COOP_MAX_TORSIONS 10
#endif

using ActiveConfig = WarpCoopConfig<
    WARP_COOP_OPTS_PER_WARP,
    WARP_COOP_MAX_ATOMS,
    WARP_COOP_MAX_TORSIONS
>;
```

---

## Memory Layout

### Per-Optimizer State

```cpp
template<typename Config>
struct WarpCoopState {
    // Distributed across THREADS_PER_OPT threads via shuffle access

    // Conformations (N_CONF = 17 for 10 torsions)
    float x[Config::CONF_PER_THREAD];           // Current conf
    float x_new[Config::CONF_PER_THREAD];       // Trial conf
    float best_conf[Config::CONF_PER_THREAD];   // Best found

    // Gradients and search vectors (N_CHANGE = 16)
    float g[Config::CHANGE_PER_THREAD];         // Current gradient
    float g_new[Config::CHANGE_PER_THREAD];     // Trial gradient
    float p[Config::CHANGE_PER_THREAD];         // Search direction
    float y[Config::CHANGE_PER_THREAD];         // Gradient difference

    // Hessian (N_HESSIAN = 136 triangular)
    float h[Config::HESS_PER_THREAD];

    // Node transforms
    float node_origins[Config::NODES_PER_THREAD * 3];
    float node_orientations[Config::NODES_PER_THREAD * 9];
    float node_axes[Config::NODES_PER_THREAD * 3];
    float node_forces[Config::NODES_PER_THREAD * 3];
    float node_torques[Config::NODES_PER_THREAD * 3];

    // Atom data (each thread owns a slice)
    float my_coords[Config::ATOMS_PER_THREAD * 3];
    float my_forces[Config::ATOMS_PER_THREAD * 3];

    // Scalars (only lane 0 uses)
    float energy;
    float best_energy;
};
```

### Thread Indexing

```cpp
template<typename Config>
struct ThreadIndex {
    int warp_lane;          // 0-31: position within warp
    int opt_in_warp;        // 0 to OPTS_PER_WARP-1: which optimizer in this warp
    int local_lane;         // 0 to THREADS_PER_OPT-1: position within optimizer
    int opt_base_lane;      // Base warp lane for this optimizer
    unsigned opt_mask;      // Shuffle mask for this optimizer's threads

    __device__ ThreadIndex() {
        warp_lane = threadIdx.x % 32;
        opt_in_warp = warp_lane / Config::THREADS_PER_OPT;
        local_lane = warp_lane % Config::THREADS_PER_OPT;
        opt_base_lane = opt_in_warp * Config::THREADS_PER_OPT;

        // Create mask for this optimizer's threads
        opt_mask = ((1u << Config::THREADS_PER_OPT) - 1) << opt_base_lane;
    }

    // Get global optimizer ID
    __device__ int global_opt_id(int block_warps = 4) const {
        int warp_id = threadIdx.x / 32;
        return blockIdx.x * block_warps * Config::OPTS_PER_WARP
             + warp_id * Config::OPTS_PER_WARP
             + opt_in_warp;
    }
};
```

---

## Shuffle Primitives

### Basic Shuffle Operations

```cpp
template<typename Config>
struct ShuffleOps {
    const ThreadIndex<Config>& idx;

    __device__ ShuffleOps(const ThreadIndex<Config>& i) : idx(i) {}

    // Shuffle within optimizer group (handles any THREADS_PER_OPT)
    __device__ float shfl(float val, int src_local_lane) const {
        int src_warp_lane = idx.opt_base_lane + src_local_lane;
        return __shfl_sync(0xffffffff, val, src_warp_lane);
    }

    // Broadcast from local lane 0
    __device__ float broadcast(float val) const {
        return __shfl_sync(0xffffffff, val, idx.opt_base_lane);
    }

    // Reduce sum across optimizer's threads
    __device__ float reduce_sum(float val) const {
        if constexpr (Config::THREADS_PER_OPT >= 32) {
            for (int offset = 16; offset > 0; offset /= 2)
                val += __shfl_down_sync(0xffffffff, val, offset);
        } else if constexpr (Config::THREADS_PER_OPT >= 16) {
            for (int offset = 8; offset > 0; offset /= 2)
                val += __shfl_down_sync(idx.opt_mask, val, offset);
        } else if constexpr (Config::THREADS_PER_OPT >= 8) {
            val += __shfl_down_sync(idx.opt_mask, val, 4);
            val += __shfl_down_sync(idx.opt_mask, val, 2);
            val += __shfl_down_sync(idx.opt_mask, val, 1);
        } else if constexpr (Config::THREADS_PER_OPT >= 4) {
            val += __shfl_down_sync(idx.opt_mask, val, 2);
            val += __shfl_down_sync(idx.opt_mask, val, 1);
        } else if constexpr (Config::THREADS_PER_OPT >= 2) {
            val += __shfl_down_sync(idx.opt_mask, val, 1);
        }
        // Broadcast result to all threads in optimizer
        return broadcast(val);
    }

    // Reduce max across optimizer's threads
    __device__ float reduce_max(float val) const {
        if constexpr (Config::THREADS_PER_OPT >= 8) {
            val = fmaxf(val, __shfl_down_sync(idx.opt_mask, val, 4));
            val = fmaxf(val, __shfl_down_sync(idx.opt_mask, val, 2));
            val = fmaxf(val, __shfl_down_sync(idx.opt_mask, val, 1));
        } else if constexpr (Config::THREADS_PER_OPT >= 4) {
            val = fmaxf(val, __shfl_down_sync(idx.opt_mask, val, 2));
            val = fmaxf(val, __shfl_down_sync(idx.opt_mask, val, 1));
        } else if constexpr (Config::THREADS_PER_OPT >= 2) {
            val = fmaxf(val, __shfl_down_sync(idx.opt_mask, val, 1));
        }
        return broadcast(val);
    }
};
```

### Distributed Array Access

```cpp
template<typename Config>
struct DistributedArray {
    // Get element from distributed array
    // Array is striped: thread i owns elements i, i+T, i+2T, ... where T = THREADS_PER_OPT
    __device__ static float get(
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
    __device__ static void set(
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

    // Parallel iteration helper
    __device__ static void for_each(
        int count,
        const ThreadIndex<Config>& idx,
        auto&& func  // func(int global_idx, int local_slot)
    ) {
        for (int i = idx.local_lane; i < count; i += Config::THREADS_PER_OPT) {
            int local_slot = i / Config::THREADS_PER_OPT;
            func(i, local_slot);
        }
    }
};
```

---

## Algorithm Implementation

### Atom Assignment

```cpp
template<typename Config>
struct AtomPartition {
    int start;
    int end;
    int count;

    __device__ AtomPartition(int local_lane, int num_atoms) {
        // Evenly distribute atoms across threads
        int base = num_atoms / Config::THREADS_PER_OPT;
        int remainder = num_atoms % Config::THREADS_PER_OPT;

        start = local_lane * base + min(local_lane, remainder);
        end = start + base + (local_lane < remainder ? 1 : 0);
        count = end - start;
    }
};
```

### Set Conf (Coordinate Transform)

```cpp
template<typename Config>
__device__ void set_conf_warp_coop(
    const ScoringContext& ctx,
    WarpCoopState<Config>& state,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    using DA = DistributedArray<Config>;

    // === Step 1: Root node transform ===
    float root_pos[3], root_quat[4], root_mat[9];

    // Gather root position and quaternion
    root_pos[0] = DA::get(state.x, 0, shfl);
    root_pos[1] = DA::get(state.x, 1, shfl);
    root_pos[2] = DA::get(state.x, 2, shfl);
    root_quat[0] = DA::get(state.x, 3, shfl);
    root_quat[1] = DA::get(state.x, 4, shfl);
    root_quat[2] = DA::get(state.x, 5, shfl);
    root_quat[3] = DA::get(state.x, 6, shfl);

    // Normalize quaternion and convert to matrix
    normalize_quat(root_quat);
    quat_to_matrix(root_quat, root_mat);

    // Store root transform (distributed)
    DA::for_each(3, idx, [&](int i, int slot) {
        state.node_origins[slot] = root_pos[i];
    });
    DA::for_each(9, idx, [&](int i, int slot) {
        state.node_orientations[slot] = root_mat[i];
    });

    // === Step 2: Torsion nodes (sequential by layer) ===
    for (int layer = 1; layer < ctx.num_layers; layer++) {
        for (int nid = 1; nid < ctx.num_nodes; nid++) {
            if (ctx.tree_nodes[nid].layer != layer) continue;

            int parent = ctx.tree_nodes[nid].parent;

            // Gather parent transform
            float parent_origin[3], parent_mat[9];
            for (int i = 0; i < 3; i++)
                parent_origin[i] = DA::get(state.node_origins, parent * 3 + i, shfl);
            for (int i = 0; i < 9; i++)
                parent_mat[i] = DA::get(state.node_orientations, parent * 9 + i, shfl);

            // Get torsion angle
            float torsion = DA::get(state.x, 7 + nid - 1, shfl);

            // Compute transform
            float origin[3], mat[9], axis[3];
            compute_torsion_node(ctx.tree_nodes[nid], parent_origin, parent_mat,
                                 torsion, origin, mat, axis);

            // Store (distributed)
            for (int i = 0; i < 3; i++) {
                DA::set(state.node_origins, nid * 3 + i, origin[i], idx);
                DA::set(state.node_axes, nid * 3 + i, axis[i], idx);
            }
            for (int i = 0; i < 9; i++) {
                DA::set(state.node_orientations, nid * 9 + i, mat[i], idx);
            }
        }
    }

    // === Step 3: Atom coordinates (PARALLEL) ===
    AtomPartition<Config> atoms(idx.local_lane, ctx.num_atoms);

    for (int a = atoms.start; a < atoms.end; a++) {
        int owner = ctx.atom_owners[a];
        int local_slot = a - atoms.start;

        // Gather owner node transform
        float origin[3], mat[9];
        for (int i = 0; i < 3; i++)
            origin[i] = DA::get(state.node_origins, owner * 3 + i, shfl);
        for (int i = 0; i < 9; i++)
            mat[i] = DA::get(state.node_orientations, owner * 9 + i, shfl);

        // Local coords
        float lx = ctx.atom_local_data[a].coords.x;
        float ly = ctx.atom_local_data[a].coords.y;
        float lz = ctx.atom_local_data[a].coords.z;

        // Transform: world = origin + mat * local
        state.my_coords[local_slot * 3 + 0] = origin[0] + mat[0]*lx + mat[3]*ly + mat[6]*lz;
        state.my_coords[local_slot * 3 + 1] = origin[1] + mat[1]*lx + mat[4]*ly + mat[7]*lz;
        state.my_coords[local_slot * 3 + 2] = origin[2] + mat[2]*lx + mat[5]*ly + mat[8]*lz;
    }
}
```

### Energy Evaluation

```cpp
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
        if (atype >= ctx.ngrids) continue;

        float e, fx, fy, fz;
        trilinear_interp(ctx.grids[atype], x, y, z,
                         ctx.slope, ctx.forcecap, &e, &fx, &fy, &fz);

        my_energy += e;
        state.my_forces[local_slot * 3 + 0] = fx;
        state.my_forces[local_slot * 3 + 1] = fy;
        state.my_forces[local_slot * 3 + 2] = fz;
    }

    // Reduce across threads
    return shfl.reduce_sum(my_energy);
}
```

### Gradient Computation

```cpp
template<typename Config>
__device__ void compute_gradient_warp_coop(
    const ScoringContext& ctx,
    WarpCoopState<Config>& state,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    using DA = DistributedArray<Config>;
    AtomPartition<Config> atoms(idx.local_lane, ctx.num_atoms);

    // === Step 1: Local force/torque accumulation ===
    float local_node_f[Config::MAX_NODES * 3] = {0};
    float local_node_t[Config::MAX_NODES * 3] = {0};

    for (int a = atoms.start; a < atoms.end; a++) {
        int local_slot = a - atoms.start;
        int owner = ctx.atom_owners[a];

        float fx = state.my_forces[local_slot * 3 + 0];
        float fy = state.my_forces[local_slot * 3 + 1];
        float fz = state.my_forces[local_slot * 3 + 2];

        // Accumulate force
        local_node_f[owner * 3 + 0] += fx;
        local_node_f[owner * 3 + 1] += fy;
        local_node_f[owner * 3 + 2] += fz;

        // Get atom position and node origin for torque
        float ax = state.my_coords[local_slot * 3 + 0];
        float ay = state.my_coords[local_slot * 3 + 1];
        float az = state.my_coords[local_slot * 3 + 2];

        float ox = DA::get(state.node_origins, owner * 3 + 0, shfl);
        float oy = DA::get(state.node_origins, owner * 3 + 1, shfl);
        float oz = DA::get(state.node_origins, owner * 3 + 2, shfl);

        float rx = ax - ox, ry = ay - oy, rz = az - oz;

        // Torque = r × f
        local_node_t[owner * 3 + 0] += ry * fz - rz * fy;
        local_node_t[owner * 3 + 1] += rz * fx - rx * fz;
        local_node_t[owner * 3 + 2] += rx * fy - ry * fx;
    }

    // === Step 2: Reduce node forces/torques across threads ===
    for (int nid = 0; nid < ctx.num_nodes; nid++) {
        for (int d = 0; d < 3; d++) {
            float f = shfl.reduce_sum(local_node_f[nid * 3 + d]);
            float t = shfl.reduce_sum(local_node_t[nid * 3 + d]);
            DA::set(state.node_forces, nid * 3 + d, f, idx);
            DA::set(state.node_torques, nid * 3 + d, t, idx);
        }
    }

    // === Step 3: Tree reduction (child → parent) ===
    for (int layer = ctx.num_layers - 1; layer > 0; layer--) {
        for (int nid = ctx.num_nodes - 1; nid >= 1; nid--) {
            if (ctx.tree_nodes[nid].layer != layer) continue;
            int parent = ctx.tree_nodes[nid].parent;

            // Gather child force/torque
            for (int d = 0; d < 3; d++) {
                float cf = DA::get(state.node_forces, nid * 3 + d, shfl);
                float ct = DA::get(state.node_torques, nid * 3 + d, shfl);

                // Add to parent (read-modify-write via lane 0)
                float pf = DA::get(state.node_forces, parent * 3 + d, shfl);
                float pt = DA::get(state.node_torques, parent * 3 + d, shfl);
                DA::set(state.node_forces, parent * 3 + d, pf + cf, idx);
                DA::set(state.node_torques, parent * 3 + d, pt + ct, idx);
            }
        }
    }

    // === Step 4: Compute gradient (parallel over DOFs) ===
    DA::for_each(Config::N_CHANGE, idx, [&](int i, int slot) {
        float g_val;
        if (i < 3) {
            // Translation: gradient = root force
            g_val = DA::get(state.node_forces, i, shfl);
        } else if (i < 6) {
            // Rotation: gradient = root torque
            g_val = DA::get(state.node_torques, i - 3, shfl);
        } else {
            // Torsion: gradient = dot(torque, axis)
            int nid = i - 5;  // Node for this torsion (1-indexed)
            float t0 = DA::get(state.node_torques, nid * 3 + 0, shfl);
            float t1 = DA::get(state.node_torques, nid * 3 + 1, shfl);
            float t2 = DA::get(state.node_torques, nid * 3 + 2, shfl);
            float a0 = DA::get(state.node_axes, nid * 3 + 0, shfl);
            float a1 = DA::get(state.node_axes, nid * 3 + 1, shfl);
            float a2 = DA::get(state.node_axes, nid * 3 + 2, shfl);
            g_val = t0 * a0 + t1 * a1 + t2 * a2;
        }
        state.g[slot] = g_val;
    });
}
```

### Hessian Operations

```cpp
template<typename Config>
__device__ void hessian_vector_multiply(
    WarpCoopState<Config>& state,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    using DA = DistributedArray<Config>;

    // p = -H * g (parallel over rows)
    DA::for_each(Config::N_CHANGE, idx, [&](int i, int slot) {
        float sum = 0.0f;
        for (int j = 0; j < Config::N_CHANGE; j++) {
            int h_idx = (i <= j) ? (i + j * (j + 1) / 2) : (j + i * (i + 1) / 2);
            float h_val = DA::get(state.h, h_idx, shfl);
            float g_val = DA::get(state.g, j, shfl);
            sum += h_val * g_val;
        }
        state.p[slot] = -sum;
    });
}

template<typename Config>
__device__ void bfgs_hessian_update(
    WarpCoopState<Config>& state,
    float alpha,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    using DA = DistributedArray<Config>;

    // Compute y = g_new - g (parallel)
    DA::for_each(Config::N_CHANGE, idx, [&](int i, int slot) {
        state.y[slot] = state.g_new[slot] - state.g[slot];
    });

    // Compute yp = y · p
    float my_yp = 0.0f;
    DA::for_each(Config::N_CHANGE, idx, [&](int i, int slot) {
        my_yp += state.y[slot] * state.p[slot];
    });
    float yp = shfl.reduce_sum(my_yp);

    if (yp < 1e-10f) return;  // Skip update

    float ypy = yp * alpha;

    // Update Hessian (parallel over elements)
    DA::for_each(Config::N_HESSIAN, idx, [&](int k, int slot) {
        // Convert triangular index to (i, j)
        int i = 0, j = k;
        while (j >= i + 1) { j -= (i + 1); i++; }
        // Now k = i + j*(j+1)/2, with i <= j

        float h_ij = state.h[slot];
        float p_i = DA::get(state.p, i, shfl);
        float p_j = DA::get(state.p, j, shfl);
        float y_i = DA::get(state.y, i, shfl);
        float y_j = DA::get(state.y, j, shfl);

        float h_new = h_ij
            + (1.0f + (y_i * y_j) / ypy) * (alpha * p_i * p_j / yp)
            - (alpha / ypy) * (y_i * p_j + y_j * p_i);

        state.h[slot] = h_new;
    });
}
```

### Line Search

```cpp
template<typename Config>
__device__ float line_search_warp_coop(
    const ScoringContext& ctx,
    WarpCoopState<Config>& state,
    float f0,
    const ThreadIndex<Config>& idx,
    const ShuffleOps<Config>& shfl
) {
    using DA = DistributedArray<Config>;

    // Compute slope = g · p
    float my_slope = 0.0f;
    DA::for_each(Config::N_CHANGE, idx, [&](int i, int slot) {
        my_slope += state.g[slot] * state.p[slot];
    });
    float slope = shfl.reduce_sum(my_slope);

    if (slope >= 0.0f) return 0.0f;  // Not descent direction

    const float ALF = 1e-4f;
    float alpha = 1.0f;

    for (int iter = 0; iter < 20; iter++) {
        // x_new = x + alpha * p
        DA::for_each(Config::N_CONF, idx, [&](int i, int slot) {
            float x_i = state.x[slot];
            float p_i = (i < Config::N_CHANGE) ? DA::get(state.p, i, shfl) : 0.0f;
            state.x_new[slot] = x_i + alpha * p_i;
        });

        // Swap x and x_new temporarily for energy eval
        // (In practice, use x_new in set_conf)

        set_conf_warp_coop<Config>(ctx, state, idx, shfl);  // Uses x_new
        float f_new = eval_energy_warp_coop<Config>(ctx, state, idx, shfl);

        // Armijo condition
        if (f_new <= f0 + ALF * alpha * slope) {
            compute_gradient_warp_coop<Config>(ctx, state, idx, shfl);
            // Copy g to g_new
            DA::for_each(Config::N_CHANGE, idx, [&](int i, int slot) {
                state.g_new[slot] = state.g[slot];
            });
            state.energy = f_new;
            return alpha;
        }

        alpha *= 0.5f;
    }

    return 0.0f;
}
```

### Main Kernel

```cpp
template<typename Config>
__global__ void __launch_bounds__(128, 8)
bfgs_warp_cooperative_kernel(
    const ScoringContext* __restrict__ contexts,
    const float* __restrict__ initial_confs,
    float* __restrict__ out_energies,
    float* __restrict__ out_confs,
    int max_iterations,
    int n_optimizers
) {
    // Thread indexing
    ThreadIndex<Config> idx;
    ShuffleOps<Config> shfl(idx);

    int global_opt_id = idx.global_opt_id();
    if (global_opt_id >= n_optimizers) return;

    const ScoringContext& ctx = contexts[0];
    WarpCoopState<Config> state;

    // === Initialize ===
    init_optimizer<Config>(state, initial_confs, global_opt_id, idx);

    // === Initial energy and gradient ===
    set_conf_warp_coop<Config>(ctx, state, idx, shfl);
    state.energy = eval_energy_warp_coop<Config>(ctx, state, idx, shfl);
    compute_gradient_warp_coop<Config>(ctx, state, idx, shfl);
    state.best_energy = state.energy;
    copy_to_best<Config>(state, idx);

    // === BFGS iterations ===
    for (int iter = 0; iter < max_iterations; iter++) {
        // Search direction: p = -H * g
        hessian_vector_multiply<Config>(state, idx, shfl);

        // Line search
        float alpha = line_search_warp_coop<Config>(ctx, state, state.energy, idx, shfl);

        if (alpha == 0.0f) break;

        // Hessian update
        bfgs_hessian_update<Config>(state, alpha, idx, shfl);

        // Accept step: x = x_new, g = g_new
        swap_conf<Config>(state, idx);
        swap_gradient<Config>(state, idx);

        // Track best
        if (state.energy < state.best_energy) {
            state.best_energy = state.energy;
            copy_to_best<Config>(state, idx);
        }
    }

    // === Output (lane 0 only) ===
    if (idx.local_lane == 0) {
        out_energies[global_opt_id] = state.best_energy;
        write_conf<Config>(state, out_confs + global_opt_id * Config::N_CONF, idx, shfl);
    }
}

// Instantiate common configurations
template __global__ void bfgs_warp_cooperative_kernel<WarpCoop4>(...);
template __global__ void bfgs_warp_cooperative_kernel<WarpCoop8>(...);
```

---

## Launch Configuration

```cpp
template<typename Config>
void launch_warp_cooperative_bfgs(
    const ScoringContext* d_contexts,
    const float* d_initial_confs,
    float* d_out_energies,
    float* d_out_confs,
    int max_iterations,
    int n_optimizers,
    cudaStream_t stream = 0
) {
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 128
    constexpr int OPTS_PER_BLOCK = WARPS_PER_BLOCK * Config::OPTS_PER_WARP;

    int num_blocks = (n_optimizers + OPTS_PER_BLOCK - 1) / OPTS_PER_BLOCK;

    bfgs_warp_cooperative_kernel<Config><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_contexts,
        d_initial_confs,
        d_out_energies,
        d_out_confs,
        max_iterations,
        n_optimizers
    );
}
```

---

## Performance Characteristics

### Configurations Comparison (MAX_ATOMS=50, MAX_TORSIONS=10)

| Config | Threads/Opt | Opts/Warp | Opts/SM (48 warps) | Atom Parallelism | Register Fit |
|--------|-------------|-----------|--------------------|--------------------|--------------|
| Coop1  | 32          | 1         | 48                 | 32-way             | Yes (75 regs) |
| Coop2  | 16          | 2         | 96                 | 16-way             | Yes (99 regs) |
| **Coop4** | **8**    | **4**     | **192**            | **8-way**          | Yes (148 regs) |
| Coop8  | 4           | 8         | 384                | 4-way              | Yes (246 regs) |
| Current| 1           | 32        | 1,536              | 1-way (serial)     | No (spills)   |

### Expected Performance Trade-offs

| Metric | Current (1 thread/opt) | Coop4 (8 threads/opt) |
|--------|------------------------|------------------------|
| Memory latency | 32-273 cycles | ~5 cycles |
| Atom loop | 50 serial | ~7 parallel |
| Optimizers/SM | 1,536 | 192 |
| Register spill | Yes | No |
| **Net throughput** | Baseline | **~similar or better** |

### When to Use Each Configuration

| Scenario | Recommended Config |
|----------|-------------------|
| Small ligands (≤20 atoms) | Coop8 or Coop16 |
| Medium ligands (20-50 atoms) | **Coop4** (default) |
| Large ligands (50-100 atoms) | Coop2 or Coop1 |
| Memory-bound workloads | Lower OPTS_PER_WARP |
| Compute-bound workloads | Higher OPTS_PER_WARP |

---

## Implementation Phases

### Phase 1: Infrastructure
- [ ] `WarpCoopConfig` template with validation
- [ ] `ThreadIndex` and `ShuffleOps` primitives
- [ ] `DistributedArray` access helpers
- [ ] Unit tests for shuffle operations

### Phase 2: Core Algorithms
- [ ] `set_conf_warp_coop` with parallel atom transform
- [ ] `eval_energy_warp_coop` with reduction
- [ ] `compute_gradient_warp_coop` with tree reduction
- [ ] Unit tests comparing to serial implementation

### Phase 3: BFGS Loop
- [ ] `hessian_vector_multiply`
- [ ] `line_search_warp_coop`
- [ ] `bfgs_hessian_update`
- [ ] Full BFGS iteration loop

### Phase 4: Integration
- [ ] Kernel launch wrapper
- [ ] Integration with existing `run_parallel_bfgs_docking`
- [ ] Benchmark vs current implementation

### Phase 5: Tuning
- [ ] Profile with Nsight Compute
- [ ] Test different OPTS_PER_WARP values
- [ ] Optimize shuffle patterns
- [ ] Final performance validation
