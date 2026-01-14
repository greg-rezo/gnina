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

// Forward declarations
struct gpu_data;

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

    // Dimensions
    int n_conf;    // 7 * nlig_roots + n_torsions
    int n_change;  // 6 * nlig_roots + n_torsions
    int n_torsions;
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
void run_parallel_bfgs_docking(
    const struct gpu_data& gdata,
    const struct GPUCacheInfo& cacheInfo,
    int n_poses,
    int max_iterations,
    const gfloat3& box_min,
    const gfloat3& box_max,
    unsigned int seed,
    std::vector<float>& out_energies,
    std::vector<std::vector<float>>& out_conformations
);

// High-level interface for local minimization from input pose
// Returns: final energy (inter-adjusted), intramolecular energy
// out_conf: optimized conformation in flat format
void run_parallel_bfgs_minimize(
    const struct gpu_data& gdata,
    const struct GPUCacheInfo& cacheInfo,
    const std::vector<float>& input_conf,  // Initial conformation in flat format
    int max_iterations,
    float& out_energy,
    float& out_intramolecular,
    std::vector<float>& out_conf
);

// Helper: convert conf struct to flat array format
// flat_conf must be pre-allocated with size >= 7 * nlig_roots + n_torsions
void conf_to_flat(
    const struct conf& c,
    unsigned nlig_roots,
    unsigned n_torsions,
    float* flat_conf
);

// Helper: convert flat array format back to conf struct
void flat_to_conf(
    const float* flat_conf,
    unsigned nlig_roots,
    unsigned n_torsions,
    struct conf& c
);

// Launch parallel BFGS kernel
void launch_parallel_bfgs(
    const LigandBatch& batch,
    BFGSBatchMemory& mem,
    int max_iterations,
    const gfloat3& box_min,
    const gfloat3& box_max,
    unsigned int random_seed
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
    float* node_orientations
);

// Single-thread grid-based energy evaluation
__device__ float eval_energy_grid_single_thread(
    const ScoringContext& ctx,
    const float* coords,
    float* forces
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

#endif // BFGS_PARALLEL_H
