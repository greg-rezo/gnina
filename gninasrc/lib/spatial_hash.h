/*
 * spatial_hash.h
 *
 * Spatial hash data structure for efficient neighbor finding.
 * Used by direct pairwise scoring to quickly find receptor atoms near each ligand atom.
 *
 * The spatial hash divides space into cubic cells of size equal to the cutoff distance.
 * To find all receptor atoms within cutoff of a ligand atom, only the 27 neighboring
 * cells (3x3x3) need to be checked.
 */

#ifndef SPATIAL_HASH_H
#define SPATIAL_HASH_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "scoring_lut.h"

// ============================================================================
// Receptor Atom Structure
// ============================================================================

// Compact representation of a receptor atom for GPU
struct ReceptorAtomGPU {
    float x, y, z;          // Position (12 bytes)
    uint8_t smina_type;     // SMINA atom type (1 byte)
    uint8_t radius_group;   // Radius group for LUT lookup (1 byte)
    uint8_t flags;          // Hydrophobic, donor, acceptor flags (1 byte)
    uint8_t padding;        // Alignment padding (1 byte)
};  // Total: 16 bytes

static_assert(sizeof(ReceptorAtomGPU) == 16, "ReceptorAtomGPU should be 16 bytes");

// ============================================================================
// Spatial Hash Structure
// ============================================================================

struct SpatialHashGPU {
    // Grid dimensions
    int3 grid_dim;          // Number of cells in each dimension
    float3 origin;          // Box corner (minimum x, y, z)
    float cell_size;        // Cell size (typically = cutoff distance)
    float inv_cell_size;    // 1 / cell_size for fast division

    // Device pointers (set by host)
    int* cell_start;        // [num_cells] Start index in sorted_atoms
    int* cell_count;        // [num_cells] Number of atoms in each cell
    ReceptorAtomGPU* atoms; // Sorted receptor atoms

    int num_cells;          // Total number of cells
    int num_atoms;          // Total number of receptor atoms
};

// ============================================================================
// Host-side Spatial Hash Builder
// ============================================================================

class SpatialHashBuilder {
public:
    // Grid parameters
    int3 grid_dim;
    float3 origin;
    float3 box_size;
    float cell_size;
    int num_cells;

    // Host-side data
    std::vector<int> cell_start;
    std::vector<int> cell_count;
    std::vector<ReceptorAtomGPU> sorted_atoms;

    // Original receptor data (for reference)
    std::vector<ReceptorAtomGPU> original_atoms;

    SpatialHashBuilder() : cell_size(8.0f), num_cells(0) {
        grid_dim = make_int3(0, 0, 0);
        origin = make_float3(0, 0, 0);
        box_size = make_float3(0, 0, 0);
    }

    // Build spatial hash from receptor atoms
    // box_min/max define the search region (typically the docking box + cutoff margin)
    void build(
        const std::vector<float>& coords,  // [num_atoms * 3] x,y,z coordinates
        const std::vector<uint8_t>& types, // [num_atoms] SMINA atom types
        float3 box_min,
        float3 box_max,
        float cutoff = 8.0f
    ) {
        cell_size = cutoff;
        int num_atoms = coords.size() / 3;

        // Expand box by cutoff to capture atoms on the boundary
        origin.x = box_min.x - cutoff;
        origin.y = box_min.y - cutoff;
        origin.z = box_min.z - cutoff;

        box_size.x = (box_max.x + cutoff) - origin.x;
        box_size.y = (box_max.y + cutoff) - origin.y;
        box_size.z = (box_max.z + cutoff) - origin.z;

        // Compute grid dimensions (at least 1 cell in each dimension)
        grid_dim.x = std::max(1, (int)ceilf(box_size.x / cell_size));
        grid_dim.y = std::max(1, (int)ceilf(box_size.y / cell_size));
        grid_dim.z = std::max(1, (int)ceilf(box_size.z / cell_size));

        num_cells = grid_dim.x * grid_dim.y * grid_dim.z;

        // Initialize cell counts
        cell_count.assign(num_cells, 0);
        cell_start.assign(num_cells, 0);

        // First pass: count atoms per cell
        original_atoms.resize(num_atoms);
        std::vector<int> atom_cells(num_atoms);

        for (int i = 0; i < num_atoms; i++) {
            float x = coords[i * 3 + 0];
            float y = coords[i * 3 + 1];
            float z = coords[i * 3 + 2];

            // Create atom struct
            original_atoms[i].x = x;
            original_atoms[i].y = y;
            original_atoms[i].z = z;
            original_atoms[i].smina_type = types[i];
            original_atoms[i].radius_group = SMINA_TYPE_TO_RADIUS_GROUP_HOST[types[i]];
            original_atoms[i].flags = SMINA_TYPE_FLAGS_HOST[types[i]];
            original_atoms[i].padding = 0;

            // Compute cell index
            int cell_idx = get_cell_index(x, y, z);
            atom_cells[i] = cell_idx;

            if (cell_idx >= 0 && cell_idx < num_cells) {
                cell_count[cell_idx]++;
            }
        }

        // Compute cell start indices (exclusive prefix sum)
        int running_sum = 0;
        for (int i = 0; i < num_cells; i++) {
            cell_start[i] = running_sum;
            running_sum += cell_count[i];
        }

        // Second pass: place atoms in sorted order
        sorted_atoms.resize(num_atoms);
        std::vector<int> cell_offset(num_cells, 0);

        for (int i = 0; i < num_atoms; i++) {
            int cell_idx = atom_cells[i];
            if (cell_idx >= 0 && cell_idx < num_cells) {
                int dest_idx = cell_start[cell_idx] + cell_offset[cell_idx];
                sorted_atoms[dest_idx] = original_atoms[i];
                cell_offset[cell_idx]++;
            }
        }
    }

    // Get cell index for a position
    int get_cell_index(float x, float y, float z) const {
        int cx = (int)((x - origin.x) / cell_size);
        int cy = (int)((y - origin.y) / cell_size);
        int cz = (int)((z - origin.z) / cell_size);

        // Clamp to valid range
        cx = std::max(0, std::min(cx, grid_dim.x - 1));
        cy = std::max(0, std::min(cy, grid_dim.y - 1));
        cz = std::max(0, std::min(cz, grid_dim.z - 1));

        return cx + cy * grid_dim.x + cz * grid_dim.x * grid_dim.y;
    }

    // Get cell coordinates from index
    int3 get_cell_coords(int cell_idx) const {
        int3 coords;
        coords.z = cell_idx / (grid_dim.x * grid_dim.y);
        int remainder = cell_idx % (grid_dim.x * grid_dim.y);
        coords.y = remainder / grid_dim.x;
        coords.x = remainder % grid_dim.x;
        return coords;
    }

    // Allocate and copy to GPU (only available when compiling with CUDA)
#ifdef __CUDACC__
    void upload_to_gpu(SpatialHashGPU& gpu_hash) const {
        gpu_hash.grid_dim = grid_dim;
        gpu_hash.origin = origin;
        gpu_hash.cell_size = cell_size;
        gpu_hash.inv_cell_size = 1.0f / cell_size;
        gpu_hash.num_cells = num_cells;
        gpu_hash.num_atoms = sorted_atoms.size();

        // Allocate and copy cell_start
        cudaMalloc(&gpu_hash.cell_start, num_cells * sizeof(int));
        cudaMemcpy(gpu_hash.cell_start, cell_start.data(),
                   num_cells * sizeof(int), cudaMemcpyHostToDevice);

        // Allocate and copy cell_count
        cudaMalloc(&gpu_hash.cell_count, num_cells * sizeof(int));
        cudaMemcpy(gpu_hash.cell_count, cell_count.data(),
                   num_cells * sizeof(int), cudaMemcpyHostToDevice);

        // Allocate and copy atoms
        cudaMalloc(&gpu_hash.atoms, sorted_atoms.size() * sizeof(ReceptorAtomGPU));
        cudaMemcpy(gpu_hash.atoms, sorted_atoms.data(),
                   sorted_atoms.size() * sizeof(ReceptorAtomGPU), cudaMemcpyHostToDevice);
    }

    // Free GPU memory
    static void free_gpu(SpatialHashGPU& gpu_hash) {
        if (gpu_hash.cell_start) cudaFree(gpu_hash.cell_start);
        if (gpu_hash.cell_count) cudaFree(gpu_hash.cell_count);
        if (gpu_hash.atoms) cudaFree(gpu_hash.atoms);
        gpu_hash.cell_start = nullptr;
        gpu_hash.cell_count = nullptr;
        gpu_hash.atoms = nullptr;
    }
#endif // __CUDACC__

    // Validate spatial hash by brute-force comparison
    bool validate(float cutoff = 8.0f) const {
        float cutoff_sq = cutoff * cutoff;
        int num_atoms = original_atoms.size();

        // For each atom, verify that all atoms within cutoff are found
        // by checking the 27 neighboring cells
        for (int i = 0; i < num_atoms; i++) {
            const auto& atom_i = original_atoms[i];

            // Find neighbors using spatial hash
            std::vector<int> hash_neighbors;
            find_neighbors_host(atom_i.x, atom_i.y, atom_i.z, cutoff, hash_neighbors);

            // Find neighbors using brute force
            std::vector<int> brute_neighbors;
            for (int j = 0; j < num_atoms; j++) {
                if (i == j) continue;
                const auto& atom_j = original_atoms[j];
                float dx = atom_i.x - atom_j.x;
                float dy = atom_i.y - atom_j.y;
                float dz = atom_i.z - atom_j.z;
                float d_sq = dx*dx + dy*dy + dz*dz;
                if (d_sq < cutoff_sq) {
                    brute_neighbors.push_back(j);
                }
            }

            // Check that all brute-force neighbors are found by hash
            std::sort(hash_neighbors.begin(), hash_neighbors.end());
            std::sort(brute_neighbors.begin(), brute_neighbors.end());

            // Hash neighbors should be a superset of brute neighbors
            // (hash may include atoms just outside cutoff due to cell boundary)
            for (int j : brute_neighbors) {
                if (std::find(hash_neighbors.begin(), hash_neighbors.end(), j)
                    == hash_neighbors.end()) {
                    printf("Validation FAILED: atom %d missing neighbor %d\n", i, j);
                    return false;
                }
            }
        }

        return true;
    }

    // Find neighbors on host (for validation)
    void find_neighbors_host(float x, float y, float z, float cutoff,
                             std::vector<int>& neighbors) const {
        neighbors.clear();
        float cutoff_sq = cutoff * cutoff;

        // Get cell containing query point
        int cx = (int)((x - origin.x) / cell_size);
        int cy = (int)((y - origin.y) / cell_size);
        int cz = (int)((z - origin.z) / cell_size);

        // Check 27 neighboring cells
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ncx = cx + dx;
                    int ncy = cy + dy;
                    int ncz = cz + dz;

                    // Skip out-of-bounds cells
                    if (ncx < 0 || ncx >= grid_dim.x) continue;
                    if (ncy < 0 || ncy >= grid_dim.y) continue;
                    if (ncz < 0 || ncz >= grid_dim.z) continue;

                    int cell_idx = ncx + ncy * grid_dim.x + ncz * grid_dim.x * grid_dim.y;
                    int start = cell_start[cell_idx];
                    int count = cell_count[cell_idx];

                    for (int i = start; i < start + count; i++) {
                        const auto& atom = sorted_atoms[i];
                        float ddx = x - atom.x;
                        float ddy = y - atom.y;
                        float ddz = z - atom.z;
                        float d_sq = ddx*ddx + ddy*ddy + ddz*ddz;
                        if (d_sq < cutoff_sq) {
                            neighbors.push_back(i);
                        }
                    }
                }
            }
        }
    }

    // Get statistics
    void print_stats() const {
        int num_atoms = sorted_atoms.size();
        int non_empty_cells = 0;
        int max_atoms_per_cell = 0;
        int total_atoms_in_cells = 0;

        for (int i = 0; i < num_cells; i++) {
            if (cell_count[i] > 0) {
                non_empty_cells++;
                total_atoms_in_cells += cell_count[i];
                if (cell_count[i] > max_atoms_per_cell) {
                    max_atoms_per_cell = cell_count[i];
                }
            }
        }

        float avg_atoms_per_cell = non_empty_cells > 0 ?
            (float)total_atoms_in_cells / non_empty_cells : 0;

        printf("Spatial Hash Statistics:\n");
        printf("  Grid dimensions: %d x %d x %d = %d cells\n",
               grid_dim.x, grid_dim.y, grid_dim.z, num_cells);
        printf("  Cell size: %.2f A\n", cell_size);
        printf("  Receptor atoms: %d\n", num_atoms);
        printf("  Non-empty cells: %d (%.1f%%)\n",
               non_empty_cells, 100.0f * non_empty_cells / num_cells);
        printf("  Avg atoms per non-empty cell: %.1f\n", avg_atoms_per_cell);
        printf("  Max atoms per cell: %d\n", max_atoms_per_cell);
    }
};

// ============================================================================
// Device-side Spatial Hash Functions
// ============================================================================

#ifdef __CUDA_ARCH__
#define SPATIAL_DEVICE __device__ __forceinline__
#else
#define SPATIAL_DEVICE inline
#endif

// Portable min/max that work in both CUDA device code and CPU code
SPATIAL_DEVICE int spatial_min(int a, int b) { return (a < b) ? a : b; }
SPATIAL_DEVICE int spatial_max(int a, int b) { return (a > b) ? a : b; }

// Get cell index for a position on GPU
SPATIAL_DEVICE int3 get_cell_coords_gpu(
    float x, float y, float z,
    const SpatialHashGPU& hash
) {
    int3 coords;
    coords.x = (int)((x - hash.origin.x) * hash.inv_cell_size);
    coords.y = (int)((y - hash.origin.y) * hash.inv_cell_size);
    coords.z = (int)((z - hash.origin.z) * hash.inv_cell_size);

    // Clamp to valid range
    coords.x = spatial_max(0, spatial_min(coords.x, hash.grid_dim.x - 1));
    coords.y = spatial_max(0, spatial_min(coords.y, hash.grid_dim.y - 1));
    coords.z = spatial_max(0, spatial_min(coords.z, hash.grid_dim.z - 1));

    return coords;
}

// Get linear cell index from coordinates
SPATIAL_DEVICE int cell_coords_to_index(
    int3 coords,
    const SpatialHashGPU& hash
) {
    return coords.x + coords.y * hash.grid_dim.x +
           coords.z * hash.grid_dim.x * hash.grid_dim.y;
}

// Check if cell coordinates are valid
SPATIAL_DEVICE bool is_valid_cell(
    int3 coords,
    const SpatialHashGPU& hash
) {
    return coords.x >= 0 && coords.x < hash.grid_dim.x &&
           coords.y >= 0 && coords.y < hash.grid_dim.y &&
           coords.z >= 0 && coords.z < hash.grid_dim.z;
}

#endif // SPATIAL_HASH_H
