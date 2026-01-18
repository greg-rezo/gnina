# GNINA GPU Optimization TODO

## Potential Optimizations

### 1. Adaptive Spatial Sorting Cell Count

**Current state:** Spatial sorting uses fixed 8³ = 512 cells regardless of grid size.

**Issue:** This may over-fragment the spatial domain. The optimal cell count should be based on:
- Grid memory size (typically 12-18 MB for drug-like molecules)
- L1 cache size per SM (128 KB on L4)

**Proposed solution:**
```cpp
int calculate_optimal_divisions(size_t grid_bytes) {
    const size_t L1_CACHE = 128 * 1024;  // 128 KB per SM

    // Target: each cell's grid region should be ~75% of L1
    size_t target_per_cell = L1_CACHE * 3 / 4;  // 96 KB
    int cells_needed = (grid_bytes + target_per_cell - 1) / target_per_cell;

    // Round up to nearest cube root
    int divisions = (int)ceilf(cbrtf((float)cells_needed));
    return max(4, min(divisions, 16));  // Clamp to 4-16
}
```

**Analysis:**
| Grid Size | Optimal Cells | Current (512) |
|-----------|---------------|---------------|
| 14 MB | 5³ = 125 | Over-fragmented |
| 20 MB | 6³ = 216 | Over-fragmented |
| 40 MB | 7³ = 343 | Slightly over |

**Files to modify:** `gninasrc/lib/bfgs_parallel.cu` - `sort_poses_spatially()` function

**Priority:** Low - requires NCU profiling to validate improvement

---

## Completed Optimizations

### Spatial Sorting for Cache Locality (2026-01-17)
- Added Morton code Z-order curve sorting to `bfgs_parallel.cu`
- Poses sorted by 3D position before BFGS optimization
- Adjacent GPU threads process spatially-nearby poses
- Expected improvement: L1 hit rate from ~49% to 60-70%

### `__ldg()` Read-Only Cache for Grid Access (2026-01-16)
- Added texture cache intrinsic for grid trilinear interpolation
- 13-14% speedup at all exhaustiveness levels

### Warp-Cooperative BFGS (Disabled)
- Experimental kernel using warp shuffles instead of shared memory
- Disabled due to correctness issues and slower performance than standard kernel
- Code remains in `warp_coop_bfgs.cu` for reference
