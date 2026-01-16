# GNINA Build Instructions

## Building

**ALWAYS use the pod-based build script:**
```bash
./build_on_pod.sh
```

Do NOT use `build_docker_incremental.sh` - use the pod-based build instead.

## GPU Kernel Development

The main GPU BFGS kernel is in:
- `gninasrc/lib/bfgs_parallel.cu` - kernel implementation
- `gninasrc/lib/bfgs_parallel.h` - structures and declarations
- `gninasrc/lib/bfgs_diagnostics.h` - runtime diagnostics

### Launch Bounds

Kernels use `__launch_bounds__(128, 8)` to target:
- 128 threads per block
- 8 blocks per SM minimum
- 1024 threads per SM total
- ~64 registers per thread on L4 GPU

### Diagnostics

Set `verbosity >= 1` when calling `run_parallel_bfgs_docking()` or `run_parallel_bfgs_minimize()` to get:
- GPU properties (L2 cache size, SM count)
- Per-optimizer memory usage
- Kernel register count
- L2 cache fit warnings for grids
