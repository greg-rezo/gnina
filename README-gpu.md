# GNINA GPU Parallel BFGS Branch

This branch (`greg/gpu-parallel`) adds massively parallel GPU-accelerated docking to GNINA, replacing the traditional Monte Carlo search with a parallel BFGS optimizer that processes tens of thousands of poses simultaneously.

## Summary of Changes

**79 files changed, 16,200 insertions, 1,246 deletions**

## Key Features

### 1. Parallel BFGS Optimizer (`--gpu` flag)

The core addition is a new GPU-based BFGS optimizer that:
- Processes 10,000-100,000+ poses in parallel per batch
- Uses CUDA for massively parallel gradient computation and optimization
- Achieves **50-500x speedup** over CPU Monte Carlo for high exhaustiveness values
- Maintains scoring accuracy comparable to CPU mode

**New files:**
- `gninasrc/lib/bfgs_parallel.cu` - Main CUDA kernel implementation (2,310 lines)
- `gninasrc/lib/bfgs_parallel.h` - Data structures and declarations (446 lines)
- `gninasrc/lib/bfgs_diagnostics.h` - GPU resource monitoring and diagnostics

### 2. Multi-Ligand Batch Processing

Efficiently dock multiple ligands in a single run:
- Groups ligands by size for optimal GPU memory utilization
- Processes batches of ~50,000 poses at a time
- Single grid cache population shared across all ligands

**New files:**
- `gninasrc/lib/ligand_batch_manager.cpp` - Batch management logic (1,129 lines)
- `gninasrc/lib/ligand_batch_manager.h` - Batch manager interface

### 3. SMILES Input Support (GPU-only)

Direct SMILES-to-docking pipeline:
- Parallel RDKit 3D coordinate generation with MMFF minimization
- Multi-threaded embedding (16+ threads)
- `--parallel_embed N` flag generates N conformers per SMILES with different ring puckerings
- `--prune_rms_thresh` for diversity filtering of conformers

**New files:**
- `gninasrc/lib/RDKitConverter.cpp` - RDKit molecule conversion (222 lines)
- `gninasrc/lib/RDKitTreeBuilder.cpp` - Build gnina tree from RDKit (651 lines)

### 4. Batched CNN Scoring

Optimized CNN inference for large batches:
- Batched voxelization across all poses
- Cached voxelization for CNN ensemble models
- `--cnn_scoring_mult N` controls CNN scoring pose count (default: 20x num_modes)

**Modified files:**
- `gninasrc/lib/cnn_torch_scorer.cpp` - Added batch scoring methods
- `gninasrc/lib/torch_model.cpp` - Extended for batch inference

### 5. Symmetry-Aware RMSD

Proper handling of symmetric molecules:
- Uses RDKit's `GetBestRMS()` for symmetry-aware RMSD calculation
- Applied to both clustering and `referenceRMSD` output
- Fixes incorrect RMSD values for molecules with equivalent atoms

## New Command-Line Options

| Option | Description |
|--------|-------------|
| `--gpu` | Enable GPU parallel BFGS docking (required for SMILES input) |
| `--parallel_embed N` | Generate N conformers per SMILES molecule |
| `--prune_rms_thresh X` | RMSD threshold for pruning similar conformers (default: 0) |
| `--bfgs_iterations N` | Max BFGS iterations per pose (default: 50) |
| `--cnn_scoring_mult N` | Multiplier for CNN scoring poses (default: 20) |
| `--no_rdkit_smiles` | Use OpenBabel instead of RDKit for SMILES (debugging) |
| `--direct_pairwise` | Experimental: direct pairwise scoring without grids |

## Performance Results

### Throughput Scaling (1000 ChEMBL molecules, L4 GPU)

| Exhaustiveness | Poses/sec | Ligands/sec | Time (1000 mols) |
|----------------|-----------|-------------|------------------|
| 8 | 35,000 | 4,375 | 0.2s |
| 64 | 85,000 | 1,328 | 0.8s |
| 512 | 105,000 | 205 | 4.9s |
| 1,000 | 108,000 | 108 | 9.3s |
| 10,000 | 110,000 | 11 | 91s |

### Timing Breakdown (typical run)

| Phase | Time | Notes |
|-------|------|-------|
| SMILES 3D generation | 1.5s | 16-thread parallel RDKit |
| Grid cache population | 0.5-2s | Depends on box size |
| BFGS optimization | 0.1-0.5s | Per batch of 50k poses |
| CNN scoring | 8-10s | Dominates for CNN-enabled runs |

## Architecture

### GPU Kernel Design

The `bfgs_parallel_kernel` processes one pose per CUDA thread:
- 128 threads per block, targeting 8 blocks per SM
- ~64 registers per thread on L4 GPU
- Spatial sorting of poses for L1 cache locality (93% hit rate achieved)

### Memory Management

- Thread-local Hessian matrices (n_dof × n_dof floats)
- Shared grid cache across all poses
- Dynamic batching based on available GPU memory

### Data Flow

```
SMILES/SDF → RDKit 3D → Batch Manager → GPU BFGS → CNN Scoring → Output
                ↓
         Grid Cache (populated once)
```

## Bug Fixes

1. **Torsion ordering mismatch** - Fixed CPU DFS vs GPU BFS tree traversal order
2. **BFGS Hessian sign bug** - Corrected gradient update direction
3. **Slope mismatch** - Set slope=10 to match CPU refine_structure behavior
4. **Quaternion handling** - Fixed orientation gradient computation
5. **Multi-conformer pooling** - Now correctly pools poses across conformers before trimming
6. **SMILES 3D geometry** - Fixed RDKit/OpenBabel coordinate generation issues

## Build Instructions

### Using Docker (recommended)

```bash
./build_on_pod.sh [POD_NAME]
```

### Local build

```bash
./build_linux.sh
```

### Profiling

```bash
./profile_on_vm.sh [exhaustiveness] [bfgs_iterations]
```

## Testing

```bash
# Run GPU tests on a pod
./run_tests_on_gpu.sh

# Python-based BFGS accuracy tests
python test/gnina/test_bfgs_only.py
```

## Files Changed

### New Core Files
- `bfgs_parallel.cu/h` - GPU BFGS kernel
- `ligand_batch_manager.cpp/h` - Multi-ligand batching
- `RDKitConverter.cpp/h` - RDKit molecule conversion
- `RDKitTreeBuilder.cpp/h` - Tree structure from RDKit
- `bfgs_diagnostics.h` - GPU diagnostics
- `scoring_lut.h` - Scoring lookup tables (experimental)
- `spatial_hash.h` - Spatial hashing (experimental)

### Modified Core Files
- `main.cpp` - Added GPU batch docking mode (+1,584 lines)
- `molgetter.cpp` - RDKit/OpenBabel integration
- `cnn_torch_scorer.cpp` - Batched CNN scoring
- `torch_model.cpp` - Extended batch inference
- `monte_carlo.cpp` - Refactored for GPU compatibility

### Build/Test Infrastructure
- `build_on_pod.sh` - K8s pod-based building
- `build_docker_incremental.sh` - Incremental Docker builds
- `profile_on_vm.sh` - NCU profiling script
- `run_tests_on_gpu.sh` - GPU test runner
- `RESULTS.md` - Benchmark results (1,611 lines)
- `CLAUDE.md` - Development instructions

## Removed Files

- `bfgs.cu` - Old single-pose GPU BFGS (replaced)
- `non_cache_gpu.cu/h` - Unused GPU scoring cache
- `test_gpucode.cpp/h` - Outdated GPU tests
- `parallel_mc.cpp` - Experimental parallel Monte Carlo

## Known Limitations

1. SMILES input requires `--gpu` flag (CPU mode needs pre-generated 3D structures)
2. Grid cache population is single-threaded CPU (can be slow for large boxes)
3. CNN scoring dominates runtime for CNN-enabled docking
4. Memory usage scales with exhaustiveness × max_atoms × max_torsions
