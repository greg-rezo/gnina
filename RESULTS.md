# GNINA GPU vs CPU Docking Results

**Git commit**: `bec9fce7d09df1ae25040275c4934569f6ed89d4`

## Test Configuration
- **Receptor**: 1dmp_rec.pdb
- **Ligand**: 1dmp_rand_pos_fix.sdf (perturbed DMQ pose)
- **Reference**: Original crystal ligand pose
- **Seed**: 12345
- **CNN scoring**: disabled

---

## Exhaustiveness 1024

### GPU (Parallel BFGS)
- **Mode**: `--gpu --bfgs_iterations 50`
- **Throughput**: 3978 poses/sec

| mode | affinity (kcal/mol) | referenceRMSD (Å) |
|------|---------------------|-------------------|
| 1 | -13.92 | **0.23** |
| 2 | -12.57 | 1.02 |
| 3 | -11.44 | 1.08 |
| 4 | -11.14 | 2.24 |
| 5 | -10.24 | 1.06 |
| 6 | -9.93 | **0.99** |
| 7 | -9.42 | 2.19 |
| 8 | -9.01 | 2.52 |
| 9 | -8.57 | 3.85 |

## CPU (Monte Carlo + BFGS)
- **Mode**: `--num_mc_steps 1`

| mode | affinity (kcal/mol) | referenceRMSD (Å) |
|------|---------------------|-------------------|
| 1 | -13.20 | 1.27 |
| 2 | -12.70 | **0.67** |
| 3 | -11.15 | 1.04 |
| 4 | -10.74 | 2.14 |
| 5 | -10.23 | 2.34 |
| 6 | -10.11 | 1.62 |
| 7 | -10.09 | 2.03 |
| 8 | -9.97 | 1.57 |
| 9 | -9.93 | 2.04 |

### Summary (Exhaustiveness 1024)

| Metric | GPU | CPU |
|--------|-----|-----|
| Best affinity | **-13.92** | -13.20 |
| Best RMSD | **0.23 Å** | 0.67 Å |
| Sub-1Å poses | 2 | 0 |
| Sub-2Å poses | 5 | 4 |

---

## Exhaustiveness 4096

### GPU (Parallel BFGS)
- **Mode**: `--gpu --bfgs_iterations 50`
- **Throughput**: 14678 poses/sec

| mode | affinity (kcal/mol) | referenceRMSD (Å) |
|------|---------------------|-------------------|
| 1 | -14.13 | **0.24** |
| 2 | -12.57 | 1.02 |
| 3 | -10.75 | 1.59 |
| 4 | -10.62 | 2.49 |
| 5 | -10.24 | 1.06 |
| 6 | -10.05 | 2.19 |
| 7 | -9.93 | **0.99** |
| 8 | -9.44 | 1.52 |
| 9 | -9.41 | 1.79 |

### CPU (Monte Carlo + BFGS)
- **Mode**: `--num_mc_steps 1`

| mode | affinity (kcal/mol) | referenceRMSD (Å) |
|------|---------------------|-------------------|
| 1 | -14.07 | **0.25** |
| 2 | -11.89 | **0.97** |
| 3 | -11.53 | 2.47 |
| 4 | -10.98 | 2.11 |
| 5 | -10.75 | 2.43 |
| 6 | -10.52 | 1.27 |
| 7 | -10.24 | 2.38 |
| 8 | -9.93 | 2.04 |
| 9 | -9.80 | 1.34 |

### Summary (Exhaustiveness 4096)

| Metric | GPU | CPU |
|--------|-----|-----|
| Best affinity | **-14.13** | -14.07 |
| Best RMSD | **0.24 Å** | 0.25 Å |
| Sub-1Å poses | 2 | 2 |
| Sub-2Å poses | 6 | 4 |

---

## BFGS Iterations Benchmark (L4 GPU)

**Test**: DMQ ligand, exhaustiveness=1024, single molecule

| BFGS Iterations | Total Time | Throughput | Time/Pose |
|-----------------|------------|------------|-----------|
| 100 (default)   | 407 ms     | 2,515 poses/sec | 398 µs |
| **50**          | **257 ms** | **3,990 poses/sec** | **251 µs** |

**Speedup with 50 iterations**: 37% faster, 59% higher throughput

Recommended default: `--bfgs_iterations 50` for good balance of speed and convergence.

---

## High Exhaustiveness Scaling (L4 GPU)

**Test**: DMQ ligand, bfgs_iterations=50, seed=12345

| Exhaustiveness | Total Time | Throughput | Time/Pose |
|----------------|------------|------------|-----------|
| 1,024          | 233 ms     | 4,398 poses/sec | 227 µs |
| 7,000          | 700 ms     | 10,005 poses/sec | 100 µs |
| 16,384         | 882 ms     | 18,582 poses/sec | 54 µs |
| **65,536**     | **3,525 ms** | **18,593 poses/sec** | **54 µs** |

**Key findings**:
- Throughput improves with higher exhaustiveness (better GPU utilization)
- Peak throughput: ~18,500 poses/sec at 16K+ exhaustiveness
- L4 GPU can easily handle 65K+ poses without memory issues
- Bug fix in `normalize_angle_device()` resolved infinite loops at high exhaustiveness

---

## Direct Pairwise vs Grid-Based Scoring (2026-01-16)

**Goal**: Test if direct pairwise scoring with LUT reduces L2 cache pressure at high exhaustiveness.

**Test Setup**:
- Receptor: 1dmp_rec.pdb (1862 atoms)
- Ligand: DMQ.sdf
- GPU: NVIDIA L4 (48 MB L2 cache, 58 SMs)
- BFGS iterations: 50

### Benchmark Results

| Exhaustiveness | Method | BFGS Kernel Time | Throughput | Time/Pose | Top Pose Energy |
|----------------|--------|------------------|------------|-----------|-----------------|
| 1,024 | Grid-based | 253 ms | 4,047 poses/sec | 247 µs | -13.62 kcal/mol |
| 1,024 | Direct pairwise | 798 ms | 1,283 poses/sec | 779 µs | -12.05 kcal/mol |
| 10,000 | Grid-based | 361 ms | 27,680 poses/sec | 36 µs | -13.76 kcal/mol |
| 10,000 | Direct pairwise | 1,270 ms | 7,854 poses/sec | 127 µs | -10.66 kcal/mol |

### Analysis

**Performance Ratio**: Direct pairwise is ~3.5x slower than grid-based at both exhaustiveness levels.

**Scaling**:
- Grid-based: 253ms → 361ms (1.43x) for 9.8x more poses - excellent GPU utilization
- Direct pairwise: 798ms → 1270ms (1.59x) for 9.8x more poses - good scaling but higher baseline

**Conclusion**: The L2 cache pressure hypothesis was not validated. The per-pose computational cost of direct pairwise (LUT lookups + spatial hash traversal) is inherently higher than optimized grid trilinear interpolation.

### Direct Pairwise Configuration

- Spatial hash cells: 125 (5 × 5 × 5)
- Cutoff: 8.0 Å
- LUT: 55 VdW groups × 100 distance bins
- Tables: base_energy, base_deriv, hydro_energy, hydro_deriv, hbond_energy, hbond_deriv

### Optimization Attempts

#### 1. Shared Memory LUT (Failed)
- Loaded 42KB LUT into shared memory per block
- **Result**: 37% **slower** (1089 ms vs 798 ms at 1024 poses)
- **Cause**: 42KB shared memory forces 1 block/SM instead of 8 (occupancy killed)
- Shared memory is per-block, not shared across blocks on an SM

#### 2. `__ldg()` Read-Only Cache (Success)
- Used `__ldg()` intrinsic for LUT access (uses 128KB texture cache per SM)
- No occupancy penalty - maintains 8 blocks/SM
- **Result**: 20% faster at low exhaustiveness, 5% faster at high

| Exhaustiveness | Original Global | Shared Memory | `__ldg()` Cache |
|----------------|-----------------|---------------|-----------------|
| 1,024 | 798 ms | 1,089 ms (+36%) | **665 ms (-17%)** |
| 10,000 | 1,270 ms | N/A | **1,204 ms (-5%)** |

### Remaining Optimization Ideas

1. **Spatial hash optimization** - Reduce number of receptor atoms checked per ligand atom
2. **Reduce LUT size** - Use 32 bins instead of 64 for smaller memory footprint
3. **Receptor atom culling** - Only include binding site atoms in spatial hash

---

## Grid-Based Scoring `__ldg()` Optimization (2026-01-16)

Added `__ldg()` read-only cache intrinsic to grid trilinear interpolation for texture cache access.

| Exhaustiveness | Before `__ldg()` | After `__ldg()` | Improvement |
|----------------|------------------|-----------------|-------------|
| 1,024 | 253 ms, 4,047 p/s | **219 ms, 4,669 p/s** | **13% faster** |
| 10,000 | 361 ms, 27,680 p/s | **310 ms, 32,270 p/s** | **14% faster** |

The texture cache (128KB per SM) provides better caching for read-only grid data with spatial locality.

---

## GPU vs CPU Throughput Comparison (2026-01-16)

**Test**: DMQ ligand, bfgs_iterations=50, seed=12345

### GPU Throughput Scaling (with `__ldg()`)

| Exhaustiveness | Kernel Time | Throughput | Time/Pose |
|----------------|-------------|------------|-----------|
| 1,024 | 219 ms | 4,669 p/s | 214 µs |
| 10,000 | 310 ms | 32,270 p/s | 31 µs |
| 15,000 | 374 ms | 40,145 p/s | 25 µs |
| **20,000** | **473 ms** | **42,243 p/s** | **24 µs** |
| 25,000 | 617 ms | 40,510 p/s | 25 µs |
| 40,000 | 1,089 ms | 36,722 p/s | 27 µs |
| 65,536 | 1,952 ms | 33,581 p/s | 30 µs |
| 100,000 | 2,555 ms | 39,143 p/s | 26 µs |

**Peak GPU throughput: ~42,000 poses/sec at 20K exhaustiveness**

### CPU Throughput (single-core, `--cpu 1`)

| Exhaustiveness | Total Time | Δ Time | Throughput |
|----------------|------------|--------|------------|
| 512 | 9.25s | - | - |
| 1024 | 10.05s | 0.80s | ~640 p/s |
| 2048 | 12.20s | 2.15s | ~476 p/s |

**Single-core CPU throughput: ~500-600 poses/sec**

### CPU Throughput (16-core threaded)

| Exhaustiveness | Total Time | Compute Time | Throughput |
|----------------|------------|--------------|------------|
| 4,096 | 10.3s | ~2.3s | ~1,780 p/s |
| 8,192 | 12.2s | ~4.2s | ~1,950 p/s |
| 16,384 | 16.3s | ~8.3s | ~1,970 p/s |

**16-core CPU throughput: ~2,000 poses/sec** (only 3.6x scaling from 16 cores)

### Summary: GPU vs CPU Speedup

| Platform | Throughput | vs 1 CPU | vs 16 CPU |
|----------|------------|----------|-----------|
| CPU (1 core) | ~550 p/s | 1x | - |
| CPU (16 × 1-core jobs) | ~8,800 p/s | 16x | - |
| CPU (16-core threaded) | ~2,000 p/s | 3.6x | 1x |
| **GPU (L4)** | **42,000 p/s** | **76x** | **21x** |

**Key findings:**
- GPU is **76x faster** than single-core CPU
- GPU is **21x faster** than 16-core threaded CPU
- GPU is **4.8x faster** than 16 parallel single-core CPU jobs
- CPU multi-threading achieves only 23% efficiency (2,000 vs 8,800 ideal)

### Memory Usage

| Configuration | RAM per Job | Total RAM |
|---------------|-------------|-----------|
| CPU (1 job) | ~1.4 GB | 1.4 GB |
| CPU (16 parallel jobs) | ~410 MB | **6.4 GB** |
| GPU (L4) | N/A | ~24 GB VRAM (grids + CNN) |

Memory per job drops significantly with parallel jobs due to shared memory-mapped files (CNN model, libraries) and copy-on-write pages.

---

## GPU BFGS Memory Analysis Per Thread

Analysis of memory usage per optimizer thread in the parallel BFGS kernel.

### Default Configuration (MAX_ATOMS=100, MAX_TORSIONS=64)

| Parameter | Value |
|-----------|-------|
| `PARALLEL_MAX_ATOMS` | 100 |
| `PARALLEL_MAX_TORSIONS` | 64 |
| `PARALLEL_MAX_CONF_SIZE` | 71 (= 7 + 64) |
| `PARALLEL_MAX_CHANGE_SIZE` | 70 (= 6 + 64) |
| `PARALLEL_MAX_HESSIAN_SIZE` | 2,485 (= 70×71/2) |
| `max_nodes` | ~65 (= 1 + torsions) |

#### Memory Breakdown (Default)

| Category | Arrays | Floats | Bytes | % |
|----------|--------|--------|-------|---|
| **Hessian** | `h` | 2,485 | 9,940 | **50.4%** |
| **Local/stack** | `node_origins`, `node_orientations`, `node_axes` | 960 | 3,840 | 19.5% |
| **Atom data** | `coords`, `forces` | 600 | 2,400 | 12.2% |
| **Node data** | `node_forces`, `node_torques` | 390 | 1,560 | 7.9% |
| **Gradients** | `g`, `g_new`, `p`, `y` | 280 | 1,120 | 5.7% |
| **Conformations** | `x`, `x_new`, `best_confs` | 213 | 852 | 4.3% |
| **Scalars** | `energies`, `best_energies` | 2 | 8 | 0.0% |
| **TOTAL** | | **3,970** | **19,720** | |

**Total: ~19.3 KB per thread** → 143 MB for 7,424 threads (L4 default)

### Reduced Configuration (MAX_ATOMS=50, MAX_TORSIONS=8)

For small/rigid ligands (≤50 atoms, ≤8 rotatable bonds):

| Parameter | Default | Reduced |
|-----------|---------|---------|
| `MAX_ATOMS` | 100 | **50** |
| `MAX_TORSIONS` | 64 | **8** |
| `MAX_CONF_SIZE` | 71 | **15** |
| `MAX_CHANGE_SIZE` | 70 | **14** |
| `MAX_HESSIAN_SIZE` | 2,485 | **105** |
| `max_nodes` | ~65 | **~9** |

#### Memory Breakdown (Reduced)

| Category | Default (bytes) | Reduced (bytes) | Savings |
|----------|-----------------|-----------------|---------|
| **Hessian** | 9,940 | 420 | **95.8%** |
| **Local/stack** | 3,840 | 480 | 87.5% |
| **Atom data** | 2,400 | 1,200 | 50.0% |
| **Node data** | 1,560 | 216 | 86.2% |
| **Gradients** | 1,120 | 224 | 80.0% |
| **Conformations** | 852 | 180 | 78.9% |
| **Scalars** | 8 | 8 | 0% |
| **TOTAL** | **19,720** | **2,728** | **86.2%** |

### Summary

| Metric | Default | Reduced | Improvement |
|--------|---------|---------|-------------|
| **Memory per thread** | 19.3 KB | 2.7 KB | **7.2× smaller** |
| **7,424 threads total** | 143 MB | 19.7 MB | **7.2× smaller** |
| **Threads in 1 GB** | ~53,000 | ~385,000 | **7.3× more** |

**Key insight**: The Hessian matrix dominates memory at 50% for default config (O(n²) scaling with DOF). With reduced torsions, Hessian drops to 15% and atom data becomes dominant. Most drug-like molecules (20-40 atoms, 2-10 torsions) would fit the reduced configuration.

---

## Conclusion

GPU parallel BFGS finds excellent results competitive with or better than CPU Monte Carlo:
- At exhaustiveness 1024: GPU finds better affinity (-13.92 vs -13.20) and better RMSD (0.23 Å vs 0.67 Å)
- At exhaustiveness 4096: Both find near-native poses (~0.25 Å), with GPU slightly better affinity (-14.13 vs -14.07)
- GPU throughput: **4,700-42,000 poses/sec** (peak at 20K exhaustiveness), **21-76x faster than CPU**

The parallel BFGS approach benefits from massively parallel local optimization from many random starting poses, which with sufficient exhaustiveness can find excellent near-native poses.

**Direct pairwise scoring**: After `__ldg()` optimization, still 2.6-3.3x slower than grid-based. The fundamental computational cost of pairwise distance calculations + spatial hash traversal exceeds grid trilinear interpolation.

---

## Warp-Cooperative BFGS Kernel (2026-01-16)

**Git commit**: `228fdaaf` (with gradient bug fixes)

New GPU kernel using warp shuffle instructions (~5 cycle latency) instead of shared/global memory (~30-273 cycles) for inter-thread communication within optimizer groups.

### Configuration
- 4 optimizers per warp, 8 threads per optimizer (WarpCoop4)
- Distributed state storage in registers across cooperating threads
- DistributedArray pattern for register-distributed data access

### Bug Fixes Applied
Two critical bugs were fixed in the gradient computation:
1. **Sign convention**: `trilinear_interp_warp` was returning force (F=-∇E) instead of gradient
2. **Missing cross product**: Tree reduction was missing `cross(r, child_force)` term in torque propagation

### Performance Comparison: Warp-Coop vs Standard GPU

**Test**: 1dmp_rec.pdb + DMQ.sdf, autobox, seed=42, bfgs_iterations=50

| Exhaustiveness | Metric | Standard GPU | Warp-Coop | Speedup |
|----------------|--------|--------------|-----------|---------|
| **256** | Throughput | 1,163 poses/sec | 7,427 poses/sec | **6.4x** |
| | Best affinity | -13.26 kcal/mol | -11.78 kcal/mol | |
| | Best CNN score | 0.9955 | 0.9900 | |
| **1,024** | Throughput | 4,113 poses/sec | 26,389 poses/sec | **6.4x** |
| | Best affinity | -13.26 kcal/mol | -11.72 kcal/mol | |
| | Best CNN score | 0.9955 | 0.9810 | |
| **7,000** | Throughput | 27,218 poses/sec | 51,989 poses/sec | **1.9x** |
| | Best affinity | -13.71 kcal/mol | -12.49 kcal/mol | |
| | Best CNN score | 0.9956 | 0.9911 | |

### Summary

| Aspect | Standard GPU | Warp-Coop |
|--------|--------------|-----------|
| **Throughput @ low exh** | Baseline | **6.4x faster** |
| **Throughput @ high exh** | Baseline | **1.9x faster** |
| Peak throughput | 27,218 p/s | **51,989 p/s** |
| Best affinity quality | Slightly better (~1 kcal/mol) | Good |
| Memory usage | 1858 MB | 756 MB (**59% less**) |

**Key findings**:
- Warp-coop is **6.4x faster** at low-medium exhaustiveness (256-1024)
- Speedup decreases to **1.9x** at high exhaustiveness (7000) as standard GPU becomes fully utilized
- Docking quality is ~1 kcal/mol worse (may need tuning)
- 59% less GPU memory usage
- Best for high-throughput virtual screening where speed matters more than exhaustive sampling

---

## Warp-Cooperative BFGS: Correctness Fix Performance Impact (2026-01-17)

**Investigation**: The original warp-coop kernel (commit `8aa85953`) was fast but produced incorrect results. Commit `d4776d94` fixed the bugs but introduced a significant performance regression.

### Bug Fixes in Commit d4776d94

1. **Quaternion multiplication for orientation updates** - Original used simple addition, fixed uses proper quaternion math
2. **Mandatory shuffle participation** - Original had `continue` statements that caused divergent shuffles (undefined behavior)
3. **Torsion index formula** - Fixed for multi-root ligand support (`nid + 6 * nlig_roots` instead of `7 + nid - 1`)
4. **Gradient sign handling** - Changed trilinear interp to return gradients directly instead of forces

### Performance Comparison

**Test**: 1dmp_rec.pdb + DMQ.sdf, seed=42, bfgs_iterations=50

| Version | Exhaustiveness | Kernel Time | Throughput | Quality |
|---------|----------------|-------------|------------|---------|
| Original (buggy) | 7,000 | 155 ms | 45,179 p/s | All poses positive energy, ligands outside box |
| Original (buggy) | 50,000 | 821 ms | **60,932 p/s** | All poses positive energy, ligands outside box |
| Fixed (correct) | 7,000 | 954 ms | 7,339 p/s | Best: -14.06 kcal/mol |
| Fixed (correct) | 50,000 | 5,903 ms | **8,470 p/s** | Best: -14.08 kcal/mol |

### Analysis

| Metric | Original (buggy) | Fixed (correct) | Difference |
|--------|------------------|-----------------|------------|
| Throughput @ 50k | 60,932 p/s | 8,470 p/s | **7.2x slower** |
| Result quality | BROKEN | Correct | N/A |
| Memory access pattern | Divergent (UB) | Uniform | Required for correctness |

**Root cause of slowdown**: The original kernel allowed threads to skip shuffle operations with `continue`, which is undefined behavior in CUDA. The fix requires ALL threads in an optimizer group to participate in shuffles, even when some don't need the result. This eliminates the performance benefit of early exits but is required for correctness.

### Architectural Constraint

In warp-cooperative algorithms using shuffle instructions:
- **ALL threads in the shuffle mask MUST participate** in every shuffle operation
- Divergent code paths that skip shuffles cause undefined behavior (data corruption)
- The performance cost of uniform execution is inherent to correct warp-cooperative algorithms

### WarpCoop Configuration Comparison

**Test**: Fixed (correct) version, 50k exhaustiveness, seed=42

| Configuration | Threads/Opt | Opts/Warp | Kernel Time | Throughput |
|---------------|-------------|-----------|-------------|------------|
| **WarpCoop4** (default) | 8 | 4 | 5,903 ms | **8,470 p/s** |
| WarpCoop8 | 4 | 8 | 6,539 ms | 7,646 p/s |

**Analysis**: WarpCoop8 is **10% slower** than WarpCoop4 despite processing 2x more optimizers per warp. The reduced threads per optimizer (4 vs 8) causes:
- Higher register pressure per thread (more distributed state per thread)
- Register spills to local memory
- Less parallelism for reduction operations within each optimizer

The header file note is correct: "OPTS_PER_WARP must be <= 4 to avoid register spills."

---

## Standard GPU BFGS: Throughput vs Exhaustiveness Scaling (2026-01-17)

**Git commit**: `d4776d94`

**Test configuration**: 1dmp_rec.pdb + 1dmp_rand_pos_fix.sdf (DMQ), NVIDIA L4 GPU, bfgs_iterations=50, cnn_scoring=none

### Results

| Exhaustiveness | BFGS Kernel Time | Throughput | Time per Pose | Best Affinity |
|----------------|------------------|------------|---------------|---------------|
| 7,000 | 304 ms | 22,863 poses/sec | 43.74 µs | -14.10 kcal/mol |
| 10,000 | 450 ms | 22,101 poses/sec | 45.25 µs | -14.12 kcal/mol |
| 20,000 | 552 ms | **35,953 poses/sec** | 27.81 µs | -14.12 kcal/mol |
| 50,000 | 1,513 ms | 32,828 poses/sec | 30.46 µs | -14.11 kcal/mol |
| 100,000 | 2,881 ms | 34,500 poses/sec | 28.99 µs | -14.14 kcal/mol |

### Key Findings

1. **Non-linear throughput scaling**: There's a ~60% throughput jump between 10k and 20k exhaustiveness
   - Below 20k: ~22k poses/sec (43-45 µs/pose)
   - At/above 20k: ~33-36k poses/sec (28-30 µs/pose)

2. **Peak throughput at 20k**: The sweet spot is around 20k exhaustiveness where GPU utilization is optimal

3. **Root cause - L2 cache warming effect**:
   - Scoring function grids need to be loaded into L2 cache at kernel start
   - At low exhaustiveness (7-10k), we finish before fully amortizing this cache warm-up cost
   - At 20k+, the grid data becomes L2-resident and highly reused across poses

4. **Docking quality is consistent**: All configurations find best scores of ~-14.1 kcal/mol

### Recommendations

- **For benchmarking**: Use ≥20k exhaustiveness to measure true sustained GPU throughput
- **For production**: The ~35k poses/sec throughput is achievable at any exhaustiveness ≥20k
- **Standard GPU kernel vs Warp-Coop**: At 50k exhaustiveness, standard GPU (32,828 p/s) outperforms fixed warp-coop (8,470 p/s) by **3.9x**

---

## Multi-Ligand Batch Docking (2026-01-17)

**Git commit**: `290fad2f`

New batch docking mode that processes multiple ligands together in GPU batches for efficient virtual screening.

### Features

- **Automatic batching**: Enabled by default when `--gpu` is passed
- **Smart grouping**: Ligands sorted by size and grouped to minimize memory waste from padding
- **Configurable batch size**: `--batch_size N` (default: 50,000 poses per batch)
- **Memory-aware**: Respects GPU memory limits when creating batches

### Test: 1000 ChEMBL Molecules (SDF Input)

**Configuration**:
- Receptor: 184l_rec.pdb
- Ligands: 1000 drug-like molecules from ChEMBL (MW 150-500, Lipinski compliant)
- Exhaustiveness: 1024
- BFGS iterations: 50
- Batch size: 50,000 poses
- GPU: NVIDIA L4

**Results**:

| Metric | With CNN | Without CNN |
|--------|----------|-------------|
| Ligands loaded | 1000 | 1000 |
| Ligands docked | 1000 | 1000 |
| Total poses | 1,024,000 | 1,024,000 |
| Total batches | 21 | 21 |
| **Total time** | 114.3 sec | **32.8 sec** |
| **Ligand throughput** | 8.8 lig/sec | **30.5 lig/sec** |
| **Pose throughput** | 8,961 p/sec | **31,220 p/sec** |

### Batch-Level Performance

The batching algorithm grouped ligands by atom count (15-42 atoms) into 21 batches:

| Batch | Ligands | Max Atoms | Max Torsions | Est. Memory | Throughput |
|-------|---------|-----------|--------------|-------------|------------|
| 1 | 48 | 15 | 3 | 42.8 MB | **471,745 p/s** |
| 2 | 48 | 16 | 4 | 48.2 MB | 440,839 p/s |
| 3 | 48 | 18 | 4 | 50.4 MB | 325,285 p/s |
| 4 | 48 | 19 | 4 | 51.6 MB | 273,920 p/s |
| 5 | 48 | 20 | 5 | 57.2 MB | 244,585 p/s |
| ... | ... | ... | ... | ... | ... |
| 18 | 48 | 33 | 14 | 120.8 MB | 74,794 p/s |
| 19 | 48 | 34 | 14 | 121.9 MB | 66,771 p/s |
| 20 | 48 | 36 | 13 | 117.9 MB | 58,878 p/s |
| 21 | 40 | 42 | 11 | 94.1 MB | **50,023 p/s** |

### Performance Analysis

1. **CNN scoring is the bottleneck**: With CNN, 93% of time is spent on refinement (not GPU BFGS)
   - GPU BFGS batch processing: ~8.5 sec for 1M poses = **120,000 poses/sec**
   - CNN adds 3.3x overhead; use `--cnn_scoring none` for high-throughput screening

2. **Throughput scales with molecule size** (GPU kernel only):
   - Small molecules (15 atoms): ~470k poses/sec
   - Medium molecules (25 atoms): ~125k poses/sec
   - Large molecules (42 atoms): ~50k poses/sec

3. **GPU kernel efficiency**: 88-97% of batch time is spent in BFGS kernel (setup/collection minimal)

### Memory Efficiency

Batch memory ranged from 42.8 MB (small molecules) to 121.9 MB (large molecules).

### Known Limitations

- **SMILES input not supported in batch mode**: Currently causes internal error in tree.h(185). Use SDF input with 3D coordinates.
- Input molecules must have 3D coordinates pre-generated

### Usage

```bash
# Default batch mode (enabled with --gpu)
gnina --gpu -r receptor.pdb -l ligands.sdf --autobox_ligand ref.sdf \
    --exhaustiveness 1024 -o output.sdf

# Custom batch size
gnina --gpu --batch_size 100000 -r receptor.pdb -l ligands.sdf ...

# Disable batching (single-ligand mode)
gnina --gpu --no_batch -r receptor.pdb -l ligands.sdf ...
```
