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

## Conclusion

GPU parallel BFGS finds excellent results competitive with or better than CPU Monte Carlo:
- At exhaustiveness 1024: GPU finds better affinity (-13.92 vs -13.20) and better RMSD (0.23 Å vs 0.67 Å)
- At exhaustiveness 4096: Both find near-native poses (~0.25 Å), with GPU slightly better affinity (-14.13 vs -14.07)
- GPU throughput: **4,000-18,500 poses/sec** depending on exhaustiveness (higher = better utilization)

The parallel BFGS approach benefits from massively parallel local optimization from many random starting poses, which with sufficient exhaustiveness can find excellent near-native poses.
