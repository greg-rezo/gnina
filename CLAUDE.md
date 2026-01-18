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

## GPU Profiling with NVIDIA Tools

### Quick Start: Profile bfgs_parallel_kernel

Use the `profile_on_vm.sh` script to profile the BFGS kernel's L1 cache hit rate:

```bash
./profile_on_vm.sh [exhaustiveness] [bfgs_iterations]

# Examples:
./profile_on_vm.sh 128 1    # Fast: 128 poses, 1 BFGS iteration
./profile_on_vm.sh 1024 5   # More comprehensive
```

The script:
- Syncs code to VM and runs NCU in a podman container
- Uses `--replay-mode kernel` for faster profiling
- Skips 6 kernel launches to profile `bfgs_parallel_kernel` (not setup kernels)
- Reports L1 cache hit rate (`l1tex__t_sector_hit_rate`)

**Important flags for gnina:**
- `--gpu` - Required to use GPU BFGS (otherwise uses CPU Monte Carlo)
- `--bfgs_iterations N` - Fewer iterations = faster NCU profiling

### Kernel Launch Order

NCU profiles kernels in launch order. For `--gpu` mode:
1. `generate_random_confs_kernel`
2-4. Thrust radix sort kernels (spatial sorting)
5. `compute_pose_cell_ids_kernel`
6. `reorder_confs_kernel`
7. **`bfgs_parallel_kernel`** ← target (use `--launch-skip 6`)

### VM Setup (One-time)

The VM `gnina-profile-vm` should already exist. If not:

```bash
gcloud compute instances create gnina-profile-vm \
  --zone=us-west1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=rocky-linux-8-optimized-gcp \
  --image-project=rocky-linux-cloud \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE
```

Then install drivers and authenticate:
```bash
gcloud compute ssh gnina-profile-vm --zone=us-west1-a --tunnel-through-iap
sudo dnf install -y nvidia-driver nvidia-container-toolkit
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
TOKEN=$(gcloud auth print-access-token)
echo "$TOKEN" | sudo podman login -u oauth2accesstoken --password-stdin us-central1-docker.pkg.dev
```

### Syncing Source Code to VM

```bash
gcloud compute scp --zone=us-west1-a --tunnel-through-iap \
  gninasrc/lib/*.cu gninasrc/lib/*.h \
  gnina-profile-vm:/home/greg_rezotx_com/gnina/gninasrc/lib/
```

Then rebuild in container:
```bash
gcloud compute ssh gnina-profile-vm --zone=us-west1-a --tunnel-through-iap --command="sudo podman run --rm --privileged --device nvidia.com/gpu=all \
  -v /home/greg_rezotx_com/gnina:/gnina \
  us-central1-docker.pkg.dev/gke-test-421317/flyte/gnina-build-base:latest \
  bash -c 'cd /gnina/build && ninja -j16'"
```

### Key Metrics

- **L1 cache hit rate**: Target >90% (spatial sorting achieved 93%)
- **Achieved Occupancy**: Target >50%
- **SM Busy**: Target >50%

### Notes

- Use `--tunnel-through-iap` if direct SSH times out
- Use `--bfgs_iterations 1` for fast NCU profiling
- Container image: `us-central1-docker.pkg.dev/gke-test-421317/flyte/gnina-build-base:latest`
