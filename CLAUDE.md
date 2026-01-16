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

### VM Setup (One-time)

1. Create a GCE VM with GPU support:
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

2. SSH into the VM (use IAP if direct SSH times out):
```bash
gcloud compute ssh gnina-profile-vm --zone=us-west1-a --tunnel-through-iap
```

3. Install NVIDIA drivers and container tools:
```bash
sudo dnf install -y nvidia-driver nvidia-container-toolkit
```

4. Configure podman for GPU access:
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

5. Authenticate with Artifact Registry:
```bash
# First run: gcloud auth login
TOKEN=$(gcloud auth print-access-token)
echo "$TOKEN" | sudo podman login -u oauth2accesstoken --password-stdin us-central1-docker.pkg.dev
```

6. Copy the gnina binary and test files to the VM:
```bash
gcloud compute scp ./gnina_bin gnina-profile-vm:~/ --zone=us-west1-a --tunnel-through-iap
gcloud compute scp ./test_files/*.pdb gnina-profile-vm:~/ --zone=us-west1-a --tunnel-through-iap
gcloud compute scp ./test_files/*.sdf gnina-profile-vm:~/ --zone=us-west1-a --tunnel-through-iap
```

### Running Profiling

#### Nsight Systems (Timeline Profiling)

Create a script `run_profile.sh`:
```bash
#!/bin/bash
cd /data
apt-get update
apt-get install -y nsight-systems-2025.5.2
nsys profile -o /data/gnina_profile /data/gnina_bin --gpu --exhaustiveness 1024 --bfgs_iterations 50 \
  -r 1dmp_rec.pdb -l 1dmp_rand_pos_fix.sdf --autobox_ligand 1dmp_rand_pos_fix.sdf -o /data/gnina_nsys_out.sdf
nsys stats /data/gnina_profile.nsys-rep
```

Run in container:
```bash
sudo podman run --rm --security-opt label=disable --device nvidia.com/gpu=all \
  -v /home/$(whoami):/data -w /data \
  us-central1-docker.pkg.dev/gke-test-421317/flyte/gnina-build-base:latest \
  /data/run_profile.sh
```

#### Nsight Compute (Detailed Kernel Analysis)

Create a script `run_ncu.sh`:
```bash
#!/bin/bash
cd /data
apt-get update
apt-get install -y nsight-compute-2025.4.0
ncu --target-processes all --kernel-name "bfgs_parallel_kernel" --set full \
  -o /data/gnina_ncu /data/gnina_bin --gpu --exhaustiveness 128 --bfgs_iterations 50 \
  -r 1dmp_rec.pdb -l 1dmp_rand_pos_fix.sdf --autobox_ligand 1dmp_rand_pos_fix.sdf -o /data/gnina_ncu_out.sdf
```

Then analyze:
```bash
ncu --import /data/gnina_ncu.ncu-rep --print-summary per-kernel
```

### Downloading Profile Files

```bash
gcloud compute scp gnina-profile-vm:~/gnina_profile.nsys-rep ./ --zone=us-west1-a --tunnel-through-iap
gcloud compute scp gnina-profile-vm:~/gnina_ncu.ncu-rep ./ --zone=us-west1-a --tunnel-through-iap
```

Open `.nsys-rep` files with Nsight Systems GUI and `.ncu-rep` files with Nsight Compute GUI.

### Key Metrics to Look For

- **Grid Size**: Should have multiple blocks (at least 58 for L4 GPU with 58 SMs)
- **Achieved Occupancy**: Target >50%
- **Warp Divergence**: Avg. Active Threads Per Warp should be close to 32
- **SM Busy**: Target >50%
- **L1/L2 Hit Rate**: Higher is better, 95%+ is good

### Container Image

Uses: `us-central1-docker.pkg.dev/gke-test-421317/flyte/gnina-build-base:latest`

This image contains:
- CUDA 12.1
- libtorch (compatible with gnina build)
- Required dependencies for gnina

### Notes

- Use `--security-opt label=disable` to work around SELinux volume mount issues
- Use `--tunnel-through-iap` for gcloud SSH/SCP if direct SSH times out
- Profile with lower exhaustiveness (128-1024) for ncu to avoid very long runtimes
- Profile with higher exhaustiveness (7000-20000) for nsys to see real-world performance
