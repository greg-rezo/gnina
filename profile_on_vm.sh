#!/bin/bash
# Profile GNINA BFGS kernel on VM using NCU
#
# Usage: ./profile_on_vm.sh [exhaustiveness] [bfgs_iterations]
#
# Example: ./profile_on_vm.sh 128 1

EXH=${1:-128}
BFGS_ITERS=${2:-1}

echo "Profiling with exhaustiveness=$EXH, bfgs_iterations=$BFGS_ITERS"

gcloud compute ssh gnina-profile-vm --zone=us-west1-a --tunnel-through-iap --command="sudo podman run --rm --privileged --device nvidia.com/gpu=all \
  -v /home/greg_rezotx_com/gnina:/gnina \
  -v /home/greg_rezotx_com:/data \
  us-central1-docker.pkg.dev/gke-test-421317/flyte/gnina-build-base:latest \
  bash -c 'ncu \
    --target-processes all \
    --replay-mode kernel \
    --metrics l1tex__t_sector_hit_rate \
    --launch-skip 6 \
    --launch-count 1 \
    /gnina/build/bin/gnina \
      --gpu \
      --cnn_scoring none \
      --exhaustiveness $EXH \
      --bfgs_iterations $BFGS_ITERS \
      --seed 42 \
      -r /data/1dmp_rec.pdb \
      -l /data/1dmp_rand_pos_fix.sdf \
      --autobox_ligand /data/1dmp_rand_pos_fix.sdf \
      -o /data/test_ncu.sdf \
    2>&1'"
