#!/bin/bash
# Simple build script using gnina's official Docker image as base
# This is more reliable as it uses the same build environment as gnina developers
#
# Usage: ./build_docker_simple.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="gnina-custom"
IMAGE_TAG="latest"
OUTPUT_DIR="$SCRIPT_DIR/build-docker"

echo "=== Building gnina using official gnina base image ==="
echo "Source directory: $SCRIPT_DIR"
echo ""

# Build the Docker image
echo "=== Building Docker image (this may take a while on first run) ==="
docker build -f Dockerfile.gnina-base -t "${IMAGE_NAME}:${IMAGE_TAG}" .

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract the built binary
echo ""
echo "=== Extracting built binaries ==="
CONTAINER_ID=$(docker create "${IMAGE_NAME}:${IMAGE_TAG}")
docker cp "$CONTAINER_ID:/gnina/src/build/bin/gnina" "$OUTPUT_DIR/gnina"
docker rm "$CONTAINER_ID"

echo ""
echo "=== Build complete ==="
echo "Binary: $OUTPUT_DIR/gnina"
echo ""
echo "To test locally (requires nvidia-docker on Linux with GPU):"
echo "  docker run --gpus all -v \$(pwd):/data ${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "    ./build/bin/gnina -r /data/receptor.pdb -l /data/ligand.sdf --help"
echo ""
echo "To copy to a remote GPU machine and run:"
echo "  scp -r $OUTPUT_DIR user@gpu-machine:/path/to/gnina"
echo "  ssh user@gpu-machine 'cd /path/to/gnina && ./gnina --help'"
