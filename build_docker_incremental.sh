#!/bin/bash
# Incremental build script using Docker volume for build cache
# First run builds everything, subsequent runs only rebuild changed files
#
# Usage: ./build_docker_incremental.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_IMAGE="gnina-build-base"
VOLUME_NAME="gnina-build-cache"
OUTPUT_DIR="$SCRIPT_DIR/build-docker"

# Check if base image exists, if not build it
if ! docker image inspect "$BASE_IMAGE" &> /dev/null; then
    echo "=== Building base image (one-time setup, ~15 min) ==="
    docker build -f Dockerfile.gnina-base-deps -t "$BASE_IMAGE" .
fi

# Create volume if it doesn't exist
if ! docker volume inspect "$VOLUME_NAME" &> /dev/null; then
    echo "=== Creating build cache volume ==="
    docker volume create "$VOLUME_NAME"
fi

echo "=== Running incremental build ==="
# Mount:
# - Source files from host (read-only for safety, except our modified files)
# - Build volume for persistent build artifacts
# - Our modified source files

docker run --rm --platform linux/amd64 \
    -v "$VOLUME_NAME:/gnina/build" \
    -v "$SCRIPT_DIR/gninasrc/lib/bfgs_parallel.h:/gnina/src/gninasrc/lib/bfgs_parallel.h:ro" \
    -v "$SCRIPT_DIR/gninasrc/lib/bfgs_parallel.cu:/gnina/src/gninasrc/lib/bfgs_parallel.cu:ro" \
    -v "$SCRIPT_DIR/gninasrc/lib/user_opts.h:/gnina/src/gninasrc/lib/user_opts.h:ro" \
    -v "$SCRIPT_DIR/gninasrc/main/main.cpp:/gnina/src/gninasrc/main/main.cpp:ro" \
    "$BASE_IMAGE" \
    bash -c '
        cd /gnina/build

        # If CMakeCache exists, just run make; otherwise run cmake first
        if [ ! -f CMakeCache.txt ]; then
            echo "First build - running cmake..."
            cmake /gnina/src \
                -DCMAKE_BUILD_TYPE=Release \
                -DLIBMOLGRID_LIBRARY=/usr/local/lib/libmolgrid.so \
                -DLIBMOLGRID_INCLUDE=/usr/local/include \
                -DCMAKE_PREFIX_PATH="/opt/libtorch" \
                -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89"
        fi

        echo "Building gnina..."
        make -j$(nproc) gnina

        echo "Build complete!"
    '

# Extract the built binary
mkdir -p "$OUTPUT_DIR"
echo ""
echo "=== Extracting binary ==="

# Create a temporary container to copy from the volume
CONTAINER_ID=$(docker run -d --platform linux/amd64 -v "$VOLUME_NAME:/gnina/build" "$BASE_IMAGE" sleep 10)
docker cp "$CONTAINER_ID:/gnina/build/gninasrc/gnina" "$OUTPUT_DIR/gnina" 2>/dev/null || \
docker cp "$CONTAINER_ID:/gnina/build/bin/gnina" "$OUTPUT_DIR/gnina" 2>/dev/null || \
echo "Warning: Could not find gnina binary"
docker rm -f "$CONTAINER_ID" > /dev/null

echo ""
echo "=== Build complete ==="
echo "Binary: $OUTPUT_DIR/gnina"
