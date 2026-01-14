#!/bin/bash
# Incremental build script using Docker volume for build cache
# First run builds everything, subsequent runs only rebuild changed files
#
# Usage: ./build_docker_incremental.sh [--tests]
#   --tests    Build test binary (gninacheck) instead of gnina

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_IMAGE="gnina-build-base"
VOLUME_NAME="gnina-build-cache"
OUTPUT_DIR="$SCRIPT_DIR/build-docker"

# Parse arguments
BUILD_TESTS=false
if [[ "$1" == "--tests" ]]; then
    BUILD_TESTS=true
fi

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

# Base volume mounts (source files)
MOUNTS=(
    -v "$VOLUME_NAME:/gnina/build"
    -v "$SCRIPT_DIR/gninasrc/lib/bfgs_parallel.h:/gnina/src/gninasrc/lib/bfgs_parallel.h:ro"
    -v "$SCRIPT_DIR/gninasrc/lib/bfgs_parallel.cu:/gnina/src/gninasrc/lib/bfgs_parallel.cu:ro"
    -v "$SCRIPT_DIR/gninasrc/lib/user_opts.h:/gnina/src/gninasrc/lib/user_opts.h:ro"
    -v "$SCRIPT_DIR/gninasrc/lib/result_info.h:/gnina/src/gninasrc/lib/result_info.h:ro"
    -v "$SCRIPT_DIR/gninasrc/lib/result_info.cpp:/gnina/src/gninasrc/lib/result_info.cpp:ro"
    -v "$SCRIPT_DIR/gninasrc/lib/PDBQTUtilities.cpp:/gnina/src/gninasrc/lib/PDBQTUtilities.cpp:ro"
    -v "$SCRIPT_DIR/gninasrc/main/main.cpp:/gnina/src/gninasrc/main/main.cpp:ro"
    -v "$SCRIPT_DIR/gninasrc/pygnina/bindings.cpp:/gnina/src/gninasrc/pygnina/bindings.cpp:ro"
    -v "$SCRIPT_DIR/gninasrc/CMakeLists.txt:/gnina/src/gninasrc/CMakeLists.txt:ro"
)

# Add test file mounts if building tests
if $BUILD_TESTS; then
    MOUNTS+=(
        -v "$SCRIPT_DIR/test/gnina/test_bfgs_parallel.cu:/gnina/src/test/gnina/test_bfgs_parallel.cu:ro"
        -v "$SCRIPT_DIR/test/gnina/test_bfgs_parallel.h:/gnina/src/test/gnina/test_bfgs_parallel.h:ro"
        -v "$SCRIPT_DIR/test/gnina/test_runner.cpp:/gnina/src/test/gnina/test_runner.cpp:ro"
        -v "$SCRIPT_DIR/test/gnina/CMakeLists.txt:/gnina/src/test/gnina/CMakeLists.txt:ro"
    )
fi

# Set build target and cmake flags based on mode
if $BUILD_TESTS; then
    BUILD_TARGET="gninacheck"
    CMAKE_EXTRA="-DBUILD_TESTING=ON"
    echo "=== Running incremental build (tests) ==="
else
    BUILD_TARGET="gnina"
    CMAKE_EXTRA=""
    echo "=== Running incremental build ==="
fi

# Run the build
docker run --rm --platform linux/amd64 \
    "${MOUNTS[@]}" \
    "$BASE_IMAGE" \
    bash -c "
        cd /gnina/build

        # Always run cmake for tests (needs BUILD_TESTING=ON), otherwise only if no cache
        if $BUILD_TESTS || [ ! -f CMakeCache.txt ]; then
            echo 'Running cmake...'
            cmake /gnina/src \
                -DCMAKE_BUILD_TYPE=Release \
                $CMAKE_EXTRA \
                -DLIBMOLGRID_LIBRARY=/usr/local/lib/libmolgrid.so \
                -DLIBMOLGRID_INCLUDE=/usr/local/include \
                -DCMAKE_PREFIX_PATH=\"/opt/libtorch\" \
                -DCMAKE_CUDA_ARCHITECTURES=\"70;75;80;86;89\"
        fi

        echo 'Building $BUILD_TARGET...'
        make -j\$(nproc) $BUILD_TARGET

        echo 'Build complete!'
        if $BUILD_TESTS; then
            ls -la /gnina/build/test/gnina/
        fi
    "

# Extract the built binary
mkdir -p "$OUTPUT_DIR"
echo ""
echo "=== Extracting binary ==="

# Create a temporary container to copy from the volume
CONTAINER_ID=$(docker run -d --platform linux/amd64 -v "$VOLUME_NAME:/gnina/build" "$BASE_IMAGE" sleep 10)

if $BUILD_TESTS; then
    docker cp "$CONTAINER_ID:/gnina/build/test/gnina/gninacheck" "$OUTPUT_DIR/gninacheck" 2>/dev/null || \
    echo "Warning: Could not find gninacheck binary"
    BINARY_NAME="gninacheck"
else
    docker cp "$CONTAINER_ID:/gnina/build/gninasrc/gnina" "$OUTPUT_DIR/gnina" 2>/dev/null || \
    docker cp "$CONTAINER_ID:/gnina/build/bin/gnina" "$OUTPUT_DIR/gnina" 2>/dev/null || \
    echo "Warning: Could not find gnina binary"
    BINARY_NAME="gnina"
fi

docker rm -f "$CONTAINER_ID" > /dev/null

echo ""
echo "=== Build complete ==="
echo "Binary: $OUTPUT_DIR/$BINARY_NAME"
