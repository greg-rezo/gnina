#!/bin/bash
# Incremental build script using Docker volume for build cache
# First run builds everything, subsequent runs only rebuild changed files
#
# Usage: ./build_docker_incremental.sh [--push POD_NAME]
#
# Builds both gnina and gninacheck binaries to build-docker/
# With --push, also copies the gnina binary to the specified K8s pod

set -e

# Parse arguments
PUSH_POD=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_POD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--push POD_NAME]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_IMAGE="gnina-build-base"
VOLUME_NAME="gnina-build-cache"
OUTPUT_DIR="$SCRIPT_DIR/build-docker"
LOCK_DIR="$SCRIPT_DIR/.build.lock.d"

# Acquire exclusive lock to prevent concurrent builds (portable mkdir-based lock)
acquire_lock() {
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        echo "Another build is already running. Waiting..."
        sleep 2
    done
    # Clean up lock on exit
    trap "rmdir '$LOCK_DIR' 2>/dev/null" EXIT
}
acquire_lock
echo "Build lock acquired"

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

# Base volume mounts - mount entire source directories
MOUNTS=(
    -v "$VOLUME_NAME:/gnina/build"
    -v "$SCRIPT_DIR/gninasrc:/gnina/src/gninasrc:ro"
    -v "$SCRIPT_DIR/test:/gnina/src/test:ro"
    -v "$SCRIPT_DIR/build_linux.sh:/gnina/build_linux.sh:ro"
)

echo "=== Running incremental build ==="

# Run the build using the standalone build script
docker run --rm --platform linux/amd64 \
    "${MOUNTS[@]}" \
    "$BASE_IMAGE" \
    /gnina/build_linux.sh

# Extract the built binaries
mkdir -p "$OUTPUT_DIR"
echo ""
echo "=== Extracting binaries ==="

# Create a temporary container to copy from the volume
CONTAINER_ID=$(docker run -d --platform linux/amd64 -v "$VOLUME_NAME:/gnina/build" "$BASE_IMAGE" sleep 10)

docker cp "$CONTAINER_ID:/gnina/build/gninasrc/gnina" "$OUTPUT_DIR/gnina" 2>/dev/null || \
docker cp "$CONTAINER_ID:/gnina/build/bin/gnina" "$OUTPUT_DIR/gnina" 2>/dev/null || \
echo "Warning: Could not find gnina binary"

docker cp "$CONTAINER_ID:/gnina/build/test/gnina/gninacheck" "$OUTPUT_DIR/gninacheck" 2>/dev/null || \
echo "Warning: Could not find gninacheck binary"

docker rm -f "$CONTAINER_ID" > /dev/null

echo ""
echo "=== Build complete ==="
echo "Binaries: $OUTPUT_DIR/gnina, $OUTPUT_DIR/gninacheck"

# Push to K8s pod if requested
if [ -n "$PUSH_POD" ]; then
    echo ""
    echo "=== Pushing to K8s pod: $PUSH_POD ==="
    kubectl cp "$OUTPUT_DIR/gnina" "$PUSH_POD:/gnina/gnina" -n development
    echo "Binary pushed to $PUSH_POD:/gnina/gnina"
fi
