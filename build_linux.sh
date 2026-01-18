#!/bin/bash
# Linux build script for gnina
# Run this directly on a Linux machine or GPU pod with dependencies installed
#
# Usage: ./build_linux.sh [OPTIONS]
#
# Options:
#   -j N         Use N parallel jobs (default: auto-detect)
#   --clean      Remove build directory and start fresh
#   --src DIR    Source directory (default: /gnina/src)
#   --build DIR  Build directory (default: /gnina/build)
#
# Environment:
#   Expects libtorch at /opt/libtorch and libmolgrid installed to /usr/local
#   Works with gnina-build-base Docker image or equivalent environment

set -e

# Defaults
JOBS=""
CLEAN=false
DEBUG=false
SRC_DIR="/gnina/src"
BUILD_DIR="/gnina/build"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -j)
            JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --src)
            SRC_DIR="$2"
            shift 2
            ;;
        --build)
            BUILD_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -j N         Use N parallel jobs (default: auto-detect)"
            echo "  --clean      Remove build directory and start fresh"
            echo "  --src DIR    Source directory (default: /gnina/src)"
            echo "  --build DIR  Build directory (default: /gnina/build)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-detect number of jobs if not specified
if [ -z "$JOBS" ]; then
    JOBS=$(nproc 2>/dev/null || echo 4)
fi

# Set up environment
export LD_LIBRARY_PATH=/opt/libtorch/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
export PATH="/usr/local/bin:/usr/lib/ccache:$PATH"

echo "=== GNINA Linux Build ==="
echo "Source:  $SRC_DIR"
echo "Build:   $BUILD_DIR"
echo "Jobs:    $JOBS"
echo ""

# Handle clean build
if [ "$CLEAN" = true ]; then
    echo "Cleaning build directory..."
    if [ -d "$BUILD_DIR" ]; then
        # Move to trash instead of rm -rf for safety
        TRASH_DIR="/tmp/gnina-build-trash-$$"
        mv "$BUILD_DIR" "$TRASH_DIR" 2>/dev/null || true
    fi
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Only run cmake if build.ninja doesn't exist (first run or after clean)
if [ ! -f build.ninja ]; then
    if [ "$DEBUG" = true ]; then
        echo "Running cmake with Ninja (DEBUG mode)..."
        cmake "$SRC_DIR" \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DBUILD_TESTING=ON \
            -DLIBMOLGRID_LIBRARY=/usr/local/lib/libmolgrid.so \
            -DLIBMOLGRID_INCLUDE=/usr/local/include \
            -DCMAKE_PREFIX_PATH="/opt/libtorch" \
            -DCMAKE_CUDA_ARCHITECTURES="89"
    else
        echo "Running cmake with Ninja..."
        cmake "$SRC_DIR" \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=ON \
            -DLIBMOLGRID_LIBRARY=/usr/local/lib/libmolgrid.so \
            -DLIBMOLGRID_INCLUDE=/usr/local/include \
            -DCMAKE_PREFIX_PATH="/opt/libtorch" \
            -DCMAKE_CUDA_ARCHITECTURES="89"
    fi
    echo ""
fi

echo "Building gnina with $JOBS parallel jobs..."
ninja -j"$JOBS" gnina

echo ""
echo "=== Build complete ==="
echo "Binaries:"
if [ -f "$BUILD_DIR/bin/gnina" ]; then
    echo "  $BUILD_DIR/bin/gnina"
elif [ -f "$BUILD_DIR/gninasrc/gnina" ]; then
    echo "  $BUILD_DIR/gninasrc/gnina"
fi
if [ -f "$BUILD_DIR/test/gnina/gninacheck" ]; then
    echo "  $BUILD_DIR/test/gnina/gninacheck"
fi
