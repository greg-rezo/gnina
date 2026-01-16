#!/bin/bash
# Wrapper script to build gnina on a K8s GPU pod
# Copies source files, then runs build_linux.sh on the pod
#
# Usage: ./build_on_pod.sh [POD_NAME] [OPTIONS]
#
# Arguments:
#   POD_NAME     Name of the K8s pod (default: gpu-gnina-build-t5k2w8)
#
# Options:
#   -j N         Use N parallel jobs (default: 16)
#   --clean      Do a clean build (remove build directory first)
#   --no-copy    Skip copying source files (just run build)
#
# Example:
#   ./build_on_pod.sh gpu-gnina-build-abc123 -j 16 --clean

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
POD_NAME="${GNINA_BUILD_POD:-gpu-gnina-build-x7k2m9}"
NAMESPACE="development"
JOBS=16
CLEAN=""
COPY_FILES=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -j)
            JOBS="$2"
            shift 2
            ;;
        --clean)
            CLEAN="--clean"
            shift
            ;;
        --no-copy)
            COPY_FILES=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [POD_NAME] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  POD_NAME     Name of the K8s pod (default: $POD_NAME)"
            echo ""
            echo "Options:"
            echo "  -j N         Use N parallel jobs (default: $JOBS)"
            echo "  --clean      Do a clean build"
            echo "  --no-copy    Skip copying source files"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            POD_NAME="$1"
            shift
            ;;
    esac
done

echo "=== GNINA Build on Pod ==="
echo "Pod:   $POD_NAME"
echo "Jobs:  $JOBS"
echo "Clean: ${CLEAN:-no}"
echo ""

if [ "$COPY_FILES" = true ]; then
    echo "=== Syncing source files to pod via rsync ==="

    # Ensure rsync is installed on pod
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- which rsync >/dev/null 2>&1 || {
        echo "Installing rsync on pod..."
        kubectl exec "$POD_NAME" -n "$NAMESPACE" -- apt-get update -qq
        kubectl exec "$POD_NAME" -n "$NAMESPACE" -- apt-get install -y -qq rsync
    }

    # Create wrapper script for rsync to use kubectl exec as transport
    # rsync calls: <rsh> <host> <command...>
    # We ignore <host> and pass <command...> to kubectl exec
    KRSYNC="/tmp/krsync-$$.sh"
    cat > "$KRSYNC" << EOF
#!/bin/bash
shift  # skip the dummy host argument
kubectl exec -i $POD_NAME -n $NAMESPACE -- \$@
EOF
    chmod +x "$KRSYNC"

    # Make ALL destination files writable (needed for rsync to update them)
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- chmod -R 777 /gnina/src/ 2>/dev/null || true

    # rsync using kubectl exec as remote shell (dummy host required by rsync syntax)
    # Use --relative to preserve path structure (test/gnina -> test/gnina, not gnina)
    # Use --delete to remove files that don't exist locally (like test_gpucode.cpp)
    # Use --checksum to force content comparison (timestamps unreliable over kubectl)
    rsync -avR --progress --delete --checksum \
        --exclude='*.o' \
        --exclude='*.a' \
        --exclude='.git' \
        --exclude='build' \
        --exclude='build-docker' \
        --exclude='__pycache__' \
        --rsync-path="rsync" \
        -e "$KRSYNC" \
        CMakeLists.txt gninasrc test build_linux.sh \
        pod:/gnina/src/

    rm -f "$KRSYNC"

    # Move build script to correct location
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- mv /gnina/src/build_linux.sh /gnina/build_linux.sh 2>/dev/null || true
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- chmod +x /gnina/build_linux.sh

    echo "Source files synced"
    echo ""
fi

echo "=== Running build on pod ==="
kubectl exec "$POD_NAME" -n "$NAMESPACE" -- /gnina/build_linux.sh -j "$JOBS" $CLEAN

echo ""
echo "=== Build complete ==="
echo ""
echo "To copy binary locally:"
echo "  kubectl cp $POD_NAME:/gnina/build/bin/gnina ./build-docker/gnina -n $NAMESPACE"
echo ""
echo "To run gnina on the pod:"
echo "  kubectl exec -it $POD_NAME -n $NAMESPACE -- /gnina/build/bin/gnina --help"
