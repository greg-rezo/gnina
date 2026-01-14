#!/bin/bash
# Run gnina CUDA unit tests on a GPU pod
#
# Usage: ./run_tests_on_gpu.sh <pod_id> [namespace]
#
# Prerequisites:
#   - kubectl configured with access to the cluster
#   - gninacheck binary built (run: ./build_docker_incremental.sh --tests)
#   - Pod must have CUDA GPU available
#
# Example:
#   ./run_tests_on_gpu.sh my-gpu-pod-abc123
#   ./run_tests_on_gpu.sh my-gpu-pod-abc123 my-namespace

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
if [[ -z "$1" ]]; then
    echo "Usage: $0 <pod_id> [namespace]"
    echo ""
    echo "Example:"
    echo "  $0 my-gpu-pod-abc123"
    echo "  $0 my-gpu-pod-abc123 my-namespace"
    exit 1
fi

POD_ID="$1"
NAMESPACE="${2:-default}"
NAMESPACE_ARG="-n $NAMESPACE"

# Paths
LOCAL_BINARY="$SCRIPT_DIR/build-docker/gninacheck"
REMOTE_DIR="/tmp/gnina-tests"

# Check if binary exists
if [[ ! -f "$LOCAL_BINARY" ]]; then
    echo "Error: gninacheck binary not found at $LOCAL_BINARY"
    echo "Build it first with: ./build_docker_incremental.sh --tests"
    exit 1
fi

echo "=== Running gnina CUDA tests on GPU pod ==="
echo "Pod: $POD_ID"
echo "Namespace: $NAMESPACE"
echo ""

# Create remote directory
echo "=== Setting up remote directory ==="
kubectl exec $NAMESPACE_ARG "$POD_ID" -- mkdir -p "$REMOTE_DIR"

# Copy the test binary
echo "=== Copying gninacheck binary ==="
kubectl cp $NAMESPACE_ARG "$LOCAL_BINARY" "$POD_ID:$REMOTE_DIR/gninacheck"

# Make executable
kubectl exec $NAMESPACE_ARG "$POD_ID" -- chmod +x "$REMOTE_DIR/gninacheck"

# Check GPU availability
echo ""
echo "=== Checking GPU availability ==="
kubectl exec $NAMESPACE_ARG "$POD_ID" -- nvidia-smi --query-gpu=name,memory.total --format=csv || {
    echo "Warning: nvidia-smi failed. GPU may not be available."
}

# Run the tests
echo ""
echo "=== Running CUDA unit tests ==="
echo "Running: $REMOTE_DIR/gninacheck --run_test=bfgs_parallel"
echo ""

kubectl exec $NAMESPACE_ARG "$POD_ID" -- \
    "$REMOTE_DIR/gninacheck" \
    --run_test=bfgs_parallel \
    --log_level=message \
    --log_sink="$REMOTE_DIR/test.log"

TEST_EXIT_CODE=$?

# Fetch and display log
echo ""
echo "=== Test log ==="
kubectl exec $NAMESPACE_ARG "$POD_ID" -- cat "$REMOTE_DIR/test.log" 2>/dev/null || echo "(no log file)"

# Cleanup (optional - comment out to keep files for debugging)
echo ""
echo "=== Cleanup ==="
kubectl exec $NAMESPACE_ARG "$POD_ID" -- rm -rf "$REMOTE_DIR"

echo ""
if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo "=== All tests PASSED ==="
else
    echo "=== Some tests FAILED (exit code: $TEST_EXIT_CODE) ==="
fi

exit $TEST_EXIT_CODE
