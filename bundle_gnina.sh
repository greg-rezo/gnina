#!/bin/bash
# Bundle gnina with all dependencies into a self-contained directory
# Uses patchelf to set RPATH so gnina finds libs relative to itself
set -e

OUTPUT_DIR="/Users/gregfriedland/src/external/gnina/build-docker"
BUNDLE_DIR="$OUTPUT_DIR/gnina-bundle"

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/lib"

VOLUME_NAME="gnina-build-cache"

docker run --rm --platform linux/amd64 \
    -v "$VOLUME_NAME:/gnina/build" \
    -v "$OUTPUT_DIR:/out" \
    gnina-build-base \
    bash -c '
        set -e
        BUNDLE=/out/gnina-bundle
        mkdir -p $BUNDLE/lib

        # Copy gnina binary
        cp /gnina/build/bin/gnina $BUNDLE/gnina

        # Function to copy a library and its dependencies
        copy_lib() {
            local lib="$1"
            local basename=$(basename "$lib")
            if [ -f "$lib" ] && [ ! -f "$BUNDLE/lib/$basename" ]; then
                cp "$lib" "$BUNDLE/lib/" 2>/dev/null || true
            fi
        }

        # Copy all libraries gnina needs
        for lib in $(ldd /gnina/build/bin/gnina 2>/dev/null | grep "=>" | awk "{print \$3}" | sort -u); do
            copy_lib "$lib"
        done

        # Copy libtorch libraries
        cp /opt/libtorch/lib/*.so* $BUNDLE/lib/ 2>/dev/null || true

        # Copy libmolgrid
        cp /usr/local/lib/libmolgrid.so* $BUNDLE/lib/ 2>/dev/null || true

        # Copy boost libraries
        cp /usr/lib/x86_64-linux-gnu/libboost*.so* $BUNDLE/lib/ 2>/dev/null || true

        # Copy openbabel
        cp /usr/lib/x86_64-linux-gnu/libopenbabel.so* $BUNDLE/lib/ 2>/dev/null || true

        # Copy other system libs
        cp /usr/lib/x86_64-linux-gnu/libhdf5*.so* $BUNDLE/lib/ 2>/dev/null || true
        cp /usr/lib/x86_64-linux-gnu/libjsoncpp*.so* $BUNDLE/lib/ 2>/dev/null || true
        cp /usr/lib/x86_64-linux-gnu/libprotobuf*.so* $BUNDLE/lib/ 2>/dev/null || true
        cp /usr/lib/x86_64-linux-gnu/libinchi*.so* $BUNDLE/lib/ 2>/dev/null || true
        cp /usr/lib/x86_64-linux-gnu/libsz*.so* $BUNDLE/lib/ 2>/dev/null || true
        cp /usr/lib/x86_64-linux-gnu/libaec*.so* $BUNDLE/lib/ 2>/dev/null || true

        # Use patchelf to set RPATH
        patchelf --set-rpath "\$ORIGIN/lib" $BUNDLE/gnina

        # Create wrapper script
        cat > $BUNDLE/run_gnina.sh << "EOF"
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"
exec "$SCRIPT_DIR/gnina" "$@"
EOF
        chmod +x $BUNDLE/run_gnina.sh

        echo "Bundle created with $(ls $BUNDLE/lib | wc -l) libraries"
        echo "Binary RPATH: $(patchelf --print-rpath $BUNDLE/gnina)"
    '

# Create tarball
echo "Creating tarball..."
tar -czf "$OUTPUT_DIR/gnina-portable.tar.gz" -C "$OUTPUT_DIR" gnina-bundle

echo ""
echo "=== Bundle complete ==="
echo "Directory: $BUNDLE_DIR"
echo "Tarball: $OUTPUT_DIR/gnina-portable.tar.gz"
echo ""
echo "Usage: Extract and run ./gnina-bundle/run_gnina.sh or ./gnina-bundle/gnina"
