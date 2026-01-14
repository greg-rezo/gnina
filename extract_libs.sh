#!/bin/bash
# Extract gnina binary and all required libraries
set -e

OUTPUT_DIR="/Users/gregfriedland/src/external/gnina/build-docker"
mkdir -p "$OUTPUT_DIR/libs"

docker run --rm --platform linux/amd64 \
    -v gnina-build-cache:/gnina/build \
    -v "$OUTPUT_DIR:/out" \
    gnina-build-base \
    bash -c '
        mkdir -p /out/libs

        # Copy gnina binary
        cp /gnina/build/gninasrc/gnina /out/gnina

        # Copy all required shared libraries
        for lib in $(ldd /gnina/build/gninasrc/gnina 2>/dev/null | grep "=>" | awk "{print \$3}" | sort -u); do
            if [ -f "$lib" ]; then
                cp "$lib" /out/libs/ 2>/dev/null || true
            fi
        done

        # Copy libtorch libraries
        cp /opt/libtorch/lib/*.so* /out/libs/ 2>/dev/null || true

        # Copy libmolgrid
        cp /usr/local/lib/libmolgrid.so* /out/libs/ 2>/dev/null || true

        # Copy boost libraries
        cp /usr/lib/x86_64-linux-gnu/libboost*.so* /out/libs/ 2>/dev/null || true

        # Copy openbabel libraries
        cp /usr/lib/x86_64-linux-gnu/libopenbabel.so* /out/libs/ 2>/dev/null || true
        cp /usr/lib/x86_64-linux-gnu/openbabel /out/libs/ -r 2>/dev/null || true

        # Copy other system libs that might be needed
        cp /usr/lib/x86_64-linux-gnu/libhdf5*.so* /out/libs/ 2>/dev/null || true
        cp /usr/lib/x86_64-linux-gnu/libjsoncpp*.so* /out/libs/ 2>/dev/null || true
        cp /usr/lib/x86_64-linux-gnu/libprotobuf*.so* /out/libs/ 2>/dev/null || true
        cp /usr/lib/x86_64-linux-gnu/libz*.so* /out/libs/ 2>/dev/null || true
        cp /usr/lib/x86_64-linux-gnu/libinchi*.so* /out/libs/ 2>/dev/null || true

        echo "Extracted libraries:"
        ls -la /out/libs/ | head -20
        echo "..."
        echo "Total: $(ls /out/libs/ | wc -l) libraries"
    '

echo "Libraries extracted to $OUTPUT_DIR/libs"
