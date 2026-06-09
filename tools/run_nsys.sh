#!/bin/bash
# Nsight Systems profiling wrapper
# Usage: bash tools/run_nsys.sh <binary> [args]

NSYS=/usr/local/cuda-13.0/bin/nsys
BINARY=$1
shift

if [ -z "$BINARY" ]; then
    echo "Usage: $0 <binary> [args]"
    exit 1
fi

OUTPUT=$(basename "$BINARY")_$(date +%Y%m%d_%H%M%S)

echo "=== NSYS Profiling: $BINARY -> ${OUTPUT}.nsys-rep ==="
$NSYS profile --trace=cuda,nvtx,osrt \
              --output="$OUTPUT" \
              --force-overwrite=true \
              "$BINARY" "$@"
echo "View with: nsys-ui ${OUTPUT}.nsys-rep"
