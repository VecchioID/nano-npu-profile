#!/bin/bash
# Nsight Compute profiling wrapper
# Usage: bash tools/run_ncu.sh <binary> [args]

NCU=/usr/local/cuda-13.0/bin/ncu
BINARY=$1
shift

if [ -z "$BINARY" ]; then
    echo "Usage: $0 <binary> [args]"
    exit 1
fi

echo "=== NCU Profiling: $BINARY ==="
$NCU --target-processes all \
     --launch-skip 0 \
     --launch-count 1 \
     --set basic \
     --print-summary per-kernel \
     "$BINARY" "$@"
