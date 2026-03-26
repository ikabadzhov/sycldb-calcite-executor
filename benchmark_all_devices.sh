#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/benchmark-env.sh"

# ALL target devices. The client reads the SSB suite definition itself.
DEVICES=("cuda:0;opencl:0" "hip:0;opencl:0" "level_zero:0;opencl:0" "opencl:0")
DEVICE_NAMES=("NVIDIA_L40S" "AMD_MI210" "Intel_Flex_170" "Intel_Xeon_CPU")

ensure_benchmark_env

RESULTS_FILE="benchmark_results_final.csv"
rm -f "$RESULTS_FILE"

for i in "${!DEVICES[@]}"; do
    DEV=${DEVICES[$i]}
    DEV_NAME=${DEVICE_NAMES[$i]}
    echo "========================================"
    echo "Testing Device: $DEV_NAME ($DEV)"
    echo "========================================"

    export ONEAPI_DEVICE_SELECTOR="$DEV"
    CLIENT_ARGS=(
        --benchmark-ssb
        --suite benchmark/ssb_queries.txt
        --results "$RESULTS_FILE"
        --device-name "$DEV_NAME"
    )
    if [[ -f "$RESULTS_FILE" ]]; then
        CLIENT_ARGS+=(--append-results)
    fi

    ./client "${CLIENT_ARGS[@]}"
    sleep 1
done

echo "Comprehensive Benchmark complete. Results saved to $RESULTS_FILE"
