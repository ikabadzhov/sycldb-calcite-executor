#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/benchmark-env.sh"
ensure_benchmark_env

RESULTS_FILE="benchmark_results_nvidia_staged.csv"
rm -f "$RESULTS_FILE"

cuda_device_count=$(sycl-ls --ignore-device-selectors | grep -c '^\[cuda:gpu\]')
if (( cuda_device_count < 1 )); then
    echo "No CUDA GPU devices found."
    exit 1
fi

SELECTORS=("cuda:0;opencl:0")
DEVICE_NAMES=("NVIDIA_1GPU")

if (( cuda_device_count >= 2 )); then
    SELECTORS+=("cuda:0;cuda:1;opencl:0")
    DEVICE_NAMES+=("NVIDIA_2GPU")
fi

for i in "${!SELECTORS[@]}"; do
    export ONEAPI_DEVICE_SELECTOR="${SELECTORS[$i]}"
    DEVICE_NAME="${DEVICE_NAMES[$i]}"

    echo "========================================"
    echo "Testing Stage: $DEVICE_NAME ($ONEAPI_DEVICE_SELECTOR)"
    echo "========================================"

    CLIENT_ARGS=(
        --benchmark-ssb
        --suite benchmark/ssb_queries.txt
        --results "$RESULTS_FILE"
        --device-name "$DEVICE_NAME"
    )
    if [[ -f "$RESULTS_FILE" ]]; then
        CLIENT_ARGS+=(--append-results)
    fi

    ./client "${CLIENT_ARGS[@]}"
    sleep 1
done

echo "Staged NVIDIA benchmark complete. Results saved to $RESULTS_FILE"
