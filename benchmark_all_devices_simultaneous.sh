#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/benchmark-env.sh"

ensure_benchmark_env
export ONEAPI_DEVICE_SELECTOR="cuda:0;hip:0;level_zero:0;opencl:0"

RESULTS_FILE="benchmark_results_all_devices_simultaneous.csv"
rm -f "$RESULTS_FILE"

./client \
    --benchmark-ssb \
    --suite benchmark/ssb_queries.txt \
    --results "$RESULTS_FILE" \
    --device-name ALL_SIMULTANEOUS

echo "Benchmark complete. Results saved to $RESULTS_FILE"
