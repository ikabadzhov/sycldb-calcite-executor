#!/bin/bash

set -eo pipefail

source /home/eugenio/develop-env-vars.sh

export LD_LIBRARY_PATH="/opt/intel/oneapi/tcm/1.3/lib:/opt/intel/oneapi/umf/0.10/lib:/opt/intel/oneapi/tbb/2022.1/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/pti/0.11/lib:/opt/intel/oneapi/mpi/2021.15/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.15/lib:/opt/intel/oneapi/mkl/2025.1/lib:/opt/intel/oneapi/ippcp/2025.1/lib/:/opt/intel/oneapi/ipp/2022.1/lib:/opt/intel/oneapi/dnnl/2025.1/lib:/opt/intel/oneapi/debugger/2025.1/opt/debugger/lib:/opt/intel/oneapi/dal/2025.4/lib:/opt/intel/oneapi/compiler/2025.1/opt/compiler/lib:/opt/intel/oneapi/compiler/2025.1/lib:/opt/intel/oneapi/ccl/2021.15/lib/"
export ONEAPI_DEVICE_SELECTOR="cuda:0;hip:0;level_zero:0;opencl:0"

QUERIES=(
  "q11.sql" "q12.sql" "q13.sql"
  "q21.sql" "q22.sql" "q23.sql"
  "q31.sql" "q32.sql" "q33.sql" "q34.sql"
  "q41.sql" "q42.sql" "q43.sql"
)

RESULTS_FILE="benchmark_results_all_devices_simultaneous.csv"
printf "Query,Device,AvgTime_ms\n" > "$RESULTS_FILE"

for QUERY in "${QUERIES[@]}"; do
    echo "Running $QUERY..."
    set +e
    OUTPUT=$(./client "queries/transformed/$QUERY" 2>&1)
    STATUS=$?
    set -e

    if [[ $STATUS -ne 0 ]]; then
        AVG_TIME="ERROR"
        echo "FAILED: $QUERY"
    else
        AVG_TIME=$(printf "%s\n" "$OUTPUT" | awk '/Repetition/ { if (++count > 1) { sum += $(NF-4); warm += 1 } } END { if (warm > 0) print sum / warm; else print "ERROR" }')
        if [[ -z "$AVG_TIME" ]]; then
            AVG_TIME="ERROR"
        fi
    fi

    printf "%s,ALL_SIMULTANEOUS,%s\n" "$QUERY" "$AVG_TIME" >> "$RESULTS_FILE"
    echo "$QUERY => $AVG_TIME"
    sleep 1
done

echo "Benchmark complete. Results saved to $RESULTS_FILE"
