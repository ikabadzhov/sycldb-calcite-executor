#!/bin/bash

# Configuration
QUERIES=("q11.sql" "q12.sql" "q13.sql" "q21.sql" "q22.sql" "q23.sql" "q31.sql" "q32.sql" "q33.sql" "q34.sql" "q41.sql" "q42.sql" "q43.sql")

# ALL Target Devices (including CPU to allow the cpu_queue to be created)
DEVICES=("cuda:0;opencl:0" "hip:0;opencl:0" "level_zero:0;opencl:0" "opencl:0")
DEVICE_NAMES=("NVIDIA_L40S" "AMD_MI210" "Intel_Flex_170" "Intel_Xeon_CPU")

# Environment
export LD_LIBRARY_PATH="/opt/intel/oneapi/tcm/1.3/lib:/opt/intel/oneapi/umf/0.10/lib:/opt/intel/oneapi/tbb/2022.1/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/pti/0.11/lib:/opt/intel/oneapi/mpi/2021.15/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.15/lib:/opt/intel/oneapi/mkl/2025.1/lib:/opt/intel/oneapi/ippcp/2025.1/lib/:/opt/intel/oneapi/ipp/2022.1/lib:/opt/intel/oneapi/dnnl/2025.1/lib:/opt/intel/oneapi/debugger/2025.1/opt/debugger/lib:/opt/intel/oneapi/dal/2025.4/lib:/opt/intel/oneapi/compiler/2025.1/opt/compiler/lib:/opt/intel/oneapi/compiler/2025.1/lib:/opt/intel/oneapi/ccl/2021.15/lib/"

RESULTS_FILE="benchmark_results_final.csv"
echo "Query,Device,AvgTime_ms" > $RESULTS_FILE

for i in "${!DEVICES[@]}"; do
    DEV=${DEVICES[$i]}
    DEV_NAME=${DEVICE_NAMES[$i]}
    echo "========================================"
    echo "Testing Device: $DEV_NAME ($DEV)"
    echo "========================================"
    
    export ONEAPI_DEVICE_SELECTOR="$DEV"
    
    for QUERY in "${QUERIES[@]}"; do
        echo "Running $QUERY on $DEV_NAME..."
        OUTPUT=$(./client queries/transformed/$QUERY 2>&1)
        
        # Check if client failed
        if [[ $? -ne 0 ]]; then
            echo "CLIENT FAILED for $QUERY on $DEV_NAME: $OUTPUT"
            AVG_TIME="ERROR"
        else
            # Extract average total (warm) time (skipping first rep)
            AVG_TIME=$(echo "$OUTPUT" | grep "Repetition" | awk 'NR>1 {sum+=$(NF-4); count+=1} END {if (count>0) print sum/count; else print "ERROR"}')
            if [[ "$AVG_TIME" == "ERROR" ]]; then
                 echo "PARSING FAILED for $QUERY on $DEV_NAME. OUTPUT: $OUTPUT"
            fi
        fi
        
        echo "Result: $AVG_TIME ms"
        echo "$QUERY,$DEV_NAME,$AVG_TIME" >> $RESULTS_FILE
        sleep 1
    done
done

echo "Comprehensive Benchmark complete. Results saved to $RESULTS_FILE"
