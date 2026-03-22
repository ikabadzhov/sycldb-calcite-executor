#!/bin/bash

# Ensure we are in the right directory
cd "$(dirname "$0")"

mkdir -p logs

RESULTS_CSV="mordred_sf100_results_breakdown.csv"
echo "Query,DiskLoad_ms,GPULoad_ms,Kernel_ms" > $RESULTS_CSV

QUERIES=("q11" "q12" "q13" "q21" "q22" "q23" "q31" "q32" "q33" "q34" "q41" "q42" "q43")

for Q in "${QUERIES[@]}"; do
    echo "Running $Q..."
    
    # Run GPU
    echo "  GPU..."
    GPU_LOG="logs/${Q}_gpu_breakdown.log"
    ./bin/ssb/$Q --t=1 > "$GPU_LOG" 2>&1
    
    DISK_LOAD=$(grep "DISK_LOAD" "$GPU_LOG" | awk '{sum+=$2} END {print sum}')
    GPU_LOAD=$(grep "GPU_LOAD" "$GPU_LOG" | awk '{sum+=$2} END {print sum}')
    KERNEL_TIME=$(grep -oP '"time_query":\K[0-9.]+' "$GPU_LOG" | tail -n 1)
    
    MAJOR=${Q:1:1}
    MINOR=${Q:2:1}
    FMT_Q="Q${MAJOR}.${MINOR}"
    
    echo "$FMT_Q,$DISK_LOAD,$GPU_LOAD,$KERNEL_TIME" >> $RESULTS_CSV
    echo "  Results: Disk=${DISK_LOAD}ms, GPU=${GPU_LOAD}ms, Kernel=${KERNEL_TIME}ms"
done

echo "Done. Results saved to $RESULTS_CSV"
