#!/bin/bash

# Ensure we are in the right directory
cd "$(dirname "$0")"

mkdir -p logs

RESULTS_CSV="mordred_sf100_results_new.csv"
echo "Query,CPU_ms,GPU_ms" > $RESULTS_CSV

QUERIES=("q11" "q12" "q13" "q21" "q22" "q23" "q31" "q32" "q33" "q34" "q41" "q42" "q43")

for Q in "${QUERIES[@]}"; do
    echo "Running $Q..."
    
    # Run CPU
    echo "  CPU..."
    CPU_LOG="logs/${Q}_cpu.log"
    ./bin/cpu/ssb/$Q --t=3 > "$CPU_LOG" 2>&1
    CPU_TIME=$(grep -oP '"time_query":\K[0-9.]+' "$CPU_LOG" | tail -n 1)
    
    # Run GPU
    echo "  GPU..."
    GPU_LOG="logs/${Q}_gpu.log"
    ./bin/ssb/$Q --t=3 > "$GPU_LOG" 2>&1
    GPU_TIME=$(grep -oP '"time_query":\K[0-9.]+' "$GPU_LOG" | tail -n 1)
    
    echo "$Q,$CPU_TIME,$GPU_TIME" >> $RESULTS_CSV
    echo "  Results: CPU=${CPU_TIME}ms, GPU=${GPU_TIME}ms"
done

echo "Done. Results saved to $RESULTS_CSV"
