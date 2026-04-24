#!/bin/bash
# benchmark/profile_hardware.sh
# Unified profiling script for NVIDIA, AMD, and Intel GPUs

if [ "$#" -lt 3 ]; then
    echo "Usage: ./profile_hardware.sh <device_type: nvidia|amd|intel> <query_id: 1|2> <impl: unfused|fused|hardcoded>"
    exit 1
fi

HW_TYPE=$1
QUERY_ID=$2
IMPL=$3
DEV_IDX=0
REPS=1

OUT_DIR="profiling/$HW_TYPE"
mkdir -p "$OUT_DIR"

# Ensure binary is compiled
if [ ! -f ./harness ]; then
    /media/ACPP/AdaptiveCpp-25.10.0/install/bin/acpp -O3 -std=c++20 --acpp-targets=generic harness.cpp -o harness
fi

case $HW_TYPE in
    nvidia)
        echo "Profiling NVIDIA..."
        export ACPP_VISIBILITY_MASK=cuda
        METRICS="dram__bytes.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum.per_cycle_elapsed,smsp__warps_active.avg.per_cycle_active,smsp__warps_eligible.avg.per_cycle_active,smsp__warp_issue_stalled.avg.per_cycle_active"
        ncu --metrics $METRICS --csv --log-file "$OUT_DIR/profile_nvidia_q${QUERY_ID}_${IMPL}.csv" \
            ./harness $DEV_IDX $QUERY_ID $IMPL $REPS
        ;;
    amd)
        echo "Profiling AMD..."
        export ACPP_VISIBILITY_MASK=hip
        rocprof -i amd_metrics.txt -o "$OUT_DIR/rocprof_q${QUERY_ID}_${IMPL}.csv" \
            ./harness $DEV_IDX $QUERY_ID $IMPL $REPS
        ;;
    intel)
        echo "Profiling Intel..."
        export ACPP_VISIBILITY_MASK=opencl
        DEV_IDX=1 # Intel GPU is usually at index 1 on OpenCL platform
        vtune -collect gpu-hotspots -result-dir "$OUT_DIR" -quiet ./harness $DEV_IDX $QUERY_ID $IMPL $REPS \
            || vtune -collect hotspots -result-dir "$OUT_DIR" -quiet ./harness $DEV_IDX $QUERY_ID $IMPL $REPS
        vtune -report summary -result-dir "$OUT_DIR" -format csv > "$OUT_DIR/vtune_q${QUERY_ID}_${IMPL}.csv"
        ;;
    *)
        echo "Unknown hardware type: $HW_TYPE"
        exit 1
        ;;
esac

echo "Profiling complete. Results in $OUT_DIR"
