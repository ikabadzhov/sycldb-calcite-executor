#!/bin/bash
# benchmark/run_bench.sh
# Final setup for running 100-iteration benchmarks across hardware

ACPP_BIN=/media/ACPP/AdaptiveCpp-25.10.0/install/bin/acpp
$ACPP_BIN -O3 -std=c++20 --acpp-targets=generic harness.cpp -o harness

REPS=100
echo "Device,Query,Mode,Iteration,TimeMS" > results.csv

# NVIDIA (SF100)
echo "Running on NVIDIA L40S (SF100)..."
export SSB_PATH=/media/ssb/s100_columnar/
export ACPP_VISIBILITY_MASK=cuda
./harness 0 1 $REPS | grep ITER_RESULT | cut -d',' -f2- >> results.csv || echo "NVIDIA Failed"

# AMD (SF100)
echo "Running on AMD MI210 (SF100)..."
export SSB_PATH=/media/ssb/s100_columnar/
export ACPP_VISIBILITY_MASK=hip
./harness 0 1 $REPS | grep ITER_RESULT | cut -d',' -f2- >> results.csv || echo "AMD Failed"

# Intel (SF40)
echo "Running on Intel Flex 170 (SF40)..."
export SSB_PATH=/media/ssb/s40_columnar/
export ACPP_VISIBILITY_MASK=opencl
./harness 0 1 $REPS | grep ITER_RESULT | cut -d',' -f2- >> results.csv || echo "Intel Failed"

echo "Benchmark complete. Results saved to results.csv"
echo "You can now run: python3 plot_results.py"
