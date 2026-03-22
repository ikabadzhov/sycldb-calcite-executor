# SYCLDB Calcite Executor

A high-performance query execution engine leveraging SYCL for multi-vendor GPU acceleration (NVIDIA, AMD, Intel) and Intel Xeon CPUs. This project integrates with Apache Calcite for plan parsing and focuses on Star Schema Benchmark (SSB) performance at Scale Factor 100 (SF100).

## 🚀 Key Features

- **Cross-Architecture Support**: Native backend support for NVIDIA (CUDA), AMD (HIP), and Intel (Level Zero).
- **On-Demand Columnar Orchestration**: Intelligent data placement that minimizes host-to-device transfers by moving only required data segments.
- **Detailed Performance Instrumentation**: Built-in tracking for kernel execution, on-demand data loading, and Thrift/Parse overheads.
- **Comprehensive Visualization**: Automated suite for comparing results against world-class engines like DuckDB, HeavyAI, and Mordred.

## 🛠 Prerequisites

- **Intel oneAPI Base Toolkit**: Specifically `icpx` (SYCL compiler) and `oneapi/dpl`.
- **Apache Thrift**: Used for communicating with the Calcite-based planner.
- **SQL Source**: Calcite Server running to provide the relational plans.

## 🏗 Build Instructions

Use the provided Makefile to compile the multi-backend client:

```bash
make clean
make client
```

This will produce a `client` binary capable of targeting any SYCL-supported device.

## 📊 Benchmarking & Reproduction Guide

To reproduce the results visualized in the comprehensive performance reports, follow the device-specific instructions below.

### 1. SYCLDB Hardware Targets (Multi-Vendor)
SYCLDB uses the `ONEAPI_DEVICE_SELECTOR` for targeting. Ensure you perform 5 repetitions for each query (warm results skip the first repetition).

- **NVIDIA L40S GPU**:
  ```bash
  export ONEAPI_DEVICE_SELECTOR="cuda:0"
  ./client queries/transformed/q11.sql
  ```
- **AMD MI210 GPU**:
  ```bash
  export ONEAPI_DEVICE_SELECTOR="hip:0"
  ./client queries/transformed/q11.sql
  ```
- **Intel Flex 170 GPU**:
  ```bash
  export ONEAPI_DEVICE_SELECTOR="level_zero:0"
  ./client queries/transformed/q11.sql
  ```
- **Intel Xeon CPU**:
  ```bash
  export ONEAPI_DEVICE_SELECTOR="opencl:cpu"
  ./client queries/transformed/q11.sql
  ```

Alternatively, use the automated suite: `bash benchmark_all.sh`.

### 2. External Engine Baselines

#### **Mordred (Fused GPU/CPU Kernels)**
Mordred is located in the `sot/Mordred` subdirectory.
1. `cd sot/Mordred`
2. Run the automated benchmark script: `bash run_all.sh`
3. Results are logged to `logs/q*_gpu.log` and summarized in `mordred_sf100_results_breakdown.csv`.

#### **DuckDB (Vectorized CPU)**
DuckDB results are obtained using its internal profiling and benchmarking suite.
1. Run queries via DuckDB Python or CLI:
   ```python
   import duckdb
   duckdb.query("PRAGMA benchmark(q11, iterations=10)")
   ```
2. Extracted breakdowns are saved to `duckdb_breakdown_final.csv`.

#### **HeavyAI (GPU Analytical Database)**
**Note**: HeavyAI is executed within a dedicated **Docker container**.
1. Start the container: `sot/HeavyAI/start_heavyai.sh`
2. Execute SQL via the internal tool:
   ```bash
   /heavydb/bin/heavysql -u admin -p <pass> heavydb < queries/q11.sql
   ```
3. Use the "Execution time" metric reported by the tool (excluding initial data load).

## 📈 Data Visualization

After data collection, generate the unified comparative plot (log-scale):

1. **Prerequisites**: Ensure `benchmark_results_final.csv`, `duckdb_breakdown_final.csv`, and `mordred_sf100_results_breakdown.csv` are in the project root.
2. **Execute**:
   ```bash
   python3 visualize_final_breakdown_full.py
   ```
3. **Artifact**: The final plot is saved as `comparison_breakdown_full_sf100.png`.

---

## 📝 Recent Development Log / Major Changes

Below is a summary of the critical architectural changes implemented in this branch:

### 1. Multi-Backend SYCL Integration
Standardized the build system to support diverse backend targets (`nvptx64-nvidia-cuda`, `amdgcn-amd-amdhsa`, and `spir64/level_zero`). This allows the same code to run natively across NVIDIA L40S, AMD MI210, and Intel Flex 170 GPUs.

### 2. On-Demand Columnar Data Movement
Implemented a segment-aware data loader within the execution pipeline. Instead of pre-loading entire tables, the engine now dynamically identifies required columns from the Calcite plan and moves only the necessary segments to the GPU, significantly improving "cold" query performance.

### 3. High-Fidelity Instrumentation
Added detailed timing hooks into `ddor_execute_result` to isolate:
- **Engine Kernel Execution**: Pure GPU computation time.
- **Data Placement Overheads**: Transfer times and segment allocation logic.
- **Other (Parse/Optimizer)**: Time spent on planning and relational algebra mapping.

### 4. Memory Management Hardening
Introduced vendor-specific allocation strategies, particularly for Intel memory models, to solve stability issues with SF100-sized datasets on integrated/discrete Intel backends.

### 5. Cross-Engine Comparative Suite
Developed a unified visualization framework that normalizes time metrics from external engines (DuckDB's vectorized CPU execution, HeavyAI's GPU runtime, and Mordred's fused kernels) for an "oranges-to-oranges" component-level comparison.
