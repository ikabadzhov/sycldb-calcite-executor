# SYCLDB: GPU Acceleration & JIT Fusion Cookbook

This guide provides a comprehensive "recipe" for running SYCLDB on high-performance GPUs, applying advanced kernel optimizations, and understanding the performance impact of JIT Fusion.

---

## 🚀 1. How to Run on GPU

SYCLDB uses **AdaptiveCpp** to target heterogeneous hardware via a single "generic" JIT binary.

### Build and Launch
```bash
# Build the project using all available cores
make -j$(nproc)

# Run a single query on the default GPU (e.g. NVIDIA L40S)
./client --query queries/transformed/q11.sql

# Run the full SSB suite (Scale Factor 100)
./client --benchmark-ssb --suite benchmark/ssb_queries.txt --results ssb_results.csv
```

### Controlling Execution Model
You can toggle JIT Fusion using environment variables:
*   **Enable Fusion (Default):** `unset SYCLDB_DISABLE_FUSION`
*   **Disable Fusion (Modular Path):** `export SYCLDB_DISABLE_FUSION=1`

---

## 🛠️ 2. Core Optimizations

To achieve peak performance on NVIDIA (L40S) and AMD (MI210) hardware, the following optimizations have been baked into the JIT kernel templates:

### A. Memory Access & Aliasing (`__restrict__`)
In `SYCLDBContext`, all pointer members are marked with `__restrict__`. 
*   **Why:** This informs the compiler that column buffers do not overlap.
*   **Impact:** Enables common subexpression elimination across rows and allows the hardware to use the **L1 Read-Only Cache** (LDG instructions on NVIDIA), significantly increasing effective bandwidth.

### B. Aggressive Loop Inlining 
JIT-fused slots and helper functions are decorated with `__attribute__((always_inline))`.
*   **Why:** standard `inline` is just a hint; `always_inline` forces the JIT compiler to collapse the entire execution plan (Filter -> Join -> Aggregation) into a single optimized instruction stream.
*   **Impact:** Eliminates function call overhead and allows the compiler to optimize register allocation across different query operators.

### C. Managed Warp Divergence
Internal JIT filters use `if(pass)` checks rather than purely branchless logic.
*   **Why:** On modern high-end GPUs, the cost of a branch is lower than the cost of a global memory load. 
*   **Impact:** By checking `if(pass)`, the kernel skips memory loads for rows already filtered out, preserving memory bus bandwidth for active rows.

---

## 📊 3. Fusion Impact Analysis

Based on real-world benchmarking at **Scale Factor 100** (approx. 600 million rows), the impact of JIT Fusion in SYCLDB is transformative.

### Performance Comparison (NVIDIA L40S)

| Metric | Modular Execution (No Fusion) | JIT Fused Execution | Gain |
| :--- | :--- | :--- | :--- |
| **Q1.1 Kernel Time** | 57.3 ms | 39.9 ms | **+44%** |
| **Q1.2 Kernel Time** | 59.7 ms | 31.6 ms | **+89%** |
| **Q1.3 Kernel Time** | 59.7 ms | 31.0 ms | **+92%** |
| **JIT Overhead** | 0 ms | < 0.01 ms | Negligible |

### Why Fusion Wins
1.  **Reduced Global Memory Traffic:** Modular execution writes intermediate "flag" buffers to VRAM after every filter. JIT Fusion keeps the `pass` flag in a register throughout the entire query.
2.  **Kernel Launch Savings:** Instead of launching 3-5 separate kernels (Filter, Filter, Filter, Agg), JIT Fusion launches 1. This significantly reduces the Host-to-Device dispatch latency.
3.  **Instruction Specialization:** The JIT compiler can replace generic operator parameters with literal constants during compilation, enabling dead-code elimination and constant propagation.

---

## 💡 4. Real-World Execution Recipe

To get the most out of the system in production:

1.  **Warm-up:** Always run a small query first to trigger the initial JIT compilation and VRAM allocation. This prevents the "First Run Cold" latency (~10s) from affecting SLAs.
2.  **Check Residency:** Monitor VRAM residency during large runs. JIT Fusion reduces the need for temporary buffers, allowing larger datasets (SF100+) to fit comfortably in memory.
3.  **Validate Selectivity:** If a query has very low selectivity (most rows pass), the JIT fusion speedup is dominated by bandwidth. If selectivity is high (many rows filtered), the `if(pass)` optimization provides an additional 20-30% boost by skipping loads.
