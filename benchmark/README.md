# SYCLDB Performance Validation & Profiling Suite

This directory contains the finalized, hardened setup for conducting 100-iteration performance benchmarks and hardware-specific profiling across **NVIDIA L40S**, **AMD MI210**, and **Intel Flex 170**.

## Directory Contents

- **`harness.cpp`**: Core C++ benchmarking engine. Hardened for non-dense keys and multi-scale (SF40/SF100).
- **`run_bench.sh`**: Automated script to run the full 1800-run performance matrix.
- **`profile_hardware.sh`**: Unified profiling utility for NCU (NVIDIA), ROCProf (AMD), and VTune (Intel).
- **`plot_results.py`**: Generates publication-ready 4-subplot performance matrices.
- **`amd_metrics.txt`**: Metric definitions for AMD ROCProf.

## Quick Start: Benchmarking

To run the full 100-iteration suite across all available hardware:

```bash
chmod +x run_bench.sh
./run_bench.sh
```

This will generate `results.csv`. After completion, generate the plots:

```bash
python3 plot_results.py
```

## Quick Start: Profiling

To profile a specific configuration (e.g., NVIDIA, Q1.1, Fused):

```bash
chmod +x profile_hardware.sh
./profile_hardware.sh nvidia 1 fused
```

Profiling results will be saved in the `profiling/` sub-directory.

## Hardware Configurations
- **NVIDIA L40S**: SF100 scale.
- **AMD MI210**: SF100 scale.
- **Intel Flex 170**: SF40 scale (optimized for 16GB VRAM).

## Dependencies
- **AdaptiveCpp**: Expected at `/media/ACPP/AdaptiveCpp-25.10.0/install/bin/acpp`.
- **SSB Data**: Expected at `/media/ssb/s100_columnar/` and `/media/ssb/s40_columnar/`.
- **Python**: `pandas`, `seaborn`, `matplotlib`.
