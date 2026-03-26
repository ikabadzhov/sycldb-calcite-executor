# SYCLDB Calcite Executor

This repository is organized around one supported product: reproducible execution of the transformed SSB benchmark queries through the DDOR execution path.

## Supported Workflow

- build the `client`
- connect to a Calcite planner on `localhost:5555`
- run one transformed SSB query or the full SSB benchmark suite
- emit repeatable CSV benchmark artifacts

The top-level application is now split into:

- `app/`: CLI, SQL loading, planner lifecycle, benchmark orchestration
- `runtime/`: queue discovery, allocator policy, base-table loading
- `executor/`: DDOR execution driver and result writing
- `benchmark/`: SSB suite definition
- `models/`, `operations/`, `kernels/`: engine internals

See [docs/ARCHITECTURE.md](/Users/eugenio/CLionProjects/sycldb-calcite-executor/docs/ARCHITECTURE.md) for the module map.

## Prerequisites

- Intel oneAPI with `icpx`
- Apache Thrift libraries and generated Calcite client bindings
- SSB columnar data at the path configured by `DATA_DIR` in [common.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/common.hpp)
- A Calcite planner listening on `localhost:5555`

## Build

```bash
make clean
make client
```

## Run One Query

```bash
export ONEAPI_DEVICE_SELECTOR="cuda:0;opencl:0"
./client queries/transformed/q11.sql
```

The client prints repetition timings and writes per-query performance logs for the active execution mode.

## Run The SSB Benchmark Suite

The suite definition lives in [benchmark/ssb_queries.txt](/Users/eugenio/CLionProjects/sycldb-calcite-executor/benchmark/ssb_queries.txt).

Run the per-device sweep:

```bash
bash benchmark_all_devices.sh
```

Run the staged NVIDIA sweep first:

```bash
bash benchmark_nvidia_staged.sh
```

That script runs `cuda:0;opencl:0` first, then `cuda:0;cuda:1;opencl:0` if a second NVIDIA GPU is present, and stops there.

Run the simultaneous multi-device sweep:

```bash
bash benchmark_all_devices_simultaneous.sh
```

Both scripts now invoke the supported benchmark mode directly instead of re-encoding the SSB query list themselves.

## Remote Workflow

For this repository, the intended validation flow is:

1. Edit locally.
2. Sync changed files to `eugenio@devenv2:/tmp/tmp.5M8Fsnsuez/sycldb-calcite-executor`.
3. Source `/home/eugenio/develop-env-vars.sh` remotely.
4. Build and run remotely.

Helper scripts:

- [scripts/sync-relative.sh](/Users/eugenio/CLionProjects/sycldb-calcite-executor/scripts/sync-relative.sh)
- [scripts/remote-build.sh](/Users/eugenio/CLionProjects/sycldb-calcite-executor/scripts/remote-build.sh)
- [remote_start_calcite.sh](/Users/eugenio/CLionProjects/sycldb-calcite-executor/remote_start_calcite.sh)
- [remote_run_benchmark.sh](/Users/eugenio/CLionProjects/sycldb-calcite-executor/remote_run_benchmark.sh)

## Notes

- The repository is optimized for the DDOR benchmarkable path, not for preserving every historical execution mode.
- Legacy/reference code under `sot/` remains separate from the supported benchmark flow.
