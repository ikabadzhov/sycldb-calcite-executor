# Architecture

## Supported Product

This repository supports one primary workflow:

- execute transformed SSB queries through the DDOR engine
- run repeatable benchmark sweeps over the SSB suite
- use Apache Calcite as the planner front-end

The repository is no longer organized around multiple equally supported execution modes.

## Module Layout

- `app/`
  - command-line parsing
  - SQL loading
  - planner client lifecycle
  - top-level query and benchmark orchestration
- `runtime/`
  - queue discovery
  - device ordering
  - allocator sizing and reset policy
  - base-table loading and runtime summaries
- `executor/`
  - DDOR plan execution
  - final result materialization and writing
- `benchmark/`
  - SSB suite definition
- `models/`
  - table, column, segment, and transient-table state
- `kernels/`
  - SYCL kernel definitions
- `operations/`
  - planner preprocessing and allocator primitives

## Control Flow

1. `main.cpp` delegates to `app::run_app(...)`.
2. `app/` parses arguments and loads SQL or the SSB suite definition.
3. `runtime/` discovers execution devices, builds queues, sizes allocators, and loads the base tables.
4. `app/` opens the Calcite planner connection and requests a `PlanResult`.
5. `executor/` runs the DDOR execution flow against `models/` and `kernels/`.
6. `app/` either prints per-query repetitions or emits benchmark CSV rows.

## Reproducibility Surface

The benchmarkable workflow depends on:

- `DATA_DIR` from `common.hpp`
- a reachable Calcite planner at `localhost:5555` by default
- the transformed SSB queries in `queries/transformed/`
- the suite definition in `benchmark/ssb_queries.txt`
- the remote build/run workflow documented in the helper scripts

## Notes

- The segmented residency and synchronization logic still lives in `models/` because it is core engine behavior, not benchmark orchestration.
- The legacy single-queue execution path is intentionally no longer the supported top-level path.
