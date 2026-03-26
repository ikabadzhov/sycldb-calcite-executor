# Async Refactor Change Log

This document records the async execution changes implemented in the current branch of `sycldb-calcite-executor`, the validation performed on `devenv2`, and the main remaining gaps.

## Scope

The work focused on the active DDOR execution path in:

- [main.cpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/main.cpp)
- [models/transient_table.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/transient_table.hpp)
- [models/models.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/models.hpp)
- [kernels/common.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/kernels/common.hpp)
- [kernels/selection.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/kernels/selection.hpp)
- [kernels/join.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/kernels/join.hpp)
- [operations/memory_manager.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/operations/memory_manager.hpp)

The intent was to reduce blocking synchronization, separate transfer work from compute work, make host materialization explicit and deferred, and keep the system benchmarkable during the refactor.

## Remote Workflow

All implementation and validation followed this workflow:

1. Edit locally in this repository.
2. Copy changed files to `eugenio@devenv2:/tmp/tmp.5M8Fsnsuez/sycldb-calcite-executor`.
3. Source `/home/eugenio/develop-env-vars.sh` remotely.
4. Build with `make client`.
5. Validate on `devenv2`.

Helper scripts added for reproducibility:

- [benchmark_all_devices.sh](/Users/eugenio/CLionProjects/sycldb-calcite-executor/benchmark_all_devices.sh)
- [remote_start_calcite.sh](/Users/eugenio/CLionProjects/sycldb-calcite-executor/remote_start_calcite.sh)
- [remote_run_benchmark.sh](/Users/eugenio/CLionProjects/sycldb-calcite-executor/remote_run_benchmark.sh)

Latest pulled benchmark artifact:

- [benchmark_results_final.devenv2.csv](/Users/eugenio/CLionProjects/sycldb-calcite-executor/benchmark_results_final.devenv2.csv)

## Major Functional Changes

### 1. Async flag counting

Changed in [kernels/common.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/kernels/common.hpp) and [models/transient_table.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/transient_table.hpp).

- Added `AsyncCountResult`.
- Added async flag count path via `count_true_flags_async(...)`.
- `TransientTable::count_flags_true(...)` now returns the async result rather than forcing an immediate scalar wait.

### 2. Operator-interleaved execution

Changed in [models/transient_table.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/transient_table.hpp).

- Reworked `execute_pending_kernels()` to submit per segment and carry explicit dependency chains.
- Removed the earlier queue-grouped submission pattern.
- Preserved CPU and per-device dependency tracking in returned event vectors.

### 3. Async row-id compaction path

Changed in [models/transient_table.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/transient_table.hpp).

- Replaced the old row-id path with `AsyncRowIds`.
- Device now produces:
  - row-id buffer
  - selected-count buffer
- Host receives explicit count and row-id copies through copy queues.
- `compress_and_sync()` now consumes those outputs without queue-wide waits.

### 4. Ping-pong flag buffers

Changed in [models/transient_table.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/transient_table.hpp), [kernels/selection.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/kernels/selection.hpp), and [kernels/join.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/kernels/join.hpp).

- Added:
  - `flags_host_secondary`
  - `flags_devices_secondary`
- Added helpers to allocate and swap secondary flag buffers only when needed.
- Filter and join kernels now support separate input and output flag pointers.
- This removed in-place flag hazards between consecutive phases.

### 5. Lazy secondary flags

Changed in [models/transient_table.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/transient_table.hpp).

- Secondary flag buffers are no longer allocated eagerly.
- This was a key memory fix for the previously problematic `q42` and `q43` runs on NVIDIA.

### 6. Background H2D staging and copy queues

Changed in [models/models.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/models.hpp) and [main.cpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/main.cpp).

- Added secondary device buffers and background copy tracking to `Segment`.
- Added background copy activation and promotion helpers.
- DDOR now stages required columns to device in the background using dedicated copy queues.
- Initial preload barriers were removed from the DDOR path.
- More D2H paths also use copy queues instead of compute queues.

### 7. Lazy host buffers for materialized device results

Changed in [models/models.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/models.hpp).

- Materialized on-device segments no longer allocate host backing buffers eagerly.
- Host buffers are created only on demand.
- `copy_on_host()` and `compress_sync()` now ensure host buffer allocation before use.
- Host-side getters trigger deferred copy only when the host view is actually requested.

### 8. Explicit host-state materialization

Changed in [models/transient_table.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/transient_table.hpp).

- Added:
  - `ensure_primary_host_flags(...)`
  - `ensure_host_state_available(...)`
  - `materialize_host_view(...)`
- CPU fallback paths now request the exact host state they need instead of relying on implicit `copy_on_host().wait()` behavior during operator construction.
- This was applied to:
  - CPU filter fallbacks
  - CPU project fallbacks
  - CPU join build/probe fallbacks
  - final result materialization

### 9. Final dependency-scoped waits

Changed in [main.cpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/main.cpp).

- Added `wait_for_dependencies_and_throw(...)`.
- DDOR no longer uses broad queue-wide waits after `execute_pending_kernels()`.
- Final completion now waits only on the exact returned dependency sets.
- Final result saving now calls the table-level host materialization API instead of open-coding per-device syncs in `main.cpp`.

## Operator Target-Selection Fixes

Several kernel-builder methods were corrected to choose host or device inputs according to the actual execution target, not current residency.

Changed in [models/models.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/models.hpp).

- `search_operator(...)`
- `filter_operator(...)`
- `aggregate_operator(...)`
- `build_keys_hash_table(...)`
- `semi_join_operator(...)`
- `build_key_vals_hash_ht(...)`
- `full_join_operator(...)`
- `group_by_aggregate_operator(...)`

Before these fixes, CPU fallback paths could still accidentally assume that `data_host` was already valid.

## Stability And Correctness Fixes

### Materialized column storage

Changed in [models/transient_table.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/transient_table.hpp).

- `materialized_columns` changed from `std::vector` to `std::deque`.
- This removed pointer invalidation in `current_columns`.

### Host/device allocator corrections

Changed in [models/models.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/models.hpp) and [models/transient_table.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/transient_table.hpp).

- Fixed several host buffers that were incorrectly allocated with device-oriented semantics.
- This was one of the direct fixes behind early runtime failures in `q11`.

### Constructor sync removal

Changed in [models/models.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/models/models.hpp).

- Removed the constructor-time SYCL reduction wait for segment `min/max`.
- Base segments now compute `min/max` on the host after the host copy.

### Repetition-safe background staging

Changed in [main.cpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/main.cpp).

- Background table-scan staging is now idempotent for already-resident columns.
- Repeated runs no longer keep re-allocating secondary device buffers for the same base columns.

### Queue-reset safety

Changed in [main.cpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/main.cpp).

- Copy queues are explicitly waited before allocator reset between repetitions.

### Memory manager behavior

Changed in [operations/memory_manager.hpp](/Users/eugenio/CLionProjects/sycldb-calcite-executor/operations/memory_manager.hpp).

- Moved from eager pool reservation to lazier region growth.
- This reduced allocator pressure during the previously failing NVIDIA runs.

## Validation History

Validation was performed incrementally on `devenv2` throughout the refactor.

### Guard queries

The main repeated validation set was:

- `q11.sql`
- `q42.sql`
- `q43.sql`

These were chosen because:

- `q11` is a fast representative sanity check.
- `q42` and `q43` were the historically unstable queries during the refactor.

### Major runtime outcomes

- Early implementation stages triggered memory pressure and instability on NVIDIA.
- After lazy flag allocation, allocator fixes, and staging fixes, `q42` and `q43` became stable.
- The guard queries were rerun after most major patches and continued to pass.

### Full benchmark outcome

The full benchmark sweep completed successfully and produced:

- [benchmark_results_final.devenv2.csv](/Users/eugenio/CLionProjects/sycldb-calcite-executor/benchmark_results_final.devenv2.csv)

Observed device status:

- `NVIDIA_L40S`: stable and fastest across all completed queries
- `Intel_Xeon_CPU`: stable
- `AMD_MI210`: functional but much slower
- `Intel_Flex_170`: all queries reported `ERROR`

## Current Benchmark Snapshot

From [benchmark_results_final.devenv2.csv](/Users/eugenio/CLionProjects/sycldb-calcite-executor/benchmark_results_final.devenv2.csv):

- `NVIDIA_L40S`: average `83.695 ms`
- `Intel_Xeon_CPU`: average `239.627 ms`
- `AMD_MI210`: average `2378.632 ms`
- `Intel_Flex_170`: no successful timings

Notable point:

- `q42` and `q43`, which were the most problematic queries during development, now complete reliably in the full sweep on NVIDIA.

## Remaining Gaps

The branch is benchmarkable and materially more asynchronous, but it is not the architectural end-state.

Remaining important gaps:

- metadata/control flow is not fully device-resident
- some CPU fallback decisions still exist
- final output still has a host boundary by design
- true data-column ping-pong is not implemented end-to-end
- `Intel_Flex_170` support remains unresolved

## Recommended Next Work

If development continues from this point, the highest-value next steps are:

1. Push more control metadata fully device-side so fewer CPU fallbacks are needed.
2. Reduce host participation in decision-making around compaction/join preparation.
3. Investigate `Intel_Flex_170` failures separately from the async pipeline work.
4. Optionally add a small benchmark/report script that summarizes the pulled CSV automatically.
