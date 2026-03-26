# Refactor Plan For Reproducible SSB Benchmarking

## Status

Initial implementation pass completed on 2026-03-26.

Completed in this pass:

- Task 1 through Task 12
- module split into `app/`, `runtime/`, `executor/`, `benchmark/`, and `docs/`
- thin `main.cpp`
- dedicated SSB suite definition
- benchmark scripts moved onto the supported benchmark mode
- remote sync/build helper scripts
- updated architecture and workflow documentation

Validated in this pass:

- remote build on `devenv2`
- CLI help path in the remote environment
- end-to-end remote execution of `queries/transformed/q11.sql` with the new structure

## Goal

Restructure the repository so that it is easy to understand, reproducible, and centered on one supported product:

- run the SSB transformed queries
- produce benchmark results in a repeatable way
- keep the active DDOR execution path
- separate benchmark orchestration from execution-engine code

## Non-Goals

- preserving the legacy single-queue execution path as a first-class mode
- keeping unrelated experimental code mixed into the benchmarkable path
- optimizing for use cases outside the SSB benchmark workflow in this pass

## Phase 1: Structural Decomposition

### Task 1: Define the supported product boundary

Subtasks:
- Write down the exact supported workflow: planner connection, query input, execution mode, benchmark mode, output artifacts.
- Explicitly mark what is in scope for the refactor: DDOR path, SSB transformed queries, benchmark scripts, result files, remote build/run workflow.
- Explicitly mark what is out of scope or legacy: old execution path, ad hoc experiments, unused scripts, temporary artifacts.
- Define the minimum reproducibility contract: required environment variables, required data layout, expected output files, supported devices.

Done when:
- There is one unambiguous statement of what the repository is supposed to do after the refactor.

### Task 2: Define the target module layout

Subtasks:
- Design a small top-level module map for the benchmarkable product.
- Separate code into these conceptual areas:
  - CLI and app entrypoints
  - planner client and SQL loading
  - runtime setup and device selection
  - execution engine
  - table and segment data model
  - benchmark orchestration
  - documentation and reproducibility assets
- Decide which existing files stay, which move, which are split, and which become legacy or are removed.
- Define stable interfaces between modules before moving code.

Done when:
- There is a target directory and ownership map that can guide the implementation phase.

### Task 3: Isolate application bootstrap from execution logic

Subtasks:
- Split `main.cpp` responsibilities into separate units:
  - argument handling
  - SQL loading
  - Calcite/Thrift setup
  - device and queue setup
  - allocator construction
  - benchmark loop
  - execution invocation
- Define a small application entry API that can run either:
  - one query
  - a benchmark suite
- Move logging and timing orchestration out of the core execution code where possible.

Done when:
- `main.cpp` becomes a thin entrypoint instead of the repository's main implementation file.

### Task 4: Isolate runtime setup and dependency management

Subtasks:
- Extract queue discovery, preferred-device ordering, copy-queue setup, and allocator sizing into runtime-specific code.
- Centralize wait helpers, dependency handling, and queue reset behavior.
- Replace scattered runtime policy decisions with one runtime configuration object.
- Ensure runtime setup is reusable by both single-query execution and benchmark runners.

Done when:
- Runtime construction and synchronization policy live in one place instead of being spread across the application.

### Task 5: Separate the data model from operator orchestration

Subtasks:
- Keep table, column, and segment ownership/state code together.
- Keep operator planning and execution scheduling together.
- Reduce cross-header coupling between `models/*`, `operations/*`, and `kernels/*`.
- Identify functions that belong to the data model versus functions that belong to orchestration.
- Define clearer boundaries for:
  - residency and movement
  - pending kernel scheduling
  - host materialization
  - flag-buffer lifecycle

Done when:
- It is clear which code represents state and which code transforms state.

### Task 6: Separate benchmark orchestration from engine internals

Subtasks:
- Define one benchmark entry flow for the SSB suite.
- Move query list, repetition policy, result-file naming, and benchmark summaries into benchmark-specific code or scripts.
- Keep the engine API free from benchmark-specific file naming and output parsing.
- Define where benchmark assets live:
  - suite definitions
  - scripts
  - result outputs
  - reference outputs

Done when:
- Benchmarks are driven by dedicated orchestration instead of being mixed into the engine entrypoint.

### Task 7: Define the reproducibility surface

Subtasks:
- Replace hard-coded operational assumptions with explicit configuration points where appropriate.
- Identify which values must be configurable:
  - data directory
  - planner host/port
  - repetitions
  - output directory
  - device selector policy
- Define one documented runbook for local editing plus remote build/run validation.
- Decide which generated artifacts belong in version control and which do not.

Done when:
- A new contributor can understand how to reproduce the benchmark flow without reverse-engineering scripts and constants.

## Phase 2: Implementation And Cleanup

### Task 8: Implement the new module layout incrementally

Subtasks:
- Create the new directories and translation units.
- Move bootstrap code out of `main.cpp` first.
- Move runtime setup second.
- Move execution-driver code third.
- Keep behavior stable after each move.
- Build after each extraction step and keep commits narrowly scoped.

Done when:
- The active code path matches the target module layout from Phase 1.

### Task 9: Remove or quarantine legacy paths

Subtasks:
- Decide whether the old `execute_result` path is deleted or moved under an explicitly named legacy area.
- Remove dead code, commented-out experiments, and duplicate bootstrap logic from the supported path.
- Remove obsolete scripts and artifacts that are no longer part of the reproducible benchmark workflow.
- Keep only one documented execution path for SSB benchmarking.

Done when:
- The repository has one clearly supported path and legacy code no longer pollutes the active design.

### Task 10: Normalize interfaces inside the execution engine

Subtasks:
- Simplify kernel dispatch and submission interfaces.
- Reduce duplicated host/device decision logic across filter, project, aggregate, and join paths.
- Consolidate common synchronization helpers and buffer-swap logic.
- Standardize naming for residency, materialization, and execution concepts.
- Add small focused tests or validation checks around the extracted interfaces where possible.

Done when:
- The engine is organized by coherent responsibilities instead of by historical accumulation.

### Task 11: Rebuild the benchmark runner on top of the cleaned interfaces

Subtasks:
- Create one benchmark runner that calls the cleaned application and engine interfaces.
- Encode the SSB suite in one place.
- Standardize output file naming and result collection.
- Ensure the runner can be used in the intended remote environment without manual code edits.

Done when:
- Running the SSB benchmark suite no longer depends on ad hoc script behavior spread across the repo.

### Task 12: Final validation and documentation

Subtasks:
- Validate single-query execution on representative SSB queries.
- Validate full benchmark execution in the supported environment.
- Update `README.md` to match the new structure and supported workflow only.
- Add a short architecture note describing the major modules and their responsibilities.
- Document the expected artifacts and the exact commands for reproducible runs.

Done when:
- The simplified repository structure is documented and the supported benchmark workflow has been revalidated end to end.

## Suggested Execution Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6
7. Task 7
8. Task 8
9. Task 9
10. Task 10
11. Task 11
12. Task 12

## Notes

- The plan intentionally optimizes for one reproducible benchmark product, not for preserving every historical execution mode.
- The highest-risk area is the boundary between runtime setup, transient-table orchestration, and benchmark-specific control flow.
- The first implementation milestones should reduce `main.cpp` size and make the DDOR path the only clearly supported execution flow.
