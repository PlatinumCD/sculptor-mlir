# Sculptor-mlir — Working Notes

## Overview

Sculptor is an out-of-tree MLIR compiler that maps neural-network tensor programs to **logical tiles** on a physical mesh. It targets analog/digital accelerator architectures (the "Golem" platform).

## Compiler Pipeline (13 stages)

| # | Pass | Description |
|---|---|---|
| 1 | `--sculptor-canonicalize-layers` | Normalize NN layers |
| 2 | `--sculptor-extract-layers` | Expose layer ops explicitly |
| 3 | `--sculptor-convert-layers` | Decompose to tensor + `sculptor.mvm` |
| 4 | `--sculptor-expand-mvm-to-golem` | Expand MVM to 6 Golem ops: matrix setup, vector tile, array load, array execute, array store, recombine |
| 5 | `--sculptor-expand-digital-work` | Expand digital work into mappable units |
| 6 | `--sculptor-build-ra-tree` | Build Resource Allocation tree |
| 7 | `--sculptor-plan-mapping` | Select spatial/temporal cuts |
| 8 | `--sculptor-apply-mapping-plan` | Apply the plan to operations |
| 9 | `--sculptor-place-logical-tiles` | Map logical tiles to mesh coords |
| 10 | `--sculptor-outline-tile-routines` | Create tensor-level tile routines |
| 11 | `--sculptor-materialize-tile-runtime-graph` | Create tile-local task graph |
| 12 | `--sculptor-extract-tile-module` | Isolate one tile module |
| 13 | `--sculptor-plan-tile-scratchpad` | Plan local scratchpad storage |

## Key Data Structures

- **ComputeGraph**: records op dependencies and stable execution order
- **ResourceAllocationTree (RA Tree)**: tree of TemporalCuts, SpatialCuts, and Leaves
- **MappingPlan**: ordered planner decisions + selected realization
- **LogicalTileGraph**: groups planned work, records inter-tile communication

## Four Mapping Strategies

| Strategy | Purpose |
|---|---|
| `setup-first` | Place matrix setup before dependent exec |
| `mvm-wave` | Group independent MVM into spatial waves |
| `fan-out-cut` | Expose parallel consumers after fan-out |
| `composer-bound-fill` | Bind fill work to its consumer |

## Hardware Model

Each logical tile has:
- 1 digital lane (for tensor ops)
- N analog lanes (for matrix setup + array exec)

## Tools

- `sculptor-mlir-opt` — compiler driver
- `sculptor-ra-tree-report` — interactive HTML/JSON RA-tree visualizer
- `runtime/` — freestanding Golem runtime library (`libgolem-runtime.a`)

## Test Coverage

- `tests/python_tests/` — lowering tests for linear, conv, lstm, gru, rnn, transformer
- `tests/model_tests/` — GPT-2 model regression test
- `runtime/tests/` — host unit tests for the runtime library

## Open Questions / Next Steps

- Deeper dive into RA tree construction algorithm
- Understand how placement islands are (or aren't) used
- Examine the Golem operation set in detail
- Look at the mapping planner interface