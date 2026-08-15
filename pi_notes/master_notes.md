# Sculptor-MLIR — Technical Architecture

## 1. Introduction

Sculptor is an out-of-tree MLIR compiler that translates neural-network tensor programs into tile-local code for the Golem analog/digital accelerator architecture. The compiler separates **logical mapping** (the "pivot" path) from **runtime lowering** (generating tile routines and deployment code).

## 2. Compiler Pipeline

The pivot pipeline executes 13 passes in sequence:

```
Input (MLIR)
  │
  ▼
───[1] sculptor-canonicalize-layers ────
  │ Normalize NN layers (RNN, LSTM, GRU, Conv, Linear, Transformer)
  ▼
───[2] sculptor-extract-layers ────
  │ Expose explicit layer operations
  ▼
───[3] sculptor-convert-layers ────
  │ Decompose layers → tensor ops + sculptor.mvm ops
  ▼
───[4] sculptor-expand-mvm-to-golem ────
  │ Each MVM → [matrix_setup, vector_tile, array_load, array_execute,
  │             array_store, recombine]
  ▼
───[5] sculptor-expand-digital-work ────
  │ Expand digital work into mappable units
  ▼
───[6] sculptor-build-ra-tree ────
  │ Build Resource Allocation tree
  ▼
───[7] sculptor-plan-mapping ────
  │ Select spatial & temporal cuts (planner strategies)
  ▼
───[8] sculptor-apply-mapping-plan ────
  │ Apply plan annotations to operations
  ▼
───[9] sculptor-place-logical-tiles ────
  │ Map logical tiles → mesh coordinates
  ▼
───[10] sculptor-outline-tile-routines ────
  │ Create tensor-level routines per tile
  ▼
───[11] sculptor-materialize-tile-runtime-graph ────
  │ Create tile-local task/route metadata
  ▼
───[12] sculptor-extract-tile-module ────
  │ Isolate one tile module
  ▼
───[13] sculptor-plan-tile-scratchpad ────
  │ Plan local scratchpad storage
  ▼
Output (per-tile MLIR module + runtime metadata)
```

## 3. Core Data Structures

### ComputeGraph
Records operation dependencies and stable operation order. Each operation gets a unique ID. The graph distinguishes between:
- `Structured` — full network ops
- `LogicalMVM` — single matrix multiply
- `MatrixSetup` — weight setup for analog lane
- `DigitalStage` / `VectorTile` / `PhysicalMVM` / `TileRecombine`

### ResourceAllocationTree (RA Tree)
A tree with three node kinds:
- **Temporal cut** — orders children sequentially
- **Spatial cut** — allows children to execute in parallel
- **Leaf** — references one compute operation or one tiled work unit

The RA tree preserves spatial and temporal structure throughout the pipeline until physical placement.

### MappingPlan
Records the ordered planning decisions and the selected realization. Each planner strategy transforms the RA tree.

### LogicalTileGraph
Groups planned work into logical tiles. Each logical tile owns:
- Its compute operations
- Digital and analog lane assignments
- Incoming and outgoing communication
- Resource demand
- Dependency order

## 4. Hardware Model

A logical tile contains:
- **1 digital lane** — for tensor/digital operations
- **N analog lanes** — for matrix setup and array execution

The planner enforces lane bindings: each matrix setup operation is bound to its dependent array operations, ensuring they execute on the same analog lane.

## 5. Planning Strategies

| Strategy | Purpose |
|---|---|
| `setup-first` | Places matrix setup work before dependent execution |
| `mvm-wave` | Groups independent MVM work into spatial waves |
| `fan-out-cut` | Exposes parallel consumers after a fan-out |
| `consumer-bound-fill` | Binds fill work to its consumer |

## 6. Deployment Boundary

- **`--sculptor-outline-tile-routines`** creates tensor-level routines. Matrix setup becomes boot routines; other work becomes dispatch routines.
- **`--sculptor-materialize-tile-runtime-graph`** creates tile-local task and route metadata.
- **`--sculptor-extract-tile-module`** selects one tile.
- **`--sculptor-plan-tile-scratchpad`** plans local storage.

The legacy island scheduler and island timing model are **not** part of this architecture.

## 7. Source Layout

```
include/                         Public headers and TableGen definitions
lib/Dialect/Sculptor/
  ├── IR/                      Operations, attrs, types (TableGen + C++)
  │   └── Ops/               Individual op implementations
  ├── Conversion/golem/      MVM → Golem lowering
  ├── Transforms/
  │   ├── canonicalizers/    Layer normalization
  │   ├── converters/      Layer decomposition
  │   ├── Golem/           MVM expansion
  │   ├── extractors/      Layer extraction
  │   ├── mapping/         RA tree, compute graph, planning, placement
  │   └── planners/        Mapping strategies
  └── Support/             Shared utilities
tools/
  ├── sculptor-mlir-opt/     Compiler driver
  ├── ra-tree-report/      Interactive RA-tree visualizer
  └── maintenance/         Audit and test utilities
runtime/                     Freestanding Golem runtime library
tests/
  ├── python_tests/        Lowering regression tests
  └── model_tests/         End-to-end model tests (GPT-2)
```

## 8. Runtime Library

The Golem runtime (`libgolem-runtime.a`) implements:
- Task and tensor ABI types
- Tile boot, dispatch, route, and model-I/O tables
- `DeploymentRuntime` — schedules ready tasks, manages tensor transport
- Fixed 16-entry task-instance pool and 16-entry FIFO ready queue
- Framed 32-bit word transport with optional DMA support
- Optional `DeploymentTrace` callback for task start/finish events

## 9. Tooling

- **`sculptor-mlir-opt`** — main compiler driver (C++)
- **`sculptor-ra-tree-report`** — generates interactive HTML/JSON RA-tree visualization
- **Python test harness** — uses `torch-mlir` to compile models to MLIR, then runs sculptor passes

## 10. Test Coverage

| Test Suite | Coverage |
|---|---|
| `tests/python_tests/` | Linear, Conv, LSTM, GRU, RNN, Transformer |
| `tests/model_tests/` | GPT-2 end-to-end regression |
| `runtime/tests/` | Host unit tests for runtime library |

## 11. Research Paper Framing

This is a compiler infrastructure project targeting analog/digital hybrid accelerators. The key contributions include:
- A pivot-style mapping pipeline that preserves computation structure until final placement
- RA tree representation for spatial/temporal cut selection
- Logical tile abstraction replacing placement islands
- A freestanding runtime library for tile execution