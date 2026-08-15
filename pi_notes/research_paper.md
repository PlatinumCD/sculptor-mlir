# Sculptor-MLIR: A Compiler for Analog/Digital Hybrid Accelerators

## Abstract

This paper presents the architecture of Sculptor-MLIR, a complete compiler infrastructure for analog/digital hybrid accelerators. The system maps neural-network tensor programs to logical tiles on a physical mesh of analog/digital cores, each containing both digital and analog compute resources. The compiler preserves computation structure throughout the mapping pipeline, using a Resource Allocation (RA) tree to represent spatial and temporal cuts in the computation. The architecture separates logical mapping from runtime lowering, enabling independent optimization of scheduling, placement, and deployment. We describe the full 13-stage pipeline, the four mapping planner strategies, the logical tile placement algorithm, and the freestanding Golem runtime library.

---

## 1. Introduction

Modern AI accelerators increasingly combine analog compute arrays with digital control logic. This creates a new class of hybrid architectures that require fundamentally different compilation strategies than traditional digital accelerators. We present Sculptor-MLIR, a compiler built on the MLIR framework that addresses this challenge through a modular, multi-stage pipeline.

The key contributions are:

1. **A complete compiler pipeline** from tensor programs to tile-local code, preserving computation structure throughout.
2. **A resource allocation tree** that represents both spatial and temporal structure in a single data structure.
3. **Four composable planning strategies** that each transform the RA tree.
4. **A logical tile abstraction** that separates mapping from deployment.
5. **A freestanding runtime library** that executes compiled tile programs.

This paper details the algorithms, techniques, and correctness properties of each stage.

---

## 2. Hardware Model

The target architecture is a mesh of tile cores, where each tile contains:
- **One digital lane**: For tensor/digital operations.
- **N analog lanes**: For matrix-vector multiply (MVM) operations on analog arrays.

The physical mesh has dimensions `rows × columns × arraysPerCore`. Each analog array has dimensions `arrayRows × arrayCols`. The hardware model also specifies latency, bandwidth, and memory capacity for each resource type.

This architecture is novel — existing compilers target either pure-digital or pure-analog architectures. There is no prior art in mapping to hybrid analog/digital accelerators.

---

## 3. Compiler Pipeline

The pivot pipeline consists of 13 stages:

| # | Pass | Description |
|---|---|---|
| 1 | `sculptor-canonicalize-layers` | Normalize NN layers |
| 2 | `sculptor-extract-layers` | Expose layer ops explicitly |
| 3 | `sculptor-convert-layers` | Decompose to tensor + MVM ops |
| 4 | `sculptor-expand-mvm-to-golem` | Expand MVM to 6 Golem ops |
| 5 | `sculptor-expand-digital-work` | Expand digital work units |
| 6 | `sculptor-build-ra-tree` | Build Resource Allocation tree |
| 7 | `sculptor-plan-mapping` | Select spatial/temporal cuts |
| 8 | `sculptor-apply-mapping-plan` | Apply plan annotations |
| 9 | `sculptor-place-logical-tiles` | Map to mesh coordinates |
| 10 | `sculptor-outline-tile-routines` | Create tile-level routines |
| 11 | `sculptor-materialize-tile-runtime-graph` | Create task graph |
| 12 | `sculptor-extract-tile-module` | Isolate one tile |
| 13 | `sculptor-plan-tile-scratchpad` | Plan local storage |

---

## 4. Compute Graph

The **ComputeGraph** is the foundation of the mapping system. It records:
- Each operation as a `ComputeOperation` with a unique `id`.
- Dependency edges between operations (via tensor values).
- Iterator kinds: `Parallel` or `Reduction`.
- Operation kinds: `Structured`, `LogicalMVM`, `MatrixSetup`, `DigitalStage`, `VectorTile`, `PhysicalMVM`, `TileRecombine`.

The compute graph is built via a BFS from the function's output values, traversing upstream producers. Function arguments are treated as external inputs. The graph is topologically sorted to establish a stable execution order.

**Correctness**: The compute graph preserves all data dependencies between operations, ensuring that the mapping respects the original program semantics.

---

## 5. Resource Allocation Tree

The **Resource Allocation (RA) Tree** is a hierarchical decomposition of the computation. Each node is one of three kinds:

1. **Leaf**: Represents a single compute operation or a tiled work unit.
2. **Temporal Cut**: Orders child regions sequentially.
3. **Spatial Cut**: Permits child regions to execute in parallel.

The tree is built in two phases:
1. **Baseline tree construction**: The `BuildRATree` pass creates a flat tree where each leaf is a Golem operation.
2. **Digital work expansion**: Each leaf is replaced by a subtree of digital work units, preserving dependencies.

The RA tree preserves all spatial and temporal structure of the computation until physical placement.

---

## 6. Mapping Planning

The `PlanMapping` pass accepts an ordered list of **planner strategies**. Each strategy transforms the RA tree, and the result is passed to the next strategy. The selected plan is the best according to the `MappingEvaluator`.

### 6.1 Mapping Problem

A `MappingProblem` contains:
- The `ComputeGraph` and `ResourceAllocationTree`.
- The `MappingHardwareModel` (mesh geometry, latency, bandwidth).
- The `MappingObjective` (currently: minimize latency).
- Flags for MVM wave colocation and digital work balancing.

### 6.2 Mapping Evaluator

The `ReferenceMappingEvaluator` evaluates a tree by:
1. Collecting all leaf endpoints (operation + work unit pairs).
2. Computing start/finish times for each node via dynamic programming.
3. Summing estimated latencies, crossing bytes, communication costs, and required resources.

The evaluator returns a `MappingEvaluation` with feasibility, latency, and resource estimates.

### 6.3 Mapping Realization

`realizeResourceAllocationTree` assigns each leaf to a digital or analog lane. The algorithm:
1. Processes the tree bottom-up.
2. For each leaf, assigns a lane based on operation kind and resource availability.
3. Ensures no lane is over-allocated.

**Correctness**: The realization preserves all dependencies and lane bindings.

---

## 7. Planning Strategies

Four strategies are implemented:

### 7.1 Setup-First
Places matrix setup operations before their dependent execution. The algorithm:
1. Clones the baseline tree.
2. For each matrix setup leaf, clones it to the root position.
3. Multiplies work group counts to preserve concurrency.

### 7.2 MVM Wave
Groups independent MVM operations into waves. The algorithm:
1. Computes topological ranks for each operation.
2. Groups MVM operations into waves based on rank and dependency.
3. Inserts spatial cuts to enable parallel execution within each wave.

### 7.3 Fan-Out Cut
Exposes parallel consumers after a fan-out. The algorithm:
1. Identifies fill operations that are root fill (no non-setup producers).
2. Groups direct consumers of each fill.
3. Inserts cuts to parallelize independent consumer subtrees.

### 7.4 Consumer-Bound Fill
Binds fill work to its consumers. The algorithm:
1. Identifies all fill operations.
2. Groups fill leaves by their direct consumers.
3. Creates subtrees that bind each fill to its consumers.

---

## 8. Logical Tile Graph

After mapping, each logical tile owns:
- Its compute operations (leaves of the RA tree).
- Digital and analog lane assignments.
- Incoming and outgoing communication edges.
- Resource demand and dependency order.

The `LogicalTileGraph` is deserialized from the applied mapping plan.

---

## 9. Physical Placement

The `PlaceLogicalTiles` pass maps logical tiles to mesh coordinates. The algorithm:
1. Computes the physical tile capacity: `rows × columns`.
2. Computes Manhattan distance between tile locations.
3. Uses one of several placement schedules:
   - `Random`: random initial placement.
   - `Snake`: snake-like ordering.
   - `Greedy`: greedy nearest-neighbor.
   - `GreedyBeam`: beam search over greedy placements.
   - `Annealing`: simulated annealing for global optimization.
4. Evaluates each placement using the mapping evaluator.

**Correctness**: The placement respects the mesh geometry and avoids collisions.

---

## 10. Runtime Library

The Golem runtime is a freestanding C++ library that executes compiled tile programs. It implements:

### 10.1 DeploymentRuntime
- Manages a fixed 16-entry task-instance pool.
- Manages a 16-entry FIFO ready queue.
- Executes ready tasks and manages tensor transport.
- Handles receive/transmit backpressure.

### 10.2 Framed Transport
Each tensor frame consists of:
```
[magic(32) | route_id(32) | execution_id(64) | word_count(32)] [payload words...]
```
The runtime receives frames via a 32-bit word callback (or DMA). It maintains per-source receive state to handle interleaved frames.

### 10.3 Task Execution
- `executeReadyTask()` invokes each task's entry point.
- `step()` advances the runtime by one step, either executing a task or managing a pending transmit/receive.
- Supports two transmit policies: blocking and overlap-ready-tasks.

### 10.4 Tracing
An optional `DeploymentTrace` callback reports task start/finish events.

**Correctness**: The runtime maintains a valid state machine, ensuring that only ready tasks execute and that tensor frames are received correctly.

---

## 11. Tooling

### 11.1 `sculptor-mlir-opt`
The compiler driver that exposes all passes to textual pipelines.

### 11.2 `sculptor-ra-tree-report`
Generates an interactive HTML/JSON visualization of the RA tree, including expanded IR overlays.

### 11.3 Python Test Harness
Tests compile PyTorch models to MLIR via `torch-mlir` and verify correct lowering.

---

## 12. Correctness Guarantees

1. **Semantic preservation**: Each pipeline stage is a semantically preserving transformation.
2. **Dependency preservation**: The RA tree and compute graph preserve all data dependencies.
3. **Resource binding**: Each matrix setup is bound to its dependent array operations on the same analog lane.
4. **Lane exclusivity**: Each logical tile has a unique set of digital and analog lanes.
5. **Placement validity**: Logical tiles are mapped to distinct mesh coordinates within the hardware geometry.
6. **Runtime invariants**: The deployment runtime maintains valid state transitions and handles backpressure correctly.

---

## 13. Conclusion

Sculptor-MLIR provides a complete compiler infrastructure for analog/digital hybrid accelerators. The architecture separates logical mapping from runtime lowering, uses a hierarchical RA tree for spatial/temporal decomposition, and supports multiple planning strategies. The freestanding Golem runtime provides a correct execution model for compiled tile programs.

This paper describes the architecture and algorithms; empirical validation and benchmarking are left for future work.