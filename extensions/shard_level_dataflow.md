# Shard-Level Data Flow

## Status

This document defines an optional extension to digital work expansion.
The extension preserves independent shard values from producer to consumer.

It does not change the runtime packet protocol.
Each shard becomes an ordinary tensor resource with its own readiness and route records.

## Problem

`--sculptor-expand-digital-work` currently creates declarative work units for one operation at a time.
Each work unit records static result and iteration slices.

`BuildRATree.cpp` creates exact work-unit edges only for a narrow destination-style pattern.
Other consumers keep a full-tensor dependency.

`OutlineTileRoutines.cpp` already materializes work units with `TilingInterface`.
It also bypasses full reassembly for some static subset consumers.
It rebuilds a full result when an unsupported whole-value consumer remains.

This fallback creates a full-tensor barrier.
The consumer cannot start from an early shard.

## Recommended Interface

Do not add a second digital-expansion pass.
Extend the existing pass:

```text
--sculptor-expand-digital-work="
  parallel-workers=8
  dataflow=sharded
  shard-propagation-depth=0
"
```

Use these options:

| Option | Values | Default |
|---|---|---|
| `dataflow` | `bulk`, `sharded` | `bulk` |
| `shard-propagation-depth` | `0` or a positive integer | `0` |
| `require-complete-shard-chain` | Boolean | `false` |

`0` means that propagation continues until a legal boundary stops it.

The default `bulk` mode must preserve current behavior.

## Existing Infrastructure To Reuse

Reuse these components:

| Component | Existing capability |
|---|---|
| `TilingInterface` | Maps iteration tiles to result tiles and creates tiled implementations |
| `MappingWorkUnit` | Stores exact static offsets and sizes |
| `MappingWorkUnitEdge` | Refines operation-level dependencies |
| `ExpandedDigitalWorkTreeBuilder` | Creates one RA leaf per work unit |
| `LogicalTileDependency` | Preserves exact source and target endpoints |
| `OutlineTileRoutines::materializeWorkUnits()` | Creates tiled operations after placement |
| `partsByFullValue` | Reuses producer tiles in downstream tiled operations |
| `TileRoutineRouteAttr` | Represents one routed tensor value |

Do not create a new shard graph beside `ComputeGraph` and `ResourceAllocationTree`.

## Data Model Changes

### File Change Table

| File | Change |
|---|---|
| `include/.../mapping/ShardDataflow.h` | Define shard planning and verification APIs |
| `lib/.../mapping/ShardDataflow.cpp` | Seed and propagate static shard groups |
| `ExpandDigitalWork.h/.cpp` | Add options and call the shard planner |
| `mapping/ResourceAllocationTree.h/.cpp` | Extend work-unit and exact-edge records |
| `BuildRATree.cpp` | Consume verified exact edges instead of rediscovering them |
| `mapping/LogicalTile.h` | Preserve tensor and operand identity on dependencies |
| `mapping/LogicalTileGraph.cpp` | Build exact shard dependencies and suppress bulk fallbacks |
| `OutlineTileRoutines.cpp` | Materialize exact shard values and assembly boundaries |
| `SculptorAttrs.td` | Extend work-unit and work-unit-edge attributes |
| `mapping/CMakeLists.txt` | Compile the shard planner |
| `tools/ra-tree-report` | Export shard groups, slices, and routes |

Extend `MappingWorkUnit`:

```cpp
int64_t shardGroupId = -1;
int64_t shardIndex = -1;
int64_t shardCount = -1;
```

Extend `MappingWorkUnitEdge`:

```cpp
int64_t tensorId = -1;
int64_t sourceResultNumber = -1;
int64_t targetOperandNumber = -1;
```

The existing operation and work-unit IDs remain the endpoint identity.
The shard fields describe cross-operation shard membership.

Add the same fields to `MappingWorkUnitAttr` and `MappingWorkUnitEdgeAttr`.
Bump the RA-tree attribute version.

Extend `LogicalTileDependency` with the target operand number.
This removes operand discovery from routine outlining.

Use a typed route shard attribute only when reports need explicit slice coordinates.
Do not change the runtime route ABI for the first version.

## Shard Planning Algorithm

Implement the planner in a new library:

```text
include/sculptor-mlir/Dialect/Sculptor/Transforms/mapping/
  ShardDataflow.h

lib/Dialect/Sculptor/Transforms/mapping/
  ShardDataflow.cpp
```

`ExpandDigitalWork.cpp` calls this library when `dataflow=sharded`.

### Step 1: Seed Work Units

Use the current worker-factor algorithm.
Create stable result tiles for each eligible root operation.

Assign one shard group to the root result.
Order shard indices by lexicographic result offset.

### Step 2: Propagate Through Consumers

Visit consumers in compute-graph topological order.

Propagate a shard group only when all conditions are true:

1. The consumer implements `TilingInterface`.
2. Tensor shapes and tile coordinates are static.
3. The producer result maps to one unambiguous consumer operand tile.
4. The consumer result tile can be calculated statically.
5. All required tiled inputs are available for the same shard.
6. The operation has no unsupported side effects.
7. The operation does not reduce across the selected shard dimension.

Create aligned work units for the consumer.
Create one exact `MappingWorkUnitEdge` for each producer and consumer shard pair.

### Step 3: Stop At A Boundary

Stop propagation at these boundaries:

- A model output that requires a full tensor.
- A dynamic shape.
- An unsupported operation.
- An ambiguous tile map.
- A reduction without enabled reduction-tree support.
- A consumer that needs data from multiple incompatible shard groups.
- A non-contiguous route payload that the runtime cannot describe.

Mark the boundary as an assembly point.
Do not silently claim shard-level readiness after this point.

### Step 4: Verify Coverage

For each sharded result, verify these properties:

- Shards do not overlap.
- Shards cover the declared result when complete coverage is required.
- Each shard has one producer work unit.
- Each exact edge references a valid source and target work unit.
- The edge byte size matches the shard tensor type.

## General-Purpose Legality

Do not classify operations by Transformer names.
Use `TilingInterface`, iterator kinds, tensor types, and memory effects.

The first version supports:

- Static ranked tensors.
- Unit-stride rectangular slices.
- Destination-style elementwise operations.
- Broadcast operands when the tiled implementation gives an exact static slice.
- One or more outputs when each output has an exact tile map.

The first version rejects:

- Dynamic dimensions.
- Non-unit-stride route slices.
- Reduction-dimension sharding without reduction-tree support.
- Overlapping writes.
- Operations with unknown memory effects.
- A view that is not contiguous under the runtime storage contract.

## Build RA Tree Changes

Replace the narrow `buildDestinationStyleEdges()` behavior with a general exact-edge builder.

The builder must consume the edges created by `ShardDataflow`.
It must not rediscover shard relationships from SSA use patterns.

Keep whole-tensor dependencies for operation pairs without an exact refinement.
Do not emit both an exact shard edge and the fallback full-tensor edge.

The RA-tree verifier must check:

- Stable shard IDs.
- Valid endpoint IDs.
- Valid tensor and operand IDs.
- Exact byte sizes.
- Acyclic work-unit dependencies.

## Logical-Tile Graph Changes

`LogicalTileGraphBuilder::addDependencies()` already prefers work-unit edges.
Extend it to preserve the tensor ID and target operand number.

Each shard dependency must remain separate in `LogicalTileEdge::dependencies`.
Only the edge's `byteSize` field is an aggregate.

Temporal scheduling must use the detailed dependency list.
It must not schedule one aggregate edge as one full-tensor barrier.

## Routine Outlining Changes

Keep work-unit materialization inside `OutlineTileRoutines`.
The physical placement is locked at this stage.

Extend `materializeWorkUnits()`:

1. Materialize operations in topological shard order.
2. Resolve generated input slices from exact `MappingWorkUnitEdge` records.
3. Give each produced shard a separate `Value` and resource ID.
4. Rebuild a full tensor only at a marked assembly boundary.
5. Keep each shard producer and consumer in separate atomic routines when placement requires it.

The current `partsByFullValue` map remains useful.
Change it from pattern-discovery state into verified shard-materialization state.

For fan-out, reuse one source shard resource and create one route per destination.
The current runtime materializer already supports this route form.

## Runtime And Storage Effects

No runtime scheduler change is required.
Each shard route arrives as an independent runtime resource.

The tile ABI already supports static tensor dimensions and separate resources.
Resource finalization allocates each routed shard according to its local lifetime.

The first version must create a contiguous buffer for each routed shard.
Same-tile consumers can use tensor views when bufferization preserves the view safely.

## IR Provenance

Add these function attributes:

```text
sculptor.mapping.dataflow_mode = "bulk" | "sharded"
sculptor.mapping.shard_group_count
sculptor.mapping.shard_edge_count
sculptor.mapping.assembly_boundary_count
```

Preserve these values through placement and outlining.
Add them to the placement summary and RA-tree report.

## Diagnostics

Emit a compiler error for these conditions:

- Duplicate shard group or shard index.
- Overlapping result shards.
- Incomplete required coverage.
- Invalid work-unit endpoint.
- Ambiguous consumer tile mapping.
- Inconsistent shard tensor type.
- Edge byte size that disagrees with the shard type.
- Unsupported non-contiguous routed slice.
- Full-tensor fallback when `require-complete-shard-chain=true`.
- Routine graph cycle after shard materialization.

## Tests

Add focused tests in `tests/python_tests/test_shard_dataflow.py`.

Use small static linalg graphs before GPT-2.
The tests must cover:

1. One elementwise producer and one aligned consumer.
2. Two chained elementwise consumers.
3. A producer shard routed before another shard finishes in the temporal model.
4. Fan-out from one shard to two consumers.
5. A same-tile shard binding.
6. A cross-tile shard route.
7. One legal full-tensor assembly boundary.
8. No assembly for a fully aligned chain.
9. Dynamic-shape rejection.
10. Overlap and coverage diagnostics.
11. Acyclic outlined routines.
12. RISC-V object generation for every active tile.

Add a GPT-2 test with two or four workers.
Verify that the compiler creates smaller route resources than the full hidden-state tensor.

## Incremental Delivery

### Phase 1

Add shard IDs and exact-edge fields to the mapping attributes.
Keep `dataflow=bulk` as the only active behavior.

### Phase 2

Propagate one partition through one destination-style elementwise consumer.
Materialize the direct shard edge in the outliner.

### Phase 3

Propagate through an arbitrary legal elementwise chain.
Add fan-out support and explicit assembly boundaries.

### Phase 4

Connect shard readiness to the temporal evaluator.

### Phase 5

Add reduction-tree consumers.

## Non-Goals

This extension does not:

- Add dynamic-shape sharding.
- Change the packet protocol.
- Add partial execution inside one runtime task.
- Shard matrix setup state.
- Select the best worker count.
- Force every operation into a shard chain.

## Easiest Effective First Version

Propagate one static output partition through elementwise `TilingInterface` consumers.
Use exact work-unit edges and the existing outliner materialization.

This version removes common full-tensor barriers without a new dialect or runtime change.
