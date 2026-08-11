# Distributed Reduction Trees

## Status

This document defines an optional graph transformation for associative reductions.
The extension replaces a central fan-in with a deterministic binary tree.

The extension requires explicit permission to reassociate floating-point operations.
It does not change the runtime task ABI or network protocol.

## Problem

Distributed work often creates many partial tensors.
The current graph can send all partials to one assembly task.

This structure has three costs:

- One task waits for every producer.
- Many routes target one tile at similar times.
- The slowest producer controls the next stage.

A balanced binary tree limits each reduction task to two inputs.
It also exposes independent reduction branches to mapping and placement.

## Existing Infrastructure To Reuse

The compiler already has these useful pieces:

| Component | Existing capability |
|---|---|
| `TaskReductionAttr` | Marks add, max, or min reductions and reassociation permission |
| `ExpandMVMToGolem.cpp` | Marks eligible one-row MVM recombination as associative addition |
| `SemanticOperationNames.h` | Defines `digital.reduction` |
| `TilingInterface` | Creates shard computations and static result tiles |
| `ComputeGraph` | Exposes explicit reduction operations and dependencies |
| `recursive-fork-join` planner | Recovers parallel branch structure |
| `LogicalTileGraph` | Preserves detailed reduction dependencies |
| `OutlineTileRoutines` | Creates atomic tasks and routes for explicit values |

The current `sculptor.task.reduction_tree_id`, level, and width names apply after task creation.
Do not use those task attributes as the primary pre-task mapping representation.

## Recommended Interface

Keep this feature in `--sculptor-expand-digital-work` because it expands distributed digital work.

```text
--sculptor-expand-digital-work="
  parallel-workers=8
  dataflow=sharded
  reduction-tree=balanced
"
```

Use these options:

| Option | Values | Default |
|---|---|---|
| `reduction-tree` | `none`, `balanced`, `ready-time` | `none` |
| `reduction-fan-in` | `2` | `2` |
| `reduction-min-width` | Integer greater than one | `3` |

Implement only `none` and `balanced` in the first change.
Reserve `ready-time` for the calibrated temporal model.

The default `none` mode must preserve current reduction order and placement.

## Why This Is A Graph Transformation

A reduction tree changes producer and consumer dependencies.
It is not only an RA-tree cut.

Create explicit tensor operations before `--sculptor-build-ra-tree`.
Then the existing compute graph, RA tree, placement, and outliner see the same dependency structure.

Do not create a hidden reduction plan that only the runtime understands.

## Reduction Contract

An eligible reduction group must meet all conditions:

1. The operation carries `TaskReductionAttr`.
2. The attribute sets `reassociate=true`.
3. The reduction kind is add, max, or min.
4. Every input and output has the same static ranked tensor type.
5. The tensor element type is supported.
6. Internal reduction nodes have no external consumers.
7. The reduction has no side effects.
8. The group has at least `reduction-min-width` leaves.

Use metadata and SSA relationships.
Do not match function names or semantic display names.

## Compiler Data Model

### File Change Table

| File | Change |
|---|---|
| `include/.../mapping/ReductionTree.h` | Define group discovery, legality, and tree-building APIs |
| `lib/.../mapping/ReductionTree.cpp` | Flatten marked reductions and create deterministic trees |
| `ExpandDigitalWork.h/.cpp` | Add tree options and call the tree builder |
| `mapping/ComputeGraph.h/.cpp` | Preserve reduction tree identity on compute operations |
| `mapping/ResourceAllocationTree.cpp` | Verify reduction-node identity and stable leaves |
| `OutlineTileRoutines.cpp` | Translate mapping metadata to task-level reduction metadata |
| `SculptorAttrs.td` | Add pre-task reduction-tree attributes |
| `mapping/CMakeLists.txt` | Compile the reduction-tree source |
| `tools/ra-tree-report` | Display reduction branches, levels, and placement |

Add these files:

```text
include/sculptor-mlir/Dialect/Sculptor/Transforms/mapping/
  ReductionTree.h

lib/Dialect/Sculptor/Transforms/mapping/
  ReductionTree.cpp
```

Use these records:

```cpp
struct ReductionLeaf {
  int64_t ordinal = -1;
  Value value;
  double estimatedReadyNs = 0.0;
};

struct ReductionTreeNode {
  int64_t treeId = -1;
  int64_t nodeId = -1;
  int64_t level = -1;
  int64_t ordinal = -1;
  int64_t leftNodeId = -1;
  int64_t rightNodeId = -1;
};
```

Add pre-task mapping attributes:

```text
sculptor.mapping.reduction_tree_id
sculptor.mapping.reduction_node_id
sculptor.mapping.reduction_level
sculptor.mapping.reduction_ordinal
sculptor.mapping.reduction_width
```

`OutlineTileRoutines` translates these values to the existing task-level reduction attributes.

Add matching optional fields to `ComputeOperation`.
The compute graph verifier must reject incomplete reduction metadata.

## Group Discovery

Discover each reduction group from its final result.
Walk backward through compatible marked reduction operations.

Flatten an internal node only when:

- Its kind matches the root kind.
- Its tensor type matches the root type.
- It permits reassociation.
- It has one use inside the group.

Keep all external inputs as ordered leaves.
Use stable topological operation order and operand order for leaf ordinals.

Do not flatten a concatenation.
For a multi-row MVM, reduce column partials within each row group.
Then preserve the existing row concatenation.

## Balanced Tree Algorithm

Build the first version with deterministic adjacent pairing.

```text
level 0: [0, 1, 2, 3, 4]
level 1: [(0,1), (2,3), 4]
level 2: [((0,1),(2,3)), 4]
level 3: [(((0,1),(2,3)),4)]
```

If a level has an odd value, carry the last value to the next level.
Do not create an identity reduction task.

Create one `linalg.add`, `linalg.max`, or `linalg.min` operation for each internal node.
Use `linalg.generic` only when no named operation exists.

Copy the source-layer and semantic reduction metadata.
Assign unique mapping-stage identity to each internal node.

The tree depth must be:

```text
ceil(log2(number_of_leaves))
```

The maximum input count of each generated node is two.

## Ready-Time Policy

Implement `ready-time` only after calibrated temporal evaluation works.

The policy uses each leaf producer's predicted ready time.
It builds a deterministic priority queue with this key:

```text
estimated_ready_time, leaf_ordinal, node_id
```

Each combine operation receives the two earliest available values.
Its predicted ready time is:

```text
max(left_ready, right_ready) + calibrated_reduction_cost
```

This policy balances time, not only tree depth.
Physical placement still decides route distance and contention.

## RA Tree And Planner Behavior

`BuildRATree` sees each generated reduction node as one compute leaf.
The explicit SSA graph gives the correct fork-join dependencies.

Run `recursive-fork-join` after the tree transformation.
It must preserve independent reduction branches as spatial children.

Do not add a new mapping strategy in the first implementation.
Add a `reduction-tree-cut` strategy only if exact tests show that recursive fork-join loses the structure.

The physical placement objective uses these edges:

- Transfer cost places a reduction near its inputs.
- Makespan cost considers producer readiness and link contention.

Do not hard-code a physical reduction tile in the graph transformation.

## Shard Integration

Shard-level data flow supplies independent partial values.
Each reduction leaf must name one shard resource or one analog partial result.

The reduction output becomes a new shard only when the output covers the same logical slice.
Otherwise, mark it as an assembly boundary.

Each tree edge must become an exact `MappingWorkUnitEdge` when both endpoints are work units.
This permits each partial result to route independently.

## Routine And Runtime Behavior

Each internal reduction node becomes one digital compute routine.
The routine has two inputs and one output.

Same-tile edges become local bindings.
Cross-tile edges become routes.

The current atomic task ABI remains valid.
A reduction routine starts only after its two inputs are ready.

`verifyRoutineDependencyDAG()` must run after outlining.
The tree transformation must never introduce a routine cycle.

## Floating-Point Semantics

Reassociation changes floating-point rounding.
The extension must require explicit reassociation permission.

Use deterministic leaf order and deterministic pairing.
Record the tree policy and tree fingerprint in the IR.

Tests must use an explicit absolute and relative tolerance.
Do not claim bitwise equality for floating-point addition.

If strict order is required, keep `reduction-tree=none`.

## IR Provenance

Add these function attributes:

```text
sculptor.mapping.reduction_tree_policy
sculptor.mapping.reduction_tree_count
sculptor.mapping.reduction_node_count
sculptor.mapping.maximum_reduction_fan_in
```

Add this information to the RA-tree report:

- Reduction tree ID.
- Reduction kind.
- Leaf count.
- Depth.
- Node level.
- Logical and physical tile.
- Predicted ready and finish time.

## Diagnostics

Emit a compiler error for these conditions:

- Reduction metadata without reassociation permission.
- Mixed reduction kinds in one group.
- Mismatched input or output tensor types.
- Dynamic tensor shape.
- External use of an internal reduction node.
- Duplicate tree or node ID.
- Reduction fan-in other than two in the first version.
- Generated tree depth that exceeds the expected bound.
- Cyclic reduction or routine graph.

If an unmarked operation blocks flattening, leave the original graph unchanged.
Do not guess that the operation is associative.

## Tests

Add focused tests in `tests/python_tests/test_reduction_tree.py`.

The tests must cover:

1. Four-input add reduction with depth two.
2. Five-input add reduction with depth three.
3. Max and min reductions.
4. One-row MVM column-partial reduction.
5. Multi-row MVM reduction followed by row concatenation.
6. Stable node IDs and tree fingerprints.
7. Maximum fan-in of two.
8. Independent routes for first-level partials.
9. Local bindings when a parent is near its children.
10. Floating-point output within declared tolerance.
11. Reassociation-permission rejection.
12. RISC-V object generation for every active tile.

Add one timing test where a reduction tree beats central fan-in under the full network model.

## Incremental Delivery

### Phase 1

Discover and verify marked reduction groups.
Emit a report without changing the graph.

### Phase 2

Rewrite add reductions into deterministic balanced binary trees.
Preserve all metadata and numerical tolerance.

### Phase 3

Add max and min.
Connect exact shard edges and route records.

### Phase 4

Verify recursive fork-join mapping and makespan placement.

### Phase 5

Add the ready-time policy with calibrated costs.

## Non-Goals

This extension does not:

- Reassociate unmarked floating-point operations.
- Change analog MVM execution.
- Add in-network reduction.
- Add multicast packets.
- Select the digital worker count.
- Force a fixed physical topology for every reduction.

## Easiest Effective First Version

Rewrite only marked static `add` reductions with three or more leaves.
Build a balanced binary tree before RA-tree construction.

Then use the existing recursive fork-join planner, physical placer, outliner, and runtime graph.
