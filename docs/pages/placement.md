# Placement

Placement is the step that turns an assembled `sculptor.task_graph` into a
runtime-facing execution plan. The input graph already has task nodes,
resources, dependencies, task kinds, source-layer names, and logical analog
arrays. The scheduler decides where that work should run, then the pass records
task order, core placement, analog-array placement, transfer summaries, scoring
metadata, and runtime resource layout.

The selected strategy is controlled by the `schedule` option on
`sculptor-schedule-task-graph`. The current tree registers:

| Schedule | Placement idea |
|---|---|
| `random` | Baseline randomized physical-array order. |
| `snake` | Deterministic mesh traversal that fills local arrays before moving to the next core. |
| `greedy-heavy-edge` | Greedy placement around the heaviest communicating setup groups. |
| `manhattan-cut` | Deterministic projection of the group graph onto mesh rows or columns. |
| `boundary-aware-cut` | Recursive rectangular mesh partitioning with region capacity. |
| `boundary-aware-cut-optimized` | Boundary-aware cut plus endpoint, local-cost, and compact-path refinements. |


<details class="doc-section" open markdown="1">
<summary markdown="block">## Placement Pass Flow</summary>


`sculptor-schedule-task-graph` does more than call a strategy. The strategy
chooses placements, but the pass owns the full scheduling/finalization flow:

1. Validate the hardware budget and attach module-level budget metadata.
2. Parse each task graph function into a `TaskGraphDAG`.
3. Look up the selected `TaskGraphScheduler`.
4. Let the scheduler attach task placement attributes.
5. Fuse recognized task routines.
6. Erase unused task callees and temporary resources.
7. Rebuild the runtime execution plan.
8. Reparse the compacted task graph.
9. Finalize schedule metadata, transfer summaries, graph score, and resource
   layout.

That ordering matters. Routine fusion can remove task nodes and graph
resources, so the pass reparses the graph before final metadata is computed.
The final score and runtime resource layout describe the graph that will
actually be consumed by later lowering and runtime emission.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Hardware Model</summary>


The placement pass receives a hardware budget through pass parameters.

| Parameter | Role |
|---|---|
| `cores` | Number of runtime cores available for task placement. |
| `arrays-per-core` | Number of physical analog arrays attached to each core. |
| `topology` | Interconnect topology used for transfer-cost summaries. The current implementation supports `mesh`. |
| `mesh-rows` | Number of rows in the mesh topology. |
| `mesh-cols` | Number of columns in the mesh topology. If set to `0`, it is inferred from `cores` and `mesh-rows`. |
| `schedule` | Name of a registered placement strategy. |
| `random-seed` | Seed used by randomized schedulers. The default is `0` for reproducible output. |

The total analog array budget is:

```text
cores * arrays-per-core
```

Each physical analog array belongs to exactly one core. The pass derives the
owning core and the core-local array id from the physical array id:

```text
core_id        = physical_array_id / arrays_per_core
local_array_id = physical_array_id % arrays_per_core
```

For example, `cores=4 arrays-per-core=2` gives physical arrays `0..7`.
Physical arrays `0,1` belong to core `0`, arrays `2,3` belong to core `1`,
arrays `4,5` belong to core `2`, and arrays `6,7` belong to core `3`.

Mesh coordinates are derived from the core id:

```text
row = core_id / mesh_cols
col = core_id % mesh_cols
```

The scorer and placement algorithms use Manhattan distance over those mesh
coordinates.

```bash
sculptor-mlir-opt model.mlir \
  --sculptor-schedule-task-graph="cores=4 arrays-per-core=2 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake"
```

If `schedule` is omitted, the pass reports that a task graph schedule name is
required.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Task Graph Placement Model</summary>


The scheduler sees the graph as `TaskGraphDAG`:

| Field | Meaning |
|---|---|
| `nodes` | Parsed `sculptor.task.create` operations in graph order. |
| `predecessors` / `successors` | Dependency edges between task nodes. |
| `nodeIndexByTaskResult` | Mapping from each task result value to the producing task node. |
| `logicalArrayResources` | Task graph resources whose payload type is `!sculptor.logical.array`. |
| `dependencyCount` | Number of explicit task dependencies. |

The main placement unit is a matrix setup group. A group starts at one
`sculptor.matrix_setup` task and includes the direct `sculptor.mvm` or
`sculptor.conv_tile_mvm` tasks that consume the logical array produced by that
setup. The setup and its MVM tasks must use the same physical analog array:

```text
matrix setup -> logical array -> one or more MVM tasks
```

The shared placement helper materializes a strategy's group placement in three
steps:

1. Attach analog placement to the matrix setup and its associated MVM tasks.
2. Place unambiguous same-source-layer core-only tasks on that same core.
3. Place any remaining core-only tasks from already placed graph neighbors.

The neighbor fallback is deterministic:

1. Prefer the most recent placed producer.
2. Otherwise use the earliest placed consumer.

This is why strategy implementations usually only choose physical arrays for
matrix setup groups. The common helper then expands those decisions to the
surrounding graph.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Scoring Model</summary>


The default scorer is `mesh-transfer`. It computes the cost of data movement
created by a placement. For each task input that is produced by another task,
the scorer compares the producer core and consumer core:

```text
transfer_cost = resource_bytes * ManhattanDistance(source_core, destination_core)
```

If both tasks are on the same core, that edge contributes no inter-core
transfer cost. Otherwise the scorer accumulates:

| Metadata | Meaning |
|---|---|
| `sculptor.schedule.core_transfer_bytes` | Flattened `source_core * num_cores + destination_core` byte matrix. |
| `sculptor.schedule.core_transfer_cost` | Flattened transfer-cost matrix with the same indexing. |
| `sculptor.schedule.inter_core_transfer_bytes` | Total bytes crossing core boundaries. |
| `sculptor.schedule.total_transfer_cost` | Total Manhattan-distance weighted transfer cost. |
| `sculptor.schedule.boundary_penalty` | Extra penalty when first and last tasks do not share a mesh boundary. |
| `sculptor.schedule.graph_score` | `total_transfer_cost + boundary_penalty`. |

The boundary penalty is a soft endpoint constraint. If the first and last tasks
do not share at least one mesh edge boundary, the scorer adds a 20% surcharge:

```text
boundary_penalty = ceil(total_transfer_cost / 5)
```

This makes endpoint placement visible in the score without making non-boundary
placements illegal.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Shared Routine Fusion</summary>


Routine fusion is a post-scheduler cleanup step, not a property of one
placement strategy. It currently recognizes the convolution tile pipeline:

```text
digital.conv_patch
-> digital.vector_tile
-> sculptor.mvm
-> digital.tile_recombine
-> digital.bias_add
```

When the pattern appears as a task chain, the fuser combines it into a single
`sculptor.conv_tile_mvm` task. The fused routine keeps only the external ABI
that is still needed:

```text
activation input, logical array input -> final output
```

Internal temporary task graph resources that only connected fused tasks are
erased. After fusion, the pass rebuilds resource slots, temporary indices,
temporary offsets, workspace size, and task indices so downstream runtime code
sees a compact graph.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Random Schedule</summary>


`random` is the baseline randomized scheduler. It builds the physical array
order from the hardware budget, shuffles that order with `random-seed`, then
assigns matrix setup groups in graph order through the shared placement helper.

The scheduler is intentionally simple:

```text
physical_array_order = shuffled(budget.analog_arrays, random_seed)
group[i]             = physical_array_order[i % physical_array_order.size()]
```

Because the seed defaults to `0`, test output is reproducible unless the pass is
invoked with a different `random-seed`.

This schedule is useful as a baseline. It exercises all shared placement,
fusion, metadata, and scoring machinery without adding algorithmic placement
policy.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Snake Schedule</summary>


`snake` is a deterministic mesh traversal scheduler. It walks the mesh from the
top-left core across the first row, drops to the next row, reverses direction,
and repeats.

For a `2x3` mesh, the core order is:

```text
0 -> 1 -> 2
          |
5 <- 4 <- 3
```

For each core on the path, `snake` uses every local physical array before moving
to the next core. With `arrays-per-core=2` on a `2x2` mesh, the physical array
order is:

```text
core path:       0,      1,      3,      2
physical arrays: [0, 1], [2, 3], [6, 7], [4, 5]
```

Like `random`, `snake` is mainly an ordering strategy. It produces a stable
physical-array order and lets the shared helper place the MVMs, same-source
tasks, and remaining core-only tasks.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Greedy Heavy-Edge Schedule</summary>


`greedy-heavy-edge` tries to keep high-traffic setup groups close together. It
first builds a reduced weighted graph:

```text
node = matrix setup group
edge = task-resource flow between two setup groups
weight = sum(resource byte sizes crossing between those groups)
```

Edges are undirected for placement purposes. A resource produced by one group
and consumed by another contributes its byte size to the edge between those
groups. Multiple resources between the same two groups accumulate onto one
edge.

The algorithm then runs in three stages:

1. Sort edges by weight.
2. Select the heaviest edges until they cover 80% of total group communication.
3. Build connected components from those heavy edges and place those components
   first.

Inside each heavy component, the first anchor is the unplaced group with the
largest incident communication weight. The next group comes from the component
frontier: the strongest edge from an already placed group to an unplaced group
wins, with incident weight and group index as deterministic tie-breakers.

Slot selection is cost-based. For a candidate physical array slot:

```text
candidate_cost =
  sum(edge_bytes * ManhattanDistance(candidate_core, placed_neighbor_core))
```

Only currently least-used slots are considered first, which keeps placement
balanced when there are more groups than physical arrays. The scheduler also
prefers slots near already placed neighbors before falling back to the full slot
set.

With multiple arrays per core, every local array is a separate candidate slot.
Two heavily communicating groups can therefore occupy different arrays on the
same core and pay zero mesh-transfer cost.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Manhattan Cut Schedule</summary>


`manhattan-cut` uses the same weighted setup-group graph as
`greedy-heavy-edge`, but it creates a deterministic global ordering instead of
growing from a frontier.

The scheduler computes two graph-coordinate vectors:

1. `phiX`, using group `0` and the farthest finite group from it as anchors.
2. `phiY`, using the highest-incident-weight group and the farthest finite
   group from it as anchors.

Distances are weighted so heavier communication behaves like a shorter graph
distance:

```text
edge_length = max(1, max_edge_weight / edge_weight)
coordinate  = distance_from_anchor_a - distance_from_anchor_b
```

Those coordinates are projected onto the mesh in two ways:

```text
x-first: sort by phiX, slice by mesh column, sort each slice by phiY
y-first: sort by phiY, slice by mesh row, sort each slice by phiX
```

For each projection, the scheduler also tries primary and secondary orientation
flips. Each candidate placement is scored with the exact weighted group transfer
cost over the mesh. The candidate score also includes the same soft boundary
surcharge used by the final graph scorer, so endpoint placement participates in
candidate selection.

Arrays-per-core acts as slice capacity. For `x-first`, each mesh column can
hold:

```text
mesh_rows * arrays_per_core
```

groups before placement moves to the next column. For `y-first`, each row can
hold:

```text
mesh_cols * arrays_per_core
```

groups. The final placement remains physical-array based, while the projection
and score reason about each slot's owning core.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Boundary-Aware Cut Schedule</summary>


`boundary-aware-cut` recursively partitions the physical mesh. It treats the
mesh as rectangular regions and gives each region capacity equal to:

```text
region_cores * arrays_per_core
```

The root region is the whole mesh. A region becomes a leaf when either:

1. The region is one core.
2. The assigned groups fit inside one core's local array capacity.

Otherwise, the scheduler evaluates real mesh cuts. For a rectangular region it
tries horizontal and vertical splits near the midpoint, including midpoint
`-1`, midpoint, and midpoint `+1` when those cuts are valid. The longer region
dimension is considered first.

For each candidate split, group capacity is divided proportionally to child
region capacity. Groups are then assigned to the two child regions by a greedy
partitioning rule:

1. Seed the first child with a high-incident-weight group.
2. Seed the second child with a group that is weakly connected to the first
   seed.
3. Assign remaining groups to the side where they have stronger existing
   connection, while respecting child capacity.

The split score is:

```text
split_score = cut_weight + balance_waste
```

`cut_weight` is the total byte weight of group edges crossing the split.
`balance_waste` penalizes unused region capacity. The lowest-score feasible
split is chosen, and recursion continues.

At a leaf, groups are assigned to physical array slots in that region. The
baseline version is intentionally direct: it gives a real recursive,
capacity-aware placement, but does not do graph coarsening, FM refinement, or
uncoarsening repair.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Boundary-Aware Cut Optimized Schedule</summary>


`boundary-aware-cut-optimized` starts with the recursive rectangular placement
idea from `boundary-aware-cut`, then adds bounded refinements that are designed
to improve the score without making the scheduler expensive or unpredictable.

### Endpoint corridors

The final scorer adds a penalty when the first and last tasks do not share a
mesh boundary edge. The optimized scheduler makes that preference explicit for
setup-group placement. It maps the first and last DAG nodes back to their
matrix setup groups when task-group ownership can identify them. If both
endpoint groups are known, the scheduler tries four endpoint corridors:

```text
top, bottom, left, right
```

For each corridor, both endpoint groups must be placed on that same boundary.
Recursive partitioning is constrained so a child region must retain enough
boundary capacity for any endpoint group assigned to it. After all corridors are
evaluated, the scheduler keeps the feasible placement with the lowest exact
weighted transfer score.

The final scorer still treats boundary alignment as a soft penalty. The
optimized scheduler uses the same idea as an active placement preference when
both endpoint groups are known. If either endpoint group cannot be identified,
the optimized scheduler falls back to unconstrained recursive placement.

### Cost-aware leaf assignment

The baseline boundary-aware scheduler assigns leaf groups in sorted order. The
optimized scheduler uses the same leaf regions, but chooses unused slots by
incremental communication cost.

At each leaf:

1. Endpoint groups are considered first so corridor constraints are preserved.
2. Higher-incident-weight groups are placed before lower-traffic groups.
3. Each group chooses the unused slot with the lowest cost to already placed
   neighboring groups.

The incremental cost is:

```text
sum(edge_bytes * ManhattanDistance(candidate_core, placed_neighbor_core))
```

This makes local leaf assignment aware of communication that was already fixed
by earlier recursive choices.

### Pair-swap refinement

After recursive placement, the optimizer runs a bounded exact swap pass. It
tries swapping the physical arrays of every pair of placed setup groups and
computes the exact weighted transfer score. A swap is accepted only when:

1. It improves the exact score.
2. It preserves the endpoint corridor constraint.

The current implementation runs at most four swap passes. This catches simple
local inversions without turning the scheduler into a full search.

### Hot-edge moves

Pair swaps can only exchange occupied slots. Hot-edge moves use unused physical
array slots when extra capacity exists.

The optimizer sorts weighted group edges by their current transfer contribution:

```text
edge_contribution = edge_bytes * ManhattanDistance(group_a_core, group_b_core)
```

For the hottest edges, it considers moving one endpoint into an unused slot. A
move is considered only if it does not make that endpoint farther from the
other endpoint of the hot edge. The move is accepted only if it improves the
exact total weighted transfer score and preserves any endpoint corridor rule.

This refinement is most useful when `arrays-per-core` or mesh capacity leaves
unused slots that can absorb high-traffic groups.

### Compact-chain path repacking

Some graphs are naturally chain-like. A legal low-cost placement can still look
spread out, leaving little contiguous space for future work. The compact-chain
refinement searches for a denser path placement without increasing exact
transfer cost.

The optimizer builds connected snake paths through compact rectangular mesh
regions between the first endpoint core and a candidate endpoint core on the
selected corridor. It then repacks ordered setup groups monotonically along
each path:

```text
group order follows path order
each core can hold arrays_per_core groups
first endpoint stays at the path start
last endpoint stays at the path end
```

Candidate assignments are exact-scored first. A candidate is rejected if it
increases weighted transfer cost. If the transfer score is equal, the optimizer
uses a compactness tie-breaker:

```text
compact_score =
  bounding_box_area * 1000
  + occupied_core_count * 100
  + squared_adjacent_chain_distance
```

This favors placements that use fewer cores, a smaller mesh rectangle, and
shorter consecutive chain hops. The implementation bounds this search to small
ordered group sets, so it remains a local refinement rather than an exhaustive
global optimizer.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Scheduler Interface</summary>


Placement strategies are implemented through a shared C++ interface:

```cpp
class TaskGraphScheduler {
public:
  virtual ~TaskGraphScheduler() = default;

  virtual StringRef getName() const = 0;

  virtual LogicalResult
  schedule(ModuleOp module, func::FuncOp taskGraphFunc,
           const HardwareBudget &budget, const TaskGraphDAG &dag) const = 0;
};
```

A scheduler usually computes a list of `MatrixSetupGroupPlacement` records and
then calls the shared helper:

```cpp
placeMatrixSetupGroupsAndSurroundingTasks(module, taskGraphFunc, budget, dag,
                                          groupPlacements);
```

Once registered with the task graph scheduler registry, a strategy can be
selected with:

```text
schedule=<registered-strategy-name>
```

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Scheduled Metadata</summary>


After scheduling and finalization, the task graph and task functions carry
runtime-facing metadata.

| Metadata | Meaning |
|---|---|
| Task index | Stable runtime order for task execution. |
| Core id | Runtime core assigned to the task. |
| Physical array id | Analog array assigned to matrix setup and MVM tasks. |
| Local array id | Core-local analog array id derived from physical array id. |
| Logical array placement | Mapping from logical arrays to physical arrays. |
| Transfer summary | Inter-core byte movement and mesh-distance transfer cost. |
| Graph score | Transfer cost plus any boundary penalty. |
| Digital op count | Static estimate of digital work attached to scheduled tasks. |
| Runtime resource layout | Resource slots, temporary indices, temporary offsets, and workspace size after any task fusion. |

Later lowering and graph emission consume this metadata instead of recomputing
placement decisions from the task graph.

</details>
