# Placement

Placement is the step that decides where an assembled task graph should run.
The input is already expressed as `sculptor.task.create` operations, task graph
resources, explicit dependencies, and logical analog arrays. A scheduler turns
that symbolic graph into runtime-facing metadata: task order, core assignment,
array assignment, a mesh-based graph score, and transfer summaries.

The compiler keeps placement as an extension point. The
`sculptor-schedule-task-graph` pass builds the hardware budget, parses the task
graph, and dispatches through the scheduler interface. This tree registers
`random`, `snake`, `greedy-heavy-edge`, `manhattan-cut`,
`boundary-aware-cut`, and `boundary-aware-cut-optimized` schedulers.
`random` assigns matrix setup/MVM groups to a shuffled physical-array order.
`snake` assigns those same setup groups by walking the mesh in a deterministic
snake order. `greedy-heavy-edge` builds a weighted communication graph between
setup groups and places the heaviest communicating groups first.
`manhattan-cut` builds deterministic cut coordinates, projects the group graph
onto rows and columns, and keeps the lower-cost projection.
`boundary-aware-cut` recursively partitions setup groups onto rectangular mesh
regions with explicit region capacity. `boundary-aware-cut-optimized` starts
from that recursive strategy and then applies local communication-cost
refinements. All registered schedulers use the shared matrix-setup-group
placement helper, so associated MVMs and surrounding same-source-layer digital
tasks stay with their setup group.


<details class="doc-section" open markdown="1">
<summary markdown="block">## Placement Parameters</summary>


The placement pass receives a hardware budget through pass parameters.

| Parameter | Role |
|---|---|
| `cores` | Number of runtime cores available for task placement. |
| `arrays-per-core` | Number of physical analog arrays attached to each core. |
| `topology` | Interconnect topology used for transfer-cost summaries. The current implementation supports `mesh`. |
| `mesh-rows` | Number of rows in the mesh topology. |
| `mesh-cols` | Number of columns in the mesh topology. If set to `0`, it is inferred from `cores` and `mesh-rows`. |
| `schedule` | Name of a registered placement strategy, currently `random`, `snake`, `greedy-heavy-edge`, `manhattan-cut`, `boundary-aware-cut`, or `boundary-aware-cut-optimized`. |
| `random-seed` | Seed used by randomized schedulers. The default is `0` for reproducible output. |

The total analog array budget is:

```text
cores * arrays-per-core
```


For example, this provides a four-core hardware budget:

```bash
sculptor-mlir-opt model.mlir \
  --sculptor-schedule-task-graph="cores=4 arrays-per-core=2 schedule=random"
```

If `schedule` is omitted, the pass reports that a task graph schedule name is
required.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Random Schedule</summary>


`random` is the current built-in scheduler. It uses a deterministic random seed
so test output is stable, but the placement policy is intentionally simple.

The scheduler first finds each `sculptor.matrix_setup` task and the
`sculptor.mvm` tasks that depend on it. Each setup group is assigned to one
physical analog array from the hardware budget. The owning core and local array
id are derived from that physical array id:

```text
core = physical-array-id / arrays-per-core
local-array = physical-array-id % arrays-per-core
```


The matrix setup and all of its associated MVM tasks receive the same physical
array placement. Digital tasks from the same unambiguous source layer are placed
on that source layer's core. Remaining core-only tasks are placed by looking at
already placed neighboring tasks:

- Prefer the most recent placed producer.
- Otherwise use the earliest placed consumer.

After placement, the pass runs task-routine fusion. The main implemented fusion
pattern recognizes:

```text
digital.conv_patch
-> digital.vector_tile
-> sculptor.mvm
-> digital.tile_recombine
-> digital.bias_add
```


That chain becomes a single `sculptor.conv_tile_mvm` task whose external ABI
keeps only the original activation input, the logical array input, and the final
bias-add output. Temporary task-graph resources that only represented internal
fused boundaries are erased, and the runtime execution plan is recomputed so
slots, temporary offsets, workspace size, and task indices match the compacted
graph.

Final schedule metadata also scores the graph for the configured mesh. The base
score is the total inter-core transfer cost: each produced resource consumed on
a different core contributes `resource-bytes * Manhattan mesh distance`. The
scorer also adds a boundary penalty when the first and last tasks do not share a
mesh boundary edge. The penalty is a 20% surcharge on the total transfer cost.
The final result is recorded as
`sculptor.schedule.graph_score`, alongside `boundary_penalty`,
`core_transfer_bytes`, `core_transfer_cost`, `inter_core_transfer_bytes`, and
`total_transfer_cost`.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Snake Schedule</summary>


`snake` is a deterministic mesh traversal scheduler. It assigns matrix setup
groups by walking the mesh from the top-left core across each row, reversing
direction on alternating rows. For a `2x3` mesh, the core order is:

```text
0 -> 1 -> 2
          |
5 <- 4 <- 3
```

For each core on the path, `snake` uses every local physical array before moving
to the next core. With `arrays-per-core=2` on a `2x2` mesh, the physical array
order is:

```text
core path:      0, 1, 3, 2
physical arrays [0, 1], [2, 3], [6, 7], [4, 5]
```

As with `random`, each matrix setup's associated MVM tasks are placed on the
same physical array, same-source-layer digital tasks are placed on the owning
core, and remaining core-only tasks are placed from already mapped graph
neighbors.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Greedy Heavy-Edge Schedule</summary>


`greedy-heavy-edge` places matrix setup groups by first building a reduced
weighted graph. Each matrix setup group is one node, and each task-resource flow
between groups contributes its byte size to an undirected weighted edge. The
scheduler keeps the heaviest edges until they cover 80% of total group
communication, places those heavy components first, and then attaches the
remaining groups with a frontier priority rule.

Placement is still physical-array based. With multiple arrays per core, each
local array is a separate candidate slot:

```text
physicalArrayId = coreId * arraysPerCore + localArrayId
```

Candidate cost uses the candidate slot's core:

```text
edgeBytes * ManhattanDistance(candidateCore, placedNeighborCore)
```

So heavily communicating groups can land on different local arrays of the same
core with zero mesh-transfer cost before the algorithm spills to nearby cores.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Manhattan Cut Schedule</summary>


`manhattan-cut` uses the same matrix setup group graph as
`greedy-heavy-edge`, but it chooses placements by projection instead of frontier
growth. The scheduler computes two deterministic graph coordinates from weighted
communication distances, sorts groups into `x` and `y` orderings, and evaluates
both projection directions:

```text
x-first: slice by x ordering, sort each slice by y
y-first: slice by y ordering, sort each slice by x
```

For each projection, the scheduler also tries orientation flips. Each candidate
is scored with exact weighted group transfer cost over the mesh cores, with the
same soft boundary surcharge used by the graph scorer. The best candidate is
then materialized through the shared matrix setup placement helper.

Multiple arrays per core are handled as physical array capacity inside each
row or column slice. The placement remains physical-array based, while
projection and scoring use the slot's owning core.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Boundary-Aware Cut Schedule</summary>


`boundary-aware-cut` is the first recursive region scheduler. It represents the
mesh as rectangular regions and gives each region capacity equal to:

```text
region cores * arrays-per-core
```

Each split is a real mesh cut. The scheduler evaluates candidate horizontal and
vertical splits near the region midpoint, partitions matrix setup groups into
the child regions with capacity constraints, and chooses the feasible split with
the lowest weighted cut traffic. Recursion continues until a region is a single
core or its groups fit within one core's local-array capacity.

At the leaves, groups are assigned to physical array slots inside the region:

```text
physicalArrayId = coreId * arraysPerCore + localArrayId
```

This implementation is intentionally the first operational version of the
multilevel idea. It has real region capacity and recursive cut placement, but
does not include graph coarsening, FM refinement, or uncoarsening repair.
`boundary-aware-cut-optimized` is registered separately so refinements can be
developed and compared without replacing the baseline. It currently adds:

- explicit endpoint corridor placement: top, bottom, left, and right boundary
  corridors are evaluated independently, and the first and last task groups must
  stay on the chosen corridor;
- cost-aware assignment inside leaf regions, using already placed neighboring
  groups to choose lower-cost physical array slots;
- bounded exact pair-swap refinement over placed setup groups;
- bounded hot-edge endpoint moves into unused physical array slots when extra
  local-array capacity is available.
- bounded compact-chain path repacking, which tries connected snake paths over
  compact mesh rectangles and repacks ordered setup groups onto those paths
  without increasing exact transfer cost.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## The Placement Problem</summary>


The placement problem is to map graph-level work onto hardware resources while
preserving task dependencies and resource ownership.

The scheduler must answer several questions:

- Which task runs first, second, third, and so on?
- Which core owns each task?
- Which physical analog array backs each logical array?
- Which tasks need analog array placement metadata?
- Which task outputs move between cores?
- What transfer cost should be recorded for those movements?

The task graph gives the scheduler the dependency structure. The hardware budget
gives it the available resources. The selected strategy decides how to connect
the two.

The result is not a new graph shape. Instead, the pass annotates the existing
task graph and task functions with runtime-facing metadata that later lowering
and runtime emission can consume.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Strategy Interface</summary>


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


A new strategy derives from this interface, gives itself a name, and implements
`schedule`. Once registered, it can be selected with the `schedule` parameter.

```text
schedule=<registered-strategy-name>
```


This is an internal extension point. New strategies are added to the compiler
and registered with the task graph scheduler registry.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Strategy Inputs</summary>


Every placement strategy receives the same core inputs.

| Input | What it provides |
|---|---|
| `ModuleOp module` | The whole MLIR module. This lets a strategy inspect or annotate task callee functions. |
| `func::FuncOp taskGraphFunc` | The task graph function being scheduled. This is where graph-level schedule metadata is attached. |
| `HardwareBudget budget` | The validated hardware budget: cores, arrays per core, topology, mesh dimensions, and physical analog array ids. |
| `TaskGraphDAG dag` | Parsed task nodes, dependencies, successors, predecessors, task result mapping, and logical array resources. |

These inputs are enough for a strategy to compute task placement without having
to rediscover the graph from raw IR. The pass has already parsed the task graph
into a DAG and validated the hardware budget before calling the selected
strategy.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Scheduled Metadata</summary>


After a strategy runs, the task graph and task functions carry runtime-facing
metadata.

| Metadata | Meaning |
|---|---|
| Task index | Stable runtime order for task execution. |
| Core id | Runtime core assigned to the task. |
| Physical array id | Analog array assigned to matrix setup and MVM tasks. |
| Logical array placement | Mapping from logical arrays to physical arrays. |
| Transfer summary | Inter-core byte movement and mesh-distance transfer cost. |
| Digital op count | Static estimate of digital work attached to scheduled tasks. |
| Runtime resource layout | Resource slots, temporary indices, temporary offsets, and workspace size after any task fusion. |

This metadata is consumed by later runtime lowering and graph emission passes.

</details>
