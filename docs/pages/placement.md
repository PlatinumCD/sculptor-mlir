# Placement

Placement is the step that decides where an assembled task graph should run.
The input is already expressed as `sculptor.task.create` operations, task graph
resources, explicit dependencies, and logical analog arrays. A scheduler turns
that symbolic graph into runtime-facing metadata: task order, core assignment,
array assignment, and transfer summaries.

The compiler keeps placement as an extension point. The
`sculptor-schedule-task-graph` pass builds the hardware budget, parses the task
graph, and dispatches through the scheduler interface. This tree registers a
`random` scheduler that assigns matrix setup/MVM groups to physical arrays,
places related digital tasks on cores, fuses recognized task routines, and
rebuilds the runtime resource layout.


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
| `schedule` | Name of a registered placement strategy, currently `random`. |

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
