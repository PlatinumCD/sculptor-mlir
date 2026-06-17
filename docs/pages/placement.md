# Placement

Placement is the step that decides where an assembled task graph should run.
The input is already expressed as `sculptor.task.create` operations, task graph
resources, explicit dependencies, and logical analog arrays. The scheduler turns
that symbolic graph into runtime-facing metadata: task order, core assignment,
array assignment, and transfer summaries.

Placement is controlled by `sculptor-schedule-task-graph`, which is included in
the `sculptor-lower-golem-to-task-graph` pipeline.

```bash
sculptor-mlir-opt model.mlir \
  --sculptor-lower-golem-to-task-graph="cores=4 arrays-per-core=2 schedule=simple-budget"
```


<details class="doc-section" open markdown="1">
<summary markdown="block">## Placement Parameters</summary>


The placement pass receives a hardware budget and a strategy selection through
pass parameters.

| Parameter | Role |
|---|---|
| `cores` | Number of runtime cores available for task placement. |
| `arrays-per-core` | Number of physical analog arrays attached to each core. |
| `topology` | Interconnect topology used for transfer-cost summaries. The current implementation supports `mesh`. |
| `mesh-rows` | Number of rows in the mesh topology. |
| `mesh-cols` | Number of columns in the mesh topology. If set to `0`, it is inferred from `cores` and `mesh-rows`. |
| `schedule` | Placement strategy to run. Current strategies are `simple-budget` and `layer-placement`. |
| `placement` | Optional strategy-specific placement vector. It is currently used by `layer-placement`. |

The total analog array budget is:

```text
cores * arrays-per-core
```


For example, this asks for four cores, two arrays per core, and explicit
boundary placement:

```bash
sculptor-mlir-opt model.mlir \
  --sculptor-lower-golem-to-task-graph="cores=4 arrays-per-core=2 schedule=layer-placement placement=0,2,1,3"
```


In that example, `placement=0,2,1,3` is interpreted by the selected strategy.
For `layer-placement`, each entry maps one placement boundary to one core.

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
           const HardwareBudget &budget, const TaskGraphDAG &dag,
           const TaskGraphScheduleOptions &options) const = 0;
};
```


A new strategy derives from this interface, gives itself a name, and implements
`schedule`. Once registered, it can be selected with the `schedule` parameter.

```text
schedule=simple-budget
schedule=layer-placement
schedule=my-new-strategy
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
| `TaskGraphScheduleOptions options` | Strategy-specific options. Currently this carries the parsed `placement` vector. |

These inputs are enough for a strategy to compute task placement without having
to rediscover the graph from raw IR. The pass has already parsed the task graph
into a DAG and validated the hardware budget before calling the selected
strategy.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Available Strategies</summary>


`simple-budget` is the default strategy. It maps logical arrays to physical
arrays in order and derives the owning core from the physical array id.

```text
logical array 0 -> physical array 0
logical array 1 -> physical array 1
logical array 2 -> physical array 2
```


```text
core = physical-array-id / arrays-per-core
```


This is deterministic and useful as a baseline. It is not intended to be a
cost optimizer.

`layer-placement` uses the same hardware budget, but requires an explicit
placement vector. It forms placement boundaries from logical analog arrays and
maps each boundary to a requested core.

```text
placement=0,2,1,3

boundary 0 -> core 0
boundary 1 -> core 2
boundary 2 -> core 1
boundary 3 -> core 3
```


The current implementation uses one-boundary-per-core mode. Each core must
appear exactly once, and each boundary's logical array is mapped to the first
physical array on that core.

```text
physical array = core * arrays-per-core
```


</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Boundary Formation</summary>


`layer-placement` creates boundaries from logical analog arrays.

1. A `sculptor.matrix_setup` task creates a boundary for its output logical
   array.
2. A `sculptor.mvm` task joins the boundary for the logical array that it
   consumes.
3. Remaining tasks are attached by walking task graph dependencies.

For remaining tasks, the scheduler looks at already-attached neighbors:

- If predecessors already belong to boundaries, the task joins the highest
  predecessor boundary.
- Otherwise, if successors already belong to boundaries, the task joins the
  lowest successor boundary.
- If the task cannot attach to any boundary, scheduling fails.

The resulting model is:

```text
logical array -> placement boundary
matrix setup -> creates boundary
MVM -> joins logical-array boundary
other tasks -> attach through dependency neighbors
```


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

This metadata is consumed by later runtime lowering and graph emission passes.

</details>
