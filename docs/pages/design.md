# IR Design

The Sculptor dialect is organized around a staged lowering model. Each group of
operations represents a different level of intent: semantic neural-network
layers, analog execution, accelerator-facing array operations, task boundaries,
and task graph construction.

<details class="doc-section" open markdown="1">
<summary markdown="block">## Passes And Pipelines</summary>


The pass structure follows the same staged model as the IR. Early passes recover
or preserve model meaning. Middle passes expose analog execution and task
boundaries. Later passes assemble, schedule, and lower the task graph into
runtime-facing code.

### Pipelines

Sculptor provides two named pipelines for the main lowering path.

| Pipeline | Role |
|---|---|
| `sculptor-lower-to-golem` | Lowers recognized model structure into callable Golem task functions. |
| `sculptor-lower-golem-to-task-graph` | Builds the task graph, places/schedules it, and rewrites Golem array operations to runtime shim calls. |

The pipelines are intentionally split at the Golem/task boundary. The first
pipeline produces task-shaped Golem IR. The second pipeline consumes that shape
and attaches the runtime execution plan.

```bash
sculptor-mlir-opt model.mlir \
  --sculptor-lower-to-golem="array-rows=128 array-cols=128" \
  --sculptor-lower-golem-to-task-graph="cores=4 arrays-per-core=2 schedule=random"
```


`sculptor-lower-to-golem` accepts `array-rows` and `array-cols`, which define the
physical array tile size used when expanding `sculptor.mvm`.
`sculptor-lower-golem-to-task-graph` accepts hardware budget options such as
`cores`, `arrays-per-core`, `topology`, `mesh-rows`, and `mesh-cols`, plus a
registered scheduler name. The current tree provides `schedule=random`, which
places matrix setup/MVM groups on physical arrays, places related digital work
on cores, fuses recognized task routines, and recomputes the runtime resource
layout.

### `sculptor-lower-to-golem`

This pipeline turns model-level IR into task-shaped Golem IR.

1. `sculptor-canonicalize-layers`
   Recovers supported layer structure from Torch-MLIR or `linalg`-style IR and
   rewrites it as inline `sculptor.nn.*` operations.
2. `sculptor-extract-layers`
   Outlines recognized layer regions from `forward` into separate layer
   functions.
3. `sculptor-convert-layers`
   Lowers extracted `sculptor.nn.*` layer functions to `sculptor.mvm` plus
   standard tensor, linalg, math, or control-flow glue.
4. `sculptor-expand-mvm-to-golem`
   Expands each `sculptor.mvm` into Golem array setup, vector tiling, array
   execution, store, and recombine task regions.
5. `sculptor-materialize-tasks`
   Turns `sculptor.task_region` boundaries into private task functions with task
   metadata, then rewrites `forward` to call those tasks.

### `sculptor-lower-golem-to-task-graph`

This pipeline turns materialized Golem tasks into a scheduled runtime graph.

1. `sculptor-assemble-task-graph`
   Builds `generate_task_graph` with `sculptor.task_graph.*` resources and
   `sculptor.task.create` nodes. The materialized `forward` function may still
   be present at this point as a direct call form of the same tasks.
2. `sculptor-schedule-task-graph`
   Attaches task order, core assignment, logical array placement, transfer
   metadata, routine fusion, and runtime resource layout. Once the task graph
   is the live representation, the pass removes the stale materialized
   `forward` entry point and unused generated task callees.
3. `sculptor-lower-golem-to-llvm-shims`
   Rewrites scheduled Golem array operations into LLVM-callable runtime shim
   calls.

### Main Lowering Passes

| Pass | Input shape | Output shape |
|---|---|---|
| `sculptor-canonicalize-layers` | Torch-MLIR or `linalg`-style layer bodies in `forward`. | Inline `sculptor.nn.*` layer ops. |
| `sculptor-extract-layers` | Inline recognized layer regions in `forward`. | Separate layer functions called by `forward`. |
| `sculptor-convert-layers` | Extracted `sculptor.nn.*` layer functions. | `sculptor.mvm` plus standard tensor, linalg, math, or control-flow glue. |
| `sculptor-expand-mvm-to-golem` | `sculptor.mvm` inside layer/helper functions. | Golem array setup, vector tiling, array execution, store, and recombine task regions. |
| `sculptor-materialize-tasks` | `sculptor.task_region` boundaries. | Private task functions with task metadata, called from `forward`. |
| `sculptor-assemble-task-graph` | A `forward` function that calls materialized task functions. | Materialized task functions plus `generate_task_graph` with `sculptor.task_graph.*` resources and `sculptor.task.create` nodes. |
| `sculptor-schedule-task-graph` | An assembled task graph. | Scheduled task graph metadata, live private task functions, and no stale materialized `forward` entry point. |
| `sculptor-lower-golem-to-llvm-shims` | Scheduled task functions containing Golem array operations. | Calls to LLVM-callable Golem runtime shims. |

### Export And Runtime Passes

These passes sit beside the main lowering path. They consume task-graph or
runtime-shaped IR and produce external artifacts or final backend forms.

| Pass | Role |
|---|---|
| `sculptor-export-task-graph-vis` | Writes an assembled task graph visualization as DOT or GraphML. |
| `sculptor-export-task-graph-sim-model` | Writes a scheduled task graph model for external placement or simulation tooling. |
| `sculptor-finalize-golem-intrinsics` | Rewrites LLVM Golem shim calls into target Golem ISA intrinsics. |
| `sculptor-emit-runtime-graph` | Emits generic runtime graph metadata and task-entry shims after the task graph has runtime layout metadata. |

After the Sculptor-specific pipeline, normal MLIR passes handle bufferization,
conversion to LLVM-compatible dialects, and final cleanup.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Custom Types</summary>


The shared custom types keep the IR explicit about when a value is still
tensor-shaped data, when it has become an analog container, and when it is a
task graph handle.

| Type group | Types | Role |
|---|---|---|
| Analog containers | `!sculptor.matrix`, `!sculptor.vector` | Matrix and vector values intended for analog execution. |
| Analog views | `!sculptor.matrix.grid`, `!sculptor.vector.slice` | Tiled views over matrix/vector containers. |
| Array handles | `!sculptor.logical.array`, `!sculptor.array.result` | Logical accelerator array state and opaque execution results. |
| Task graph handles | `!sculptor.task_graph`, `!sculptor.task` | Symbolic task graph and task node handles. |
| Runtime/resource handles | `!sculptor.runtime_handle`, `!sculptor.task_resource<T>` | Runtime-owned state and graph resource slots carrying typed payloads. |

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Operation Groups</summary>


The dialect currently has five operation groups.

### Neural-Network Ops

Neural-network ops represent semantic layers after the compiler has recognized
patterns from Torch-MLIR or `linalg` IR. They preserve layer meaning before the
program is lowered into analog execution primitives.

| Operation family | Operations |
|---|---|
| Feed-forward layers | `sculptor.nn.linear`, `sculptor.nn.conv1d`, `sculptor.nn.conv2d`, `sculptor.nn.grouped_conv2d`, `sculptor.nn.conv3d` |
| Recurrent cells | `sculptor.nn.rnn_cell`, `sculptor.nn.lstm_cell`, `sculptor.nn.gru_cell` |
| Single recurrent layers | `sculptor.nn.rnn_layer`, `sculptor.nn.lstm_layer`, `sculptor.nn.gru_layer` |
| Whole recurrent modules | `sculptor.nn.rnn`, `sculptor.nn.lstm`, `sculptor.nn.gru` |

This group is intentionally high level. It is the IR that says, "this is a
linear layer" or "this is an LSTM layer," instead of exposing every tensor slice
and elementwise operation that originally produced the same result.

### MVM Execution Ops

The execution group currently contains:

| Operation | Role |
|---|---|
| `sculptor.mvm` | Represents a row-vector by matrix-vector multiply at the analog execution level. |

`sculptor.mvm` is the narrow bridge between semantic neural-network layers and
accelerator-facing array IR. Layer conversion reduces supported layer work into
one or more MVM-shaped units before later passes expand those units for Golem
execution.

### Golem Ops

Golem ops describe accelerator-facing matrix, vector, and logical array
operations.

| Operation family | Operations |
|---|---|
| Matrix containers | `sculptor.matrix.from_tensor`, `sculptor.matrix.partition` |
| Vector containers | `sculptor.vector.from_tensor`, `sculptor.vector.partition` |
| Placement | `sculptor.array.matrix.place`, `sculptor.array.vector.place` |
| Logical array execution | `sculptor.array.set`, `sculptor.array.load`, `sculptor.array.execute`, `sculptor.array.store` |

This group is lower than `sculptor.mvm`. It makes matrix/vector tiling, placement,
array programming, execution, and result storage visible in the IR.

### Task Region Ops

Task region ops describe compiler-internal task boundaries before the task graph
exists.

| Operation | Role |
|---|---|
| `sculptor.task_region` | Groups a fragment of IR that should become one task-stage boundary. |
| `sculptor.yield` | Terminates a task region and returns values to the parent IR. |

`sculptor.task_region` is structural IR. It is useful because earlier passes can
mark task boundaries without committing to graph resources, scheduling metadata,
or runtime layout.

### Task Graph Ops

Task graph ops describe the symbolic runtime graph before it is emitted as
runtime metadata.

| Operation family | Operations |
|---|---|
| Graph construction | `sculptor.task_graph.create` |
| Graph resources | `sculptor.task_graph.input`, `sculptor.task_graph.output`, `sculptor.task_graph.temporary`, `sculptor.task_graph.persistent` |
| Task nodes | `sculptor.task.create` |

The task graph IR makes dependencies and resources explicit. Each task records
its callee, domain, task kind, task name, source layer, ordinal, inputs, outputs,
and dependencies.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Lowering Shape</summary>


The operation groups line up with the compiler's lowering path:

```text
Torch/Linalg IR
-> sculptor.nn.* semantic layer IR
-> sculptor.mvm execution IR
-> sculptor.matrix.* / sculptor.vector.* / sculptor.array.* Golem IR
-> sculptor.task_region task boundaries
-> sculptor.task_graph.* / sculptor.task.create symbolic runtime graph
-> runtime graph emission
```


This staging keeps each level focused. Semantic layer recognition can reason
about model structure, MVM conversion can reason about analog compute units,
Golem expansion can reason about arrays and placement, and task graph assembly
can reason about resources and execution dependencies.

</details>
