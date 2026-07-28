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
| `sculptor-lower-golem-to-task-graph` | Builds and schedules the task graph, lowers Golem array operations to runtime shim calls, and emits an isolated per-core deployment. |

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
registered scheduler name. The current tree provides several placement
strategies, including `random`, `snake`, and `greedy`. Registered schedulers use
min-cut digital placement with local digital-task refinement by default. After
the selected strategy places matrix setup/MVM groups and related digital work,
separate passes fuse same-island components, lower the logical-array ABI, and
partition active cores. Runtime resources are finalized later on one extracted
core module at a time.

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
   execution, store, and recombine task regions. Matrix and vector tiles retain
   their physical padded dimensions, while MVM stores and recombination expose
   only the valid logical output rows.
5. `sculptor-materialize-tasks`
   Turns `sculptor.task_region` boundaries into private task functions with task
   metadata, then rewrites `forward` to call those tasks.

### `sculptor-lower-golem-to-task-graph`

This pipeline turns materialized Golem tasks into a deployment containing
isolated scheduled core graphs.

1. `sculptor-assemble-task-graph`
   Builds `generate_task_graph` with `sculptor.task_graph.*` resources and
   `sculptor.task.create` nodes. The materialized `forward` function may still
   be present at this point as a direct call form of the same tasks.
2. `sculptor-build-task-graph-islands`
   Builds logical placement islands and attaches stable island IDs without
   assigning physical cores or arrays.
3. `sculptor-analyze-task-graph-timing`
   Combines explicit task dependencies with resource producer-consumer edges,
   validates the resulting execution DAG, and attaches timing metadata used by
   `greedy-timing` and preserved for post-fusion reporting. Standalone ordinary
   schedulers do not require this pass before scheduling.
4. `sculptor-schedule-task-graph`
   Consumes the prebuilt islands, assigns cores and arrays, and records transfer
   metadata and the placement score without changing graph topology.
   `sculptor-lower-scheduled-mvm-to-digital` may be inserted immediately after
   this pass to create a controlled digital-compute baseline. It preserves the
   scheduled task graph and replaces only the analog tile implementations with
   padded, tiled `linalg.matmul_transpose_b` operations.
   `sculptor-optimize-task-graph` may then apply selected
   placement-preserving graph rewrites before generic fusion. Its
   `streaming-convolution` pattern replaces a co-located patch/MVM/recombine
   chain with one bounded-buffer task while retaining setup dependencies and
   per-array placement.
5. `sculptor-fuse-task-graph`
   Fuses connected tasks only when they share both a logical island and a core,
   then removes task callees and intermediate resources made dead by fusion.
6. `sculptor-lower-golem-to-llvm-shims`
   Rewrites scheduled Golem array operations into LLVM-callable runtime shim
   calls. At the same boundary it removes logical-array resources from task
   interfaces, preserves setup ordering as explicit task dependencies, and
   retains physical/local array bindings plus graph-level placement provenance.
7. `sculptor-partition-task-graph-by-core`
   Assigns deterministic deployment task/resource IDs, converts cross-core
   tensor edges into typed route boundaries, clones each core's complete symbol
   closure, and emits modules only for active cores. Global runtime slots are
   intentionally absent.
8. `sculptor-extract-core-module`
   Selects one active core, flattens its isolated symbol closure into a
   standalone module, and retains only that core's routes and model ownership
   records. `sculptor-finalize-task-graph-resources` then assigns private local
   slots and workspace storage.
9. Standard MLIR lowering followed by `sculptor-emit-golem-tile-abi`
   Converts task implementations to `llvm.func`, then packages the declarative
   core graph as immutable Golem boot, dispatch, route, and model-I/O tables.
   The pass emits Golem `TaskExecute` adapters, removes the consumed graph and
   Sculptor metadata, and leaves a pure LLVM-dialect module ready for intrinsic
   finalization and LLVM IR translation.

The outer deployment retains the global scheduling and timing summaries once.
Each nested core graph contains local structural counts and its required
hardware context. If a model input directly feeds tasks on several cores, the
model-input manifest contains one ownership record per consuming core; this
explicitly requests host-side input replication without inventing a producer
task or a cross-core tensor route.

### Main Lowering Passes

| Pass | Input shape | Output shape |
|---|---|---|
| `sculptor-canonicalize-layers` | Torch-MLIR or `linalg`-style layer bodies in `forward`. | Inline `sculptor.nn.*` layer ops. |
| `sculptor-extract-layers` | Inline recognized layer regions in `forward`. | Separate layer functions called by `forward`. |
| `sculptor-convert-layers` | Extracted `sculptor.nn.*` layer functions. | `sculptor.mvm` plus standard tensor, linalg, math, or control-flow glue. |
| `sculptor-expand-mvm-to-golem` | `sculptor.mvm` inside layer/helper functions. | Golem array setup, vector tiling, array execution, store, and recombine task regions. |
| `sculptor-materialize-tasks` | `sculptor.task_region` boundaries. | Private task functions with task metadata, called from `forward`. |
| `sculptor-assemble-task-graph` | A `forward` function that calls materialized task functions. | Materialized task functions plus `generate_task_graph` with `sculptor.task_graph.*` resources and `sculptor.task.create` nodes. |
| `sculptor-build-task-graph-islands` | An assembled task graph. | Placement-island members annotated with stable logical island IDs. |
| `sculptor-analyze-task-graph-timing` | An island-annotated task graph. | Task-level execution order and latency metadata plus graph critical-path and island-work summaries. |
| `sculptor-schedule-task-graph` | An island-annotated task graph. | Scheduled task graph metadata, graph score, live private task functions, and no stale materialized `forward` entry point. |
| `sculptor-lower-scheduled-mvm-to-digital` | A scheduled, unfinalized Golem task graph. | A placement-preserving digital baseline: analog MVM task bodies become tiled `linalg.matmul_transpose_b` operations while core assignments, islands, tensor communication, setup dependencies, and physical tile geometry remain fixed. |
| `sculptor-optimize-task-graph` | A scheduled, unfinalized task graph plus an optional comma-separated pattern list. | A placement-preserving optimized graph with structural metadata refreshed and stale timing metadata removed. The initial `streaming-convolution` pattern eliminates full im2col resources for eligible co-located convolutions. |
| `sculptor-fuse-task-graph` | A scheduled task graph with island and core assignments. | Same-island, same-core components outlined as fused task routines. |
| `sculptor-lower-golem-to-llvm-shims` | Scheduled task functions and graph resources containing logical-array operations. | Calls to LLVM-callable Golem runtime shims plus a task graph whose executable array identity is represented by physical bindings and setup dependencies; the logical-to-physical schedule map remains as reporting metadata. |
| `sculptor-partition-task-graph-by-core` | A scheduled, fused graph after logical-array ABI lowering and before runtime finalization. | A deployment module with active per-core nested modules, route boundaries, a typed global route table, and stable global identities. |
| `sculptor-extract-core-module` | A partitioned deployment plus `core-id=N`. | One standalone core module with filtered incoming/outgoing routes and model ownership manifests. |
| `sculptor-finalize-task-graph-resources` | One standalone core graph extracted from a deployment. | Core-private runtime slots, task indices, intermediate offsets, route slot arrays, and workspace metadata. Route boundaries receive unique non-reused regions. |
| `sculptor-emit-golem-tile-abi` | One isolated, finalized core after task implementations have become `llvm.func`. | A pure LLVM-dialect tile module with Golem task adapters, separate boot and dispatch tables, route and model-I/O records, and C ABI accessors. |

### Export And Runtime Passes

These passes sit beside the main lowering path. They consume task-graph or
runtime-shaped IR and produce external artifacts or final backend forms.

| Pass | Role |
|---|---|
| `sculptor-export-task-graph-vis` | Writes an assembled task graph visualization as DOT or GraphML. |
| `sculptor-export-task-graph-sim-model` | Writes a scheduled task graph model for external placement or simulation tooling. |
| `sculptor-finalize-golem-intrinsics` | Rewrites LLVM Golem shim calls into target Golem ISA intrinsics. |
| `sculptor-emit-golem-tile-abi` | Packages one extracted core for the Golem bare-metal runtime. It is distinct from the generic runtime graph emitter and does not generate or link an ELF. |
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
| Graph resources | `sculptor.task_graph.input`, `sculptor.task_graph.output`, `sculptor.task_graph.intermediate`, `sculptor.task_graph.persistent` |
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
