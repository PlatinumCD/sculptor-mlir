# IR Design

The Sculptor dialect is organized around a staged lowering model. Each group of
operations represents a different level of intent: semantic neural-network
layers, analog execution, accelerator-facing array operations, task boundaries,
and task graph construction.

## Custom Types

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

## Operation Groups

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

## Lowering Shape

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
