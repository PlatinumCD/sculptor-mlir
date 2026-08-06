# Golem bare-metal runtime

This directory builds the reusable, freestanding runtime library described in
the [runtime and tile ABI guide](../docs/pages/runtime.md). The target artifact is
`libgolem-runtime.a`, statically linked into each tile ELF alongside that
tile's generated registry, routes, tensor plan, and compiled MLIR task objects.

The library currently implements the agreed Platform v0.1 foundations:

- task and tensor ABI types;
- the generated Sculptor tile ABI, including boot, dispatch, route, and
  model-I/O tables;
- compiler-emitted resource, flattened-dimension, workspace, and task-binding
  deployment tables;
- a basic single-execution tile runtime that runs boot tasks, dispatches a
  task by global ID, and moves route payloads as blocking 32-bit words;
- `DeploymentRuntime`, which constructs rank-generic contiguous memref
  descriptors, owns the compiler-sized workspace, schedules ready local
  tasks, and moves source-aware framed routes for execution 0 with resumable
  transmission under backpressure;
- immutable binary-search task registries;
- the fixed 16-entry task-instance pool;
- the fixed FIFO ready queue;
- transport-neutral raw and source-aware interfaces with a legacy word
  callback, an optional bounded bulk-send callback, and optional receive-DMA
  submission and completion callbacks;
- an optional `DeploymentTrace` callback that reports task start and finish
  with the global task ID and current execution ID.

The deployment frame is five 32-bit header words followed by the exact tensor
payload:

```text
magic | route_id | execution_id[31:0] | execution_id[63:32] | word_count
payload word 0 | payload word 1 | ...
```

The deployment sender submits the header and payload in bursts of at most 4096
words when the platform supplies the bulk callback. The callback is optional;
host unit tests and older tile programs fall back to the original word path.
The receiver always decodes the five-word header in software. When receive
DMA is available, it registers the source, route, tensor destination, and
payload length, then waits for a source-and-route completion before marking
that tensor ready. Without those callbacks it consumes the payload through
the original 32-bit word interface. Both modes implement the same frame ABI.
`DeploymentRuntime::step()` polls receive progress before advancing at most
one pending transmit chunk. A full NIC or network queue therefore returns
control to the runtime without losing the frame offset. This prevents two
tiles exchanging large tensors in opposite directions from both blocking
inside send while their receive queues are full.

When no task or transmit chunk can advance and an incoming route remains
outstanding, `step()` returns `DeploymentStep::WaitForReceive`. A platform
entry point should respond by invoking its blocking receive primitive; the
Mittens implementation writes `RX_WAIT` and yields over fd 41 until SST
delivers data. A transmit-backpressured step returns
`DeploymentStep::WaitForTransmit`; the Mittens implementation writes
`TX_WAIT` and SST resumes the guest only after the appropriate shared-memory
transmit ring has space.

The default transmit policy drains a completed task's outgoing routes before
executing another local task. The optional
`DeploymentTransmitPolicy::OverlapReadyTasks` policy keeps a bounded FIFO of
completed source tasks and may execute another ready task while transmission
is blocked. Before doing so, it proves that the new task's output buffers do
not overlap any unsent route buffer, preserving the compiler's workspace
liveness contract.

`DeploymentRuntime::executeReadyTask()` invokes the optional trace callback
immediately before and after `Task::execute`. The runtime remains independent
of the tracing transport. The Mittens tile entry point maps the callback to
the NIC diagnostic registers, which QEMU forwards to SST over fd 41. A null
callback emits no task markers.

One receive state is maintained per source tile, so frames from different
sources may interleave without mixing payloads. Platform v0.1 admits only
execution ID 0. The runtime does not provide an operating-system abstraction,
tensor partitioner, dynamic task mapper, wider tensor transport, or
multi-execution admission policy.

Build and install the RISC-V archive with:

```bash
./build-scripts/build-runtime.sh
```

Run the host unit tests with:

```bash
./runtime/tests/run-test.sh
```

The host suite includes a reciprocal-transfer regression with one bounded
network cell per tile. Both endpoints send simultaneously, drain incoming
frames while their own sends are backpressured, and complete their downstream
tasks with exact outputs. It also verifies that a destination returns
`WaitForReceive` while each framed word is unavailable and that transmit
backpressure returns `WaitForTransmit`. A separate receive-DMA regression interleaves
headers from two source tiles, verifies only the ten header words pass through
the CPU word callback, copies both payloads through DMA callbacks, checks both
completions, and executes the dependent task.

The installed layout is:

```text
install/runtime/include/golem/runtime/
install/runtime/lib/libgolem-runtime.a
```
