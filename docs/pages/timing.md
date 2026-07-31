# Timing and scoring contract

Sculptor reports workload, pressure, search, and elapsed-time metrics. These
categories are intentionally separate: a value from one category must not be
presented as a value from another category.

## Elapsed-time metrics

`warm` timing begins after matrix setup and runtime readiness; matrix-setup
tasks therefore contribute no placement-critical latency. `cold` timing
includes matrix-setup task cost. Every estimate records the selected boundary.
The current compiler model does not estimate ELF startup, registry
initialization, or READY signaling, so a `cold` estimate is not yet a complete
platform boot makespan.

`exposed_transport` and `exposed_contention` are elapsed-time contributions
computed by replaying the same fixed deployment under documented
counterfactual scenarios:

```text
exposed_contention = full - no_contention
exposed_transport  = no_contention - zero_network
```

These values may be divided by makespan when the replay order is reported.

## Aggregate work and pressure

`sum_task_work` is the sum of task execution times over all cores.
`sum_edge_network_service` is the sum of ideal service time over all routed
edges. `sum_edge_network_queue_delay` is the sum of queue delay observed by all
routed edges. Parallel work and hidden waits are included in these sums, so
they may exceed makespan.

Aggregate values must not be:

- displayed as stacked portions of elapsed runtime;
- divided by makespan and labeled as runtime percentages;
- called exposed communication or contention;
- used by themselves to identify the causal bottleneck.

## Spatial and search metrics

`spatial_shared_link_pressure` measures order-independent route overlap on
directed mesh links. It is a placement heuristic, not temporal contention.

Greedy timing search uses `search_completion_time_proxy`,
`search_communication_proxy`, and `search_resource_load_proxy` while expanding
partial placements. These values rank search states. They are not predicted
nanoseconds and are not exported as makespan.

Every completed timing-aware beam candidate is replayed using the full task,
core, NIC, directed-link, and receive model. The selected plan records
`timing_rerank_candidate_count` and `timing_rerank_selected_proxy_rank`, plus
the exact predicted makespan, exposed transport, exposed contention, and
word-hop objective.

## Causal metrics

A causal critical chain follows the event that determines final output
completion through task execution, core queues, NIC queues, directed links,
and receive completion. Zero slack alone does not establish a unique causal
chain. Tied parents must be represented without double-counting elapsed time.

The current IR chain contains task and route events. Route events preserve the
specific source-NIC, directed-link, or receive-DMA blocker through their causal
resource and parent edge. Reports must not reconstruct a competing
zero-slack-based chain when this recorded chain is available.

## Provenance

Every timing estimate records:

- compiler and cost-model revisions;
- timing boundary and memory backend;
- processor clock, issue width, and vector width;
- analog MVM and I/O parameters;
- network word width, bandwidth, hop latency, protocol overhead, routing, NIC,
  and receive-DMA parameters;
- runtime ready-task and transmit policies.

Timing profiles with incompatible or stale provenance are invalid inputs to a
timing-aware scheduler.

## Metadata lifetime

`sculptor.workload.*` attributes describe semantic work such as bytes,
elements, loops, and digital replacement operations.
`sculptor.timing.*` attributes are derived estimates. Placement-only changes
clear derived timing while retaining workload. Structural rewrites clear both,
advance `sculptor.task_graph.generation`, and require the final executable task
body to be costed again. Timing-aware scheduling rejects profiles whose
`sculptor.timing.generation` does not match the graph generation.

Re-analysis records body-derived analog work as
`sculptor.workload.analog_load_bytes`,
`sculptor.workload.analog_execution_count`, and
`sculptor.workload.analog_store_bytes`. Physical array rows, rather than only
the valid logical result width, determine store traffic when tile geometry is
available. This lets a fused `mixed.fused` task retain analog semantics without
inheriting pre-fusion latency estimates.

## Declared approximations

- Local memory uses the `native-untimed` backend; cache, DRAM, and bank timing
  are not implied.
- Mesh links use interval reservations with effectively sufficient buffering.
  Finite credits and router head-of-line blocking are not modeled.
- Output-buffer ownership conflicts with in-flight routes are not yet modeled.
- CPU task bodies serialize per core while NIC progress may overlap later CPU
  work.
- Multi-array streaming convolution models per-array phase overlap but no
  cross-patch pipelining.
