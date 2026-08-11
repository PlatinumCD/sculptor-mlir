# Temporal Makespan Scheduling

## Status

This document defines an optional physical-placement objective.
The extension evaluates time after the compiler creates logical tiles.

The extension does not replace the greedy, snake, random, or annealing schedules.
It changes the objective that these schedules use.

## Problem

The current physical-placement score is:

```text
sum(edge_bytes * Manhattan_hops)
```

`scoreLogicalTilePlacement()` calculates this value in `LogicalTilePlacement.cpp`.
The greedy implementation also uses the same edge-distance cost for each candidate.

This score measures spatial communication work.
It does not predict the final output time.

The current mapping evaluator calculates structural start and finish times.
These times do not know physical coordinates, NIC state, directed-link state, or tile availability.

## Correct Integration Point

Implement temporal scheduling in physical placement.
Do not add it to `BuildRATree` or `OutlineTileRoutines`.

`PlaceLogicalTiles.cpp` has all required inputs:

- The compute graph.
- The selected RA tree.
- The logical-tile graph.
- The physical mesh.
- The physical placement candidate.
- The serialized cost profile.

This is the first stage that knows the route distance between logical tiles.

## Recommended Interface

Extend `--sculptor-place-logical-tiles`:

```text
--sculptor-place-logical-tiles="
  schedule=greedy
  objective=makespan
  network-mode=full
  timing-scope=warm
  ...
"
```

Use these options:

| Option | Values | Default |
|---|---|---|
| `objective` | `transfer-cost`, `makespan` | `transfer-cost` |
| `network-mode` | `ideal`, `finite`, `full` | `finite` |
| `timing-scope` | `warm`, `cold` | `warm` |
| `emit-timing-trace` | File path | Empty |
| `temporal-candidate-limit` | Positive integer | `8` |

The default options must preserve current placement results.

Do not add `makespan` to the structural `MappingObjectiveKind` in the first change.
Use a separate `PlacementObjectiveKind` because physical makespan needs physical coordinates.

## Compiler Data Model

Add these files:

```text
include/sculptor-mlir/Dialect/Sculptor/Transforms/mapping/
  TemporalPlacementModel.h
  PlacementObjective.h

lib/Dialect/Sculptor/Transforms/mapping/
  TemporalPlacementModel.cpp
  PlacementObjective.cpp
```

Use these records:

```cpp
enum class PlacementObjectiveKind {
  TransferCost,
  Makespan,
};

enum class TemporalNetworkMode {
  Ideal,
  Finite,
  Full,
};

struct TemporalTaskEvent {
  int64_t eventId = -1;
  int64_t leafId = -1;
  int64_t logicalTileId = -1;
  int64_t operationId = -1;
  int64_t workUnitId = -1;
  LogicalLaneKind laneKind = LogicalLaneKind::Digital;
  int64_t laneIndex = -1;
  double durationNs = 0.0;
};

struct TemporalRouteEvent {
  int64_t eventId = -1;
  int64_t sourceTaskEvent = -1;
  int64_t targetTaskEvent = -1;
  int64_t tensorId = -1;
  int64_t byteSize = 0;
};

struct TemporalPlacementEvaluation {
  double makespanNs = 0.0;
  double taskTimeOnCriticalChainNs = 0.0;
  double exposedTransportNs = 0.0;
  double exposedContentionNs = 0.0;
  double maximumTileLoadNs = 0.0;
  int64_t maximumDirectedLinkWords = 0;
  SmallVector<int64_t> criticalEventIds;
};
```

`LogicalTilePlacementProblem` must reference:

```cpp
const ComputeGraph &computeGraph;
const ResourceAllocationTree &raTree;
const MappingCostProfile &costProfile;
```

Do not store owning copies in the placement problem.

## Temporal Event Graph

### File Change Table

| File | Change |
|---|---|
| `include/.../mapping/TemporalPlacementModel.h` | Define events, resource calendars, and evaluation results |
| `include/.../mapping/PlacementObjective.h` | Define the common complete-placement objective API |
| `lib/.../mapping/TemporalPlacementModel.cpp` | Build and schedule the temporal event graph |
| `lib/.../mapping/PlacementObjective.cpp` | Implement transfer-cost and makespan objectives |
| `mapping/LogicalTilePlacement.h/.cpp` | Extend the problem and placement-plan records |
| `mapping/LogicalTileSchedulers.cpp` | Route greedy and annealing scores through the objective API |
| `PlaceLogicalTiles.h/.cpp` | Add temporal options, provenance, summaries, and trace output |
| `SculptorAttrs.td` | Add the temporal-placement summary attribute |
| `mapping/CMakeLists.txt` | Compile the new placement sources |
| `tools/ra-tree-report` | Export the critical chain and time views |

Build one task event for each `LogicalTileAssignment`.
Use `leafId` as the unique event identity.
Keep `(operationId, workUnitId)` as the dependency endpoint identity.

Build precedence edges from these sources:

1. Each internal or external `LogicalTileDependency`.
2. Each temporal cut in the selected RA tree.
3. Model-input readiness at time zero.

The logical-tile dependencies already include exact `MappingWorkUnitEdge` refinements.
Do not add those edges a second time.

A temporal cut orders child regions.
Connect each sink in child `i` to each source in child `i + 1`.

Do not add an edge for a spatial cut.
The data DAG and lane resources control its legal execution.

Reject a temporal event graph that contains a cycle.

## Resource Model

Track these calendars for each physical tile:

- One digital lane.
- One execution calendar for each analog lane.
- One shared analog I/O calendar.
- One source NIC calendar.
- One destination receive-DMA calendar.

Track one calendar for each directed mesh link.

The first version uses deterministic XY routing.
Store this route policy in placement provenance.

Matrix setup events have zero warm-execution cost.
`timing-scope=cold` includes their calibrated setup cost.

## Task Scheduling

Use a deterministic list scheduler for a fixed placement.

For each ready task, calculate:

```text
dependency_ready = maximum arrival time of all required values
resource_ready = next free time of the required tile lane
task_start = max(dependency_ready, resource_ready)
task_finish = task_start + calibrated_task_cost
```

Use this tie order:

1. Earliest possible start.
2. Largest remaining critical-path cost.
3. Stable operation ID.
4. Stable work-unit ID.

Update the selected lane calendar after the event.

Analog load and store phases reserve the shared analog I/O calendar.
Analog execute reserves only its bound analog lane.
This requires phase costs from the calibrated profile.

## Route Scheduling

Use the exact dependency byte size.
Do not use an aggregate logical-tile edge when exact dependencies are available.

For `network-mode=ideal`:

```text
route_arrival = producer_finish
```

For `network-mode=finite`:

```text
route_arrival = producer_finish
              + injection_cost
              + payload_serialization
              + hop_pipeline_cost
              + receive_dma_cost
```

The finite mode has no shared network calendars.

For `network-mode=full`:

1. Wait for the source NIC.
2. Reserve each directed link on the XY path.
3. Keep payload serialization on each link.
4. Apply the hop pipeline offset between adjacent links.
5. Wait for the destination receive-DMA resource.
6. Record the causal predecessor that caused each wait.

Use this serialization term:

```text
words = ceil(byte_size * 8 / network_word_bits)
serialization_ns = words * network_word_time_ns
```

Do not report aggregate link service as elapsed runtime.

## Placement Objective Interface

Add one objective interface for complete placements:

```cpp
class PlacementObjective {
public:
  virtual FailureOr<PlacementScore>
  evaluate(ArrayRef<int64_t> physicalTiles) const = 0;
};
```

`PlacementScore` must contain a primary floating-point value and deterministic tie values.

Use these comparison keys:

```text
transfer-cost:
  total_transfer_cost, maximum_edge_cost, lexicographic placement

makespan:
  predicted_makespan_ns,
  exposed_contention_ns,
  total_transfer_cost,
  lexicographic placement
```

Keep `scoreLogicalTilePlacement()` as the transfer-cost implementation.

## Search Integration

### Annealing

Annealing already evaluates complete placements.
Replace its direct score call with the selected objective interface.

Use a normalized integer energy only for the acceptance calculation.
Keep the full double-precision makespan in the final plan and trace.

### Greedy

The current greedy search calculates an incremental edge-distance cost.
A full temporal evaluation for every mesh location is too expensive.

Use a two-stage candidate evaluation:

1. Rank all candidates with the current incremental transfer-cost lower bound.
2. Keep the best `temporal-candidate-limit` candidates.
3. Complete each candidate with the current deterministic greedy rollout.
4. Evaluate the complete provisional placement with the temporal model.
5. Commit the candidate with the best temporal score.

The lookahead value still controls the existing rollout decisions.
The provisional completion exists only to make the makespan score defined.

Cache task costs, route paths, and topology.
Do not cache placement-dependent resource times across different candidates.

### Random And Snake

Random and snake create one placement.
The temporal evaluator reports their makespan without changing their order.

## IR And Reports

Extend `LogicalTilePlacementAttr` with:

- Placement objective.
- Cost-profile name and hash.
- Network mode.
- Route policy.
- Predicted makespan.
- Critical-chain task time.
- Exposed transport time.
- Exposed contention time.
- Maximum tile load.
- Maximum directed-link load.

Add a typed temporal summary attribute instead of appending unrelated fields to every edge attribute.

Use this function attribute:

```text
sculptor.mapping.temporal_placement_summary
```

The optional timing-trace file must use JSON.
Each event must record its stable IDs, start, finish, resource, and causal predecessor.

Extend `sculptor-ra-tree-report` to show:

- The predicted critical chain.
- A tile-by-time execution view.
- Network resource waits.
- Ideal, finite, and full makespan values.

## Validation Modes

The same placement must support three evaluations:

```text
exposed_transport = finite_makespan - ideal_makespan
exposed_contention = full_makespan - finite_makespan
```

The compiler must clamp small negative floating-point differences to zero.
It must reject larger negative differences as a model error.

## Diagnostics

Emit a compiler error for these conditions:

- Missing calibrated cost profile for the makespan objective.
- Duplicate event endpoint identity.
- Unknown dependency endpoint.
- Cyclic temporal event graph.
- Negative or nonfinite task cost.
- Invalid route byte size.
- Unsupported mesh route.
- Link, word, or time arithmetic overflow.
- Analog assignment without a valid lane.
- Placement provenance that disagrees with the selected profile.

## Tests

Add focused tests in `tests/python_tests/test_temporal_placement.py`.

Use a controlled fork-join graph with one late producer.
The tests must cover:

1. Exact task start and finish times.
2. Digital-lane serialization on one tile.
3. Parallel execution on different tiles.
4. Concurrent analog execution on different array lanes.
5. Shared analog I/O serialization.
6. Ideal, finite, and full network modes.
7. Directed-link contention.
8. Source NIC serialization.
9. Critical-chain reconstruction.
10. A placement where transfer cost and makespan select different winners.
11. Deterministic greedy and annealing results.
12. Extension-off placement equivalence.

Add one GPT-2 report test in `tests/model_tests`.
Do not use GPT-2 as the first correctness test.

## Incremental Delivery

### Phase 1

Build the event graph and evaluate a completed placement.
Report ideal, finite, and full makespan.

### Phase 2

Add the makespan objective to annealing.
This is direct because annealing evaluates complete placements.

### Phase 3

Add the two-stage temporal candidate ranking to greedy.

### Phase 4

Add the JSON critical-chain trace and report views.

### Phase 5

Compare predicted event times against simulator traces.
Record the error for each calibration workload.

## Non-Goals

This extension does not:

- Change runtime scheduling semantics.
- Add partial routine execution.
- Change task boundaries.
- Select the number of digital workers.
- Implement adaptive mesh routing.
- Replace simulator validation.

## Easiest Effective First Version

Implement a complete-placement evaluator and connect it to annealing first.
Then add the bounded temporal candidate set to greedy.

This order provides a correct reference before search optimization changes the evaluation count.
