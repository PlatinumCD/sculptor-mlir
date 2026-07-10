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
| `random` | Baseline randomized physical-array order with shared island materialization and digital placement. |
| `snake` | Deterministic mesh traversal that fills local arrays before moving to the next core, with shared island materialization and digital placement. |
| `greedy` | Island-level lookahead search over configurable candidate core scopes. |
| `annealing` | Simulated annealing over physical-array orders with configurable initial schedules and perturbation move sets. |


The scheduler implementation is split by responsibility:

| File | Role |
|---|---|
| `task_schedulers/TaskGraphScheduler.h/.cpp` | Pure scheduler interface, registry, and public schedule registrations. |
| `task_schedulers/TaskGraphTypes.h` | Hardware budget and parsed DAG node types. |
| `task_schedulers/TaskGraphScheduleConfig.h/.cpp` | One-time parsing of Greedy and annealing options into typed configuration. |
| `task_graph/TaskGraphDAG.h/.cpp` | Task graph parser. |
| `TaskGraphIslands.h` | Logical placement island data model and island graph builder. |
| `TaskGraphIslandBuilder.cpp` | Matrix/MVM island seeding and island graph coordination. |
| `TaskGraphIslandDigitalAssignment.cpp` | Pre-placement min-cut digital assignment and local-affinity fallback assignment. |
| `TaskGraphIslandCommunication.cpp` | Island-level communication edge construction and compaction. |
| `TaskGraphIslandInternals.h` | Private island-builder helpers shared by the island implementation files. |
| `TaskGraphPlacementPlan.h/.cpp` | Canonical scheduler output and placement-plan validation. |
| `TaskGraphPlacementObjective.h/.cpp` | Shared island transfer and boundary objective. |
| `TaskGraphPlacement.h` | Central placement-plan commit and placement attachment API. |
| `TaskGraphPlacementUtils.cpp` | Physical placement materialization and fallback core propagation. |
| `task_graph/TaskGraphScheduleMetadata.h/.cpp` | Final schedule metadata, logical-array layout, transfer summaries, graph score, and digital op counts. |
| `GreedySearch.h`, `greedy/` | Public Greedy planner plus private heuristics, candidate expansion, traversal, and terminal repair. |
| `annealing/` | Annealing perturbations and cooling/search loop. |
| `task_graph/TaskGraphRoutineFuser.h/.cpp` | Post-schedule task routine fusion. |
| `task_graph/TaskGraphScorer.h/.cpp` | Mesh transfer scoring and boundary penalty computation. |

<details class="doc-section" open markdown="1">
<summary markdown="block">## Placement Pass Flow</summary>


`sculptor-schedule-task-graph` does more than call a strategy. The strategy
chooses placements, but the pass owns the full scheduling/finalization flow:

1. Validate the hardware budget and attach module-level budget metadata.
2. Parse each task graph function into a `TaskGraphDAG`.
3. Build logical placement islands once and look up the selected scheduler.
4. Ask the scheduler for an `IslandPlacementPlan` without mutating IR.
5. Validate and commit that plan through the shared placement materializer.
6. Fuse same-core task components.
7. Erase unused task callees and temporary resources.
8. Rebuild the runtime execution plan.
9. Reparse the compacted task graph.
10. Finalize schedule metadata, transfer summaries, graph score, and resource
   layout.

That ordering matters. Same-core routine fusion can remove task nodes and graph
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
| `greedy-heuristic` | Candidate scoring and search terms used by the `greedy` scheduler. Terms are comma-separated, such as `transfer-cost,compact-region,lookahead=3,beam=8,scope=diagonal`. Supported scoring terms are `transfer-cost`, `boundary-regret`, and `compact-region`; supported search terms are `lookahead=N`, `beam=N`, and `scope=NAME`. The default is `transfer-cost`, with `lookahead=1`, `beam=1`, and `scope=diagonal`. |
| `annealing-initial-schedule` | Initial placement schedule used by the `annealing` scheduler: `identity`, `random`, `snake`, or `greedy`. The default is `snake`. The `greedy` initializer uses the configured `greedy-heuristic` terms. |
| `annealing-move-set` | Comma-separated perturbation moves used by the `annealing` scheduler. Supported presets are `basic`, `basic-wide`, and `all`; individual moves are `move-one-position`, `move-one-relocation`, `swap-two-positions`, `adjacent-swap`, `segment-reverse`, `segment-relocation`, and `block-swap`. The default is `basic`. |
| `annealing-move-radius` | Maximum physical-array-order index distance for single-position annealing moves. `0` means unbounded/global. The default is `0`. |
| `annealing-initial-temperature` | Initial annealing temperature. If set to `0`, the scheduler infers it from the initial placement score. The default is `0`. |
| `annealing-final-temperature` | Final annealing temperature. The default is `1`. |
| `annealing-cooling-rate` | Multiplicative cooling factor applied after each temperature stage. The default is `0.9`. |
| `annealing-steps-per-temperature` | Number of perturbation attempts per temperature stage. The default is `32`. |
| `summary-output` | Optional CSV path where the scheduler appends compact score and transfer metadata for each scheduled task graph. |

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

The main placement unit is a logical island. Each island starts from one matrix
setup group: a `sculptor.matrix_setup` task and the direct `sculptor.mvm` tasks
that consume the logical array produced by that setup. The setup and its MVM
tasks must use the same physical analog array:

```text
matrix setup -> logical array -> one or more MVM tasks
```

The island builder then assigns eligible digital components to those islands
before physical placement. Clear same-source-layer components between analog
anchors are assigned by a byte-only min-cut. Components that do not have an
unambiguous analog boundary use the producer/consumer fallback.

Every strategy returns the same `IslandPlacementPlan`: one physical-array id
per logical island, in island order. The shared commit helper validates and
materializes that plan in three steps:

1. Attach analog placement to the matrix setup and its associated MVM tasks.
2. Place unambiguous same-source-layer core-only tasks on that same core.
3. Place any remaining core-only tasks from already placed graph neighbors.

The neighbor fallback is deterministic:

1. Prefer the most recent placed producer.
2. Otherwise use the earliest placed consumer.

Order-based strategies convert their order to the canonical plan before
returning. The common commit path expands the plan to analog tasks, assigned
digital work, and remaining core-only tasks.

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
placement strategy. It only fuses tasks that have already been placed on the
same core and can be represented as a single component task without violating
dependency order.

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
<summary markdown="block">## Greedy Schedule</summary>


`greedy` first forms logical matrix setup islands: every matrix setup owns its
direct MVM users, and eligible same-layer digital components are assigned into
those islands with a byte-only min-cut before any physical mesh coordinate is
chosen. The islands are then placed with greedy queue lookahead over the mesh:
the first island starts on core 0, each next island considers candidates
selected by the `scope=...` heuristic term, and `lookahead=...` controls how
many future islands are simulated when scoring the current choice. Each
simulated window recomputes transfer cost over an effective island communication
graph and applies `bytes * ManhattanDistance` to island pairs that have
simulated core placements.

When `beam=...` is greater than `1`, greedy placement keeps a bounded
frontier of partial placements at each island instead of committing to a single
path. Each expansion still uses the selected candidate scope and heuristic, but
the frontier lets the scheduler preserve alternatives that may be useful for
later global constraints such as terminal boundary placement. In beam mode, the
frontier is the search mechanism; recursive lookahead is used by the single-path
mode.

The `boundary-regret` greedy heuristic allows a bounded transfer-cost tradeoff
for boundary-aware placement. Before the final task island, it adds late-stage
pressure toward the first task island's mesh boundary only when the candidate is
within a progress-scaled regret budget of the best transfer-cost candidate. At
the final task island, it scores the actual boundary penalty, so it can choose a
boundary-compatible endpoint when the avoided boundary penalty is worth the
extra transfer cost.

The `compact-region` greedy heuristic term keeps transfer cost primary but adds
a bounded shape penalty for candidates within a small transfer-regret window.
The shape penalty looks at the occupied core bounding box after the candidate
placement and penalizes high aspect ratios, sparse boxes, and one-dimensional
growth along a mesh boundary. This is intended to prefer healthier two
dimensional regions when transfer-cost candidates are equal or near-equal. It is
intended to be composed with the transfer baseline, for example
`greedy-heuristic=transfer-cost,compact-region,lookahead=3,beam=8`.

Boundary-aware greedy heuristics also run a terminal repair step after the
initial island placement is selected. If the first and last task islands do not
share a mesh boundary, the repair tries moving the terminal island to unused
array slots on the first island's boundary and keeps the move only when the
island-level estimate of `transfer_cost + boundary_penalty` improves.

```text
island[setup] = {setup, direct_mvm_users}
island += byte_only_min_cut(eligible_digital_components)
place(island[0]) = core 0
for island in island_queue[1:]:
  candidates = candidate_scope(current_core, placed_region)
  place(island) = argmin(simulated_cost_for_next_N_islands)
```

The `cardinal` scope considers the current core, then up, left, down, and right.
The `diagonal` scope additionally considers the four diagonal neighbors. The
`producer-consumer` scope considers candidates near already placed islands that
communicate with the current island.

With `lookahead=1`, the scheduler scores only the current island. With
`lookahead=3`, it scores the current island plus two future islands
before committing the current placement.

This keeps the scheduler runnable while making the search policy explicit.

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
physical-array order and lets the shared island helper place the MVMs, assigned
digital components, and remaining core-only tasks.

## Simulated Annealing Schedule

`annealing` runs a simulated annealing search over complete physical-array
orders. It builds an initial placement from `annealing-initial-schedule`, scores
that placement over the logical island communication graph, then repeatedly
generates candidate placements with randomized perturbation moves selected from
`annealing-move-set`. Better candidates are accepted immediately; worse
candidates are accepted with probability `exp(-delta / temperature)`. The
temperature is reduced by `annealing-cooling-rate` after each stage.

Supported initial schedules are `identity`, `random`, `snake`, and `greedy`;
the `snake` initializer uses the same physical-array order builder as the real
`snake` scheduler. The `greedy` initializer builds the initial island order with
the same greedy placement logic used by `schedule=greedy`, including the
configured `greedy-heuristic` terms. The default move set, `basic`, preserves
the original annealing behavior: `move-one-position` and `swap-two-positions`.
The `basic-wide` preset adds `move-one-relocation`, which removes one physical
array assignment and reinserts it elsewhere in the order so the intervening
assignments shift together. The `all` preset also enables `adjacent-swap`,
`segment-reverse`, `segment-relocation`, and `block-swap`, which can perturb
contiguous regions of the island order instead of only changing single
positions.

`annealing-move-radius` controls the target range for `move-one-position` and
`move-one-relocation`. A radius of `0` keeps the target unbounded across the
full physical-array order; a positive radius restricts the target to that many
positions to the left or right of the selected active island position.

The score used inside the annealing loop is an approximate pre-commit score:

```text
score = sum(edge_bytes * ManhattanDistance(source_core, destination_core))
      + boundary_penalty
```

After annealing selects the best physical-array order, it converts the order to
the same canonical plan returned by every other scheduler.

### Default Min-Cut Digital Placement

Registered schedulers use min-cut digital placement by default. For an
unplaced same-source-layer digital component between placed analog MVMs, the
scheduler builds a small source/sink cut problem. Resource byte sizes become
edge weights. The predecessor analog core is the source side, the successor
analog core is the sink side, and the selected cut minimizes the bytes that
must cross between those two cores.

Components with multiple distinct predecessor or successor analog cores are left
to the normal producer/consumer fallback. This keeps the policy local to clear
same-layer analog boundaries.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Scheduler Interface</summary>


Placement strategies are implemented through a shared C++ interface:

```cpp
class TaskGraphScheduler {
public:
  virtual ~TaskGraphScheduler() = default;

  virtual StringRef getName() const = 0;

  virtual FailureOr<IslandPlacementPlan>
  buildPlacementPlan(const TaskGraphPlacementProblem &problem) const = 0;
};
```

`TaskGraphPlacementProblem` provides read-only access to the hardware budget,
parsed DAG, and logical island graph. A scheduler returns one canonical plan:

```cpp
IslandPlacementPlan plan;
plan.physicalArrayByIsland = {/* one physical array per island */};
return plan;
```

The pass validates and commits the plan centrally. Scheduler implementations
do not attach core or array attributes themselves.

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
