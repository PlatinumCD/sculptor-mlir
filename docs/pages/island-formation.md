# Island Formation

## 1. Purpose

This report describes logical placement-island formation in Sculptor MLIR.
It explains the current algorithm and identifies its main limits.
It also defines three alternatives for future development.

The report uses the following goals:

1. Keep required analog operations on a compatible core.
2. Reduce communication between cores.
3. Preserve useful task parallelism.
4. Limit digital work and live storage on each core.
5. Preserve task readiness for timing-aware placement.
6. Produce deterministic and explainable groups.

The measurements describe the GPT-2 Small task graph before this report's
experimental multi-terminal implementation.

## 2. Scope

Island formation occurs after task-graph assembly and before placement.
The pass is `--sculptor-build-task-graph-islands`.

The pass receives these inputs:

- The task DAG.
- Task-resource producer and consumer relationships.
- Explicit task dependencies.
- Task kinds and source-layer metadata.
- Matrix-setup and logical-array relationships.
- Reduction and digital-shard metadata.

The pass produces these outputs:

- One island ID for each covered task.
- One list of tasks for each island.
- One directed island execution graph.
- One undirected island affinity graph.

The pass does not use mesh geometry or core capacity.
The scheduler assigns the completed islands to physical cores.

## 3. Terms

| Term | Definition |
|---|---|
| Task DAG | The directed graph of task dependencies and task-resource flow. |
| Analog anchor | A matrix-setup task that starts an analog island. |
| Terminal island | An existing island adjacent to an unassigned digital component. |
| Digital component | A connected region of unassigned digital tasks. |
| Island execution edge | A directed dependency between two islands. |
| Island affinity edge | An undirected communication weight between two islands. |
| Entry event | A task-level transfer or dependency that enters an island. |
| Exit event | A task-level transfer or dependency that leaves an island. |
| Working set | The live local storage required during execution. |
| Parallel width | The number of independent tasks or regions that can execute concurrently. |

## 4. Current Algorithm

### 4.1 Processing sequence

The island builder applies these steps in order:

1. Collect all matrix-setup tasks.
2. Find each setup task's direct analog compute users.
3. Create one analog island for each matrix-setup task.
4. Add the related analog compute tasks to that island.
5. Create explicit islands for balanced reductions.
6. Create explicit islands for distributed digital shards.
7. Assign eligible digital components with the two-terminal min-cut routine.
8. Assign remaining digital tasks with local byte affinity.
9. Assign graph input and output tasks to nearby islands.
10. Assign uncovered digital tasks with unweighted graph search.
11. Build the island affinity graph.
12. Build the island execution graph.
13. Attach stable island IDs to all assigned tasks.

### 4.2 Analog island creation

Each matrix-setup task creates one analog island.
The setup task programs one logical analog array.

The builder adds direct analog compute users to the same island.
This rule preserves the setup-to-MVM relationship.

The island therefore contains:

- One matrix-setup task.
- One or more analog compute tasks.
- Digital tasks that later assignment steps absorb.

This setup-to-MVM grouping is a valid hard constraint.
The current implementation also makes the full island indivisible.

### 4.3 Reduction and shard islands

Balanced reduction lanes receive independent reduction islands.
The reduction root receives a separate island.

Distributed digital matmul shards also receive independent islands.
These islands use explicit metadata from earlier graph transformations.

These are the only normal cases that create standalone digital islands.
Other digital tasks must join an existing anchor island.

### 4.4 Two-terminal min-cut assignment

The builder finds connected components of unassigned digital tasks.
It then counts the adjacent terminal islands for each component.

The current behavior is:

| Terminal count | Action |
|---:|---|
| 0 | Leave the component unassigned. |
| 1 | Assign the complete component to that island. |
| 2 | Use the min-cut routine to divide the component. |
| More than 2 | Leave the component unassigned. |

The min-cut routine therefore applies only to a narrow graph shape.
It does not handle general multi-terminal digital regions.

### 4.5 Local-affinity assignment

The builder processes the remaining digital tasks in DAG order.
For each task, it calculates incident resource bytes by adjacent island.

The builder selects the island with the largest byte value.
It selects the lower island ID when byte values are equal.

The builder commits each assignment immediately.
Later tasks can use earlier digital assignments as new affinity paths.

The rule also requires matching nonempty `source_layer` values.
This metadata acts as a hard grouping boundary.

### 4.6 Endpoint and fallback assignment

The builder assigns graph endpoints after local-affinity assignment.
It uses graph distance to find a nearby assigned island.

The final fallback uses undirected breadth-first search.
The search stops at the first distance that contains an island.

The fallback selects the lowest island ID when distances are equal.
It does not use communication bytes or timing data.

### 4.7 Island graph construction

The affinity graph combines communication in both directions.
It stores one undirected byte value for each island pair.

The execution graph combines task edges with the same island pair.
It stores one directed edge for each producer-consumer island pair.

The combined execution edge contains these fields:

- A producer island ID.
- A consumer island ID.
- A control-dependency flag.
- A data-dependency flag.
- The sum of transferred bytes.

This representation does not retain the original task-level events.

## 5. Measured GPT-2 Behavior

### 5.1 Measurement scope

The analysis used the GPT-2 Small task graph before physical placement.
It compared the original graph with the separable-region optimized graph.

Both graphs contained 360 analog islands.
The optimization changed task structure but retained the island count.

### 5.2 Digital task concentration

| Metric | Original graph | Separable graph |
|---|---:|---:|
| Total tasks | 3,553 | 3,529 |
| Digital tasks | 1,753 | 1,729 |
| Islands | 360 | 360 |
| Islands with no digital tasks | 288 | 288 |
| Maximum digital tasks in one island | 62 | 60 |
| Median digital tasks in one island | 0 | 0 |
| Digital tasks in the largest 12 islands | 744 | 720 |
| Share in the largest 12 islands | 42.4% | 41.6% |
| Share in the largest 36 islands | 88.4% | 88.2% |

The builder concentrates almost all digital work in 10% of the islands.
Most islands contain only setup and MVM work.

This concentration is not a result of a digital-load objective.
It is a result of byte affinity, traversal order, and anchor selection.

### 5.3 Min-cut coverage

The original graph had 48 eligible digital components.
The terminal counts were 6, 9, 21, or 24.

The separable graph had 84 eligible digital components.
The terminal counts were also 6, 9, 21, or 24.

No measured component had exactly two terminals.
Therefore, the min-cut routine processed none of these components.

The pass name and architecture suggest general min-cut assignment.
The measured GPT-2 graph does not receive that behavior.

### 5.4 Readiness loss

The GPT-2 graph had 2,674 external task-data edges.
Island construction reduced them to 694 directed island pairs.

The reduction combined 1,980 task-level edges.
Most combined pairs represented two through four independent transfers.

The timing profile uses one combined edge for each island pair.
It uses the sum of bytes and the latest producer finish time.

The runtime can receive and consume these transfers independently.
The scheduler instead sees one bulk transfer with one readiness time.

This behavior hides token-level and branch-level pipeline opportunities.

## 6. Efficiency Problems

### 6.1 The common graph does not use min-cut

The two-terminal condition excludes normal transformer components.
These components connect to many analog tiles.

The fallback path handles the main workload.
That path does not provide a balanced graph partition.

**Effect:** The implementation pays for min-cut infrastructure without using it on the target graph.

### 6.2 Assignment depends on traversal order

Local-affinity assignment commits tasks during one DAG traversal.
Each commitment changes the choices for later tasks.

Equal byte values use the island ID as the final decision.
The island ID represents construction order, not execution value.

**Effect:** Small IR-order changes can change large digital regions.

### 6.3 The objective uses communication only

The local rule measures incident resource bytes.
It does not measure digital work, slack, or parallel width.

It also does not measure the working set.
One island can absorb a large task region without a capacity limit.

**Effect:** Communication decreases at the cost of compute balance or memory pressure.

### 6.4 Islands are indivisible

The scheduler places one complete island on one core.
It cannot split a large digital region from its analog anchor.

This rule is stronger than the actual hardware requirement.
Only the setup task and its analog users require strict co-location.

**Effect:** Early grouping removes valid placement choices before the scheduler starts.

### 6.5 The execution graph loses events

The island execution graph combines all task edges for one island pair.
It does not retain producer and consumer task indexes.

It also does not retain separate readiness times.
The timing layer must use one conservative pair-level value.

**Effect:** A scheduler cannot place early-ready transfers independently from late-ready transfers.

### 6.6 The island timing span includes remote waits

The timing profile calculates an island span from its earliest task start to latest task finish.
Tasks in one island can have remote dependencies between them.

The span can include time when the island performs no local work.
The scheduler can also add incoming readiness costs for the same wait.

**Effect:** The timing objective can count remote waiting twice.

### 6.7 `source_layer` is not a hardware constraint

The current rule blocks assignment across different source layers.
It permits large regions inside one source layer.

Source-layer metadata describes compiler provenance.
It does not describe storage capacity or execution compatibility.

**Effect:** The rule blocks useful choices and permits harmful choices.

### 6.8 The fallback has weak information

The fallback uses unweighted graph distance.
It ignores edge bytes, task direction, and task criticality.

The island ID resolves equal distances.
This value has no performance meaning.

**Effect:** Uncovered tasks receive deterministic but low-quality assignments.

### 6.9 The pass has insufficient diagnostics

The current export contains only `task_index` and `island_id`.
It does not record why the builder selected an island.

It also omits work, memory, edge multiplicity, and assignment phase.
These omissions make performance regressions difficult to explain.

**Effect:** Development relies on manual graph inspection and simulation results.

## 7. Required Design Properties

Any replacement must satisfy these requirements.

### 7.1 Correctness requirements

1. A matrix-setup task and its array users must remain compatible.
2. Every task must belong to exactly one schedulable group.
3. The group graph must remain acyclic.
4. The builder must preserve all data and control dependencies.
5. The builder must produce deterministic output.

### 7.2 Quality requirements

1. The builder must support components with more than two terminals.
2. The builder must retain independent boundary events.
3. The builder must limit digital-work concentration.
4. The builder must account for local working-set pressure.
5. The builder must preserve useful parallel branches.
6. The builder must not use provenance as a hard hardware constraint.

### 7.3 Observability requirements

The builder must report these values for each island:

- The island kind and anchor.
- The assignment method for each task.
- The analog and digital work.
- The estimated working set.
- The internal and external bytes.
- The entry-event and exit-event counts.
- The internal critical span.
- The parallel width.

The builder must report each boundary event separately.
An aggregate pair summary can remain available for transfer scoring.

## 8. Alternative 1: Multi-Terminal Balanced Assignment

An opt-in implementation is available through:

```bash
--sculptor-build-task-graph-islands="digital-assignment=multi-terminal-balanced"
```

The default remains `digital-assignment=legacy`.
The experimental policy does not replace the existing algorithm.

### 8.1 Description

This alternative retains one flat island for each analog anchor.
It replaces the current two-terminal and local-affinity stages.

For each digital component, the builder collects all terminal islands.
It then assigns each digital task to one terminal.

The assignment uses a multi-terminal objective:

```text
assignment_cost =
    external_communication_bytes
  + lambda_work * digital_work_imbalance
  + lambda_memory * working_set_pressure
  + lambda_parallel * lost_parallel_width
```

The builder can use deterministic region growth or repeated graph cuts.
It must evaluate all terminals, not only two terminals.

### 8.2 Implementation route

1. Add per-task work and storage estimates.
2. Build one terminal set for each digital component.
3. Initialize one frontier for each terminal island.
4. Assign tasks with the multi-terminal objective.
5. Refine boundary tasks with deterministic local moves.
6. Reject assignments that exceed configured limits.
7. Record the assignment cost and reason.

### 8.3 Advantages

- The change fits the current flat-island model.
- It applies to normal transformer components.
- It can reduce the current digital-work concentration.
- It preserves the existing scheduler interface.

### 8.4 Limits

- Each completed island remains indivisible.
- A poor early partition still removes scheduler choices.
- Pair-level execution edges still lose readiness unless separately fixed.

### 8.5 Potential upside

This alternative can distribute digital work across more analog anchors.
It can also reduce dependence on task order and island IDs.

The expected implementation risk is moderate.
The expected architecture change is small.

### 8.6 Initial compiler result

The first experiment used GPT-2 Small with the separable-region optimization.
The hardware had a 100-by-100 mesh and four arrays per core.

The scheduler was `greedy-timing` with lookahead three.
This experiment did not run the simulator.

| Metric | Legacy | Multi-terminal balanced | Change |
|---|---:|---:|---:|
| Inter-core bytes | 5,189,632 | 4,950,016 | -4.62% |
| Transfer score | 11,714,560 | 6,804,480 | -41.91% |
| Predicted makespan | 12,434,960 ns | 11,817,430 ns | -4.97% |
| Islands with no digital tasks | 288 | 96 | -66.67% |
| Digital tasks in the largest 36 islands | 1,525 | 1,177 | -22.82% |
| Maximum digital tasks in one island | 60 | 62 | +3.33% |

The result improves broad distribution and communication cost.
It does not remove the largest individual hotspot.

The result is compiler evidence, not runtime evidence.
Simulator validation remains necessary before a production default changes.

## 9. Alternative 2: Hierarchical and Splittable Islands

### 9.1 Description

This alternative separates hard co-location from optional locality.
It creates small atomic groups and a higher-level grouping structure.

An atomic analog group contains:

- One matrix-setup task.
- The analog tasks that use its logical array.
- Only the digital glue that cannot move safely.

Independent digital regions become separate movable groups.
The hierarchy records affinity between these groups.

The scheduler can place related groups on one core.
It can separate them when work, memory, or timing requires separation.

### 9.2 Data model

```text
PlacementRegion {
    region_id
    kind
    member_tasks
    hard_colocation_set
    parent_region
    child_regions
    entry_events
    exit_events
    work
    working_set
    affinity
}
```

The parent region is a preference, not an indivisible placement unit.
The hard co-location set expresses the actual analog constraint.

### 9.3 Implementation route

1. Define atomic setup-to-MVM groups.
2. Find digital regions between atomic groups.
3. Preserve digital branches as separate regions.
4. Build a hierarchy from communication affinity.
5. Give the scheduler both hard constraints and soft parents.
6. Let the scheduler merge or separate child regions.
7. Commit final island IDs only after placement decisions.

### 9.4 Advantages

- The model preserves more placement freedom.
- The scheduler can balance digital work across cores.
- The scheduler can respond to hardware capacity.
- The model supports future memory-aware placement.
- The model separates correctness constraints from optimization preferences.

### 9.5 Limits

- The scheduler interface requires a significant extension.
- Search complexity increases because more groups remain movable.
- Fusion and timing consumers must understand hierarchical regions.

### 9.6 Potential upside

This alternative offers the largest long-term improvement.
It can retain communication locality without forcing complete digital regions onto one core.

It also creates a clear path for memory and timing co-design.
The expected implementation risk is high.

## 10. Alternative 3: Event-Preserving Phased Islands

### 10.1 Description

This alternative retains flat placement islands.
It divides each island's internal execution into ordered phases.

A phase contains tasks that share a readiness interval.
Each boundary transfer remains a separate event.

The scheduler places one island on one core.
It uses phase timing and event readiness during placement.

### 10.2 Data model

```text
IslandPhase {
    phase_id
    member_tasks
    local_work_ns
    entry_events
    exit_events
    earliest_ready_ns
    critical_path_remaining_ns
    working_set_bytes
}

IslandBoundaryEvent {
    producer_task
    consumer_task
    producer_island
    consumer_island
    bytes
    producer_ready_ns
    consumer_pressure
}
```

The aggregate island-pair edge remains available for the legacy score.
The timing scheduler uses the event table instead.

### 10.3 Implementation route

1. Preserve task indexes during island-edge construction.
2. Emit one boundary event for each task-level edge.
3. Group internal tasks into readiness phases.
4. Calculate local work without remote wait intervals.
5. Update `greedy-timing` to consume boundary events.
6. Retain aggregate edges for legacy schedulers.
7. Compare predicted timing with simulator timing.

### 10.4 Advantages

- The change preserves the current placement unit.
- It exposes token-level and branch-level readiness.
- It removes the island-span double count.
- Legacy placement scores can remain unchanged.
- The implementation can proceed in small steps.

### 10.5 Limits

- This alternative does not correct digital-work concentration by itself.
- Large flat islands remain indivisible.
- The timing scheduler becomes more detailed and more expensive.

### 10.6 Potential upside

This alternative can recover pipeline opportunities that the current graph hides.
It directly supports the separable-region optimization.

The expected implementation risk is low to moderate.
The timing benefit can be large when several transfers share one island pair.

## 11. Comparison

| Property | Multi-terminal assignment | Hierarchical islands | Phased islands |
|---|---|---|---|
| Corrects more-than-two terminals | Yes | Yes | No |
| Reduces digital concentration | Yes | Yes | No |
| Preserves placement freedom | Partial | Strong | No change |
| Preserves task readiness | With extra work | Yes | Strong |
| Supports memory limits | Yes | Strong | Reports pressure |
| Keeps current scheduler interface | Yes | No | Mostly |
| Implementation risk | Moderate | High | Low to moderate |
| Long-term architecture value | Moderate | High | High for timing |

The alternatives are not mutually exclusive.
A complete design can combine their strongest parts.

## 12. Recommended Route

### 12.1 Stage 1: Add an island audit

Add diagnostics before the grouping policy changes.
The audit must explain every task assignment.

The first report must include:

- Assignment phase and reason.
- Anchor task and terminal set.
- Digital and analog work.
- Internal and external bytes.
- Entry-event and exit-event counts.
- Estimated working set.
- Internal critical span.
- Parallel width.

This stage gives a stable baseline.
It also prevents unsupported performance claims.

### 12.2 Stage 2: Preserve boundary events

Implement the event model from Alternative 3.
Keep the aggregate edge model for existing schedulers.

Update timing-aware placement to use separate readiness events.
Do not use the latest producer as the readiness time for all transfers.

This stage has the best benefit-to-risk ratio.
It corrects known information loss without changing placement feasibility.

### 12.3 Stage 3: Add multi-terminal assignment

Implement Alternative 1 after the audit is available.
Use communication, work, memory, and parallel-width terms.

Compare the new assignment with the current assignment.
Use identical scheduler parameters and hardware parameters.

This stage corrects the inactive min-cut path.
It also reduces dependence on DAG order and island ID.

### 12.4 Stage 4: Evaluate hierarchical islands

Use simulator evidence to decide whether flat islands remain a limit.
Implement Alternative 2 if large indivisible regions still block performance.

Start with one transformer block.
Keep setup-to-MVM groups as hard atomic units.

This stage has the largest design cost.
It also provides the largest future design space.

## 13. Evaluation Plan

### 13.1 Structural measurements

Record these values for each model:

- Total island count.
- Island count by kind.
- Task count by island.
- Digital work by island.
- Working set by island.
- Entry-event and exit-event counts.
- Multi-terminal component counts.
- Task-edge to island-edge compression ratio.
- Maximum and percentile island sizes.

### 13.2 Placement measurements

Use the same hardware and scheduler parameters for each comparison.
Record these values:

- Total transfer cost.
- Inter-core bytes.
- Transfer cost per inter-core byte.
- Predicted critical path.
- Active core count.
- Digital-work imbalance.
- Maximum local working set.

### 13.3 Runtime measurements

Use runtime results only after compiler-level validation.
Record these values:

- End-to-end execution time.
- Critical memory stall.
- Network queue delay.
- Core idle time.
- Peak live memory by core.
- First and last route-ready times.

### 13.4 Required workloads

The initial evaluation must include:

- The small large-to-tiny linear fixture.
- GPT-2 Small.
- BERT.
- ViT Base.
- ResNet-18.

These models expose different fan-in, fan-out, convolution, and transformer structures.

## 14. Success Criteria

The new implementation succeeds when all criteria are true:

1. It assigns all supported multi-terminal components without fallback BFS.
2. It reduces digital-work concentration relative to the current implementation.
3. It retains independent readiness for separate task-level transfers.
4. It does not increase invalid placement or compilation failures.
5. It preserves numerical output.
6. It produces deterministic results.
7. It explains every assignment in the audit report.
8. It improves measured placement or runtime on more than one model family.

A transfer-score reduction alone is not sufficient.
The result must also preserve timing, memory, and correctness.

## 15. Conclusion

The current island builder preserves analog setup relationships.
That part of the design is necessary and useful.

The digital assignment does not match the structure of GPT-2.
Its min-cut routine does not process the measured multi-terminal components.

The fallback then concentrates digital work in a small set of islands.
The final execution graph also removes independent readiness events.

The recommended route starts with observability and event preservation.
It then adds multi-terminal balanced assignment.

Hierarchical islands remain the strongest long-term design.
They separate hard analog constraints from optional communication locality.
