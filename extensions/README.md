# Mapping Extensions

## Purpose

This directory specifies four optional extensions for the Sculptor pivot pipeline.
The extensions correct the gap between spatial placement cost and simulated completion time.

The current compiler already has the correct major boundaries:

```text
tensor and MVM operations
  -> digital work units
  -> RA tree
  -> mapping plan
  -> logical-tile graph
  -> physical placement
  -> tile routines
  -> runtime graphs
```

The extensions add information and policies to these boundaries. They do not replace the RA tree, logical tiles, or tile routines.

## Specifications

| Extension | Specification | Primary result |
|---|---|---|
| Calibrated cost profiles | [calibrated_cost_profiles.md](calibrated_cost_profiles.md) | Reproducible task, memory, analog, and network costs |
| Temporal makespan scheduling | [temporal_makespan_scheduling.md](temporal_makespan_scheduling.md) | Placement decisions based on predicted completion time |
| Shard-level data flow | [shard_level_dataflow.md](shard_level_dataflow.md) | Independent shard readiness, routing, and consumption |
| Distributed reduction trees | [distributed_reduction_trees.md](distributed_reduction_trees.md) | Bounded fan-in and local, parallel reduction stages |

## Common Design Rules

1. Each extension is optional.
2. The default pipeline keeps its current behavior.
3. One data structure has one owner.
4. The compiler must not create a second compute graph.
5. The compiler must keep stable operation, work-unit, tile, routine, and resource identities.
6. The compiler must reject unsupported dynamic or ambiguous cases.
7. The compiler must not infer semantics from function names.
8. The compiler must preserve the atomic runtime-task ABI.
9. The compiler must emit enough provenance to reproduce each result.
10. Each extension must have a small exact test before a GPT-2 experiment.

## Shared Infrastructure

The implementation must reuse these current components:

| Existing component | Use |
|---|---|
| `mapping/ComputeGraph` | Operation and tensor DAG |
| `mapping/ResourceAllocationTree` | Spatial and temporal mapping structure |
| `MappingWorkUnit` | Static shard coordinates |
| `MappingWorkUnitEdge` | Exact work-unit dependency |
| `LogicalTileGraph` | Logical placement units and communication |
| `LogicalTilePlacementProblem` | Physical placement input |
| `OutlineTileRoutines` | Work-unit materialization and routine boundaries |
| `MaterializeTileRuntimeGraph` | Runtime resources, routes, and task dependencies |
| `sculptor-ra-tree-report` | Mapping, placement, and timing reports |

## Extension Boundary

The recommended pipeline is:

```text
--sculptor-canonicalize-layers
--sculptor-extract-layers
--sculptor-convert-layers
--sculptor-expand-mvm-to-golem
[--sculptor-duplicate-matrices]
--sculptor-expand-digital-work="
    parallel-workers=N
    dataflow=bulk|sharded
    reduction-tree=none|balanced
  "
--sculptor-build-ra-tree
--sculptor-plan-mapping="
    ...
    cost-profile=profile.json
  "
--sculptor-place-logical-tiles="
    ...
    objective=transfer-cost|makespan
    network-mode=ideal|finite|full
  "
--sculptor-outline-tile-routines
--sculptor-materialize-tile-runtime-graph
--sculptor-extract-tile-module
--sculptor-plan-tile-scratchpad
```

The cost profile and temporal objective do not change program semantics.
Shard data flow and reduction trees change dependency granularity under explicit legality rules.

## Recommended Implementation Order

1. Implement calibrated cost profiles.
2. Implement the temporal evaluator and report its predictions.
3. Use the temporal evaluator as an optional placement objective.
4. Extend digital work units into shard-level dependency chains.
5. Build balanced reduction trees for marked reductions.
6. Add timing-aware reduction pairing after the balanced form is correct.

This order is important. Temporal scheduling needs calibrated costs.
Reduction trees need independent shard values.

## Compatibility Matrix

| Profile | Placement objective | Data flow | Reduction tree | Behavior |
|---|---|---|---|---|
| Built-in legacy | `transfer-cost` | `bulk` | `none` | Current behavior |
| Calibrated | `transfer-cost` | `bulk` | `none` | Better reports, same placement objective |
| Calibrated | `makespan` | `bulk` | `none` | Time-aware placement with full-tensor barriers |
| Calibrated | `makespan` | `sharded` | `none` | Shard overlap with central assembly where required |
| Calibrated | `makespan` | `sharded` | `balanced` | Complete first implementation |

## Definition Of Done

The four extensions are complete when all statements are true:

1. The same input and options produce the same plan and timing report.
2. A profile identifies its schema, source, and content hash in the IR.
3. The compiler predicts the critical chain for a controlled test graph.
4. A completed shard can route before unrelated shards finish.
5. A supported reduction has a maximum fan-in of two.
6. All generated routine graphs remain acyclic.
7. Each active tile still lowers to a RISC-V object.
8. Extension-off results match the current baseline.
9. Numerical output remains within the declared reduction tolerance.
10. The report separates aggregate work from exposed makespan delay.
