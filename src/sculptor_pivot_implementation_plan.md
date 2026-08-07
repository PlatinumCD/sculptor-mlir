# Sculptor Pivot Implementation Plan

## Goal

The pivot maps operations through an RA tree and a logical-tile graph. It does
not reconstruct the retired placement-island model.

## Frozen Passes

The following passes have stable responsibilities:

| Order | Pass | Responsibility |
|---|---|---|
| 1 | `--sculptor-canonicalize-layers` | Normalize source layer patterns |
| 2 | `--sculptor-extract-layers` | Expose high-level layer operations |
| 3 | `--sculptor-convert-layers` | Decompose layers without task outlining |
| 4 | `--sculptor-expand-mvm-to-golem` | Expose matrix setup and array operations |
| 5 | `--sculptor-expand-digital-work` | Create independent digital work units |
| 6 | `--sculptor-build-ra-tree` | Build the baseline hierarchy |
| 7 | `--sculptor-plan-mapping` | Apply an ordered strategy list |
| 8 | `--sculptor-place-logical-tiles` | Assign logical tiles to the mesh |
| 9 | `--sculptor-outline-tile-routines` | Materialize selected work units and create boot and dispatch routines |
| 10 | `--sculptor-materialize-tile-runtime-graph` | Create tile-local task and route metadata |
| 11 | `--sculptor-extract-tile-module` | Produce one standalone tile module |
| 12 | `--sculptor-plan-tile-scratchpad` | Assign tile-local storage |

`--sculptor-apply-mapping-plan` is a terminal inspection utility. It consumes
the mapping representation and is not part of executable deployment lowering.

## Core Data Model

### Compute Graph

The compute graph records stable operation IDs, dependencies, and operation
order. It is the semantic source for all mapping constraints.

### Resource Allocation Tree

The RA tree contains spatial cuts, temporal cuts, and leaves. Each leaf refers
to one operation or one explicit tiled work unit.

### Mapping Plan

The mapping plan records planner decisions. Strategies execute in list order.
Each strategy receives the result of the previous strategy.

### Logical Tile

A logical tile contains operations, lane assignments, dependencies, resource
demand, and communication edges. It is the unit of physical placement.

## Required Invariants

1. Every planned operation has one stable compute ID.
2. Every leaf refers to a valid operation or work unit.
3. The RA tree obeys all compute-graph dependencies.
4. Matrix setup precedes each dependent MVM.
5. Bound array operations use one compatible analog lane.
6. Digital work uses the digital lane.
7. Each logical tile obeys its analog-lane capacity.
8. Physical placement does not change operation semantics.
9. Tile outlining removes the original mapped graph.
10. Extracted tile modules contain no cross-module SSA references.

## Removed Architecture

The pivot no longer contains:

- task-graph island construction;
- island compatibility adapters;
- island schedulers;
- island timing analysis;
- island-aware graph fusion;
- island visualization and simulator exporters.

The logical-tile graph owns placement input. The RA tree owns spatial and
temporal mapping structure.

## Next Cleanup

The next cleanup removes the old deployment path. This work includes:

1. Remove legacy task materialization and global task-graph assembly.
2. Remove core partitioning that consumes scheduled task graphs.
3. Remove old core extraction and resource finalization.
4. Keep tile routine outlining and tile runtime materialization.
5. Keep tile extraction and tile-local scratchpad planning.
6. Update conversion and runtime ABI passes for tile modules only.
7. Remove deployment tests that use the global scheduled graph.
8. Rebuild and run the complete RA-tree-to-tile pipeline.

## Verification

For each cleanup step:

1. Build `sculptor-mlir-opt`.
2. Run the RA-tree planner regression.
3. Run the logical-tile placement regression.
4. Run tile outlining with `--verify-each`.
5. Run tile runtime materialization with `--verify-each`.
6. Make sure that removed pass names are absent from `--help`.
