# Mapping Planners

This directory contains mapping-policy implementations used by
`sculptor-plan-mapping`.

Each policy has one directory containing its private implementation files.
Shared compute-graph, RA-tree, verification, and evaluation infrastructure
belongs in the sibling `mapping` directory.

Planner registration is explicit. A planner must not modify executable MLIR.
Each planner refines the RA tree that the preceding strategy produced.

## Registered planners

- `setup-first` is an unconditional first compiler phase. It creates one
  spatial frontier for every matrix setup and puts
  this frontier before the current compute subtree.
- `layer-cut` contracts operation dependencies into the semantic-layer graph.
  Every ready layer frontier becomes an S-cut, while successive frontiers
  become children of a T-cut. Existing operation-level S/T structure inside a
  layer is preserved under the mandatory setup-first frontier.
- `mvm-wave` preserves wave order with temporal cuts. Within each wave, vector
  preparation occurs first, physical MVM members form one spatial cut, and
  optional tile recombination occurs last.
- `recursive-fork-join` detects nested single-entry/single-exit regions in the
  compute DAG. It maps sequential regions to temporal cuts and disjoint
  fork/join branches to spatial cuts. Complete MVM waves remain structured
  bodies: global region discovery treats each wave as one atom. Independent,
  consecutive wave atoms with identical producer and consumer frontiers form
  one spatial cohort, including multi-exit regions that are not strictly
  single-entry/single-exit. Within each wave, vector preparation uses the
  logical tile's digital lane before a spatial frontier over that tile's analog
  lanes. Recombination and bias remain on the same digital lane after the
  analog frontier. Unstructured regions retain a conservative topological
  temporal order.
- `fan-out-cut` exposes independent consumers of a fan-out through a spatial
  cut.
- `consumer-bound-fill` keeps fill operations with the consumers that use
  their values.

The `strategies` option gives an ordered, comma-separated pipeline. For
example:

```text
--sculptor-plan-mapping="strategies=setup-first,mvm-wave,fan-out-cut,consumer-bound-fill \
  arrays-per-core=4"
```

This pipeline isolates setup, refines MVM waves, exposes fan-out parallelism,
and binds fill operations to their consumers. Planning does not modify
executable IR.

Work-unit dependencies refine an operation-level compute-graph edge without
replacing the compute DAG. Each edge identifies the producer operation and
work unit, the first downstream compute operation, and the transferred byte
count. Verification, evaluation, logical-tile affinity, and report rendering
use these exact endpoints. Operation-level edges remain the fallback when no
work-unit refinement exists.

After the final strategy, planning realizes the selected RA tree onto logical
tile resources. Each logical tile contains one digital lane and
`arrays-per-core` analog lanes. T-cut children inherit the same resource pool
in sequence. S-cut children receive disjoint lane subsets. The resulting
`sculptor.mapping.plan` records every node's inherited resources and every
leaf's logical tile, lane, and estimated execution interval. This realization
is the sole source of lane and time coordinates in the S-T report.

`digital-scheduling-policy=earliest-finish` makes that final realization
schedule-aware. T-cut children retain barrier order, S-cut children receive
disjoint lanes and may overlap, and each unpinned digital leaf chooses the
legal lane minimizing `max(lane availability, input arrival) + task cost`.
Input arrival charges one logical network hop before physical placement.
MVM-body work remains pinned to its resident array tile; uniform-sibling
affinity remains a tie-break preference rather than a hard constraint.

`digital-scheduling-policy=progressive` applies the same schedule-aware score
to a monotonically expanding set of flexible digital lanes. It starts with one
legal lane per encountered resource pool and opens an unused lane only when
the resulting completion-time improvement exceeds its incremental estimated
communication cost. Semantically required MVM-body placements bypass the
admission mask.

`digital-scheduling-policy=sliding-window` restricts flexible digital leaves
to a fixed-width, work-progressed window over increasing logical tile IDs.
`digital-window-size` sets the width. The head advances monotonically from the
first tile toward the last, so a flexible lane that falls behind the head can
never receive work again. Required MVM-body tiles bypass the window for
correctness. Physical `schedule=snake` maps this logical order onto an adjacent
snake path through the mesh; communication-optimizing placement schedules may
permute it.

`mvm-body-policy=first-use-window` makes matrix-home selection follow the
first scheduled physical-MVM wave for each lane-binding group. New matrices
receive a free analog lane inside the same fixed-width logical window; later
uses remain pinned to that home. The window advances only between waves and
never moves backward. When independent S-cut branches share a persistent home
tile, their digital control work is serialized by the schedule-aware lane
clock while distinct analog lanes remain available concurrently. This policy
requires `digital-scheduling-policy=sliding-window`.

`mvm-body-policy=first-use-adaptive` keeps first-use ordering and permanent
matrix homes, but treats matrix groups in sibling RA spatial branches as hard
tile conflicts. It prefers the active window and spills to a nearby tile only
when strict locality would serialize independent work; it never migrates or
reloads an already-bound matrix.

A setup-only spatial frontier consumes one logical tile for every
`arrays-per-core` setup leaves. This models analog-lane capacity without
committing those leaves to physical tiles or lane indices.

Expanded analog operations carry a deterministic
`sculptor.mapping.lane_binding_group`. One group contains exactly one
`sculptor.array.set` stage and every physical MVM stage whose load, execute,
and store operations use that logical array. A spatial cut cannot split one
binding group across multiple resource lanes.

Lane-binding groups are persistent across temporal cuts. A matrix setup and
every physical MVM that consumes its logical array therefore retain one
logical `(tile, analog lane)` assignment. Digital work, including vector
tiling and recombination, uses only the tile's digital lane.

The compute graph also derives deterministic MVM waves from SSA flow. One
wave contains the vector-tile producers, the independently executable
physical MVM stages, and the optional tile-recombine stage for one expanded
MVM. Optional bias addition is part of the same wave. `packed` fills each
logical tile's analog lanes before opening another tile; `spread` uses one
logical tile per physical MVM. Matrix setup and physical MVM work retain one
persistent lane binding. Branch-specific vector work follows its MVM, while
shared vector work, recombination, and bias use the wave's home tile.
