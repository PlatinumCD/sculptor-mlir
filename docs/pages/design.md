# Design

## Compiler Boundary

Sculptor separates logical mapping from runtime lowering. Mapping operates on
tensor operations and explicit analog operations. Runtime lowering starts after
the compiler assigns operations to physical tiles.

## Compute Representation

`--sculptor-convert-layers` decomposes neural-network layers. It preserves
`sculptor.mvm` as the analog compute operation.

`--sculptor-expand-mvm-to-golem` exposes the analog sequence:

1. Matrix setup
2. Vector tile
3. Array load
4. Array execute
5. Array store
6. Result recombination

The RA tree uses these operations as leaves. It does not hide them inside task
regions or placement islands.

## Mapping Representation

The mapping system has four main data structures.

| Data structure | Purpose |
|---|---|
| Compute graph | Records operation dependencies and stable operation order |
| Resource Allocation tree | Represents spatial cuts, temporal cuts, and operation leaves |
| Mapping plan | Records the ordered planner decisions and selected realization |
| Logical-tile graph | Groups planned work and records communication between logical tiles |

A planner changes the RA tree. A placement algorithm maps logical tiles to the
mesh. These are separate decisions.

## Hardware Model

A logical tile contains one digital lane and a configurable number of analog
lanes. Matrix setup and array execution use analog lanes. Digital tensor
operations use the digital lane.

The planner enforces lane bindings between each matrix setup and its array
operations. A physical placement pass later assigns each logical tile to a mesh
coordinate.

## Deployment Boundary

`--sculptor-outline-tile-routines` creates tensor-level routines for placed
logical tiles. Matrix setup operations become boot routines. Other work becomes
dispatch routines.

`--sculptor-materialize-tile-runtime-graph` creates tile-local task and route
metadata. `--sculptor-extract-tile-module` selects one tile. Storage planning
then operates on that isolated module.

The legacy island scheduler and island timing model are not part of this
architecture.
