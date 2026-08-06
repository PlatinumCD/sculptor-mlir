# Sculptor MLIR

Sculptor maps tensor programs to logical tiles and then to a physical mesh.
The compiler keeps computation structure visible until mapping is complete.

## Primary Pipeline

| Stage | Pass | Result |
|---|---|---|
| Normalize layers | `--sculptor-canonicalize-layers` | Canonical neural-network operations |
| Expose layers | `--sculptor-extract-layers` | Explicit layer operations |
| Decompose layers | `--sculptor-convert-layers` | General tensor and `sculptor.mvm` operations |
| Expand analog work | `--sculptor-expand-mvm-to-golem` | Matrix setup, vector tile, array load, execute, store, and recombine operations |
| Expand digital work | `--sculptor-expand-digital-work` | Independently mappable digital work units |
| Build hierarchy | `--sculptor-build-ra-tree` | A Resource Allocation tree over compute operations |
| Select a plan | `--sculptor-plan-mapping` | Ordered spatial and temporal planning decisions |
| Place logical tiles | `--sculptor-place-logical-tiles` | Logical tiles assigned to mesh coordinates |
| Outline routines | `--sculptor-outline-tile-routines` | Tensor-level routines inside tile modules |
| Build runtime graph | `--sculptor-materialize-tile-runtime-graph` | Tile-local tasks and communication resources |
| Extract one tile | `--sculptor-extract-tile-module` | A standalone module for one tile |
| Plan scratchpad | `--sculptor-plan-tile-scratchpad` | Tile-local storage offsets and sizes |

The pivot pipeline does not use placement islands. It maps explicit compute
operations through an RA tree and a logical-tile graph.

## Source Layout

| Directory | Purpose |
|---|---|
| `Transforms/mapping` | Compute graphs, RA trees, mapping plans, logical tiles, and physical placement |
| `Transforms/planners` | Registered planning strategies |
| `Transforms/Golem` | Expansion of `sculptor.mvm` into Golem operations |
| `Transforms/converters` | Layer decomposition |
| `Transforms/canonicalizers` | Layer normalization |

Use the navigation to read about the [architecture](design.md), [pass
pipeline](passes.md), [mapping and RA tree](mapping.md), [runtime and tile
ABI](runtime.md), [testing](testing.md), and [development workflow](development.md).
