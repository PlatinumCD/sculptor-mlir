# Pass Pipeline

The pivot pipeline keeps semantic operations visible while it builds the
mapping. Physical tile routines and runtime resources are created only after
the mapping is selected.

## Ordered Passes

| Order | Pass | Purpose |
|---:|---|---|
| 1 | `--sculptor-canonicalize-layers` | Normalize supported neural-network patterns. |
| 2 | `--sculptor-extract-layers` | Expose explicit layer operations. |
| 3 | `--sculptor-convert-layers` | Decompose layers into tensor operations and `sculptor.mvm`. |
| 4 | `--sculptor-expand-mvm-to-golem` | Add matrix setup, vector tile, load, execute, store, and recombine operations. |
| 5 | `--sculptor-expand-digital-work` | Create independently mappable digital work units. |
| 6 | `--sculptor-build-ra-tree` | Build spatial cuts, temporal cuts, and operation leaves. |
| 7 | `--sculptor-plan-mapping` | Apply an ordered mapping strategy list. |
| 8 | `--sculptor-place-logical-tiles` | Assign logical tiles to mesh coordinates. |
| 9 | `--sculptor-outline-tile-routines` | Materialize selected work units and create tile-local boot and dispatch routines. |
| 10 | `--sculptor-materialize-tile-runtime-graph` | Create local task and route metadata. |
| 11 | `--sculptor-extract-tile-module` | Isolate one physical tile module. |
| 12 | `--sculptor-plan-tile-scratchpad` | Plan tile-local storage. |

## Terminal Mapping Materialization

`--sculptor-apply-mapping-plan` is an inspection utility outside the executable
deployment pipeline. It materializes selected tiled work units and consumes the
RA tree, mapping plan, and logical-tile graph. Do not run placement or routine
outlining after it.

The executable path places the logical plan first. Routine outlining then
materializes the selected work units under the locked physical placement.

## Inspecting Intermediate IR

Write an output after each stage so that the representation can be inspected:

```bash
sculptor-mlir-opt input.mlir \
  --sculptor-canonicalize-layers \
  -o 01-canonicalized.mlir
```

Repeat the pattern for each pass. Use `--verify-each` when debugging a pass
boundary.

## Boundary Rules

`sculptor.mvm` remains the logical analog operation until Golem expansion.
The RA tree and logical-tile graph are mapping representations. Runtime task
graphs, slots, offsets, and tile routines are downstream deployment artifacts.
