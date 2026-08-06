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
| 8 | `--sculptor-apply-mapping-plan` | Materialize the selected tiled work units. |
| 9 | `--sculptor-place-logical-tiles` | Assign logical tiles to mesh coordinates. |
| 10 | `--sculptor-outline-tile-routines` | Create tile-local boot and dispatch routines. |
| 11 | `--sculptor-materialize-tile-runtime-graph` | Create local task and route metadata. |
| 12 | `--sculptor-extract-tile-module` | Isolate one physical tile module. |
| 13 | `--sculptor-plan-tile-scratchpad` | Plan tile-local storage. |

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

