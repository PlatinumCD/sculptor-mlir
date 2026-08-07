# Sculptor RA-Tree Report

`sculptor-ra-tree-report` creates a self-contained HTML report from MLIR that
contains `sculptor.mapping.ra_tree`. It reconstructs the SSA compute graph.
Then it verifies the tree version, graph fingerprint, counts, node identities,
and parent-child relationships before it writes the report.

The report provides three views:

- **RA hierarchy** shows temporal cuts, spatial cuts, operation leaves, and the
  expanded physical stages owned by each logical MVM leaf.
- **Compute DAG** shows producer-consumer dependencies between operations. With
  expanded IR, logical MVM nodes are replaced by their physical Golem stages
  and the corresponding physical SSA dependencies.
- **S-T graph** expands temporal cuts across normalized time and spatial cuts
  across abstract resource lanes. When expanded IR is supplied, each logical
  MVM footprint is refined into matrix-setup, vector-tile, array-MVM, and
  recombination stages in this same graph. Matrix setup and physical MVM
  operations with the same lane-binding group remain on one resource lane.
  The inspector also reports each SSA-derived MVM wave and each physical
  MVM's member index within that wave.

The report also provides these mapping details:

- **Mapping plan summary** compares every evaluated candidate and reports why
  the selected candidate won the configured objective.
- **Node inspection** shows reference latency, crossing bytes, communication,
  resource demand, and pipeline-stage estimates when a typed plan is present.

The S-T graph remains structural and does not invent physical core IDs. When a
typed mapping plan is present, the inspector displays analytic evaluator
estimates separately from normalized visual positions.

Both views support pan, zoom, operation search, semantic-block filtering,
hover details, and an inspector for iteration domains, tensor types, semantic
provenance, and compact MLIR.

## Build

```bash
cmake --build build/sculptor-mlir-pivot \
  --target sculptor-ra-tree-report
```

## Generate a report

```bash
build/sculptor-mlir-pivot/bin/sculptor-ra-tree-report \
  /path/to/plan_mapping.mlir \
  --expanded-ir /tmp/gpt2-mini-full-06-golem.mlir \
  -o /tmp/gpt2-mini-ra-tree.html \
  --json-output /tmp/gpt2-mini-ra-tree.json \
  --title "GPT-2 Mini Mapping Plan"
```

Use `--function=<symbol>` when one input module contains more than one
function with an RA tree. `--expanded-ir` is optional. The primary input remains
the validated logical mapping, while expanded stages are joined to it by
`sculptor.mapping.operation_id` and `sculptor.mapping.ra_leaf_id`. The HTML
embeds all report data and does not require a server or external JavaScript
package.

## Serve the report

```bash
python3 -m http.server 18987 \
  --bind 0.0.0.0 \
  --directory /tmp
```

Open `http://0.0.0.0:18987/gpt2-mini-ra-tree.html` on the host machine.

The GPT-2 Mini mapping checkpoint uses the dependency-level planner. The
report compares it against the deterministic temporal baseline and renders
the selected temporal and spatial cuts directly from compiler IR.
