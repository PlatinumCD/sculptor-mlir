# Mapping

## RA Tree

The Resource Allocation tree contains three node kinds:

- A temporal cut orders child regions.
- A spatial cut permits child regions to execute in parallel.
- A leaf references one compute operation or one tiled work unit.

The compute graph supplies dependency constraints. The RA tree supplies mapping
structure. A valid plan must obey both representations.

## Planning Strategies

`--sculptor-plan-mapping` accepts an ordered strategy list. Each strategy
transforms the current plan and passes its result to the next strategy.

Current strategies include:

| Strategy | Purpose |
|---|---|
| `setup-first` | Places matrix setup work before dependent execution |
| `mvm-wave` | Groups independent MVM work into spatial waves |
| `recursive-fork-join` | Recovers recursive SESE regions while preserving structured MVM bodies |
| `fan-out-cut` | Exposes parallel consumers after a fan-out |
| `consumer-bound-fill` | Binds fill work to its consumer |

## MVM Body Policy

`--sculptor-plan-mapping` accepts `mvm-body-policy=packed|spread` and defaults
to `spread`.

- `packed` fills each logical tile up to `arrays-per-core`. A ten-array wave on
  four-array tiles uses three logical tiles with occupancy `4 + 4 + 2`.
- `spread` assigns each array in a wave to a different logical tile. The same
  ten-array wave uses ten logical tiles.

Both policies preserve each matrix setup's persistent lane binding. A
branch-specific vector tile follows its physical MVM. Shared vector work,
recombination, and bias addition use the wave's deterministic home tile while
cross-tile dependencies remain explicit in the logical-tile graph.

## Setup Binding Policy

`--sculptor-plan-mapping` also accepts
`setup-binding-policy=global|consumer-anchored` and defaults to `global`.

- `global` preserves the original behavior: persistent analog bindings are
  available to unrelated digital work throughout logical-tile realization.
- `consumer-anchored` reserves each setup-bound tile for its MVM body. After
  realization, it orders active logical-tile identities by their first
  non-setup RA-tree use. Matrix setup still executes during initialization,
  but its lane appears at the point where its MVM consumer enters the S-T
  flow.

The setup policy is independent of `mvm-body-policy`. It changes the logical
RA-tree-to-S-T realization and does not perform physical mesh placement.

## Logical Tiles

The applied mapping plan creates logical tiles. A logical tile owns:

- its compute operations;
- its digital and analog lane assignments;
- its incoming and outgoing communication;
- its resource demand;
- its dependency order.

Logical tiles replace the old placement-island abstraction.

## Physical Placement

`--sculptor-place-logical-tiles` maps logical tiles to mesh coordinates. The
placement uses the logical-tile communication graph and hardware capacity.

Planning controls parallel structure. Placement controls physical distance.
Neither step changes the computation semantics.
