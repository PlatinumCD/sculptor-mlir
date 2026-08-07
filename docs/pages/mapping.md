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
to `spread`. Packed plans diagnose MVM bodies wider than the tile's analog
lane count.

- `packed` assigns every physical MVM in one body to analog lanes on one
  logical tile.
- `spread` assigns every physical MVM branch in one body to a distinct logical
  tile.

Both policies preserve matrix-setup lane bindings. The policy affects logical
tile formation, not physical mesh placement.

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
