# Calibrated Cost Profiles

## Status

This document defines an optional compiler extension.
The extension changes cost estimates only.
It does not change the compute graph, RA tree, placement space, or runtime ABI.

## Problem

The current reference evaluator uses broad formulas in `MappingEvaluator.cpp`.
It estimates digital work from iteration count and effective operations per cycle.
It estimates analog work from fixed MVM latency, shared I/O width, and recombination work.

These formulas do not distinguish expensive digital task kinds.
They also omit measured runtime, memory, partition, assembly, and descriptor costs.

The hardware options currently mix geometry and cost assumptions in `MappingHardwareModel`.
This makes calibration hard to reproduce and hard to share between planning and placement.

## Existing Infrastructure To Reuse

Use these files and types:

| File or type | Current role | Required extension |
|---|---|---|
| `mapping/MappingConfig.h` | Hardware geometry and rough cost fields | Keep geometry and capacity here |
| `mapping/ComputeGraph.h` | Stable operations, tensors, and work domains | Add stable cost features and semantic task kind |
| `mapping/MappingEvaluator.cpp` | Structural RA-tree evaluation | Call the shared cost-model API |
| `PlanMapping.cpp` | Builds the mapping problem | Load and serialize one profile |
| `PlaceLogicalTiles.cpp` | Builds the placement problem | Read the serialized profile |
| `SculptorAttrs.td` | Typed mapping attributes | Add profile and provenance attributes |
| `sculptor-ra-tree-report` | JSON and HTML reports | Export profile and cost breakdowns |

The compiler already uses `sculptor.semantic.section` for task classification.
The profile must use this semantic value before it uses an MLIR operation-name fallback.

## Recommended Interface

Do not add a new compiler pass.
Add a profile option to `--sculptor-plan-mapping`:

```text
--sculptor-plan-mapping="
  ...
  cost-profile=/path/to/golem-costs.json
"
```

An empty option selects a built-in profile named `legacy-v1`.
The built-in profile must reproduce the current formulas.

`--sculptor-plan-mapping` must serialize the resolved profile into the function IR.
Later passes must use the serialized data, not reopen the original file.

Use this attribute name:

```text
sculptor.mapping.cost_profile
```

This rule makes separate `sculptor-mlir-opt` invocations reproducible.

## Profile Format

Use JSON because LLVM already provides `llvm::json` and `MemoryBuffer`.
Do not add a YAML dependency.

The first schema can use this form:

```json
{
  "schema_version": 1,
  "name": "golem-qemu-sst-2026-08",
  "source": "qemu-sst-calibration-2026-08-01",
  "clock_frequency_hz": 1000000000,
  "digital_fallback": {
    "fixed_ns": 0.0,
    "ns_per_work_item": 0.125,
    "ns_per_input_byte": 0.0,
    "ns_per_output_byte": 0.0
  },
  "digital_task_kinds": {
    "digital.layer_norm": {
      "fixed_ns": 40.0,
      "ns_per_work_item": 1.8,
      "ns_per_input_byte": 0.02,
      "ns_per_output_byte": 0.02
    },
    "digital.tile_recombine": {
      "fixed_ns": 25.0,
      "ns_per_work_item": 0.4,
      "ns_per_input_byte": 0.03,
      "ns_per_output_byte": 0.02
    }
  },
  "analog": {
    "load_fixed_ns": 0.0,
    "load_ns_per_byte": 0.125,
    "execute_ns": 100.0,
    "store_fixed_ns": 0.0,
    "store_ns_per_byte": 0.125
  },
  "runtime": {
    "task_dispatch_ns": 0.0,
    "route_setup_ns": 0.0
  },
  "network": {
    "word_bits": 32,
    "hop_pipeline_ns": 1.0,
    "inject_fixed_ns": 0.0,
    "eject_fixed_ns": 0.0,
    "dma_ns_per_byte": 0.0
  }
}
```

The schema must reject unknown duplicate keys and invalid values.
All costs must be finite and nonnegative.

## Compiler Data Model

Add these files:

```text
include/sculptor-mlir/Dialect/Sculptor/Transforms/mapping/
  MappingCostProfile.h
  MappingCostModel.h

lib/Dialect/Sculptor/Transforms/mapping/
  MappingCostProfile.cpp
  MappingCostModel.cpp
```

Use these primary records:

```cpp
struct TaskCostRule {
  double fixedNs = 0.0;
  double nsPerWorkItem = 0.0;
  double nsPerInputByte = 0.0;
  double nsPerOutputByte = 0.0;
};

struct TaskCostFeatures {
  int64_t operationId = -1;
  int64_t workUnitId = -1;
  StringRef semanticTaskKind;
  int64_t workItems = 0;
  int64_t inputBytes = 0;
  int64_t outputBytes = 0;
};

struct TaskCostEstimate {
  double computeNs = 0.0;
  double memoryNs = 0.0;
  double runtimeNs = 0.0;
  double totalNs = 0.0;
};
```

`MappingCostProfile` owns immutable parsed coefficients and network costs.
`MappingCostModel` extracts features and calculates estimates.

Add these fields to `ComputeOperation`:

```cpp
std::string semanticTaskKind;
```

Do not copy all profile coefficients into each operation.

## Semantic Classification

Resolve a task kind in this order:

1. Use a consistent `sculptor.semantic.section` from the operation members.
2. Use the `ComputeOperationKind` fallback for analog stages.
3. Use the MLIR operation name for a structured digital fallback.
4. Use the profile's digital fallback rule.

If members have conflicting semantic sections, emit a diagnostic.
Do not select a cost from a symbol or function name.

## Cost Calculation

Use checked byte and work-item calculations.
Calculate the digital cost as:

```text
compute_ns = fixed_ns + ns_per_work_item * work_items
memory_ns = ns_per_input_byte * input_bytes
          + ns_per_output_byte * output_bytes
total_ns = compute_ns + memory_ns + runtime_task_dispatch_ns
```

Calculate an analog stage from explicit phases:

```text
total_ns = load_ns + execute_ns + store_ns + digital_support_ns
```

Keep each phase in `TaskCostEstimate`.
The temporal extension needs these phases for shared analog I/O scheduling.

The first version can keep the existing work-item definition.
A later profile version can add operation-specific feature extractors.

## IR Representation

Add typed attributes in `SculptorAttrs.td`:

```text
#sculptor.cost_rule<...>
#sculptor.cost_profile<...>
```

The profile attribute must contain:

- Schema version.
- Profile name.
- Source identifier.
- SHA-256 content hash.
- Clock value.
- Default digital rule.
- Ordered task-kind rules.
- Analog phase costs.
- Runtime costs.
- Network costs.

Sort task-kind rules by task-kind string before serialization.

Add profile identity to `MappingPlanAttr` and `LogicalTilePlacementAttr`.
Store the profile name and hash in summary CSV output.

## Integration Changes

### File Change Table

| File | Change |
|---|---|
| `include/.../mapping/MappingCostProfile.h` | Add the immutable profile model and JSON loader API |
| `include/.../mapping/MappingCostModel.h` | Add feature extraction and cost-estimate APIs |
| `lib/.../mapping/MappingCostProfile.cpp` | Parse, verify, hash, and serialize profiles |
| `lib/.../mapping/MappingCostModel.cpp` | Calculate digital, analog, memory, and runtime costs |
| `mapping/ComputeGraph.h/.cpp` | Preserve one stable semantic task kind per operation |
| `mapping/MappingProblem.h` | Add a constant profile reference |
| `mapping/MappingEvaluator.cpp` | Replace inline leaf formulas with the shared model |
| `PlanMapping.h/.cpp` | Add options, load the profile, and attach provenance |
| `PlaceLogicalTiles.cpp` | Read the serialized profile for temporal placement |
| `SculptorAttrs.td` | Add typed profile, rule, and provenance attributes |
| `mapping/CMakeLists.txt` | Compile the two new mapping sources |
| `tools/ra-tree-report` | Export task costs and profile identity |

### `PlanMapping.cpp`

1. Parse `cost-profile` once per module.
2. Verify that the profile agrees with fixed hardware widths.
3. Serialize the resolved profile on each mapped function.
4. Add a profile reference to `MappingProblem`.
5. Pass the profile into `ReferenceMappingEvaluator`.

### `MappingEvaluator.cpp`

1. Replace direct digital formulas with `MappingCostModel` calls.
2. Replace direct analog formulas with explicit phase estimates.
3. Keep current structural T-cut and S-cut composition.
4. Preserve `crossingBytes` as a separate metric.
5. Record task cost breakdowns in node evaluation data.

### `PlaceLogicalTiles.cpp`

1. Read the serialized profile from the function.
2. Reject a missing profile only when the makespan objective is selected.
3. Pass the profile to the temporal placement problem.
4. Add profile provenance to the CSV summary.

## Calibration Tool

Add an optional host tool:

```text
tools/cost-profile/fit_profile.py
```

The tool reads measurement rows with these fields:

```text
task_kind,work_items,input_bytes,output_bytes,measured_ns
```

The tool writes a versioned JSON profile and a fit report.
The compiler must never start QEMU or SST during compilation.

## Diagnostics

Emit a compiler error for these conditions:

- Unsupported profile schema.
- Missing required profile field.
- Negative or nonfinite coefficient.
- Duplicate task-kind rule.
- Arithmetic overflow during feature extraction.
- Conflicting semantic task kinds in one compute operation.
- Profile word width that conflicts with the selected hardware model.
- Unknown task kind when strict profile mode is active.

Add `strict-cost-profile=true|false` to `--sculptor-plan-mapping`.
The default is `false` for compatibility.

## Tests

Add focused tests in `tests/python_tests/test_mapping_cost_profile.py`.

The tests must cover:

1. The built-in profile reproduces current costs.
2. A task-kind override changes only matching operations.
3. Work-unit costs use work-unit extents.
4. Input and output bytes use static tensor sizes.
5. Invalid JSON produces a clear diagnostic.
6. Negative and nonfinite values fail.
7. Profile serialization is deterministic.
8. A serialized profile works in a second compiler invocation.
9. The CSV contains the profile name and hash.
10. Extension-off placement matches the current baseline.

## Incremental Delivery

### Phase 1

Add the profile parser, data model, legacy profile, and provenance.
Keep the old evaluator active behind the legacy profile.

### Phase 2

Move the current evaluator formulas into `MappingCostModel`.
Make the legacy profile produce identical values.

### Phase 3

Add measured task-kind rules and cost breakdown reports.

### Phase 4

Connect the profile to temporal placement.

## Non-Goals

This extension does not:

- Collect simulator measurements during compilation.
- Change task boundaries.
- Change worker counts.
- Change placement by itself.
- Model dynamic tensor shapes.
- Claim physical accuracy without a documented calibration set.

## Easiest Effective First Version

Implement JSON loading, a built-in legacy profile, and semantic task-kind overrides.
Then replace only the leaf-cost code in `MappingEvaluator.cpp`.

This version adds one shared cost API without changing the compiler pipeline.
