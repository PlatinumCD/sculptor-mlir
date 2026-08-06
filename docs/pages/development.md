# Development Guide

## Source Organization

```text
include/sculptor-mlir/  Public headers and TableGen definitions
lib/                    Pass and dialect implementations
runtime/                Reusable tile runtime library
tests/                  Model and lowering tests
tools/                  Compiler and inspection tools
docs/                   This documentation site
```

Mapping code belongs under `Transforms/mapping`. Mapping strategies belong
under `Transforms/planners`. Golem lowering belongs under `Transforms/Golem`
or the Golem conversion directory.

## Adding a Pass

1. Define the public pass interface in `include/`.
2. Implement the pass in the matching `lib/` directory.
3. Register the pass in the pass registry.
4. Add the source to the relevant CMake target.
5. Document its position and invariants in [Pass Pipeline](passes.md).
6. Add a focused MLIR or Python regression.
7. Run the full validation commands before committing.

## Commit Readiness

Before staging a change, verify:

- the project configures from a fresh build directory;
- the compiler and runtime targets build;
- model lowering tests pass;
- runtime tests pass when enabled;
- strict documentation builds pass;
- removed pass names have no remaining references;
- `git diff --check` is clean.

The pivot intentionally removes the old placement-island and global task-graph
deployment paths. Do not reintroduce compatibility wrappers unless a concrete
downstream consumer requires one.

