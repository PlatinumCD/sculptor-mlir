# sculptor-mlir

Sculptor is an out-of-tree MLIR compiler for tiled analog and digital
accelerators. The pivot architecture maps explicit compute operations through a
Resource Allocation tree before it creates tile routines.

## Build Requirements

The build requires:

- CMake;
- Ninja;
- a C++20 compiler;
- an LLVM and MLIR build tree.

Configure and build the compiler:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir

cmake --build build --target sculptor-mlir-opt
```

The compiler driver is `build/bin/sculptor-mlir-opt`.

## Compiler Flow

| Stage | Pass |
|---|---|
| Normalize neural-network layers | `--sculptor-canonicalize-layers` |
| Expose layer operations | `--sculptor-extract-layers` |
| Decompose layers into tensor and MVM operations | `--sculptor-convert-layers` |
| Expand MVM operations into Golem operations | `--sculptor-expand-mvm-to-golem` |
| Expand independently mappable digital work | `--sculptor-expand-digital-work` |
| Build the RA tree | `--sculptor-build-ra-tree` |
| Select spatial and temporal cuts | `--sculptor-plan-mapping` |
| Place logical tiles on the mesh | `--sculptor-place-logical-tiles` |
| Create tile routines | `--sculptor-outline-tile-routines` |
| Create tile-local runtime graphs | `--sculptor-materialize-tile-runtime-graph` |
| Extract one tile module | `--sculptor-extract-tile-module` |
| Plan tile-local scratchpad storage | `--sculptor-plan-tile-scratchpad` |

This flow does not use placement islands. The RA tree preserves spatial and
temporal structure until logical-tile placement.

## Source Layout

```text
include/                                  Public headers and TableGen files
lib/Dialect/Sculptor/Transforms/mapping/  RA-tree and logical-tile model
lib/Dialect/Sculptor/Transforms/planners/ Mapping strategies
lib/Dialect/Sculptor/Transforms/Golem/    MVM expansion
tools/ra-tree-report/                     Interactive mapping report
tools/sculptor-mlir-opt/                  Compiler driver
tests/model_tests/                        Generated model lowering tests
tests/python_tests/                       Model-family lowering tests
docs/                                     MkDocs site
```

## Documentation

Build the documentation:

```bash
python3 -m pip install -r docs/requirements.txt
mkdocs -f docs/mkdocs.yml build --strict
```

The generated site is in `site/`.
