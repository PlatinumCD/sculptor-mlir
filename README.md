# sculptor-mlir

**sculptor-mlir** is an experimental out-of-tree MLIR project for lowering a
small set of neural-network-shaped tensor and `linalg` programs into a staged
analog compute-in-memory execution model.

The repository currently contains the Sculptor dialect, compiler passes,
`sculptor-mlir-opt`, MLIR regression inputs, documentation, and the common
runtime foundation used by backend runtimes.

## Build Requirements

The project expects an existing LLVM/MLIR build tree. The important CMake
package directories are:

```text
<llvm-project-build>/lib/cmake/llvm
<llvm-project-build>/lib/cmake/mlir
```

The host build also needs:

- `cmake`
- `ninja`
- a C++20-capable compiler
- LLVM/MLIR headers, libraries, and CMake package files

Torch-MLIR is not required to build `sculptor-mlir`, but it is used by the
walkthrough flow to export PyTorch examples into Torch/Linalg MLIR.

## Build

From the repository root:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_DIR=/path/to/llvm-project-build/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm-project-build/lib/cmake/mlir \
  -DSCULPTOR_MLIR_BUILD_RUNTIME=ON

cmake --build build
```

The main compiler driver is:

```text
build/tools/sculptor-mlir-opt/sculptor-mlir-opt
```

## Compiler Flow

The compiler is staged around explicit IR boundaries:

| Stage | Boundary | Main pass |
|---|---|---|
| Canonicalize layers | tensor/`linalg` layer regions -> `sculptor.nn.*` ops | `sculptor-canonicalize-layers` |
| Extract layers | inline `sculptor.nn.*` ops -> outlined layer functions | `sculptor-extract-layers` |
| Convert layers | outlined `sculptor.nn.*` functions -> `sculptor.mvm` plus tensor/math glue | `sculptor-convert-layers` |
| Expand MVMs | `sculptor.mvm` -> matrix/vector/logical-array operations | `sculptor-expand-mvm-to-golem` |
| Materialize tasks | task regions -> callable task-stage functions | `sculptor-materialize-tasks` |
| Assemble graph | outlined work and resources -> symbolic task graph | `sculptor-assemble-task-graph` |
| Build islands | symbolic tasks -> stable logical placement islands | `sculptor-build-task-graph-islands` |
| Analyze timing | logical islands -> task critical-path and island-work metadata | `sculptor-analyze-task-graph-timing` |
| Schedule graph | island-annotated tasks -> placed/scheduled task graph metadata | `sculptor-schedule-task-graph` |
| Fuse graph | placed tasks -> same-island, same-core component tasks | `sculptor-fuse-task-graph` |
| Finalize resources | surviving logical resources -> runtime slots and offsets | `sculptor-finalize-task-graph-resources` |
| Lower shims | Sculptor execution ops -> backend-facing runtime shim calls | `sculptor-lower-golem-to-llvm-shims` |
| Emit runtime graph | scheduled graph -> runtime graph image builders | `sculptor-emit-runtime-graph` |

## Repository Layout

```text
include/                    Public headers and TableGen definitions
lib/                        Dialect, pass, conversion, and scheduling code
tools/sculptor-mlir-opt/    MLIR optimizer/pass driver
runtime/common/             Shared runtime ABI and backend foundation
tests/mlir/                 MLIR regression inputs
docs/                       MkDocs documentation site
```

Device-specific runtimes and simulator tooling are intentionally not part of the
first public tree. `runtime/common/` is the stable foundation intended for
future backend runtimes.

## Tests

The files under `tests/mlir/` are MLIR regression inputs with `RUN:` comments.
After building, they can be run manually with the built optimizer and LLVM's
`FileCheck`.

Example:

```bash
build/tools/sculptor-mlir-opt/sculptor-mlir-opt \
  tests/mlir/verify_task_region.mlir \
  --split-input-file \
  --verify-diagnostics \
  --allow-unregistered-dialect
```

## Documentation

The documentation is built with MkDocs from the `docs/` directory.

```bash
python3 -m pip install -r docs/requirements.txt
mkdocs -f docs/mkdocs.yml build --strict
```

The generated static site is written to `site/`, which is ignored by Git. The
repository includes a GitHub Pages workflow that can build and deploy the docs
from `master` once Pages is configured to use GitHub Actions.
