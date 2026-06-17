# `sculptor-mlir`

`sculptor-mlir` is an experimental MLIR compiler for lowering
neural-network-shaped tensor and `linalg` programs into an analog
compute-in-memory execution model.

`sculptor-mlir` is built as an out-of-tree MLIR project. It expects an existing
LLVM/MLIR build tree and focuses on the Sculptor dialect, compiler passes,
runtime lowering path, and simulator-facing support code.

<details class="doc-section" open markdown="1">
<summary markdown="block">## Build Requirements</summary>


The `sculptor-mlir` build needs these host tools available in `PATH`:

| Requirement | Used for |
|---|---|
| `cmake` | Configuring `sculptor-mlir` against LLVM and MLIR. |
| `ninja` | Driving the generated build tree. |
| C/C++ compiler toolchain | Building the dialect library, passes, tools, and runtime code. |
| LLVM/MLIR CMake packages | Providing `LLVM_DIR` and `MLIR_DIR` for the out-of-tree build. |

`sculptor-mlir` expects LLVM and MLIR CMake package configs from an LLVM project
build tree:

```text
<llvm-project-build>/lib/cmake/llvm
<llvm-project-build>/lib/cmake/mlir
```


</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Source Layout</summary>


The actual checkout path uses `src/`, not `source/`. This tree shows the main
source-bearing directories and omits generated output such as `site/` and hidden
tooling directories.

```text
src/
└── sculptor-mlir/
    ├── CMakeLists.txt
    ├── README.md
    ├── include/sculptor-mlir/Dialect/Sculptor/
    │   ├── Conversion/
    │   │   ├── golem/
    │   │   └── runtime/
    │   ├── IR/
    │   │   └── Ops/
    │   └── Transforms/
    │       ├── Golem/
    │       ├── Support/
    │       │   ├── Assembly/
    │       │   ├── Canonicalization/
    │       │   ├── Conversion/
    │       │   ├── Extraction/
    │       │   ├── IR/
    │       │   └── Layers/
    │       └── task_schedulers/
    ├── lib/Dialect/Sculptor/
    │   ├── Conversion/
    │   │   ├── golem/
    │   │   └── runtime/
    │   ├── IR/
    │   │   └── Ops/
    │   └── Transforms/
    │       ├── assemblers/
    │       ├── canonicalizers/
    │       ├── converters/
    │       ├── extractors/
    │       ├── Golem/
    │       ├── Support/
    │       │   └── Conversion/
    │       └── task_schedulers/
    ├── runtime/common/
    │   ├── include/
    │   └── lib/
    ├── tools/sculptor-mlir-opt/
    ├── tests/mlir/
    └── docs/
        ├── mkdocs.yml
        └── pages/
```


### `include/`

`include/` contains the public headers and TableGen files for the Sculptor
dialect: ops, types, pass declarations, conversion interfaces, and shared
compiler support.

### `lib/`

`lib/` contains the implementation behind those declarations: dialect behavior,
passes, rewrites, layer conversion, task graph construction, scheduling, Golem
expansion, and runtime lowering.

### `runtime/`

`runtime/` contains support code for lowered programs. The first public tree
includes `runtime/common/`, which provides the shared runtime ABI, graph image
model, task argument helpers, resource slot handling, and backend extension
surface.

### `tools/`

`tools/` contains the `sculptor-mlir-opt` pass driver.

### `tests/`

`tests/` contains MLIR regression inputs for parser, verifier, transformation,
conversion, scheduling, and runtime-graph emission behavior.

### `docs/`

`docs/` contains this MkDocs site: configuration and Markdown source pages.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Next</summary>


Read [Design](design.md) for the compiler pipeline, IR boundaries, pass
responsibilities, runtime model, and current limits.

</details>
