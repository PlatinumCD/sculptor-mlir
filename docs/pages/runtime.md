# Runtime and Tile ABI

The runtime is a first-class CMake component in `runtime/`. It provides the
reusable tile runtime library used by generated tile programs.

## Build Options

Enable the runtime library with:

```bash
cmake -S . -B build \
  -DSCULPTOR_MLIR_BUILD_RUNTIME=ON
cmake --build build --target golem-runtime
```

The archive is written to:

```text
build/lib/libgolem-runtime.a
```

Enable host runtime tests with:

```bash
cmake -S . -B build \
  -DSCULPTOR_MLIR_BUILD_RUNTIME=ON \
  -DSCULPTOR_MLIR_BUILD_RUNTIME_TESTS=ON
ctest --test-dir build -R golem-runtime-test --output-on-failure
```

## Compiler Boundary

The compiler emits tile-local routines, resource metadata, task bindings, and
route metadata. The runtime consumes those records to:

- execute boot tasks;
- dispatch local tasks by global task ID;
- allocate compiler-planned tensor resources;
- move routed tensor payloads;
- report receive and transmit backpressure.

The runtime does not perform placement, mapping, scheduling search, LLVM
lowering, ELF linking, or mesh route generation.

## Runtime Layout

```text
runtime/
├── include/golem/runtime/  Public ABI and runtime headers
├── src/                    Runtime implementation
├── tests/                  Host-side runtime tests
└── CMakeLists.txt          Library and test targets
```

The generated tile ABI and the runtime library are separate concerns. The
compiler emits the tile-specific tables; `golem-runtime` supplies the common
execution machinery.

