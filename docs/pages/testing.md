# Testing and Validation

Validation is divided into compiler, model-lowering, runtime, and
documentation checks.

## Build the Compiler

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-build/lib/cmake/llvm
cmake --build build -j2
```

## Python Lowering Tests

The Python tests cover supported model families, with and without biases, and
compile the resulting tile program to a RISC-V object:

```bash
../../.venv/bin/python tests/python_tests/run_all.py
```

Model generators and larger model tests are under `tests/model_tests/`:

```bash
../../.venv/bin/python tests/model_tests/run_all.py
```

## Runtime Tests

```bash
ctest --test-dir build -R golem-runtime-test --output-on-failure
```

## Documentation and Hygiene

Build the documentation in strict mode:

```bash
python -m mkdocs build -f docs/mkdocs.yml --strict
```

Before a commit, also run:

```bash
git diff --check
```

When a pass changes, inspect its intermediate MLIR and run the smallest
affected model-family test before running the complete suite.

