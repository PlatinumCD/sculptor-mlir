# Tests

The first public test corpus lives under `tests/mlir/`. These files are
standalone MLIR regression inputs for the Sculptor dialect and compiler passes.

The tests cover:

- parser and verifier behavior
- layer canonicalization
- layer extraction
- layer-to-MVM conversion
- MVM expansion into accelerator-facing IR
- task materialization
- task graph assembly
- scheduling metadata
- runtime graph emission

Most files include `RUN:` comments that show the intended `sculptor-mlir-opt`
and `FileCheck` invocation.

Example:

```bash
build/tools/sculptor-mlir-opt/sculptor-mlir-opt \
  tests/mlir/verify_task_region.mlir \
  --split-input-file \
  --verify-diagnostics \
  --allow-unregistered-dialect
```
