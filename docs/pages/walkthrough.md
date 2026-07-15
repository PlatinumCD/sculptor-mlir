# Walkthrough With Example

This walkthrough uses `linear_example.py` as a small PyTorch example. The goal
is to start with a model, export it with Torch-MLIR, and use that exported IR as
the input to the `sculptor-mlir` pipeline.

The overall flow is:

```text
linear_example.py
-> Torch dialect MLIR
-> torch-mlir linalg-on-tensors lowering
-> tensor/linalg MLIR
-> sculptor-mlir layer recognition and lowering
-> symbolic task graph and runtime graph metadata
-> backend/runtime integration
```


<details class="doc-section" open markdown="1">
<summary markdown="block">## Python Program</summary>


The starting point is a single biased linear layer. This is the full Python
file used throughout the walkthrough:

```python
import argparse

import torch
from torch_mlir import fx


class SingleLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=True)

        with torch.no_grad():
            weights = torch.arange(
                1,
                3 * 4 + 1,
                dtype=torch.float32,
            ).reshape(3, 4)
            bias = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
            self.linear.weight.copy_(weights)
            self.linear.bias.copy_(bias)

    def forward(self, x):
        return self.linear(x)


model = SingleLinearModel()
model.eval()

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("run", "mlir"),
        default="run",
    )
    args = parser.parse_args()

    if args.mode == "mlir":
        mlir_module = fx.export_and_import(
            model,
            x,
            output_type="torch",
            func_name="forward",
        )
        print(mlir_module)
        return

    with torch.no_grad():
        print(model(x))


if __name__ == "__main__":
    main()
```


The model is intentionally small and deterministic:

- Input tensor: `x`, shape `1x4`.
- Weight matrix: `linear.weight`, shape `3x4`.
- Bias vector: `linear.bias`, shape `3`.
- Output tensor: shape `1x3`.

With the fixed weights and bias, the expected result is:

```text
31.000000 69.000000 112.000000
```


</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Torch-MLIR Export</summary>


Torch-MLIR needs three things from this example:

| Piece | Role |
|---|---|
| `model` | The `torch.nn.Module` to trace and import. |
| `x` | A concrete example input that fixes shape and dtype. |
| `fx.export_and_import(...)` | The Torch-MLIR entry point that produces MLIR. |

The `--mode mlir` branch is the export path:

```python
mlir_module = fx.export_and_import(
    model,
    x,
    output_type="torch",
    func_name="forward",
)
print(mlir_module)
```


`output_type="torch"` asks Torch-MLIR to emit Torch dialect IR. `func_name`
sets the exported function name to `forward`, which is the entry point consumed
by the later compiler pipeline.

The export produces this Torch dialect module:

```mlir
module {
  func.func @forward(%arg0: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[1,3],f32> {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>) : !torch.vtensor<[3,4],f32>
    %1 = torch.vtensor.literal(dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>) : !torch.vtensor<[3],f32>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %2 = torch.aten.transpose.int %0, %int0, %int1 : !torch.vtensor<[3,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,3],f32>
    %3 = torch.aten.mm %arg0, %2 : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,3],f32> -> !torch.vtensor<[1,3],f32>
    %4 = torch.aten.add.Tensor %3, %1, %int1 : !torch.vtensor<[1,3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[1,3],f32>
    return %4 : !torch.vtensor<[1,3],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x040000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041",
      torch_tensor_3_torch.float32: "0x040000000000803F000080BF00000040"
    }
  }
#-}
```


At this point, the model is no longer Python code. It is an MLIR module that
represents the linear layer as Torch dialect operations.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Linalg-On-Tensors IR</summary>


The next stage is Torch-MLIR's backend pipeline. It lowers the Torch dialect
program into tensor and `linalg` operations for `sculptor-mlir`.

This is produced by piping the Python export into Torch-MLIR's
linalg-on-tensors backend pipeline:

```bash
python3 linear_example.py --mode mlir |
  torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline
```


After that pipeline, the IR looks like this:

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %0 = tensor.empty() : tensor<4x3xf32>
    %transposed = linalg.transpose ins(%cst_1 : tensor<3x4xf32>) outs(%0 : tensor<4x3xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x3xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<1x4xf32>, tensor<4x3xf32>) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst : tensor<1x3xf32>, tensor<3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.addf %in, %in_2 : f32
      linalg.yield %5 : f32
    } -> tensor<1x3xf32>
    return %4 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_torch.float32: "0x040000000000803F000080BF00000040",
      torch_tensor_3_4_torch.float32: "0x040000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041"
    }
  }
#-}
```


The important change is that the high-level Torch ops have been converted into
standard tensor and `linalg` operations:

- `!torch.vtensor` values become builtin `tensor<...>` values.
- `torch.aten.transpose.int` becomes `linalg.transpose`.
- `torch.aten.mm` becomes `linalg.matmul`.
- `torch.aten.add.Tensor` becomes a `linalg.generic` with `arith.addf`.

This tensor/`linalg` form is the input shape that the `sculptor-mlir` passes use
for layer recognition.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Canonicalize Layers</summary>


The next pass is `--sculptor-canonicalize-layers`. It recognizes layer-shaped
tensor/`linalg` patterns and rewrites them into explicit Sculptor dialect layer
operations.

The pass currently has canonicalizers for:

- Linear layers.
- 1D, 2D, grouped 2D, and 3D convolution layers.
- RNN cell, LSTM cell, and GRU cell layers.
- Full RNN, LSTM, and GRU layers.

For this example, the pass sees the transpose, matrix multiply, and bias add
pattern from the previous section and replaces it with `sculptor.nn.linear`:

```bash
python3 linear_example.py --mode mlir |
  torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline |
  sculptor-mlir-opt --sculptor-canonicalize-layers
```


The real output is:

```mlir
module {
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %cst = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %cst_0 = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %0 = sculptor.nn.linear %arg0, %cst_0, %cst {has_bias = true} : (tensor<1x4xf32>, tensor<3x4xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_torch.float32: "0x040000000000803F000080BF00000040",
      torch_tensor_3_4_torch.float32: "0x040000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041"
    }
  }
#-}
```


This is the first point where the program contains Sculptor dialect IR. The
linear layer is now a named semantic operation instead of a collection of
generic tensor operations.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Extract Layers</summary>


The next pass is `--sculptor-extract-layers`. It outlines recognized Sculptor layer
operations into their own functions and leaves `forward` as the top-level model
entry point.

For this example, the inline `sculptor.nn.linear` operation is moved into a new
function named `@linearwbias_0`, and `@forward` calls that function:

```bash
python3 linear_example.py --mode mlir |
  torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline |
  sculptor-mlir-opt \
    --sculptor-canonicalize-layers \
    --sculptor-extract-layers
```


The real output is:

```mlir
module {
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = call @linearwbias_0(%arg0) : (tensor<1x4xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
  func.func @linearwbias_0(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> attributes {layer_type = "linear_w_bias"} {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %cst_0 = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %0 = sculptor.nn.linear %arg0, %cst, %cst_0 {has_bias = true} : (tensor<1x4xf32>, tensor<3x4xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x040000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041",
      torch_tensor_3_torch.float32: "0x040000000000803F000080BF00000040"
    }
  }
#-}
```


This creates an explicit layer boundary so the compiler can prepare the program
for a task graph-based model. The main `forward` function now describes
model-level control flow, while the extracted layer function owns the Analog
layer operation that later passes can lower into task-sized work.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Convert Layers</summary>


The next pass is `--sculptor-convert-layers`. It lowers semantic Sculptor layer
operations into lower-level Analog work.

For this linear example, `sculptor.nn.linear` is converted into two pieces:

- `sculptor.mvm` for the matrix-vector multiply.
- `sculptor.task_region` for the remaining digital bias add.

```bash
python3 linear_example.py --mode mlir |
  torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline |
  sculptor-mlir-opt \
    --sculptor-canonicalize-layers \
    --sculptor-extract-layers \
    --sculptor-convert-layers
```


The real output is:

```mlir
module {
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = call @linearwbias_0(%arg0) : (tensor<1x4xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
  func.func @linearwbias_0(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> attributes {layer_type = "linear_w_bias"} {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %0 = sculptor.mvm %arg0, %cst : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<1x3xf32>
    %1 = sculptor.task_region kind = "digital.bias_add" name = "linear_bias_add"(%0) {
    ^bb0(%arg1: tensor<1x3xf32>):
      %cst_0 = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
      %expanded = tensor.expand_shape %cst_0 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
      %2 = tensor.empty() : tensor<1x3xf32>
      %3 = linalg.add ins(%arg1, %expanded : tensor<1x3xf32>, tensor<1x3xf32>) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
      sculptor.yield %3 : tensor<1x3xf32>
    } : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x040000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041",
      torch_tensor_3_torch.float32: "0x040000000000803F000080BF00000040"
    }
  }
#-}
```


This pass moves the program from named neural-network layers to backend-neutral
Analog compute. The matrix-vector multiply is now explicit as `sculptor.mvm`,
while the bias add remains digital work that can become its own task.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Expand MVM To Golem</summary>


The next pass is `--sculptor-expand-mvm-to-golem`. It takes backend-neutral
`sculptor.mvm` work and expands it into Golem-oriented Analog array operations.

For this example, the pass uses one `4x4` analog array tile:

```bash
python3 linear_example.py --mode mlir |
  torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline |
  sculptor-mlir-opt \
    --sculptor-canonicalize-layers \
    --sculptor-extract-layers \
    --sculptor-convert-layers \
    --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4"
```


The real output is:

```mlir
module {
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = call @linearwbias_0(%arg0) : (tensor<1x4xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
  func.func @linearwbias_0(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> attributes {layer_type = "linear_w_bias"} {
    %0 = sculptor.task_region kind = "sculptor.matrix_setup" name = "linearwbias_0_matrix_tile_0_0"() {
      %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32__tile_0_0> : tensor<4x4xf32>
      %5 = sculptor.array.set %cst : tensor<4x4xf32> -> !sculptor.logical.array
      sculptor.yield %5 : !sculptor.logical.array
    } {sculptor.source_resource = "torch_tensor_3_4_torch.float32", sculptor.tile = [0, 0], sculptor.tile_grid = [1, 1], sculptor.tile_physical_shape = [4, 4], sculptor.tile_valid_shape = [3, 4]} : () -> !sculptor.logical.array
    %1 = sculptor.task_region kind = "digital.vector_tile" name = "linearwbias_0_vector_tile_0"(%arg0) {
    ^bb0(%arg1: tensor<1x4xf32>):
      %extracted_slice = tensor.extract_slice %arg1[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> to tensor<1x4xf32>
      sculptor.yield %extracted_slice : tensor<1x4xf32>
    } {sculptor.vector_tile = 0 : i64, sculptor.vector_tile_grid = 1 : i64, sculptor.vector_tile_physical_cols = 4 : i64, sculptor.vector_tile_valid_cols = 4 : i64} : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %2 = sculptor.task_region kind = "sculptor.mvm" name = "linearwbias_0_mvm_0_0"(%1, %0) {
    ^bb0(%arg1: tensor<1x4xf32>, %arg2: !sculptor.logical.array):
      sculptor.array.load %arg1, %arg2 : tensor<1x4xf32>, !sculptor.logical.array
      %5 = sculptor.array.execute %arg2 : !sculptor.logical.array -> !sculptor.array.result
      %6 = sculptor.array.store %5 : !sculptor.array.result -> tensor<1x3xf32>
      sculptor.yield %6 : tensor<1x3xf32>
    } {sculptor.source_resource = "torch_tensor_3_4_torch.float32", sculptor.tile = [0, 0], sculptor.tile_grid = [1, 1], sculptor.tile_physical_shape = [4, 4], sculptor.tile_valid_shape = [3, 4], sculptor.vector_tile = 0 : i64, sculptor.vector_tile_grid = 1 : i64, sculptor.vector_tile_physical_cols = 4 : i64, sculptor.vector_tile_valid_cols = 4 : i64} : (tensor<1x4xf32>, !sculptor.logical.array) -> tensor<1x3xf32>
    %3 = sculptor.task_region kind = "digital.tile_recombine" name = "linearwbias_0_tile_recombine"(%2) {
    ^bb0(%arg1: tensor<1x3xf32>):
      sculptor.yield %arg1 : tensor<1x3xf32>
    } : (tensor<1x3xf32>) -> tensor<1x3xf32>
    %4 = sculptor.task_region kind = "digital.bias_add" name = "linear_bias_add"(%3) {
    ^bb0(%arg1: tensor<1x3xf32>):
      %cst = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
      %expanded = tensor.expand_shape %cst [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
      %5 = tensor.empty() : tensor<1x3xf32>
      %6 = linalg.add ins(%arg1, %expanded : tensor<1x3xf32>, tensor<1x3xf32>) outs(%5 : tensor<1x3xf32>) -> tensor<1x3xf32>
      sculptor.yield %6 : tensor<1x3xf32>
    } : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %4 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32__tile_0_0: "0x040000000000803F0000004000004040000080400000A0400000C0400000E040000000410000104100002041000030410000404100000000000000000000000000000000",
      torch_tensor_3_torch.float32: "0x040000000000803F000080BF00000040"
    }
  }
#-}
```


The single `sculptor.mvm` has become a sequence of task regions. These regions
are important because they mark the compiler's future task boundaries before a
task graph exists. Each region names a unit of work, declares the values it
depends on, and returns the values it produces. Later passes can materialize
these regions into task functions, assemble dependencies between them, and
schedule them onto the simulator's hardware model.

### `sculptor.matrix_setup`

This task programs the matrix tile into a logical analog array. In this example,
the original `3x4` weight is padded into one `4x4` tile because the pass was run
with `array-rows=4 array-cols=4`.

### `digital.vector_tile`

This task extracts the input vector tile that feeds the analog MVM. The example
only needs one vector tile because the input width is `4`.

### `sculptor.mvm`

This task performs the accelerator-facing matrix-vector multiply. It loads the
vector tile, executes the logical analog array, and stores the analog result.

### `digital.tile_recombine`

This task sums column-tile partials and concatenates row tiles when an MVM spans
multiple physical arrays. Physical matrix tiles remain `4x4`, but each array
store exposes only its valid logical rows. The single tile in this example
therefore produces `tensor<1x3xf32>` directly, and recombination is an identity.
Padded rows never become task-graph communication.

### `digital.bias_add`

This task applies the bias after the analog MVM result has been recombined. Bias
addition remains digital work in this lowering.

This is the point where the compiler has moved from a backend-neutral MVM to
explicit Golem-facing array setup, execution, and digital cleanup work.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Materialize Tasks</summary>


The next pass is `--sculptor-materialize-tasks`. It turns each inline
`sculptor.task_region` into a private task function and replaces the regions with
calls to those functions.

```bash
python3 linear_example.py --mode mlir |
  torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline |
  sculptor-mlir-opt \
    --sculptor-canonicalize-layers \
    --sculptor-extract-layers \
    --sculptor-convert-layers \
    --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" \
    --sculptor-materialize-tasks
```


The real output is:

```mlir
module {
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = call @task_linearwbias_0_matrix_tile_0_0_0() : () -> !sculptor.logical.array
    %1 = call @task_linearwbias_0_vector_tile_0_1(%arg0) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %2 = call @task_linearwbias_0_mvm_0_0_2(%1, %0) : (tensor<1x4xf32>, !sculptor.logical.array) -> tensor<1x4xf32>
    %3 = call @task_linearwbias_0_tile_recombine_3(%2) : (tensor<1x4xf32>) -> tensor<1x3xf32>
    %4 = call @task_linearwbias_0_linear_bias_add_4(%3) : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %4 : tensor<1x3xf32>
  }
  func.func private @task_linearwbias_0_matrix_tile_0_0_0() -> !sculptor.logical.array attributes {sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 0 : i64, sculptor.task_domain = "analog", sculptor.task_kind = "sculptor.matrix_setup", sculptor.task_name = "linearwbias_0_matrix_tile_0_0"} {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32__tile_0_0> : tensor<4x4xf32>
    %0 = sculptor.array.set %cst : tensor<4x4xf32> -> !sculptor.logical.array
    return %0 : !sculptor.logical.array
  }
  func.func private @task_linearwbias_0_vector_tile_0_1(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> attributes {sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 1 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.vector_tile", sculptor.task_name = "linearwbias_0_vector_tile_0"} {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> to tensor<1x4xf32>
    return %extracted_slice : tensor<1x4xf32>
  }
  func.func private @task_linearwbias_0_mvm_0_0_2(%arg0: tensor<1x4xf32>, %arg1: !sculptor.logical.array) -> tensor<1x4xf32> attributes {sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 2 : i64, sculptor.task_domain = "analog", sculptor.task_kind = "sculptor.mvm", sculptor.task_name = "linearwbias_0_mvm_0_0"} {
    sculptor.array.load %arg0, %arg1 : tensor<1x4xf32>, !sculptor.logical.array
    %0 = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %1 = sculptor.array.store %0 : !sculptor.array.result -> tensor<1x4xf32>
    return %1 : tensor<1x4xf32>
  }
  func.func private @task_linearwbias_0_tile_recombine_3(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> attributes {sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 3 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.tile_recombine", sculptor.task_name = "linearwbias_0_tile_recombine"} {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 3] [1, 1] : tensor<1x4xf32> to tensor<1x3xf32>
    return %extracted_slice : tensor<1x3xf32>
  }
  func.func private @task_linearwbias_0_linear_bias_add_4(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 4 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.bias_add", sculptor.task_name = "linear_bias_add"} {
    %cst = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %expanded = tensor.expand_shape %cst [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %0 = tensor.empty() : tensor<1x3xf32>
    %1 = linalg.add ins(%arg0, %expanded : tensor<1x3xf32>, tensor<1x3xf32>) outs(%0 : tensor<1x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32__tile_0_0: "0x040000000000803F0000004000004040000080400000A0400000C0400000E040000000410000104100002041000030410000404100000000000000000000000000000000",
      torch_tensor_3_torch.float32: "0x040000000000803F000080BF00000040"
    }
  }
#-}
```


After this pass, `forward` no longer contains inline task regions. It calls a
sequence of private task functions instead. Each task function carries metadata
such as `sculptor.task_domain`, `sculptor.task_kind`, `sculptor.task_name`, and
`sculptor.source_task_ordinal`.

This prepares the program for task graph assembly: the compiler now has named
callable units of work that can become task graph nodes.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Assemble Task Graph</summary>


The next pass is `--sculptor-assemble-task-graph`. It keeps the materialized task
functions and builds a symbolic task graph that records resources, task nodes,
and dependencies.

```bash
python3 linear_example.py --mode mlir |
  torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline |
  sculptor-mlir-opt \
    --sculptor-canonicalize-layers \
    --sculptor-extract-layers \
    --sculptor-convert-layers \
    --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" \
    --sculptor-materialize-tasks \
    --sculptor-assemble-task-graph
```


The real output is:

```mlir
module {
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = call @task_linearwbias_0_matrix_tile_0_0_0() : () -> !sculptor.logical.array
    %1 = call @task_linearwbias_0_vector_tile_0_1(%arg0) : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %2 = call @task_linearwbias_0_mvm_0_0_2(%1, %0) : (tensor<1x4xf32>, !sculptor.logical.array) -> tensor<1x4xf32>
    %3 = call @task_linearwbias_0_tile_recombine_3(%2) : (tensor<1x4xf32>) -> tensor<1x3xf32>
    %4 = call @task_linearwbias_0_linear_bias_add_4(%3) : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %4 : tensor<1x3xf32>
  }
  func.func private @task_linearwbias_0_matrix_tile_0_0_0() -> !sculptor.logical.array attributes {sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 0 : i64, sculptor.task_domain = "analog", sculptor.task_kind = "sculptor.matrix_setup", sculptor.task_name = "linearwbias_0_matrix_tile_0_0", llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32__tile_0_0> : tensor<4x4xf32>
    %0 = sculptor.array.set %cst : tensor<4x4xf32> -> !sculptor.logical.array
    return %0 : !sculptor.logical.array
  }
  func.func private @task_linearwbias_0_vector_tile_0_1(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> attributes {sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 1 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.vector_tile", sculptor.task_name = "linearwbias_0_vector_tile_0", llvm.emit_c_interface} {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> to tensor<1x4xf32>
    return %extracted_slice : tensor<1x4xf32>
  }
  func.func private @task_linearwbias_0_mvm_0_0_2(%arg0: tensor<1x4xf32>, %arg1: !sculptor.logical.array) -> tensor<1x4xf32> attributes {sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 2 : i64, sculptor.task_domain = "analog", sculptor.task_kind = "sculptor.mvm", sculptor.task_name = "linearwbias_0_mvm_0_0", llvm.emit_c_interface} {
    sculptor.array.load %arg0, %arg1 : tensor<1x4xf32>, !sculptor.logical.array
    %0 = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %1 = sculptor.array.store %0 : !sculptor.array.result -> tensor<1x4xf32>
    return %1 : tensor<1x4xf32>
  }
  func.func private @task_linearwbias_0_tile_recombine_3(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> attributes {sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 3 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.tile_recombine", sculptor.task_name = "linearwbias_0_tile_recombine", llvm.emit_c_interface} {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 3] [1, 1] : tensor<1x4xf32> to tensor<1x3xf32>
    return %extracted_slice : tensor<1x3xf32>
  }
  func.func private @task_linearwbias_0_linear_bias_add_4(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 4 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.bias_add", sculptor.task_name = "linear_bias_add", llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %expanded = tensor.expand_shape %cst [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %0 = tensor.empty() : tensor<1x3xf32>
    %1 = linalg.add ins(%arg0, %expanded : tensor<1x3xf32>, tensor<1x3xf32>) outs(%0 : tensor<1x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {sculptor.runtime.input_slots = [0], sculptor.runtime.output_slots = [1], sculptor.runtime.resource_count = 6 : i64, sculptor.runtime.temp_base_slot = 2 : i64, sculptor.runtime.temp_count = 4 : i64, sculptor.runtime.temp_offsets = [0, 0, 16, 0], sculptor.runtime.workspace_size = 32 : i64} {
    %0 = sculptor.task_graph.create : !sculptor.task_graph
    %1 = sculptor.task_graph.input %0 {sculptor.runtime.byte_size = 16 : i64, sculptor.runtime.slot = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %2 = sculptor.task_graph.output %0 {sculptor.runtime.byte_size = 12 : i64, sculptor.runtime.slot = 1 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>
    %3 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 0 : i64, sculptor.runtime.slot = 2 : i64, sculptor.runtime.temp_index = 0 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %4 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 16 : i64, sculptor.runtime.slot = 3 : i64, sculptor.runtime.temp_index = 1 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %5 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 16 : i64, sculptor.runtime.slot = 4 : i64, sculptor.runtime.temp_index = 2 : i64, sculptor.runtime.temp_offset = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %6 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 12 : i64, sculptor.runtime.slot = 5 : i64, sculptor.runtime.temp_index = 3 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>
    %7 = sculptor.task.create %0, @task_linearwbias_0_matrix_tile_0_0_0, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linearwbias_0_matrix_tile_0_0", source_layer = "linearwbias_0", source_task_ordinal = 0, inputs[], outputs[%3], deps[] {sculptor.runtime.input_slots = [], sculptor.runtime.output_slots = [2], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %8 = sculptor.task.create %0, @task_linearwbias_0_vector_tile_0_1, domain = "digital", task_kind = "digital.vector_tile", task_name = "linearwbias_0_vector_tile_0", source_layer = "linearwbias_0", source_task_ordinal = 1, inputs[%1], outputs[%4], deps[] {sculptor.runtime.input_slots = [0], sculptor.runtime.output_slots = [3], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>) -> !sculptor.task
    %9 = sculptor.task.create %0, @task_linearwbias_0_mvm_0_0_2, domain = "analog", task_kind = "sculptor.mvm", task_name = "linearwbias_0_mvm_0_0", source_layer = "linearwbias_0", source_task_ordinal = 2, inputs[%4, %3], outputs[%5], deps[%7, %8] {sculptor.runtime.input_slots = [3, 2], sculptor.runtime.output_slots = [4], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 2 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %10 = sculptor.task.create %0, @task_linearwbias_0_tile_recombine_3, domain = "digital", task_kind = "digital.tile_recombine", task_name = "linearwbias_0_tile_recombine", source_layer = "linearwbias_0", source_task_ordinal = 3, inputs[%5], outputs[%6], deps[%9] {sculptor.runtime.input_slots = [4], sculptor.runtime.output_slots = [5], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 3 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task) -> !sculptor.task
    %11 = sculptor.task.create %0, @task_linearwbias_0_linear_bias_add_4, domain = "digital", task_kind = "digital.bias_add", task_name = "linear_bias_add", source_layer = "linearwbias_0", source_task_ordinal = 4, inputs[%6], outputs[%2], deps[%10] {sculptor.runtime.input_slots = [5], sculptor.runtime.output_slots = [1], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 4 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task) -> !sculptor.task
    return %0 : !sculptor.task_graph
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32__tile_0_0: "0x040000000000803F0000004000004040000080400000A0400000C0400000E040000000410000104100002041000030410000404100000000000000000000000000000000",
      torch_tensor_3_torch.float32: "0x040000000000803F000080BF00000040"
    }
  }
#-}
```


The bottom of the IR is centered on `@generate_task_graph`. This function is
where the compiler starts describing the program as graph resources and graph
nodes instead of only as normal function calls.

### `sculptor.task_graph.create`

This creates the empty task graph object. In the example, `%0` is the graph
handle, and every later task graph operation attaches resources or tasks to that
same graph.

### `sculptor.task_graph.input`

This declares a value that the runtime must provide before the graph can run.
Here, `%1` is input slot `0`, has a byte size of `16`, and represents the
`tensor<1x4xf32>` input tensor.

### `sculptor.task_graph.output`

This declares a value that the runtime should read after the graph finishes.
Here, `%2` is output slot `1`, has a byte size of `12`, and represents the
`tensor<1x3xf32>` final result.

### `sculptor.task_graph.intermediate`

This declares storage for values produced between tasks. In this example, the
intermediate resources hold the logical array, the vector tile, the MVM result,
and the recombined tensor before bias add. These resources make the graph's
intermediate data movement explicit.

### `sculptor.task.create`

This creates a task node in the graph from one of the materialized task
functions. Each node records the function it calls, the task kind, the task
domain, its input resources, its output resources, and the tasks it depends on.

The dependency list is the graph ordering. Matrix setup and vector tiling have
no dependencies, so they can run first. The MVM task depends on both of them.
Tile recombine depends on the MVM task, and bias add depends on tile recombine.

This is still symbolic graph IR. The island pass groups the graph before
physical scheduling.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Build Islands And Schedule Task Graph</summary>


The next pass is `--sculptor-build-task-graph-islands`. It assigns every task
to a stable logical placement island without choosing a core. The timing stage
then runs before `--sculptor-schedule-task-graph`, which consumes those islands
and attaches scheduling information for a target hardware budget. The timing
pass validates the combined control/data execution DAG and records intrinsic
task latency, critical-path, and island-work estimates before placement.

A schedule is the compiler's plan for where graph work should run. This is the
point in the pipeline where placement actually happens. The pass chooses which
core owns each task, which physical analog array backs each logical analog
array, and what task order the runtime should use. It does not change the math
of the model. It adds placement, ordering, digital-operation counts, and
transfer-cost summaries that later runtime export passes can consume.

For a single-core target, this mostly records that all tasks run on the same
core and the same physical analog array. For a multi-core target, this is where
the graph starts to become a hardware placement plan: layers, matrix tiles,
analog MVM work, and digital cleanup tasks can be assigned to different cores
and arrays. The schedule also records the data movement that placement creates,
which lets later tools estimate or model communication between cores.

For this example, the schedule uses one core and one analog array:

```bash
python3 linear_example.py --mode mlir |
  torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline |
  sculptor-mlir-opt \
    --sculptor-canonicalize-layers \
    --sculptor-extract-layers \
    --sculptor-convert-layers \
    --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" \
    --sculptor-materialize-tasks \
    --sculptor-assemble-task-graph \
    --sculptor-build-task-graph-islands \
    --sculptor-analyze-task-graph-timing \
    --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 schedule=random"
```


The scheduled output no longer keeps the materialized `@forward` function once
the task graph is present. The live IR is the scheduled task graph plus the
private task functions referenced by `sculptor.task.create`:

```mlir
module attributes {sculptor.schedule.analog_arrays = [0], sculptor.schedule.arrays_per_core = 1 : i64, sculptor.schedule.mesh_cols = 1 : i64, sculptor.schedule.mesh_rows = 1 : i64, sculptor.schedule.num_analog_arrays = 1 : i64, sculptor.schedule.num_cores = 1 : i64, sculptor.schedule.topology = "mesh"} {
  func.func private @task_linearwbias_0_matrix_tile_0_0_0() -> !sculptor.logical.array attributes {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 0 : i64, sculptor.task_domain = "analog", sculptor.task_kind = "sculptor.matrix_setup", sculptor.task_name = "linearwbias_0_matrix_tile_0_0", llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32__tile_0_0> : tensor<4x4xf32>
    %0 = sculptor.array.set %cst : tensor<4x4xf32> -> !sculptor.logical.array
    return %0 : !sculptor.logical.array
  }
  func.func private @task_linearwbias_0_vector_tile_0_1(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> attributes {sculptor.runtime.core_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 1 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.vector_tile", sculptor.task_name = "linearwbias_0_vector_tile_0", llvm.emit_c_interface} {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> to tensor<1x4xf32>
    return %extracted_slice : tensor<1x4xf32>
  }
  func.func private @task_linearwbias_0_mvm_0_0_2(%arg0: tensor<1x4xf32>, %arg1: !sculptor.logical.array) -> tensor<1x4xf32> attributes {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 2 : i64, sculptor.task_domain = "analog", sculptor.task_kind = "sculptor.mvm", sculptor.task_name = "linearwbias_0_mvm_0_0", llvm.emit_c_interface} {
    sculptor.array.load %arg0, %arg1 : tensor<1x4xf32>, !sculptor.logical.array
    %0 = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %1 = sculptor.array.store %0 : !sculptor.array.result -> tensor<1x4xf32>
    return %1 : tensor<1x4xf32>
  }
  func.func private @task_linearwbias_0_tile_recombine_3(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> attributes {sculptor.runtime.core_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 3 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.tile_recombine", sculptor.task_name = "linearwbias_0_tile_recombine", llvm.emit_c_interface} {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 3] [1, 1] : tensor<1x4xf32> to tensor<1x3xf32>
    return %extracted_slice : tensor<1x3xf32>
  }
  func.func private @task_linearwbias_0_linear_bias_add_4(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {sculptor.runtime.core_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 4 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.bias_add", sculptor.task_name = "linear_bias_add", llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %expanded = tensor.expand_shape %cst [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %0 = tensor.empty() : tensor<1x3xf32>
    %1 = linalg.add ins(%arg0, %expanded : tensor<1x3xf32>, tensor<1x3xf32>) outs(%0 : tensor<1x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {sculptor.runtime.input_slots = [0], sculptor.runtime.output_slots = [1], sculptor.runtime.resource_count = 6 : i64, sculptor.runtime.temp_base_slot = 2 : i64, sculptor.runtime.temp_count = 4 : i64, sculptor.runtime.temp_offsets = [0, 0, 16, 0], sculptor.runtime.workspace_size = 32 : i64, sculptor.schedule.analog_arrays = [0], sculptor.schedule.arrays_per_core = 1 : i64, sculptor.schedule.core_transfer_bytes = [0], sculptor.schedule.core_transfer_cost = [0], sculptor.schedule.dependency_count = 4 : i64, sculptor.schedule.inter_core_transfer_bytes = 0 : i64, sculptor.schedule.logical_array_to_analog_array = [0], sculptor.schedule.mesh_cols = 1 : i64, sculptor.schedule.mesh_rows = 1 : i64, sculptor.schedule.num_analog_arrays = 1 : i64, sculptor.schedule.num_cores = 1 : i64, sculptor.schedule.num_logical_arrays = 1 : i64, sculptor.schedule.task_count = 5 : i64, sculptor.schedule.topology = "mesh", sculptor.schedule.total_digital_ops = 3 : i64, sculptor.schedule.total_transfer_cost = 0 : i64} {
    %0 = sculptor.task_graph.create : !sculptor.task_graph
    %1 = sculptor.task_graph.input %0 {sculptor.runtime.byte_size = 16 : i64, sculptor.runtime.slot = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %2 = sculptor.task_graph.output %0 {sculptor.runtime.byte_size = 12 : i64, sculptor.runtime.slot = 1 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>
    %3 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 0 : i64, sculptor.runtime.slot = 2 : i64, sculptor.runtime.temp_index = 0 : i64, sculptor.runtime.temp_offset = 0 : i64, sculptor.schedule.logical_array_index = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %4 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 16 : i64, sculptor.runtime.slot = 3 : i64, sculptor.runtime.temp_index = 1 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %5 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 16 : i64, sculptor.runtime.slot = 4 : i64, sculptor.runtime.temp_index = 2 : i64, sculptor.runtime.temp_offset = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %6 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 12 : i64, sculptor.runtime.slot = 5 : i64, sculptor.runtime.temp_index = 3 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>
    %7 = sculptor.task.create %0, @task_linearwbias_0_matrix_tile_0_0_0, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linearwbias_0_matrix_tile_0_0", source_layer = "linearwbias_0", source_task_ordinal = 0, inputs[], outputs[%3], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.input_slots = [], sculptor.runtime.output_slots = [2], sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %8 = sculptor.task.create %0, @task_linearwbias_0_vector_tile_0_1, domain = "digital", task_kind = "digital.vector_tile", task_name = "linearwbias_0_vector_tile_0", source_layer = "linearwbias_0", source_task_ordinal = 1, inputs[%1], outputs[%4], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.input_slots = [0], sculptor.runtime.output_slots = [3], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>) -> !sculptor.task
    %9 = sculptor.task.create %0, @task_linearwbias_0_mvm_0_0_2, domain = "analog", task_kind = "sculptor.mvm", task_name = "linearwbias_0_mvm_0_0", source_layer = "linearwbias_0", source_task_ordinal = 2, inputs[%4, %3], outputs[%5], deps[%7, %8] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.input_slots = [3, 2], sculptor.runtime.output_slots = [4], sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 2 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %10 = sculptor.task.create %0, @task_linearwbias_0_tile_recombine_3, domain = "digital", task_kind = "digital.tile_recombine", task_name = "linearwbias_0_tile_recombine", source_layer = "linearwbias_0", source_task_ordinal = 3, inputs[%5], outputs[%6], deps[%9] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.input_slots = [4], sculptor.runtime.output_slots = [5], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 3 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task) -> !sculptor.task
    %11 = sculptor.task.create %0, @task_linearwbias_0_linear_bias_add_4, domain = "digital", task_kind = "digital.bias_add", task_name = "linear_bias_add", source_layer = "linearwbias_0", source_task_ordinal = 4, inputs[%6], outputs[%2], deps[%10] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 3 : i64, sculptor.runtime.input_slots = [5], sculptor.runtime.output_slots = [1], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 4 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task) -> !sculptor.task
    return %0 : !sculptor.task_graph
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32__tile_0_0: "0x040000000000803F0000004000004040000080400000A0400000C0400000E040000000410000104100002041000030410000404100000000000000000000000000000000",
      torch_tensor_3_torch.float32: "0x040000000000803F000080BF00000040"
    }
  }
#-}
```


The schedule appears as metadata on several parts of the IR:

- The module records the hardware budget: one core, one array per core, one
  physical analog array, and a `1x1` mesh topology.
- `@generate_task_graph` records graph-level schedule totals, including
  `sculptor.schedule.task_count = 5`, `sculptor.schedule.dependency_count = 4`,
  and `sculptor.schedule.logical_array_to_analog_array = [0]`.
- The logical array intermediate gets `sculptor.schedule.logical_array_index = 0`,
  which gives the scheduler a stable identity for that logical array resource.
- Each task records `sculptor.runtime.core_id = 0`, because this example has only
  one core.
- The analog tasks also record `sculptor.runtime.physical_array_id = 0`, because
  the logical array was placed onto the only physical array.
- Each `sculptor.task.create` records an `sculptor.runtime.task_index`, which is the
  scheduled task order used by later runtime lowering.

In this small example, every task lands on core `0`, so the schedule is simple.
The important change is that the graph now carries enough placement and ordering
metadata for the runtime export pipeline.

For a multi-core schedule, this same section of IR is where the output begins to
differ. Logical arrays can map onto different physical arrays, tasks can point
at different cores, and transfer summaries can report communication between
those cores. In other words, the task graph is no longer only a dependency graph;
it is now a placed execution plan for the target runtime model.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Lower Golem To LLVM Shims</summary>


The next pass is `--sculptor-lower-golem-to-llvm-shims`. It lowers the scheduled
Golem-facing analog operations into stable runtime shim calls.

At this point, the compiler has already chosen the task placement. Shim lowering
uses that placement to replace logical array operations with calls that name the
local analog array selected by the schedule. The task graph is still present,
but the task bodies are now closer to code that can be lowered into LLVM and
linked with the runtime.

The runtime build runs shim lowering and then cleans up the result:

```bash
python3 linear_example.py --mode mlir |
  torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline |
  sculptor-mlir-opt \
    --sculptor-canonicalize-layers \
    --sculptor-extract-layers \
    --sculptor-convert-layers \
    --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" \
    --sculptor-materialize-tasks \
    --sculptor-assemble-task-graph \
    --sculptor-build-task-graph-islands \
    --sculptor-analyze-task-graph-timing \
    --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 schedule=random" \
    --sculptor-lower-golem-to-llvm-shims \
    --canonicalize \
    --cse
```


The shim-lowered output keeps the same scheduled graph shape. `@forward` is
still absent; the task graph is now backed by task functions whose Golem array
operations have been rewritten to runtime shim calls:

```mlir
module attributes {sculptor.schedule.analog_arrays = [0], sculptor.schedule.arrays_per_core = 1 : i64, sculptor.schedule.mesh_cols = 1 : i64, sculptor.schedule.mesh_rows = 1 : i64, sculptor.schedule.num_analog_arrays = 1 : i64, sculptor.schedule.num_cores = 1 : i64, sculptor.schedule.topology = "mesh"} {
  func.func private @golem_analog_mvm_store(memref<1x1x4xf32>, i32)
  func.func private @golem_analog_mvm_compute(i32)
  func.func private @golem_analog_mvm_load(memref<1x4xf32>, i32)
  func.func private @golem_analog_mvm_set(memref<4x4xf32>, i32)
  func.func private @task_linearwbias_0_matrix_tile_0_0_0() attributes {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 0 : i64, sculptor.task_domain = "analog", sculptor.task_kind = "sculptor.matrix_setup", sculptor.task_name = "linearwbias_0_matrix_tile_0_0", llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32__tile_0_0> : tensor<4x4xf32>
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_buffer %cst : tensor<4x4xf32> to memref<4x4xf32>
    call @golem_analog_mvm_set(%0, %c0_i32) : (memref<4x4xf32>, i32) -> ()
    return
  }
  func.func private @task_linearwbias_0_vector_tile_0_1(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> attributes {sculptor.runtime.core_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 1 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.vector_tile", sculptor.task_name = "linearwbias_0_vector_tile_0", llvm.emit_c_interface} {
    return %arg0 : tensor<1x4xf32>
  }
  func.func private @task_linearwbias_0_mvm_0_0_2(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> attributes {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 2 : i64, sculptor.task_domain = "analog", sculptor.task_kind = "sculptor.mvm", sculptor.task_name = "linearwbias_0_mvm_0_0", llvm.emit_c_interface} {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_buffer %arg0 : tensor<1x4xf32> to memref<1x4xf32>
    call @golem_analog_mvm_load(%0, %c0_i32) : (memref<1x4xf32>, i32) -> ()
    call @golem_analog_mvm_compute(%c0_i32) : (i32) -> ()
    %alloc = memref.alloc() : memref<1x4xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x4xf32>
    call @golem_analog_mvm_store(%alloc_0, %c0_i32) : (memref<1x1x4xf32>, i32) -> ()
    scf.for %arg1 = %c0 to %c4 step %c1 {
      %2 = memref.load %alloc_0[%c0, %c0, %arg1] : memref<1x1x4xf32>
      memref.store %2, %alloc[%c0, %arg1] : memref<1x4xf32>
    }
    memref.dealloc %alloc_0 : memref<1x1x4xf32>
    %1 = bufferization.to_tensor %alloc restrict writable : memref<1x4xf32> to tensor<1x4xf32>
    return %1 : tensor<1x4xf32>
  }
  func.func private @task_linearwbias_0_tile_recombine_3(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> attributes {sculptor.runtime.core_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 3 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.tile_recombine", sculptor.task_name = "linearwbias_0_tile_recombine", llvm.emit_c_interface} {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 3] [1, 1] : tensor<1x4xf32> to tensor<1x3xf32>
    return %extracted_slice : tensor<1x3xf32>
  }
  func.func private @task_linearwbias_0_linear_bias_add_4(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {sculptor.runtime.core_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 4 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.bias_add", sculptor.task_name = "linear_bias_add", llvm.emit_c_interface} {
    %cst = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %expanded = tensor.expand_shape %cst [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %0 = tensor.empty() : tensor<1x3xf32>
    %1 = linalg.add ins(%arg0, %expanded : tensor<1x3xf32>, tensor<1x3xf32>) outs(%0 : tensor<1x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {sculptor.runtime.input_slots = [0], sculptor.runtime.output_slots = [1], sculptor.runtime.resource_count = 6 : i64, sculptor.runtime.temp_base_slot = 2 : i64, sculptor.runtime.temp_count = 4 : i64, sculptor.runtime.temp_offsets = [0, 0, 16, 0], sculptor.runtime.workspace_size = 32 : i64, sculptor.schedule.analog_arrays = [0], sculptor.schedule.arrays_per_core = 1 : i64, sculptor.schedule.core_transfer_bytes = [0], sculptor.schedule.core_transfer_cost = [0], sculptor.schedule.dependency_count = 4 : i64, sculptor.schedule.inter_core_transfer_bytes = 0 : i64, sculptor.schedule.logical_array_to_analog_array = [0], sculptor.schedule.mesh_cols = 1 : i64, sculptor.schedule.mesh_rows = 1 : i64, sculptor.schedule.num_analog_arrays = 1 : i64, sculptor.schedule.num_cores = 1 : i64, sculptor.schedule.num_logical_arrays = 1 : i64, sculptor.schedule.task_count = 5 : i64, sculptor.schedule.topology = "mesh", sculptor.schedule.total_digital_ops = 3 : i64, sculptor.schedule.total_transfer_cost = 0 : i64} {
    %0 = sculptor.task_graph.create : !sculptor.task_graph
    %1 = sculptor.task_graph.input %0 {sculptor.runtime.byte_size = 16 : i64, sculptor.runtime.slot = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %2 = sculptor.task_graph.output %0 {sculptor.runtime.byte_size = 12 : i64, sculptor.runtime.slot = 1 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>
    %3 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 0 : i64, sculptor.runtime.slot = 2 : i64, sculptor.runtime.temp_index = 0 : i64, sculptor.runtime.temp_offset = 0 : i64, sculptor.schedule.logical_array_index = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %4 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 16 : i64, sculptor.runtime.slot = 3 : i64, sculptor.runtime.temp_index = 1 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %5 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 16 : i64, sculptor.runtime.slot = 4 : i64, sculptor.runtime.temp_index = 2 : i64, sculptor.runtime.temp_offset = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %6 = sculptor.task_graph.intermediate %0 {sculptor.runtime.byte_size = 12 : i64, sculptor.runtime.slot = 5 : i64, sculptor.runtime.temp_index = 3 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>
    %7 = sculptor.task.create %0, @task_linearwbias_0_matrix_tile_0_0_0, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linearwbias_0_matrix_tile_0_0", source_layer = "linearwbias_0", source_task_ordinal = 0, inputs[], outputs[%3], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.input_slots = [], sculptor.runtime.output_slots = [2], sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %8 = sculptor.task.create %0, @task_linearwbias_0_vector_tile_0_1, domain = "digital", task_kind = "digital.vector_tile", task_name = "linearwbias_0_vector_tile_0", source_layer = "linearwbias_0", source_task_ordinal = 1, inputs[%1], outputs[%4], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.input_slots = [0], sculptor.runtime.output_slots = [3], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>) -> !sculptor.task
    %9 = sculptor.task.create %0, @task_linearwbias_0_mvm_0_0_2, domain = "analog", task_kind = "sculptor.mvm", task_name = "linearwbias_0_mvm_0_0", source_layer = "linearwbias_0", source_task_ordinal = 2, inputs[%4, %3], outputs[%5], deps[%7, %8] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.input_slots = [3, 2], sculptor.runtime.output_slots = [4], sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 2 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %10 = sculptor.task.create %0, @task_linearwbias_0_tile_recombine_3, domain = "digital", task_kind = "digital.tile_recombine", task_name = "linearwbias_0_tile_recombine", source_layer = "linearwbias_0", source_task_ordinal = 3, inputs[%5], outputs[%6], deps[%9] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.input_slots = [4], sculptor.runtime.output_slots = [5], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 3 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task) -> !sculptor.task
    %11 = sculptor.task.create %0, @task_linearwbias_0_linear_bias_add_4, domain = "digital", task_kind = "digital.bias_add", task_name = "linear_bias_add", source_layer = "linearwbias_0", source_task_ordinal = 4, inputs[%6], outputs[%2], deps[%10] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 3 : i64, sculptor.runtime.input_slots = [5], sculptor.runtime.output_slots = [1], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 4 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task) -> !sculptor.task
    return %0 : !sculptor.task_graph
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32__tile_0_0: "0x040000000000803F0000004000004040000080400000A0400000C0400000E040000000410000104100002041000030410000404100000000000000000000000000000000",
      torch_tensor_3_torch.float32: "0x040000000000803F000080BF00000040"
    }
  }
#-}
```


The main change is that the `sculptor.array.*` operations are gone from the task
bodies. They have been replaced by four shim calls:

- `golem_analog_mvm_set` programs the scheduled array with the matrix tile.
- `golem_analog_mvm_load` loads the input vector into the scheduled array.
- `golem_analog_mvm_compute` runs the array operation.
- `golem_analog_mvm_store` stores the array result into a memref buffer.

The scheduled local array appears as the `i32` value passed to those shim calls.
For this example, that value is `0` because the schedule placed the only logical
array on the only physical array. The task graph remains in the module so later
passes can still emit runtime graph metadata.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Standard MLIR Lowering</summary>


After shim lowering, the runtime build runs a standard MLIR lowering sequence.
This stage is not specific to `sculptor-mlir`. Its job is to lower tensors,
bufferization, linalg, control flow, arithmetic, memrefs, and functions into
LLVM-compatible IR.

The task graph metadata stays in the module during this stage. The task bodies
and callable entry points move toward LLVM dialect, while the
`sculptor.task_graph.*` and `sculptor.task.create` operations remain available for
the runtime graph emission pass.

```bash
python3 linear_example.py --mode mlir |
  torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline |
  sculptor-mlir-opt \
    --sculptor-canonicalize-layers \
    --sculptor-extract-layers \
    --sculptor-convert-layers \
    --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" \
    --sculptor-materialize-tasks \
    --sculptor-assemble-task-graph \
    --sculptor-build-task-graph-islands \
    --sculptor-analyze-task-graph-timing \
    --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 schedule=random" \
    --sculptor-lower-golem-to-llvm-shims \
    --canonicalize \
    --cse \
    --empty-tensor-to-alloc-tensor \
    --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
    --convert-bufferization-to-memref \
    --convert-linalg-to-loops \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-math-to-llvm \
    --expand-strided-metadata \
    --lower-affine \
    --convert-arith-to-llvm \
    --convert-index-to-llvm \
    --convert-cf-to-llvm \
    --finalize-memref-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts
```


The full output is much larger than the previous stage because memref
descriptors, allocation logic, loops, constants, and C interface wrappers are
expanded into LLVM dialect. The important shape is visible in these real output
snippets.

The module now contains LLVM dialect declarations for allocation, constants, and
the Golem shim ABI:

```mlir
llvm.func @free(!llvm.ptr)
llvm.func @malloc(i64) -> !llvm.ptr
llvm.mlir.global private constant @__constant_3xf32(dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<3 x f32>
llvm.mlir.global private constant @__constant_4x4xf32(dense_resource<torch_tensor_3_4_torch.float32__tile_0_0> : tensor<4x4xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<4 x array<4 x f32>>
llvm.func @golem_analog_mvm_store(!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i32) attributes {sym_visibility = "private"}
llvm.func @golem_analog_mvm_compute(i32) attributes {sym_visibility = "private"}
llvm.func @golem_analog_mvm_load(!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i32) attributes {sym_visibility = "private"}
llvm.func @golem_analog_mvm_set(!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i32) attributes {sym_visibility = "private"}
```


The materialized `@forward` entry point has already been removed by this stage.
Task functions get LLVM-compatible forms and C interface wrappers:

```mlir
llvm.func @task_linearwbias_0_vector_tile_0_1(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {sculptor.runtime.core_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 1 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.vector_tile", sculptor.task_name = "linearwbias_0_vector_tile_0", llvm.emit_c_interface, sym_visibility = "private"}
llvm.func @_mlir_ciface_task_linearwbias_0_vector_tile_0_1(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {sculptor.runtime.core_id = 0 : i64, sculptor.source_layer = "linearwbias_0", sculptor.source_task_ordinal = 1 : i64, sculptor.task_domain = "digital", sculptor.task_kind = "digital.vector_tile", sculptor.task_name = "linearwbias_0_vector_tile_0", llvm.emit_c_interface, sym_visibility = "private"}
```


The task graph itself is still present. Its resources and task nodes have not
yet been emitted as runtime graph construction code:

```mlir
%0 = sculptor.task_graph.create : !sculptor.task_graph
%1 = sculptor.task_graph.input %0 {sculptor.runtime.byte_size = 16 : i64, sculptor.runtime.slot = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
%2 = sculptor.task_graph.output %0 {sculptor.runtime.byte_size = 12 : i64, sculptor.runtime.slot = 1 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>
%7 = sculptor.task.create %0, @task_linearwbias_0_matrix_tile_0_0_0, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linearwbias_0_matrix_tile_0_0", source_layer = "linearwbias_0", source_task_ordinal = 0, inputs[], outputs[%3], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.input_slots = [], sculptor.runtime.output_slots = [2], sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
```


After this stage, the compute side is close to LLVM, but the task graph is still
declarative IR. The next `sculptor-mlir` specific pass consumes that graph and
emits the runtime graph construction logic.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Emit Runtime Graph</summary>


The next pass is `--sculptor-emit-runtime-graph`. It consumes the symbolic task
graph and emits LLVM functions that build and run the runtime graph.

This is where `@generate_task_graph` stops being declarative IR. The pass reads
its resources, callables, task records, bindings, dependencies, and workspace
size, then emits calls into the runtime graph API. After this pass,
`@generate_task_graph` is removed from the module.

```bash
python3 linear_example.py --mode mlir |
  torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline |
  sculptor-mlir-opt \
    --sculptor-canonicalize-layers \
    --sculptor-extract-layers \
    --sculptor-convert-layers \
    --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" \
    --sculptor-materialize-tasks \
    --sculptor-assemble-task-graph \
    --sculptor-build-task-graph-islands \
    --sculptor-analyze-task-graph-timing \
    --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 schedule=random" \
    --sculptor-lower-golem-to-llvm-shims \
    --canonicalize \
    --cse \
    --empty-tensor-to-alloc-tensor \
    --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
    --convert-bufferization-to-memref \
    --convert-linalg-to-loops \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-math-to-llvm \
    --expand-strided-metadata \
    --lower-affine \
    --convert-arith-to-llvm \
    --convert-index-to-llvm \
    --convert-cf-to-llvm \
    --finalize-memref-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
    --sculptor-emit-runtime-graph
```


The emitted module gains declarations for the runtime graph API:

```mlir
llvm.func @sculptor_runtime_free_result_buffer(!llvm.ptr, !llvm.ptr)
llvm.func @sculptor_runtime_persistent_handle_create(i64, !llvm.ptr) -> !llvm.ptr
llvm.func @sculptor_runtime_copy_to_buffer(!llvm.ptr, !llvm.ptr, i64)
llvm.func @sculptor_runtime_task_set_arg_handle(!llvm.ptr, i32, !llvm.ptr) -> i32
llvm.func @sculptor_runtime_task_arg_buffer(!llvm.ptr, i32) -> !llvm.struct<(ptr, i64)>
llvm.func @sculptor_runtime_destroy(!llvm.ptr)
llvm.func @sculptor_runtime_execute(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @sculptor_runtime_init(!llvm.ptr) -> !llvm.ptr
llvm.func @sculptor_runtime_graph_set_dep(!llvm.ptr, i32, i32)
llvm.func @sculptor_runtime_graph_set_binding(!llvm.ptr, i32, i32, i16, i32, i32, i32, i32)
llvm.func @sculptor_runtime_graph_set_task(!llvm.ptr, i32, i32, i32, i16, i32, i16, i32, i32, i32)
llvm.func @sculptor_runtime_graph_set_callable(!llvm.ptr, i32, i32, !llvm.ptr, i32)
llvm.func @sculptor_runtime_graph_set_resource(!llvm.ptr, i32, i32, i32, i32, i64, i64)
llvm.func @sculptor_runtime_graph_create(i32, i32, i32, i32, i32, i32, i32, i64) -> !llvm.ptr
```


It also emits one runtime entry shim per task:

```mlir
llvm.func @__analog_rt_entry_task_linearwbias_0_matrix_tile_0_0_0(%arg0: !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
llvm.func @__analog_rt_entry_task_linearwbias_0_vector_tile_0_1(%arg0: !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
llvm.func @__analog_rt_entry_task_linearwbias_0_mvm_0_0_2(%arg0: !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
llvm.func @__analog_rt_entry_task_linearwbias_0_tile_recombine_3(%arg0: !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
llvm.func @__analog_rt_entry_task_linearwbias_0_linear_bias_add_4(%arg0: !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
```


For this example, the generated graph image has `6` resources, `5` callables,
`5` tasks, `8` bindings, `4` dependencies, and a `32` byte workspace:

```mlir
llvm.func @__analog_rt_build_graph_image() -> !llvm.ptr attributes {sym_visibility = "private"} {
  %0 = llvm.mlir.constant(6 : i32) : i32
  %1 = llvm.mlir.constant(5 : i32) : i32
  %2 = llvm.mlir.constant(5 : i32) : i32
  %3 = llvm.mlir.constant(8 : i32) : i32
  %4 = llvm.mlir.constant(4 : i32) : i32
  %5 = llvm.mlir.constant(0 : i32) : i32
  %6 = llvm.mlir.constant(0 : i32) : i32
  %7 = llvm.mlir.constant(32 : i64) : i64
  %8 = llvm.call @sculptor_runtime_graph_create(%0, %1, %2, %3, %4, %5, %6, %7) : (i32, i32, i32, i32, i32, i32, i32, i64) -> !llvm.ptr
```


The graph image builder then fills in each graph table. These are representative
real calls from the emitted output:

```mlir
llvm.call @sculptor_runtime_graph_set_resource(%8, %9, %10, %11, %12, %13, %14) : (!llvm.ptr, i32, i32, i32, i32, i64, i64) -> ()
llvm.call @sculptor_runtime_graph_set_callable(%8, %46, %47, %45, %48) : (!llvm.ptr, i32, i32, !llvm.ptr, i32) -> ()
llvm.call @sculptor_runtime_graph_set_task(%8, %65, %66, %67, %68, %69, %70, %71, %72, %73) : (!llvm.ptr, i32, i32, i32, i16, i32, i16, i32, i32, i32) -> ()
llvm.call @sculptor_runtime_graph_set_binding(%8, %110, %111, %112, %113, %114, %115, %116) : (!llvm.ptr, i32, i32, i16, i32, i32, i32, i32) -> ()
llvm.call @sculptor_runtime_graph_set_dep(%8, %166, %167) : (!llvm.ptr, i32, i32) -> ()
llvm.return %8 : !llvm.ptr
```


Finally, the pass emits public wrappers that the host program can call:

```mlir
llvm.func @runtime_init() -> !llvm.ptr {
  %0 = llvm.call @__analog_rt_build_graph_image() : () -> !llvm.ptr
  %1 = llvm.call @sculptor_runtime_init(%0) : (!llvm.ptr) -> !llvm.ptr
  llvm.return %1 : !llvm.ptr
}
llvm.func @runtime_execute(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
  %0 = llvm.call @sculptor_runtime_execute(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.return %0 : i32
}
llvm.func @runtime_destroy(%arg0: !llvm.ptr) {
  llvm.call @sculptor_runtime_destroy(%arg0) : (!llvm.ptr) -> ()
  llvm.return
}
```


After this pass, the runtime has an LLVM-callable graph image builder, task entry
points, and public lifecycle functions. The remaining build step is translating
the LLVM dialect module to LLVM IR and linking it with the runtime library.

</details>

<details class="doc-section" open markdown="1">
<summary markdown="block">## Runtime Connection</summary>


After `--sculptor-emit-runtime-graph`, the generated module and the runtime library
meet at the public lifecycle functions:

```c++
extern "C" {
void *runtime_init();
std::int32_t runtime_execute(void *runtime, const void *const *inputs,
                             void *const *outputs);
void runtime_destroy(void *runtime);
}
```


Those symbols are emitted by the compiled model. The runtime library provides the
lower-level `sculptor_runtime_*` functions that those generated wrappers call.

The build connects them in three steps:

```make
$(GENERATED_LL): $(GENERATED_MLIR)
	$(MLIR_TRANSLATE) --mlir-to-llvmir $< > $@

$(GENERATED_OBJ): $(GENERATED_LL)
	$(CXX) -Wno-override-module -c -x ir $< -o $@

$(RUNNER): $(MAIN_SRC) $(GENERATED_OBJ) $(RUNTIME_LIB)
	$(CXX) -std=c++20 \
		-I$(RUNTIME_API_INCLUDE_DIR) \
		$< $(GENERATED_OBJ) $(RUNTIME_LIB) $(PYTHON_LDFLAGS) \
		-o $@
```


The generated LLVM IR becomes an object file. That object file is linked with the
test `main.cpp` and `libMLIRSculptorRuntime.a`. The test runner only needs the
three lifecycle functions:

```c++
std::array<float, 4> inputStorage = {
    1.0f, 2.0f, 3.0f, 4.0f,
};
std::array<float, 3> outputStorage = {
    0.0f, 0.0f, 0.0f,
};

const void *inputs[] = {inputStorage.data()};
void *outputs[] = {outputStorage.data()};

void *runtime = runtime_init();
const std::int32_t rc = runtime_execute(runtime, inputs, outputs);
runtime_destroy(runtime);
```


`runtime_init` is generated code. It builds the graph image and passes it to the
runtime library:

```mlir
llvm.func @runtime_init() -> !llvm.ptr {
  %0 = llvm.call @__analog_rt_build_graph_image() : () -> !llvm.ptr
  %1 = llvm.call @sculptor_runtime_init(%0) : (!llvm.ptr) -> !llvm.ptr
  llvm.return %1 : !llvm.ptr
}
```


Inside the runtime library, `sculptor_runtime_init` validates the graph,
allocates the workspace and resource slots, and binds each task descriptor to
its generated entry function.

`runtime_execute` forwards the host input and output pointer arrays to the
runtime:

```mlir
llvm.func @runtime_execute(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
  %0 = llvm.call @sculptor_runtime_execute(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.return %0 : i32
}
```


The runtime assigns input resources to `inputs`, output resources to `outputs`,
then walks the scheduled task list and calls each task's generated entry shim.
Backend runtimes can interpret scheduling metadata such as `core_id` to place
or dispatch the task on a concrete target.

The generated entry shims read task arguments through the runtime argument API,
call the lowered task function, and write results back into runtime buffers:

```c++
BufferView sculptor_runtime_task_arg_buffer(void *opaque, uint32_t index);
void sculptor_runtime_copy_to_buffer(void *dst, const void *src, uint64_t size);
void sculptor_runtime_free_result_buffer(void *opaque, void *ptr);
```


For analog MVM work, shim lowering emits backend-facing Golem-style calls:

```c++
void golem_analog_mvm_set(..., std::int32_t arrayId);
void golem_analog_mvm_load(..., std::int32_t arrayId);
void golem_analog_mvm_compute(std::int32_t arrayId);
void golem_analog_mvm_store(..., std::int32_t arrayId);
```


Those declarations are the boundary between the compiler output and a concrete
backend runtime. The first public tree includes `runtime/common/`, which provides
the shared task graph ABI and runtime state helpers. Device-specific shim
implementations live outside this core package.

So the complete connection is:

```text
main.cpp
-> runtime_init / runtime_execute / runtime_destroy
-> generated graph image and task entry shims
-> sculptor_runtime_* library API
-> scheduled RuntimeTask execution
-> lowered task functions
-> golem_analog_mvm_* shims
-> target backend implementation
-> host output buffers
```


For the linear example, the host output buffer receives:

```text
31.000000 69.000000 112.000000
```

</details>
