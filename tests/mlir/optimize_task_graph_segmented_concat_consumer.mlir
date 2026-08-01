// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=segmented-concat-consumer require-change=true" --canonicalize --cse | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=segmented-concat-consumer require-change=true" --canonicalize --cse --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-results-to-out-params="hoist-static-allocs" --convert-bufferization-to-memref --buffer-deallocation-pipeline | FileCheck %s --check-prefix=BUFFER
// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=segmented-concat-consumer require-change=true" --symbol-dce --canonicalize --cse --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-results-to-out-params="hoist-static-allocs" --convert-bufferization-to-memref --buffer-deallocation-pipeline --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --expand-strided-metadata --convert-arith-to-llvm --convert-index-to-llvm --convert-cf-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @task_segmented_activation(
      %arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>,
      %arg2: tensor<1x4xf32>, %bias: tensor<1x1x12xf32>)
      -> tensor<1x1x12xf32> attributes {
        sculptor.task_kind = "mixed.fused"
      } {
    %concat = tensor.concat dim(1) %arg0, %arg1, %arg2
        : (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>)
          -> tensor<1x12xf32>
    %expanded = tensor.expand_shape %concat [[0, 1], [2]]
        output_shape [1, 1, 12]
        : tensor<1x12xf32> into tensor<1x1x12xf32>
    %init = tensor.empty() : tensor<1x1x12xf32>
    %result = linalg.generic {
        indexing_maps = [#identity3, #identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%expanded, %bias : tensor<1x1x12xf32>, tensor<1x1x12xf32>)
        outs(%init : tensor<1x1x12xf32>) {
      ^bb0(%input: f32, %bias_value: f32, %output: f32):
        %sum = arith.addf %input, %bias_value : f32
        %activated = arith.mulf %sum, %sum : f32
        linalg.yield %activated : f32
      } -> tensor<1x1x12xf32>
    return %result : tensor<1x1x12xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input0 = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %input1 = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %input2 = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %bias = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x12xf32>>
    %output = sculptor.task_graph.output %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x12xf32>>
    %task = sculptor.task.create %graph, @task_segmented_activation,
        domain = "digital", task_kind = "mixed.fused",
        task_name = "segmented_activation", source_layer = "test",
        source_task_ordinal = 0,
        inputs[%input0, %input1, %input2, %bias], outputs[%output], deps[] {
          sculptor.runtime.core_id = 0 : i64,
          sculptor.runtime.result_indices = array<i64: 0>
        } : (!sculptor.task_graph,
             !sculptor.task_resource<tensor<1x4xf32>>,
             !sculptor.task_resource<tensor<1x4xf32>>,
             !sculptor.task_resource<tensor<1x4xf32>>,
             !sculptor.task_resource<tensor<1x1x12xf32>>,
             !sculptor.task_resource<tensor<1x1x12xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func @task_segmented_activation(
// CHECK-NOT: tensor.concat
// CHECK: linalg.generic
// CHECK-SAME: sculptor.optimization.segmented_concat = 0 : i64
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: linalg.generic
// CHECK-SAME: sculptor.optimization.segmented_concat = 4 : i64
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: linalg.generic
// CHECK-SAME: sculptor.optimization.segmented_concat = 8 : i64
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: return

// BUFFER-LABEL: func.func @task_segmented_activation(
// BUFFER-NOT: tensor.concat
// BUFFER-NOT: memref.copy
// BUFFER: linalg.generic
// BUFFER-SAME: sculptor.optimization.segmented_concat = 0 : i64
// BUFFER: arith.addf
// BUFFER: arith.mulf
// BUFFER: linalg.generic
// BUFFER-SAME: sculptor.optimization.segmented_concat = 4 : i64
// BUFFER: arith.addf
// BUFFER: arith.mulf
// BUFFER: linalg.generic
// BUFFER-SAME: sculptor.optimization.segmented_concat = 8 : i64
// BUFFER: arith.addf
// BUFFER: arith.mulf

// LLVM-LABEL: define void @task_segmented_activation(
// LLVM-NOT: llvm.memcpy
// LLVM-NOT: @memcpy
// LLVM: fadd float
// LLVM: fmul float
