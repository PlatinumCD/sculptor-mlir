// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=elementwise-fusion require-change=true" --canonicalize --cse | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=elementwise-fusion require-change=true" --canonicalize --cse --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-results-to-out-params="hoist-static-allocs" --convert-bufferization-to-memref --buffer-deallocation-pipeline | FileCheck %s --check-prefix=BUFFER
// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=elementwise-fusion require-change=true" --symbol-dce --canonicalize --cse --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-results-to-out-params="hoist-static-allocs" --convert-bufferization-to-memref --buffer-deallocation-pipeline --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --expand-strided-metadata --convert-arith-to-llvm --convert-index-to-llvm --convert-cf-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

#identity2 = affine_map<(d0, d1) -> (d0, d1)>
#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @task_add_then_activation(
      %arg0: tensor<1x8xf32>, %arg1: tensor<1x8xf32>)
      -> tensor<1x1x8xf32> attributes {
        sculptor.task_kind = "mixed.fused"
      } {
    %add_init = tensor.empty() : tensor<1x8xf32>
    %sum = linalg.add ins(%arg0, %arg1 : tensor<1x8xf32>, tensor<1x8xf32>)
        outs(%add_init : tensor<1x8xf32>) -> tensor<1x8xf32>
    %expanded = tensor.expand_shape %sum [[0, 1], [2]] output_shape [1, 1, 8]
        : tensor<1x8xf32> into tensor<1x1x8xf32>
    %activation_init = tensor.empty() : tensor<1x1x8xf32>
    %activated = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%expanded : tensor<1x1x8xf32>)
        outs(%activation_init : tensor<1x1x8xf32>) {
      ^bb0(%input: f32, %output: f32):
        %result = arith.mulf %input, %input : f32
        linalg.yield %result : f32
      } -> tensor<1x1x8xf32>
    return %activated : tensor<1x1x8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input0 = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %input1 = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %output = sculptor.task_graph.output %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x8xf32>>
    %task = sculptor.task.create %graph, @task_add_then_activation,
        domain = "digital", task_kind = "mixed.fused",
        task_name = "add_then_activation", source_layer = "test",
        source_task_ordinal = 0,
        inputs[%input0, %input1], outputs[%output], deps[] {
          sculptor.runtime.core_id = 0 : i64,
          sculptor.runtime.result_indices = array<i64: 0>
        } : (!sculptor.task_graph,
             !sculptor.task_resource<tensor<1x8xf32>>,
             !sculptor.task_resource<tensor<1x8xf32>>,
             !sculptor.task_resource<tensor<1x1x8xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func @task_add_then_activation(
// CHECK-NOT: linalg.add
// CHECK-COUNT-1: linalg.generic
// CHECK-SAME: ins(
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: return

// BUFFER-LABEL: func.func @task_add_then_activation(
// BUFFER-NOT: memref.alloc
// BUFFER-COUNT-1: linalg.generic
// BUFFER-SAME: outs(%arg2
// BUFFER: arith.addf
// BUFFER: arith.mulf

// LLVM-LABEL: define void @task_add_then_activation(
// LLVM-NOT: malloc
// LLVM: fadd float
// LLVM: fmul float
