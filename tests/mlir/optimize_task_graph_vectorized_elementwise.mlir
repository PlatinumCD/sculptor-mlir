// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=vectorized-elementwise require-change=true" --canonicalize --cse | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=vectorized-elementwise require-change=true" --symbol-dce --canonicalize --cse --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-results-to-out-params --convert-bufferization-to-memref --buffer-deallocation-pipeline --convert-scf-to-cf --expand-strided-metadata --convert-vector-to-llvm --convert-arith-to-llvm --convert-index-to-llvm --convert-cf-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @task_residual_add(
      %lhs: tensor<1x4x16xf32>, %rhs: tensor<1x4x16xf32>)
      -> tensor<1x4x16xf32> attributes {
        sculptor.task_kind = "digital.residual_add"
      } {
    %init = tensor.empty() : tensor<1x4x16xf32>
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<1x4x16xf32>, tensor<1x4x16xf32>)
        outs(%init : tensor<1x4x16xf32>) {
      ^bb0(%left: f32, %right: f32, %output: f32):
        %sum = arith.addf %left, %right : f32
        linalg.yield %sum : f32
      } -> tensor<1x4x16xf32>
    return %result : tensor<1x4x16xf32>
  }

  func.func private @task_residual_add_nondivisible(
      %lhs: tensor<1x4x10xf32>, %rhs: tensor<1x4x10xf32>)
      -> tensor<1x4x10xf32> attributes {
        sculptor.task_kind = "digital.residual_add"
      } {
    %init = tensor.empty() : tensor<1x4x10xf32>
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<1x4x10xf32>, tensor<1x4x10xf32>)
        outs(%init : tensor<1x4x10xf32>) {
      ^bb0(%left: f32, %right: f32, %output: f32):
        %sum = arith.addf %left, %right : f32
        linalg.yield %sum : f32
      } -> tensor<1x4x10xf32>
    return %result : tensor<1x4x10xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %lhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4x16xf32>>
    %rhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4x16xf32>>
    %output = sculptor.task_graph.output %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4x16xf32>>
    %short_lhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4x10xf32>>
    %short_rhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4x10xf32>>
    %short_output = sculptor.task_graph.output %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4x10xf32>>
    %task = sculptor.task.create %graph, @task_residual_add,
        domain = "digital", task_kind = "digital.residual_add",
        task_name = "residual_add", source_layer = "test",
        source_task_ordinal = 0,
        inputs[%lhs, %rhs], outputs[%output], deps[] {
          sculptor.runtime.core_id = 0 : i64,
          sculptor.runtime.result_indices = array<i64: 0>
        } : (!sculptor.task_graph,
             !sculptor.task_resource<tensor<1x4x16xf32>>,
             !sculptor.task_resource<tensor<1x4x16xf32>>,
             !sculptor.task_resource<tensor<1x4x16xf32>>) -> !sculptor.task
    %short_task = sculptor.task.create %graph, @task_residual_add_nondivisible,
        domain = "digital", task_kind = "digital.residual_add",
        task_name = "residual_add_nondivisible", source_layer = "test",
        source_task_ordinal = 1,
        inputs[%short_lhs, %short_rhs], outputs[%short_output], deps[] {
          sculptor.runtime.core_id = 0 : i64,
          sculptor.runtime.result_indices = array<i64: 0>
        } : (!sculptor.task_graph,
             !sculptor.task_resource<tensor<1x4x10xf32>>,
             !sculptor.task_resource<tensor<1x4x10xf32>>,
             !sculptor.task_resource<tensor<1x4x10xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func @task_residual_add(
// CHECK: scf.for
// CHECK-COUNT-2: vector.transfer_read
// CHECK: arith.addf {{.*}} : vector<8xf32>
// CHECK: vector.transfer_write
// CHECK: sculptor.optimization.vector_width = 8 : i64
// CHECK-NOT: linalg.generic

// CHECK-LABEL: func.func private @task_residual_add_nondivisible(
// CHECK: linalg.generic
// CHECK-NOT: vector.transfer_read

// LLVM-LABEL: define void @task_residual_add(
// LLVM: load <8 x float>
// LLVM: load <8 x float>
// LLVM: fadd <8 x float>
// LLVM: store <8 x float>
