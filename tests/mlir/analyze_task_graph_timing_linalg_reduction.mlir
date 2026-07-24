// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing | FileCheck %s

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

module {
  func.func private @task_matmul(%lhs: tensor<2x3xf32>,
                                %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<2x4xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<2x4xf32>) -> tensor<2x4xf32>
    %result = linalg.generic {
        indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs : tensor<2x3xf32>, tensor<3x4xf32>)
        outs(%init : tensor<2x4xf32>) {
      ^bb0(%left: f32, %right: f32, %acc: f32):
        %product = arith.mulf %left, %right : f32
        %sum = arith.addf %acc, %product : f32
        linalg.yield %sum : f32
    } -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %lhs = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<2x3xf32>>
    %rhs = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<3x4xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<2x4xf32>>
    %matmul = sculptor.task.create %graph, @task_matmul, domain = "digital", task_kind = "digital.test_reduction", task_name = "matmul", source_layer = "test", source_task_ordinal = 0, inputs[%lhs, %rhs], outputs[%output], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<2x3xf32>>, !sculptor.task_resource<tensor<3x4xf32>>, !sculptor.task_resource<tensor<2x4xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// The generic executes 2 * 4 * 3 iterations with one multiply and one add.
// CHECK-LABEL: func.func private @generate_task_graph
// CHECK: task_name = "matmul"
// CHECK-SAME: sculptor.timing.digital_ops = 48 : i64
// CHECK-SAME: sculptor.timing.intrinsic_latency_ns = 6.000000e+00 : f64
