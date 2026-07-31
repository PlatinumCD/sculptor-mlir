// RUN: not sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="max-shards=2 min-ops-per-shard=1" 2>&1 | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=attention-heads max-shards=2 min-ops-per-shard=1" | FileCheck %s --check-prefix=SKIP

module {
  func.func private @task_dynamic_matmul(
      %lhs: tensor<?x8xf32>,
      %rhs: tensor<8x8xf32>) -> tensor<4x8xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<4x8xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<?x8xf32>, tensor<8x8xf32>)
        outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %result : tensor<4x8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %lhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<?x8xf32>>
    %rhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<8x8xf32>>
    %output = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<4x8xf32>>
    %matmul = sculptor.task.create %graph, @task_dynamic_matmul,
        domain = "digital",
        task_kind = "digital.matmul",
        task_name = "dynamic",
        source_layer = "test",
        source_task_ordinal = 0,
        inputs[%lhs, %rhs],
        outputs[%output],
        deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<?x8xf32>>,
           !sculptor.task_resource<tensor<8x8xf32>>,
           !sculptor.task_resource<tensor<4x8xf32>>)
        -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK: digital matmul distribution requires static ranked tensor shapes

// SKIP: task_kind = "digital.matmul"
// SKIP-SAME: task_name = "dynamic"
