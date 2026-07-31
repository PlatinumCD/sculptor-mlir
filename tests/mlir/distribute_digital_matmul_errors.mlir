// RUN: not sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=invalid" 2>&1 | FileCheck %s --check-prefix=INVALID-STRATEGY
// RUN: not sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="placement-policy=invalid" 2>&1 | FileCheck %s --check-prefix=INVALID-PLACEMENT
// RUN: not sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="max-shards=1" 2>&1 | FileCheck %s --check-prefix=INVALID-SHARDS
// RUN: not sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="min-ops-per-shard=0" 2>&1 | FileCheck %s --check-prefix=INVALID-MINIMUM
// RUN: not sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="max-shards=4 min-ops-per-shard=100000 require-change=true" 2>&1 | FileCheck %s --check-prefix=NO-CHANGE
// RUN: not sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="max-shards=4 min-ops-per-shard=1" --sculptor-build-task-graph-islands --sculptor-distribute-digital-matmul="max-shards=4 min-ops-per-shard=1" 2>&1 | FileCheck %s --check-prefix=STALE

module {
  func.func private @task_matmul(
      %lhs: tensor<4x8xf32>,
      %rhs: tensor<8x8xf32>) -> tensor<4x8xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<4x8xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf32>, tensor<8x8xf32>)
        outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %result : tensor<4x8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %lhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<4x8xf32>>
    %rhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<8x8xf32>>
    %output = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<4x8xf32>>
    %matmul = sculptor.task.create %graph, @task_matmul,
        domain = "digital",
        task_kind = "digital.matmul",
        task_name = "matmul",
        source_layer = "test",
        source_task_ordinal = 0,
        inputs[%lhs, %rhs],
        outputs[%output],
        deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<4x8xf32>>,
           !sculptor.task_resource<tensor<8x8xf32>>,
           !sculptor.task_resource<tensor<4x8xf32>>)
        -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// INVALID-STRATEGY: unknown digital matmul distribution strategy 'invalid'; expected auto, output-columns, output-rows, two-dimensional, or attention-heads
// INVALID-PLACEMENT: unknown digital matmul placement policy 'invalid'; expected unconstrained, prefer-distinct, or require-distinct
// INVALID-SHARDS: expected max-shards to be at least two
// INVALID-MINIMUM: expected min-ops-per-shard to be positive
// NO-CHANGE: expected at least one eligible digital matmul
// STALE: digital matmul distribution must run before island construction and placement
