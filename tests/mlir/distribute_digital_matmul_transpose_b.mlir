// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=3 min-ops-per-shard=1 require-change=true" | FileCheck %s

module {
  func.func private @task_matmul_transpose_b(
      %lhs: tensor<2x4xf32>,
      %rhs: tensor<5x4xf32>) -> tensor<2x5xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<2x5xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<2x5xf32>) -> tensor<2x5xf32>
    %result = linalg.matmul_transpose_b
        ins(%lhs, %rhs : tensor<2x4xf32>, tensor<5x4xf32>)
        outs(%init : tensor<2x5xf32>) -> tensor<2x5xf32>
    return %result : tensor<2x5xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %lhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<2x4xf32>>
    %rhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<5x4xf32>>
    %output = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<2x5xf32>>
    %matmul = sculptor.task.create %graph, @task_matmul_transpose_b,
        domain = "digital",
        task_kind = "digital.matmul",
        task_name = "transpose_b",
        source_layer = "test",
        source_task_ordinal = 0,
        inputs[%lhs, %rhs],
        outputs[%output],
        deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<2x4xf32>>,
           !sculptor.task_resource<tensor<5x4xf32>>,
           !sculptor.task_resource<tensor<2x5xf32>>)
        -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-DAG: func.func private @__sculptor_distribute_transpose_b_group0_partition_rhs
// CHECK-DAG: tensor.extract_slice {{.*}} tensor<5x4xf32> to tensor<2x4xf32>
// CHECK-DAG: tensor.extract_slice {{.*}} tensor<5x4xf32> to tensor<1x4xf32>
// CHECK-DAG: func.func private @__sculptor_distribute_transpose_b_group0_shard0
// CHECK-DAG: linalg.matmul_transpose_b
// CHECK-DAG: sculptor.runtime.digital_ops = 32 : i64
// CHECK-DAG: sculptor.runtime.digital_ops = 16 : i64
// CHECK-DAG: tensor.insert_slice {{.*}}[0, 4] [2, 1]
// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-COUNT-3: task_kind = "digital.matmul_shard"
