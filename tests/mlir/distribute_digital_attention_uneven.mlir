// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=attention-heads max-shards=3 min-ops-per-shard=1 require-change=true" --verify-each | FileCheck %s

module {
  func.func private @scores(
      %query: tensor<1x3x10xf32>,
      %key: tensor<1x3x10xf32>) -> tensor<1x5x3x3xf32>
      attributes {
        causal = false,
        head_dim = 2 : i64,
        sculptor.task_domain = "digital",
        sculptor.task_kind = "digital.attention_scores",
        sculptor.task_name = "scores"
      }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %query = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x3x10xf32>>
    %key = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x3x10xf32>>
    %scores = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x5x3x3xf32>>
    %task = sculptor.task.create %graph, @scores,
        domain = "digital",
        task_kind = "digital.attention_scores",
        task_name = "scores",
        source_layer = "attention",
        source_task_ordinal = 0,
        inputs[%query, %key],
        outputs[%scores],
        deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x3x10xf32>>,
           !sculptor.task_resource<tensor<1x3x10xf32>>,
           !sculptor.task_resource<tensor<1x5x3x3xf32>>)
        -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @__sculptor_distribute_scores_group0_assemble
// CHECK: tensor.insert_slice {{.*}}[0, 0, 0, 0] [1, 2, 3, 3]
// CHECK: tensor.insert_slice {{.*}}[0, 2, 0, 0] [1, 2, 3, 3]
// CHECK: tensor.insert_slice {{.*}}[0, 4, 0, 0] [1, 1, 3, 3]
// CHECK-DAG: func.func private @__sculptor_distribute_scores_group0_partition_query({{.*}}tensor<1x3x10xf32>) -> (tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x2xf32>)
// CHECK-DAG: func.func private @__sculptor_distribute_scores_group0_partition_key({{.*}}tensor<1x3x10xf32>) -> (tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<1x3x2xf32>)
// CHECK-DAG: func.func private @__sculptor_distribute_scores_group0_shard0{{.*}} -> tensor<1x2x3x3xf32> attributes {{.*}}sculptor.runtime.digital_ops = 72 : i64
// CHECK-DAG: func.func private @__sculptor_distribute_scores_group0_shard1{{.*}} -> tensor<1x2x3x3xf32> attributes {{.*}}sculptor.runtime.digital_ops = 72 : i64
// CHECK-DAG: func.func private @__sculptor_distribute_scores_group0_shard2{{.*}} -> tensor<1x1x3x3xf32> attributes {{.*}}sculptor.runtime.digital_ops = 36 : i64
