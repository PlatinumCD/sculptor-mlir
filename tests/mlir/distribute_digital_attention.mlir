// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=attention-heads max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --verify-each | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=auto max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --verify-each | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=attention-heads max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" --verify-each | FileCheck %s --check-prefix=PLACED
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=attention-heads max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" --sculptor-fuse-task-graph --sculptor-analyze-task-graph-timing --verify-each | FileCheck %s --check-prefix=FUSED
// RUN: sculptor-mlir-opt %S/convert_transformer_attention_structured.mlir --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" --sculptor-materialize-tasks --sculptor-assemble-task-graph --sculptor-distribute-digital-matmul="strategy=attention-heads max-shards=2 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --verify-each | FileCheck %s --check-prefix=TRANSFORMER
// RUN: sculptor-mlir-opt %S/convert_transformer_attention_structured.mlir --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" --sculptor-materialize-tasks --sculptor-lower-golem-to-task-graph="distribute-digital-matmuls=true digital-matmul-strategy=attention-heads digital-matmul-max-shards=2 digital-matmul-min-ops-per-shard=1 digital-matmul-placement-policy=require-distinct cores=64 arrays-per-core=4 topology=mesh mesh-rows=8 mesh-cols=8 schedule=snake" --verify-each | FileCheck %s --check-prefix=DEPLOYMENT

module {
  func.func private @attention_scores(
      %query: tensor<1x2x8xf32>,
      %key: tensor<1x2x8xf32>) -> tensor<1x4x2x2xf32>
      attributes {
        causal = true,
        head_dim = 2 : i64,
        sculptor.source_layer = "attention",
        sculptor.source_task_ordinal = 0 : i64,
        sculptor.task_domain = "digital",
        sculptor.task_kind = "digital.attention_scores",
        sculptor.task_name = "scores"
      } {
    %empty = tensor.empty() : tensor<1x4x2x2xf32>
    return %empty : tensor<1x4x2x2xf32>
  }

  func.func private @attention_apply(
      %probabilities: tensor<1x4x2x2xf32>,
      %value: tensor<1x2x8xf32>) -> tensor<1x4x2x2xf32>
      attributes {
        head_dim = 2 : i64,
        sculptor.source_layer = "attention",
        sculptor.source_task_ordinal = 1 : i64,
        sculptor.task_domain = "digital",
        sculptor.task_kind = "digital.attention_apply",
        sculptor.task_name = "apply"
      } {
    %empty = tensor.empty() : tensor<1x4x2x2xf32>
    return %empty : tensor<1x4x2x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %query = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x2x8xf32>>
    %key = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x2x8xf32>>
    %value = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x2x8xf32>>
    %scores = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x4x2x2xf32>>
    %output = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x4x2x2xf32>>
    %score_task = sculptor.task.create %graph, @attention_scores,
        domain = "digital",
        task_kind = "digital.attention_scores",
        task_name = "scores",
        source_layer = "attention",
        source_task_ordinal = 0,
        inputs[%query, %key],
        outputs[%scores],
        deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x2x8xf32>>,
           !sculptor.task_resource<tensor<1x2x8xf32>>,
           !sculptor.task_resource<tensor<1x4x2x2xf32>>)
        -> !sculptor.task
    %apply_task = sculptor.task.create %graph, @attention_apply,
        domain = "digital",
        task_kind = "digital.attention_apply",
        task_name = "apply",
        source_layer = "attention",
        source_task_ordinal = 1,
        inputs[%scores, %value],
        outputs[%output],
        deps[%score_task]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x4x2x2xf32>>,
           !sculptor.task_resource<tensor<1x2x8xf32>>,
           !sculptor.task_resource<tensor<1x4x2x2xf32>>,
           !sculptor.task)
        -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-NOT: func.func private @attention_scores
// CHECK-NOT: func.func private @attention_apply
// CHECK-DAG: func.func private @__sculptor_distribute_scores_group0_partition_query({{.*}}tensor<1x2x8xf32>) -> (tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>)
// CHECK-DAG: func.func private @__sculptor_distribute_scores_group0_shard0
// CHECK-DAG: causal = true, head_dim = 2 : i64, sculptor.runtime.digital_ops = 16 : i64
// CHECK-DAG: strategy = attention_heads
// CHECK-DAG: linalg.generic
// CHECK-DAG: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-DAG: linalg.index 2
// CHECK-DAG: linalg.index 3
// CHECK-DAG: arith.select
// CHECK-DAG: func.func private @__sculptor_distribute_apply_group1_shard0({{.*}}tensor<1x1x2x2xf32>, {{.*}}tensor<1x2x2xf32>) -> tensor<1x1x2x2xf32>
// CHECK-DAG: sculptor.runtime.digital_ops = 16 : i64
// CHECK-DAG: func.func private @__sculptor_distribute_apply_group1_assemble
// CHECK-DAG: tensor.insert_slice
// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-DAG: task_name = "scores.partition.query"
// CHECK-DAG: task_name = "scores.partition.key"
// CHECK-DAG: task_name = "scores.shard.0"
// CHECK-DAG: task_name = "scores.shard.1"
// CHECK-DAG: task_name = "scores.shard.2"
// CHECK-DAG: task_name = "scores.shard.3"
// CHECK-DAG: task_name = "scores.assemble"
// CHECK-DAG: task_name = "apply.partition.probabilities"
// CHECK-DAG: task_name = "apply.partition.value"
// CHECK-DAG: task_name = "apply.shard.0"
// CHECK-DAG: task_name = "apply.shard.1"
// CHECK-DAG: task_name = "apply.shard.2"
// CHECK-DAG: task_name = "apply.shard.3"
// CHECK-DAG: task_name = "apply.assemble"
// CHECK-NOT: task_kind = "digital.attention_scores"
// CHECK-NOT: task_kind = "digital.attention_apply"

// PLACED-LABEL: func.func private @generate_task_graph
// PLACED: task_name = "scores.shard.0"
// PLACED-SAME: sculptor.runtime.core_id = 0 : i64
// PLACED: task_name = "scores.shard.1"
// PLACED-SAME: sculptor.runtime.core_id = 1 : i64
// PLACED: task_name = "scores.shard.2"
// PLACED-SAME: sculptor.runtime.core_id = 2 : i64
// PLACED: task_name = "scores.shard.3"
// PLACED-SAME: sculptor.runtime.core_id = 3 : i64
// PLACED: task_name = "apply.shard.0"
// PLACED-SAME: sculptor.runtime.core_id = 0 : i64
// PLACED: task_name = "apply.shard.1"
// PLACED-SAME: sculptor.runtime.core_id = 1 : i64
// PLACED: task_name = "apply.shard.2"
// PLACED-SAME: sculptor.runtime.core_id = 2 : i64
// PLACED: task_name = "apply.shard.3"
// PLACED-SAME: sculptor.runtime.core_id = 3 : i64

// FUSED-LABEL: func.func private @generate_task_graph
// FUSED-DAG: task_name = "scores.shard.0"
// FUSED-DAG: task_name = "scores.shard.3"
// FUSED-DAG: task_name = "apply.shard.0"
// FUSED-DAG: task_name = "apply.shard.3"
// FUSED-NOT: task_kind = "mixed.fused"

// TRANSFORMER-LABEL: func.func private @generate_task_graph
// TRANSFORMER-DAG: task_name = "transformer_block_self_attention_scores.partition.query"
// TRANSFORMER-DAG: task_name = "transformer_block_self_attention_scores.shard.0"
// TRANSFORMER-DAG: task_name = "transformer_block_self_attention_scores.shard.1"
// TRANSFORMER-DAG: task_name = "transformer_block_self_attention_scores.assemble"
// TRANSFORMER-DAG: task_name = "transformer_block_self_attention_apply.partition.probabilities"
// TRANSFORMER-DAG: task_name = "transformer_block_self_attention_apply.shard.0"
// TRANSFORMER-DAG: task_name = "transformer_block_self_attention_apply.shard.1"
// TRANSFORMER-DAG: task_name = "transformer_block_self_attention_apply.assemble"
// TRANSFORMER-DAG: task_name = "transformer_block_cross_attention_scores.partition.query"
// TRANSFORMER-DAG: task_name = "transformer_block_cross_attention_scores.shard.0"
// TRANSFORMER-DAG: task_name = "transformer_block_cross_attention_scores.shard.1"
// TRANSFORMER-DAG: task_name = "transformer_block_cross_attention_scores.assemble"
// TRANSFORMER-DAG: task_name = "transformer_block_cross_attention_apply.partition.probabilities"
// TRANSFORMER-DAG: task_name = "transformer_block_cross_attention_apply.shard.0"
// TRANSFORMER-DAG: task_name = "transformer_block_cross_attention_apply.shard.1"
// TRANSFORMER-DAG: task_name = "transformer_block_cross_attention_apply.assemble"
// TRANSFORMER-NOT: task_kind = "digital.attention_scores"
// TRANSFORMER-NOT: task_kind = "digital.attention_apply"

// DEPLOYMENT-LABEL: module attributes
// DEPLOYMENT-SAME: sculptor.deployment.active_core_ids
// DEPLOYMENT: module @core_0
// DEPLOYMENT: func.func private @__sculptor_distribute_transformer_block_self_attention_scores_group0_shard0
// DEPLOYMENT-SAME: strategy = attention_heads
// DEPLOYMENT: module @core_8
// DEPLOYMENT: func.func private @__sculptor_distribute_transformer_block_self_attention_scores_group0_shard1
// DEPLOYMENT-SAME: strategy = attention_heads
