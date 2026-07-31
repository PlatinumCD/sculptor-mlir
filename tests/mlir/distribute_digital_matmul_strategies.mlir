// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-rows max-shards=4 min-ops-per-shard=1 require-change=true" | FileCheck %s --check-prefix=ROWS
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=two-dimensional max-shards=4 min-ops-per-shard=1 require-change=true" | FileCheck %s --check-prefix=TWO-D
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=auto max-shards=4 min-ops-per-shard=1 require-change=true" | FileCheck %s --check-prefix=AUTO
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=unconstrained require-change=true" --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=snake" | FileCheck %s --check-prefix=UNCONSTRAINED
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=prefer-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=snake" | FileCheck %s --check-prefix=PREFER

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

// ROWS-DAG: func.func private @__sculptor_distribute_matmul_group0_partition_lhs
// ROWS-DAG: tensor.extract_slice {{.*}} tensor<4x8xf32> to tensor<1x8xf32>
// ROWS-DAG: func.func private @__sculptor_distribute_matmul_group0_shard0{{.*}}tensor<1x8xf32>
// ROWS-DAG: strategy = output_rows
// ROWS-DAG: tensor.insert_slice {{.*}}[3, 0] [1, 8]
// ROWS-LABEL: func.func private @generate_task_graph
// ROWS-COUNT-4: task_kind = "digital.matmul_shard"

// TWO-D-DAG: func.func private @__sculptor_distribute_matmul_group0_partition_lhs
// TWO-D-DAG: func.func private @__sculptor_distribute_matmul_group0_partition_rhs
// TWO-D-DAG: func.func private @__sculptor_distribute_matmul_group0_shard0{{.*}}tensor<2x8xf32>{{.*}}tensor<8x4xf32>{{.*}}tensor<2x4xf32>
// TWO-D-DAG: rowShards = 2 : i64
// TWO-D-DAG: columnShards = 2 : i64
// TWO-D-DAG: strategy = two_dimensional
// TWO-D-DAG: tensor.insert_slice {{.*}}[2, 4] [2, 4]
// TWO-D-LABEL: func.func private @generate_task_graph
// TWO-D-COUNT-4: task_kind = "digital.matmul_shard"

// AUTO-DAG: strategy = output_columns
// AUTO-NOT: strategy = output_rows
// AUTO-NOT: strategy = two_dimensional
// AUTO-LABEL: func.func private @generate_task_graph
// AUTO-COUNT-4: task_kind = "digital.matmul_shard"

// UNCONSTRAINED-LABEL: func.func private @generate_task_graph
// UNCONSTRAINED: task_name = "matmul.shard.0"
// UNCONSTRAINED-SAME: sculptor.runtime.core_id = 0 : i64
// UNCONSTRAINED: task_name = "matmul.shard.1"
// UNCONSTRAINED-SAME: sculptor.runtime.core_id = 0 : i64
// UNCONSTRAINED: task_name = "matmul.shard.2"
// UNCONSTRAINED-SAME: sculptor.runtime.core_id = 0 : i64
// UNCONSTRAINED: task_name = "matmul.shard.3"
// UNCONSTRAINED-SAME: sculptor.runtime.core_id = 0 : i64

// PREFER-LABEL: func.func private @generate_task_graph
// PREFER: task_name = "matmul.shard.0"
// PREFER-SAME: sculptor.runtime.core_id = 0 : i64
// PREFER: task_name = "matmul.shard.1"
// PREFER-SAME: sculptor.runtime.core_id = 1 : i64
