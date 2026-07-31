// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=TIMING
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=random" | FileCheck %s --check-prefix=PLACED
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" | FileCheck %s --check-prefix=PLACED
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=greedy greedy-heuristic=transfer-cost,lookahead=2,beam=1,scope=cardinal" | FileCheck %s --check-prefix=PLACED
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=greedy-timing greedy-heuristic=transfer-cost,lookahead=2,beam=1,scope=cardinal" | FileCheck %s --check-prefix=PLACED
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=annealing annealing-initial-schedule=snake annealing-steps-per-temperature=1 annealing-minimum-epochs=1 annealing-plateau-patience=1 annealing-plateau-acceptance-rate=1 annealing-maximum-evaluations=4 annealing-maximum-runtime-seconds=0" | FileCheck %s --check-prefix=PLACED
// RUN: not sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=snake" 2>&1 | FileCheck %s --check-prefix=INSUFFICIENT
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" --sculptor-export-task-graph-vis="output=%t.graphml format=graphml" -o /dev/null
// RUN: FileCheck %s --check-prefix=GRAPHML --input-file=%t.graphml
// RUN: sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=4 min-ops-per-shard=1 placement-policy=require-distinct require-change=true" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" --sculptor-finalize-task-graph-resources --sculptor-export-task-graph-sim-model="output=%t.json" -o /dev/null
// RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json
// RUN: sculptor-mlir-opt %s --sculptor-lower-golem-to-task-graph="distribute-digital-matmuls=true digital-matmul-strategy=output-columns digital-matmul-max-shards=4 digital-matmul-min-ops-per-shard=1 digital-matmul-placement-policy=require-distinct cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" --verify-each | FileCheck %s --check-prefix=PIPELINE

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

// CHECK-NOT: func.func private @task_matmul(
// CHECK-DAG: func.func private @__sculptor_distribute_matmul_group0_partition_rhs
// CHECK-DAG: tensor.extract_slice
// CHECK-DAG: func.func private @__sculptor_distribute_matmul_group0_shard0
// CHECK-DAG: sculptor.runtime.digital_ops = 128 : i64
// CHECK-DAG: role = shard
// CHECK-DAG: linalg.matmul
// CHECK-DAG: func.func private @__sculptor_distribute_matmul_group0_assemble
// CHECK-DAG: tensor.insert_slice
// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-COUNT-4: task_kind = "digital.matmul_shard"
// CHECK: task_kind = "digital.matmul_assembly"
// CHECK-NOT: task_kind = "digital.matmul",

// TIMING-LABEL: func.func private @generate_task_graph
// TIMING-SAME: sculptor.timing.critical_path_ns = 7.850000e+02 : f64
// TIMING-SAME: sculptor.timing.execution_depth = 3 : i64
// TIMING-SAME: sculptor.timing.task_count = 6 : i64
// TIMING: task_name = "matmul.shard.0"
// TIMING-SAME: sculptor.runtime.digital_ops = 128 : i64
// TIMING-SAME: sculptor.workload.digital_ops = 128 : i64

// PLACED-LABEL: func.func private @generate_task_graph
// PLACED-SAME: sculptor.schedule.task_count = 6 : i64
// PLACED-SAME: sculptor.schedule.total_digital_ops = 512 : i64
// PLACED: task_name = "matmul.shard.0"
// PLACED-SAME: sculptor.runtime.core_id = 0 : i64
// PLACED: task_name = "matmul.shard.1"
// PLACED-SAME: sculptor.runtime.core_id = 1 : i64
// PLACED: task_name = "matmul.shard.2"
// PLACED-SAME: sculptor.runtime.core_id = 2 : i64
// PLACED: task_name = "matmul.shard.3"
// PLACED-SAME: sculptor.runtime.core_id = 3 : i64

// INSUFFICIENT: digital distribution group 0 requires 4 distinct cores, but the hardware has 3

// GRAPHML: <key id="distribution_group_id" for="node" attr.name="distribution_group_id" attr.type="long"/>
// GRAPHML: <key id="distribution_strategy" for="node" attr.name="distribution_strategy" attr.type="string"/>
// GRAPHML: <data key="task_name">matmul.shard.0</data>
// GRAPHML: <data key="distribution_group_id">0</data>
// GRAPHML: <data key="distribution_role">shard</data>
// GRAPHML: <data key="distribution_shard_id">0</data>
// GRAPHML: <data key="distribution_shard_count">4</data>
// GRAPHML: <data key="distribution_strategy">output_columns</data>
// GRAPHML: <data key="distribution_placement">require_distinct</data>

// JSON: "name": "matmul.shard.0"
// JSON: "core_id": 0
// JSON: "distribution_group_id": 0
// JSON: "distribution_role": "shard"
// JSON: "distribution_shard_id": 0
// JSON: "distribution_shard_count": 4
// JSON: "distribution_strategy": "output_columns"
// JSON: "distribution_placement": "require_distinct"

// PIPELINE-LABEL: module attributes
// PIPELINE-SAME: sculptor.deployment.active_core_ids = [0, 1, 2, 3]
// PIPELINE: module @core_0
// PIPELINE: sculptor.task.create {{.*}}task_name = "matmul.shard.0"{{.*}}sculptor.deployment.global_task_id = 1 : i64
// PIPELINE: module @core_1
// PIPELINE: sculptor.task.create {{.*}}task_name = "matmul.shard.1"
// PIPELINE: module @core_2
// PIPELINE: sculptor.task.create {{.*}}task_name = "matmul.shard.2"
// PIPELINE: module @core_3
// PIPELINE: sculptor.task.create {{.*}}task_name = "matmul.shard.3"
