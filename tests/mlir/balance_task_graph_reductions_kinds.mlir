// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions | FileCheck %s --check-prefix=OPS
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions | FileCheck %s --check-prefix=HELPERS
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=snake" | FileCheck %s --check-prefix=REUSE

module {
  func.func private @star_add_a(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @star_add_b(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @star_max(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @star_min(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %add_a = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %add_b = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %maximum = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %minimum = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %task_add_a = sculptor.task.create %graph, @star_add_a, domain = "digital", task_kind = "digital.reduction", task_name = "add_a", source_layer = "reduction", source_task_ordinal = 0, inputs[%input, %input, %input, %input], outputs[%add_a], deps[] {sculptor.runtime.result_indices = array<i64: 0>, sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    %task_add_b = sculptor.task.create %graph, @star_add_b, domain = "digital", task_kind = "digital.reduction", task_name = "add_b", source_layer = "reduction", source_task_ordinal = 1, inputs[%input, %input, %input, %input], outputs[%add_b], deps[] {sculptor.runtime.result_indices = array<i64: 0>, sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    %task_max = sculptor.task.create %graph, @star_max, domain = "digital", task_kind = "digital.reduction", task_name = "maximum", source_layer = "reduction", source_task_ordinal = 2, inputs[%input, %input, %input, %input], outputs[%maximum], deps[] {sculptor.runtime.result_indices = array<i64: 0>, sculptor.task.reduction = #sculptor.task_reduction<kind = max, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    %task_min = sculptor.task.create %graph, @star_min, domain = "digital", task_kind = "digital.reduction", task_name = "minimum", source_layer = "reduction", source_task_ordinal = 3, inputs[%input, %input, %input, %input], outputs[%minimum], deps[] {sculptor.runtime.result_indices = array<i64: 0>, sculptor.task.reduction = #sculptor.task_reduction<kind = min, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// HELPERS-NOT: func.func private @star_
// HELPERS-COUNT-12: attributes {sculptor.task.reduction_helper = #sculptor.task_reduction
// HELPERS-NOT: attributes {sculptor.task.reduction_helper = #sculptor.task_reduction

// OPS-DAG: arith.addf
// OPS-DAG: arith.maximumf
// OPS-DAG: arith.minimumf
// OPS-LABEL: func.func private @generate_task_graph()
// OPS: task_name = "add_a.level0.lane0"
// OPS: task_name = "add_a.level0.lane1"
// OPS: task_name = "add_a.level1.root"
// OPS: task_name = "add_b.level0.lane0"
// OPS: task_name = "add_b.level0.lane1"
// OPS: task_name = "add_b.level1.root"
// OPS: task_name = "maximum.level0.lane0"
// OPS: task_name = "maximum.level0.lane1"
// OPS: task_name = "maximum.level1.root"
// OPS: task_name = "minimum.level0.lane0"
// OPS: task_name = "minimum.level0.lane1"
// OPS: task_name = "minimum.level1.root"

// REUSE-DAG: task_name = "add_a.level0.lane0"{{.*}}sculptor.runtime.core_id = 0 : i64
// REUSE-DAG: task_name = "add_a.level0.lane1"{{.*}}sculptor.runtime.core_id = 1 : i64
// REUSE-DAG: task_name = "add_b.level0.lane0"{{.*}}sculptor.runtime.core_id = 0 : i64
// REUSE-DAG: task_name = "add_b.level0.lane1"{{.*}}sculptor.runtime.core_id = 1 : i64
// REUSE-DAG: task_name = "maximum.level0.lane0"{{.*}}sculptor.runtime.core_id = 0 : i64
// REUSE-DAG: task_name = "maximum.level0.lane1"{{.*}}sculptor.runtime.core_id = 1 : i64
// REUSE-DAG: task_name = "add_a.level1.root"{{.*}}sculptor.runtime.core_id = 0 : i64
// REUSE-DAG: task_name = "add_b.level1.root"{{.*}}sculptor.runtime.core_id = 1 : i64
