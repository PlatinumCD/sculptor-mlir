// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=4 schedule=snake" | FileCheck %s

module {
  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x2xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x2xf32>
    return %stored : tensor<1x2xf32>
  }

  func.func private @reduce_a(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>, %arg2: tensor<1x2xf32>, %arg3: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @reduce_b(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>, %arg2: tensor<1x2xf32>, %arg3: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output_a = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output_b = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %leaf0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %leaf1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %leaf2 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %leaf3 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array2 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array3 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup0 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix0", source_layer = "spatial", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix1", source_layer = "spatial", source_task_ordinal = 1, inputs[], outputs[%array1], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup2 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix2", source_layer = "spatial", source_task_ordinal = 2, inputs[], outputs[%array2], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup3 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix3", source_layer = "spatial", source_task_ordinal = 3, inputs[], outputs[%array3], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm0 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm0", source_layer = "spatial", source_task_ordinal = 4, inputs[%input, %array0], outputs[%leaf0], deps[%setup0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm1 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm1", source_layer = "spatial", source_task_ordinal = 5, inputs[%input, %array1], outputs[%leaf1], deps[%setup1] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm2 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm2", source_layer = "spatial", source_task_ordinal = 6, inputs[%input, %array2], outputs[%leaf2], deps[%setup2] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm3 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm3", source_layer = "spatial", source_task_ordinal = 7, inputs[%input, %array3], outputs[%leaf3], deps[%setup3] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %reduction_a = sculptor.task.create %graph, @reduce_a, domain = "digital", task_kind = "digital.reduction", task_name = "reduce_a", source_layer = "spatial", source_task_ordinal = 8, inputs[%leaf0, %leaf1, %leaf2, %leaf3], outputs[%output_a], deps[%mvm0, %mvm1, %mvm2, %mvm3] {sculptor.runtime.result_indices = array<i64: 0>, sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task, !sculptor.task, !sculptor.task) -> !sculptor.task
    %reduction_b = sculptor.task.create %graph, @reduce_b, domain = "digital", task_kind = "digital.reduction", task_name = "reduce_b", source_layer = "spatial", source_task_ordinal = 9, inputs[%leaf2, %leaf3, %leaf0, %leaf1], outputs[%output_b], deps[%mvm2, %mvm3, %mvm0, %mvm1] {sculptor.runtime.result_indices = array<i64: 0>, sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task, !sculptor.task, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-DAG: task_name = "mvm0"{{.*}}sculptor.runtime.core_id = 0 : i64
// CHECK-DAG: task_name = "mvm1"{{.*}}sculptor.runtime.core_id = 1 : i64
// CHECK-DAG: task_name = "mvm2"{{.*}}sculptor.runtime.core_id = 2 : i64
// CHECK-DAG: task_name = "mvm3"{{.*}}sculptor.runtime.core_id = 3 : i64

// CHECK-DAG: task_name = "reduce_a.level0.lane0"{{.*}}sculptor.runtime.core_id = 1 : i64
// CHECK-DAG: task_name = "reduce_a.level0.lane1"{{.*}}sculptor.runtime.core_id = 2 : i64
// CHECK-DAG: task_name = "reduce_a.level1.root"{{.*}}sculptor.runtime.core_id = 1 : i64

// CHECK-DAG: task_name = "reduce_b.level0.lane0"{{.*}}sculptor.runtime.core_id = 2 : i64
// CHECK-DAG: task_name = "reduce_b.level0.lane1"{{.*}}sculptor.runtime.core_id = 1 : i64
// CHECK-DAG: task_name = "reduce_b.level1.root"{{.*}}sculptor.runtime.core_id = 2 : i64
