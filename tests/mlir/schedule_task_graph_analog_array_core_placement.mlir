// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=2 schedule=simple-budget" | FileCheck %s

module {
  func.func private @task_matrix() -> !sculptor.logical.array
  func.func private @task_vector(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>
  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32>

  // CHECK: func.func private @generate_task_graph()
  // CHECK-SAME: sculptor.schedule.logical_array_to_analog_array = [0, 1, 2, 3]
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array0 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array2 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array3 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %tile = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>

    // CHECK-DAG: task_name = "matrix_0"{{.*}}sculptor.runtime.core_id = 0 : i64{{.*}}sculptor.runtime.physical_array_id = 0 : i64{{.*}}sculptor.runtime.task_index = 0 : i64
    %setup0 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix_0", source_layer = "linear_0", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // CHECK-DAG: task_name = "matrix_1"{{.*}}sculptor.runtime.core_id = 0 : i64{{.*}}sculptor.runtime.physical_array_id = 1 : i64{{.*}}sculptor.runtime.task_index = 1 : i64
    %setup1 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix_1", source_layer = "linear_1", source_task_ordinal = 0, inputs[], outputs[%array1], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // CHECK-DAG: task_name = "matrix_2"{{.*}}sculptor.runtime.core_id = 1 : i64{{.*}}sculptor.runtime.physical_array_id = 2 : i64{{.*}}sculptor.runtime.task_index = 2 : i64
    %setup2 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix_2", source_layer = "linear_2", source_task_ordinal = 0, inputs[], outputs[%array2], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // CHECK-DAG: task_name = "matrix_3"{{.*}}sculptor.runtime.core_id = 1 : i64{{.*}}sculptor.runtime.physical_array_id = 3 : i64{{.*}}sculptor.runtime.task_index = 3 : i64
    %setup3 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix_3", source_layer = "linear_3", source_task_ordinal = 0, inputs[], outputs[%array3], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task

    // CHECK-DAG: task_name = "vector_3"{{.*}}sculptor.runtime.core_id = 0 : i64{{.*}}sculptor.runtime.task_index = 4 : i64
    %vector3 = sculptor.task.create %graph, @task_vector, domain = "digital", task_kind = "digital.vector_tile", task_name = "vector_3", source_layer = "linear_3", source_task_ordinal = 1, inputs[%input], outputs[%tile], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    // CHECK-DAG: task_name = "mvm_3"{{.*}}sculptor.runtime.core_id = 1 : i64{{.*}}sculptor.runtime.physical_array_id = 3 : i64{{.*}}sculptor.runtime.task_index = 5 : i64
    %mvm3 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm_3", source_layer = "linear_3", source_task_ordinal = 2, inputs[%tile, %array3], outputs[%output], deps[%setup3, %vector3] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}
