// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=2 schedule=simple-budget" | FileCheck %s

module {
  func.func private @task_matrix() -> !sculptor.logical.array
  func.func private @task_vector(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>
  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32>
  func.func private @task_bias(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>

  // CHECK: func.func private @generate_task_graph()
  // CHECK-SAME: sculptor.schedule.analog_arrays = [0, 1, 2, 3]
  // CHECK-SAME: sculptor.schedule.arrays_per_core = 2 : i64
  // CHECK-SAME: sculptor.schedule.dependency_count = 3 : i64
  // CHECK-SAME: sculptor.schedule.logical_array_to_analog_array = [0]
  // CHECK-SAME: sculptor.schedule.num_analog_arrays = 4 : i64
  // CHECK-SAME: sculptor.schedule.num_cores = 2 : i64
  // CHECK-SAME: sculptor.schedule.num_logical_arrays = 1 : i64
  // CHECK-SAME: sculptor.schedule.task_count = 4 : i64
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    // CHECK: sculptor.task_graph.temporary {{.*}} {sculptor.schedule.logical_array_index = 0 : i64}
    %array = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %tile = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %mvm_out = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>

    // CHECK: task_name = "linear_matrix_tile_0_0"
    // CHECK-SAME: sculptor.runtime.core_id = 0 : i64
    // CHECK-SAME: sculptor.runtime.task_index = 0 : i64
    %setup = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linear_matrix_tile_0_0", source_layer = "linear_0", source_task_ordinal = 0, inputs[], outputs[%array], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // CHECK: task_name = "linear_vector_tile_0"
    // CHECK-SAME: sculptor.runtime.core_id = 0 : i64
    // CHECK-SAME: sculptor.runtime.task_index = 1 : i64
    %vector = sculptor.task.create %graph, @task_vector, domain = "digital", task_kind = "digital.vector_tile", task_name = "linear_vector_tile_0", source_layer = "linear_0", source_task_ordinal = 1, inputs[%input], outputs[%tile], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    // CHECK: task_name = "linear_mvm_0_0"
    // CHECK-SAME: sculptor.runtime.core_id = 0 : i64
    // CHECK-SAME: sculptor.runtime.task_index = 2 : i64
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "linear_mvm_0_0", source_layer = "linear_0", source_task_ordinal = 2, inputs[%tile, %array], outputs[%mvm_out], deps[%setup, %vector] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    // CHECK: task_name = "linear_bias_add"
    // CHECK-SAME: sculptor.runtime.core_id = 0 : i64
    // CHECK-SAME: sculptor.runtime.task_index = 3 : i64
    %bias = sculptor.task.create %graph, @task_bias, domain = "digital", task_kind = "digital.bias_add", task_name = "linear_bias_add", source_layer = "linear_0", source_task_ordinal = 3, inputs[%mvm_out], outputs[%output], deps[%mvm] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}
