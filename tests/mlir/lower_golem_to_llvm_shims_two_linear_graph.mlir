// RUN: sculptor-mlir-opt %s --sculptor-lower-golem-to-llvm-shims --sculptor-finalize-task-graph-resources | FileCheck %s --implicit-check-not=!sculptor.logical.array
// RUN: sculptor-mlir-opt %s --sculptor-finalize-task-graph-resources --sculptor-lower-golem-to-llvm-shims | FileCheck %s --implicit-check-not=!sculptor.logical.array

module {
  // CHECK-LABEL: func.func private @task_setup_0()
  func.func private @task_setup_0() -> !sculptor.logical.array attributes {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64}
  // CHECK-LABEL: func.func private @task_mvm_0(
  // CHECK-SAME: tensor<1x4xf32>
  // CHECK-NOT: !sculptor.logical.array
  func.func private @task_mvm_0(%arg0: tensor<1x4xf32>, %arg1: !sculptor.logical.array) -> tensor<1x3xf32> attributes {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64}
  // CHECK-LABEL: func.func private @task_setup_1()
  func.func private @task_setup_1() -> !sculptor.logical.array attributes {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 1 : i64}
  // CHECK-LABEL: func.func private @task_mvm_1(
  // CHECK-SAME: tensor<1x3xf32>
  // CHECK-NOT: !sculptor.logical.array
  func.func private @task_mvm_1(%arg0: tensor<1x3xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> attributes {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 1 : i64}

  // CHECK-LABEL: func.func private @generate_task_graph()
  // CHECK-SAME: sculptor.runtime.input_slots = [0]
  // CHECK-SAME: sculptor.runtime.output_slots = [1]
  // CHECK-SAME: sculptor.runtime.resource_count = 3 : i64
  // CHECK-SAME: sculptor.runtime.temp_count = 1 : i64
  // CHECK-SAME: sculptor.schedule.logical_array_to_analog_array = [0, 1]
  // CHECK-SAME: sculptor.schedule.num_logical_arrays = 2 : i64
  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {sculptor.schedule.logical_array_to_analog_array = [0, 1], sculptor.schedule.num_logical_arrays = 2 : i64} {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    // CHECK: %[[INPUT:.*]] = sculptor.task_graph.input
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    // CHECK: %[[OUTPUT:.*]] = sculptor.task_graph.output
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array0 = sculptor.task_graph.intermediate %graph {sculptor.schedule.logical_array_index = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    // CHECK: %[[ACTIVATION:.*]] = sculptor.task_graph.intermediate
    // CHECK-SAME: sculptor.runtime.byte_size = 12 : i64
    // CHECK-SAME: sculptor.runtime.slot = 2 : i64
    %activation = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>
    %array1 = sculptor.task_graph.intermediate %graph {sculptor.schedule.logical_array_index = 1 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>

    // CHECK: %[[SETUP0:.*]] = sculptor.task.create
    // CHECK-SAME: @task_setup_0
    // CHECK-SAME: inputs[], outputs[], deps[]
    %setup0 = sculptor.task.create %graph, @task_setup_0, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup_0", source_layer = "linear_0", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // CHECK: %[[MVM0:.*]] = sculptor.task.create
    // CHECK-SAME: @task_mvm_0
    // CHECK-SAME: inputs[%[[INPUT]]], outputs[%[[ACTIVATION]]], deps[%[[SETUP0]]]
    // CHECK-SAME: sculptor.runtime.input_slots = [0]
    // CHECK-SAME: sculptor.runtime.output_slots = [2]
    %mvm0 = sculptor.task.create %graph, @task_mvm_0, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm_0", source_layer = "linear_0", source_task_ordinal = 1, inputs[%input, %array0], outputs[%activation], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.result_indices = [0]} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x3xf32>>) -> !sculptor.task
    // CHECK: %[[SETUP1:.*]] = sculptor.task.create
    // CHECK-SAME: @task_setup_1
    // CHECK-SAME: inputs[], outputs[], deps[]
    %setup1 = sculptor.task.create %graph, @task_setup_1, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup_1", source_layer = "linear_1", source_task_ordinal = 0, inputs[], outputs[%array1], deps[] {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // CHECK: sculptor.task.create
    // CHECK-SAME: @task_mvm_1
    // CHECK-SAME: inputs[%[[ACTIVATION]]], outputs[%[[OUTPUT]]], deps[%[[MVM0]], %[[SETUP1]]]
    // CHECK-SAME: sculptor.runtime.input_slots = [2]
    // CHECK-SAME: sculptor.runtime.output_slots = [1]
    %mvm1 = sculptor.task.create %graph, @task_mvm_1, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm_1", source_layer = "linear_1", source_task_ordinal = 1, inputs[%activation, %array1], outputs[%output], deps[%mvm0] {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 1 : i64, sculptor.runtime.result_indices = [0]} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}
