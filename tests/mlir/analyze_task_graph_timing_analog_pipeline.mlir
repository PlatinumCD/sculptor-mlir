// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing | FileCheck %s

module {
  func.func private @task_setup() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<8x8xf32>
    %array = sculptor.array.set %matrix : tensor<8x8xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<1x8xf32>, %arg1: !sculptor.logical.array) -> tensor<1x8xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x8xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x8xf32>
    return %stored : tensor<1x8xf32>
  }

  func.func private @task_join(%arg0: tensor<1x8xf32>, %arg1: tensor<1x8xf32>) -> tensor<1x8xf32> {
    return %arg0 : tensor<1x8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input0 = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %input1 = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %mvm_out0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %mvm_out1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %array0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>

    %setup0 = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup0", source_layer = "left", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup1", source_layer = "right", source_task_ordinal = 0, inputs[], outputs[%array1], deps[] {sculptor.schedule.island_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm0 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm0", source_layer = "left", source_task_ordinal = 1, inputs[%input0, %array0], outputs[%mvm_out0], deps[%setup0] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task) -> !sculptor.task
    %mvm1 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm1", source_layer = "right", source_task_ordinal = 1, inputs[%input1, %array1], outputs[%mvm_out1], deps[%setup1] {sculptor.schedule.island_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task) -> !sculptor.task
    %join = sculptor.task.create %graph, @task_join, domain = "digital", task_kind = "digital.compute", task_name = "join", source_layer = "join", source_task_ordinal = 0, inputs[%mvm_out0, %mvm_out1], outputs[%output], deps[%mvm0, %mvm1] {sculptor.runtime.digital_ops = 8 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph()
// CHECK-SAME: sculptor.timing.critical_path_ns = 1.030000e+02 : f64
// CHECK-SAME: sculptor.timing.islands = [#sculptor.island_timing<islandId = 0
// CHECK-SAME: #sculptor.island_timing<islandId = 1
// CHECK-SAME: sculptor.timing.total_data_bytes = 64 : i64

// Both independent arrays start at zero and overlap in the DAG. Each array's
// own load, execute, and store phases remain sequential and sum to 102 ns.
// CHECK: task_name = "mvm0"
// CHECK-SAME: sculptor.timing.analog_execute_latency_ns = 1.000000e+02 : f64
// CHECK-SAME: sculptor.timing.analog_load_latency_ns = 1.000000e+00 : f64
// CHECK-SAME: sculptor.timing.analog_store_latency_ns = 1.000000e+00 : f64
// CHECK-SAME: sculptor.timing.earliest_finish_ns = 1.020000e+02 : f64
// CHECK-SAME: sculptor.timing.earliest_start_ns = 0.000000e+00 : f64
// CHECK-SAME: sculptor.timing.intrinsic_latency_ns = 1.020000e+02 : f64

// CHECK: task_name = "mvm1"
// CHECK-SAME: sculptor.timing.earliest_finish_ns = 1.020000e+02 : f64
// CHECK-SAME: sculptor.timing.earliest_start_ns = 0.000000e+00 : f64
// CHECK-SAME: sculptor.timing.intrinsic_latency_ns = 1.020000e+02 : f64

// CHECK: task_name = "join"
// CHECK-SAME: sculptor.timing.earliest_finish_ns = 1.030000e+02 : f64
// CHECK-SAME: sculptor.timing.earliest_start_ns = 1.020000e+02 : f64
