// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing | FileCheck %s

module {
  func.func private @task_setup() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.0> : tensor<1x1xf32>
    %array = sculptor.array.set %matrix : tensor<1x1xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_producer() -> tensor<1x4xf32> {
    %value = arith.constant dense<1.0> : tensor<1x4xf32>
    return %value : tensor<1x4xf32>
  }

  func.func private @task_consumer(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
      sculptor.schedule.mesh_cols = 3 : i64,
      sculptor.schedule.mesh_rows = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %array0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %value0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %value1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %result0 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %result1 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>

    %setup0 = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup0", source_layer = "source", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup1", source_layer = "destination", source_task_ordinal = 0, inputs[], outputs[%array1], deps[] {sculptor.runtime.core_id = 2 : i64, sculptor.schedule.island_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %producer0 = sculptor.task.create %graph, @task_producer, domain = "digital", task_kind = "digital.compute", task_name = "producer0", source_layer = "source", source_task_ordinal = 1, inputs[], outputs[%value0], deps[%setup0] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task
    %producer1 = sculptor.task.create %graph, @task_producer, domain = "digital", task_kind = "digital.compute", task_name = "producer1", source_layer = "source", source_task_ordinal = 2, inputs[], outputs[%value1], deps[%setup0] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task
    %consumer0 = sculptor.task.create %graph, @task_consumer, domain = "digital", task_kind = "digital.compute", task_name = "consumer0", source_layer = "destination", source_task_ordinal = 1, inputs[%value0], outputs[%result0], deps[%producer0] {sculptor.runtime.core_id = 2 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.schedule.island_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task
    %consumer1 = sculptor.task.create %graph, @task_consumer, domain = "digital", task_kind = "digital.compute", task_name = "consumer1", source_layer = "destination", source_task_ordinal = 2, inputs[%value1], outputs[%result1], deps[%producer1] {sculptor.runtime.core_id = 2 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.schedule.island_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph()
// CHECK-SAME: sculptor.timing.critical_path_ns = 9.000000e+00 : f64
// CHECK-SAME: sculptor.timing.network_edges = [
// CHECK-SAME: meshHops = 0 : i64
// CHECK-SAME: meshHops = 0 : i64
// CHECK-SAME: meshHops = 2 : i64
// CHECK-SAME: networkLatencyNs = 5.000000e+00 : f64
// CHECK-SAME: meshHops = 2 : i64
// CHECK-SAME: networkLatencyNs = 9.000000e+00 : f64, contentionDelayNs = 4.000000e+00 : f64>
// CHECK-SAME: sculptor.timing.placement_aware = true
// CHECK-SAME: sculptor.timing.total_network_contention_delay_ns = 4.000000e+00 : f64
// CHECK-SAME: sculptor.timing.total_network_latency_ns = 1.400000e+01 : f64

// CHECK: task_name = "consumer0"
// CHECK-SAME: sculptor.timing.earliest_start_ns = 5.000000e+00 : f64
// CHECK-SAME: sculptor.timing.incoming_network_delay_ns = 5.000000e+00 : f64

// CHECK: task_name = "consumer1"
// CHECK-SAME: sculptor.timing.earliest_start_ns = 9.000000e+00 : f64
// CHECK-SAME: sculptor.timing.incoming_network_delay_ns = 9.000000e+00 : f64
