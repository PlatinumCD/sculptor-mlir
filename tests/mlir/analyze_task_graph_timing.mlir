// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="analog-mvm-latency-ns=75 analog-io-bits-per-cycle=128 analog-io-shared=false digital-clock-ghz=1.5 digital-issue-width=4 digital-vector-bits-per-cycle=512 network-link-bits-per-cycle=64 network-hop-latency-cycles=2 network-pipelined=false" | FileCheck %s

module {
  func.func private @task_setup() -> !sculptor.logical.array {
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

  func.func private @task_digital(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @task_join(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %mvm_out = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %left_out = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %right_out = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "test", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm", source_layer = "test", source_task_ordinal = 1, inputs[%input, %array], outputs[%mvm_out], deps[%setup] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %left = sculptor.task.create %graph, @task_digital, domain = "digital", task_kind = "digital.compute", task_name = "left", source_layer = "test", source_task_ordinal = 2, inputs[%mvm_out], outputs[%left_out], deps[%mvm] {sculptor.runtime.digital_ops = 16 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %right = sculptor.task.create %graph, @task_digital, domain = "digital", task_kind = "digital.compute", task_name = "right", source_layer = "test", source_task_ordinal = 3, inputs[%mvm_out], outputs[%right_out], deps[%mvm] {sculptor.runtime.digital_ops = 32 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    // The right branch is a data predecessor without a duplicate control edge.
    %join = sculptor.task.create %graph, @task_join, domain = "digital", task_kind = "digital.compute", task_name = "join", source_layer = "test", source_task_ordinal = 4, inputs[%left_out, %right_out], outputs[%output], deps[%left] {sculptor.runtime.digital_ops = 16 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph()
// CHECK-SAME: sculptor.timing.control_edge_count = 4 : i64
// CHECK-SAME: sculptor.timing.critical_path_ns = 78.333333333333343 : f64
// CHECK-SAME: sculptor.timing.data_edge_count = 5 : i64
// CHECK-SAME: sculptor.timing.execution_depth = 4 : i64
// CHECK-SAME: sculptor.timing.execution_edge_count = 5 : i64
// CHECK-SAME: sculptor.timing.islands = [#sculptor.island_timing<islandId = 0 : i64, taskCount = 5 : i64
// CHECK-SAME: digitalWorkNs = 2.6666666666666665 : f64
// CHECK-SAME: earliestStartNs = 0.000000e+00 : f64
// CHECK-SAME: slackNs = 0.000000e+00 : f64, isCritical = true>]
// CHECK-SAME: sculptor.timing.model = #sculptor.timing_model<
// CHECK-SAME: analogIOShared = false
// CHECK-SAME: networkPipelined = false>
// CHECK-SAME: sculptor.timing.task_count = 5 : i64
// CHECK-SAME: sculptor.timing.total_data_bytes = 32 : i64

// CHECK: task_name = "setup"
// CHECK-SAME: sculptor.timing.dependency_depth = 0 : i64
// CHECK-SAME: sculptor.timing.intrinsic_latency_ns = 0.000000e+00 : f64
// CHECK-SAME: sculptor.timing.topological_index = 0 : i64

// CHECK: task_name = "mvm"
// CHECK-SAME: sculptor.timing.analog_execute_latency_ns = 7.500000e+01 : f64
// CHECK-SAME: sculptor.timing.analog_load_latency_ns = 0.66666666666666663 : f64
// CHECK-SAME: sculptor.timing.analog_store_latency_ns = 0.66666666666666663 : f64
// CHECK-SAME: sculptor.timing.intrinsic_latency_ns = 76.333333333333343 : f64
// CHECK-SAME: sculptor.timing.is_critical = true
// CHECK-SAME: sculptor.timing.slack_ns = 0.000000e+00 : f64

// CHECK: task_name = "right"
// CHECK-SAME: sculptor.timing.earliest_finish_ns = 77.666666666666671 : f64
// CHECK-SAME: sculptor.timing.is_critical = true

// CHECK: task_name = "join"
// CHECK-SAME: sculptor.timing.control_predecessor_count = 1 : i64
// CHECK-SAME: sculptor.timing.data_predecessor_count = 2 : i64
// CHECK-SAME: sculptor.timing.dependency_depth = 3 : i64
// CHECK-SAME: sculptor.timing.earliest_start_ns = 77.666666666666671 : f64
// CHECK-SAME: sculptor.timing.topological_index = 4 : i64
