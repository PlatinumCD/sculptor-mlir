// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="network-link-bits-per-cycle=32 network-hop-latency-cycles=1 network-pipelined=true" | FileCheck %s --check-prefix=PIPELINED
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="network-link-bits-per-cycle=32 network-hop-latency-cycles=1 network-pipelined=false" | FileCheck %s --check-prefix=NONPIPELINED

module {
  func.func private @task_producer() -> tensor<1x4xf32> {
    %value = arith.constant dense<1.0> : tensor<1x4xf32>
    return %value : tensor<1x4xf32>
  }

  func.func private @task_consumer(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.mesh_cols = 4 : i64,
    sculptor.schedule.mesh_rows = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %intermediate = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %producer = sculptor.task.create %graph, @task_producer, domain = "digital", task_kind = "digital.producer", task_name = "producer", source_layer = "test", source_task_ordinal = 0, inputs[], outputs[%intermediate], deps[] {sculptor.runtime.core_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>) -> !sculptor.task
    %consumer = sculptor.task.create %graph, @task_consumer, domain = "digital", task_kind = "digital.consumer", task_name = "consumer", source_layer = "test", source_task_ordinal = 1, inputs[%intermediate], outputs[%output], deps[] {sculptor.runtime.core_id = 3 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// Four 32-bit flits crossing three hops cost 4 + 3 - 1 = 6 cycles when
// pipelined, and 3 * (4 + 1 - 1) = 12 cycles when not pipelined.
// PIPELINED-LABEL: func.func private @generate_task_graph
// PIPELINED-SAME: sculptor.timing.critical_path_ns = 6.000000e+00 : f64
// PIPELINED-SAME: meshHops = 3 : i64
// PIPELINED-SAME: networkLatencyNs = 6.000000e+00 : f64
// PIPELINED-SAME: sculptor.timing.total_network_latency_ns = 6.000000e+00 : f64

// NONPIPELINED-LABEL: func.func private @generate_task_graph
// NONPIPELINED-SAME: sculptor.timing.critical_path_ns = 1.200000e+01 : f64
// NONPIPELINED-SAME: meshHops = 3 : i64
// NONPIPELINED-SAME: networkLatencyNs = 1.200000e+01 : f64
// NONPIPELINED-SAME: sculptor.timing.total_network_latency_ns = 1.200000e+01 : f64
