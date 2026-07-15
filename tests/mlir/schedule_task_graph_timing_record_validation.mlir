// RUN: not sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" 2>&1 | FileCheck %s

module {
  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.0> : tensor<1x1xf32>
    %array = sculptor.array.set %matrix : tensor<1x1xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.timing.critical_path_ns = 0.000000e+00 : f64,
    sculptor.timing.island_edges = [],
    sculptor.timing.islands = ["not-a-timing-record"],
    sculptor.timing.model = #sculptor.timing_model<analogMVMLatencyNs = 100 : i64, analogIOBitsPerCycle = 256 : i64, analogIOShared = true, digitalClockGHz = 1.000000e+00 : f64, digitalIssueWidth = 2 : i64, digitalVectorBitsPerCycle = 256 : i64, networkLinkBitsPerCycle = 32 : i64, networkHopLatencyCycles = 1 : i64, networkPipelined = true>,
    sculptor.timing.placement_aware = false,
    sculptor.timing.task_count = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %array = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix", source_layer = "test", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.timing.analog_execute_latency_ns = 0.000000e+00 : f64, sculptor.timing.analog_load_latency_ns = 0.000000e+00 : f64, sculptor.timing.analog_store_latency_ns = 0.000000e+00 : f64, sculptor.timing.control_predecessor_count = 0 : i64, sculptor.timing.critical_path_remaining_ns = 0.000000e+00 : f64, sculptor.timing.data_predecessor_count = 0 : i64, sculptor.timing.dependency_depth = 0 : i64, sculptor.timing.digital_ops = 0 : i64, sculptor.timing.earliest_finish_ns = 0.000000e+00 : f64, sculptor.timing.earliest_start_ns = 0.000000e+00 : f64, sculptor.timing.incoming_data_bytes = 0 : i64, sculptor.timing.incoming_network_delay_ns = 0.000000e+00 : f64, sculptor.timing.intrinsic_latency_ns = 0.000000e+00 : f64, sculptor.timing.is_critical = true, sculptor.timing.outgoing_data_bytes = 0 : i64, sculptor.timing.slack_ns = 0.000000e+00 : f64, sculptor.timing.topological_index = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK: expected 'sculptor.timing.islands' to contain #sculptor.island_timing records
// CHECK: failed to load pre-placement scheduling timing profile
