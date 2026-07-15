// RUN: sculptor-mlir-opt --help | FileCheck %s --check-prefix=HELP
// RUN: sculptor-mlir-opt %s --sculptor-export-task-graph-sim-model 2>&1 | FileCheck %s --check-prefix=MISSING-OUTPUT
// RUN: sculptor-mlir-opt %s --sculptor-export-task-graph-sim-model="output=%t.json" > /dev/null
// RUN: FileCheck %s --input-file=%t.json

module {
  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }
  func.func private @task_vector(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>
  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x2xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x2xf32>
    return %stored : tensor<1x2xf32>
  }
  func.func private @task_bias(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.analog_arrays = [0 : i64, 1 : i64],
    sculptor.schedule.arrays_per_core = 1 : i64,
    sculptor.schedule.core_transfer_bytes = [0 : i64, 8 : i64, 8 : i64, 0 : i64],
    sculptor.schedule.core_transfer_cost = [0 : i64, 8 : i64, 8 : i64, 0 : i64],
    sculptor.schedule.dependency_count = 3 : i64,
    sculptor.schedule.boundary_penalty = 0 : i64,
    sculptor.schedule.graph_score = 16 : i64,
    sculptor.schedule.inter_core_transfer_bytes = 16 : i64,
    sculptor.schedule.logical_array_to_analog_array = [0 : i64],
    sculptor.schedule.mesh_cols = 2 : i64,
    sculptor.schedule.mesh_rows = 1 : i64,
    sculptor.schedule.num_analog_arrays = 2 : i64,
    sculptor.schedule.num_cores = 2 : i64,
    sculptor.schedule.num_logical_arrays = 1 : i64,
    sculptor.schedule.task_count = 4 : i64,
    sculptor.schedule.topology = "mesh",
    sculptor.schedule.total_digital_ops = 5 : i64,
    sculptor.schedule.total_transfer_cost = 16 : i64,
    sculptor.timing.critical_path_ns = 1.020000e+02 : f64,
    sculptor.timing.execution_depth = 3 : i64,
    sculptor.timing.execution_edge_count = 3 : i64,
    sculptor.timing.model = #sculptor.timing_model<analogMVMLatencyNs = 100 : i64, analogIOBitsPerCycle = 256 : i64, analogIOShared = true, digitalClockGHz = 1.000000e+00 : f64, digitalIssueWidth = 2 : i64, digitalVectorBitsPerCycle = 256 : i64, networkLinkBitsPerCycle = 32 : i64, networkHopLatencyCycles = 1 : i64, networkPipelined = true>,
    sculptor.timing.network_edges = [
      #sculptor.network_edge_timing<producerTask = 0 : i64, consumerTask = 2 : i64, sourceCore = 0 : i64, destinationCore = 0 : i64, meshHops = 0 : i64, transferStartNs = 0.000000e+00 : f64, transferFinishNs = 0.000000e+00 : f64, networkLatencyNs = 0.000000e+00 : f64, contentionDelayNs = 0.000000e+00 : f64>,
      #sculptor.network_edge_timing<producerTask = 1 : i64, consumerTask = 2 : i64, sourceCore = 1 : i64, destinationCore = 0 : i64, meshHops = 1 : i64, transferStartNs = 0.000000e+00 : f64, transferFinishNs = 2.000000e+00 : f64, networkLatencyNs = 2.000000e+00 : f64, contentionDelayNs = 0.000000e+00 : f64>,
      #sculptor.network_edge_timing<producerTask = 2 : i64, consumerTask = 3 : i64, sourceCore = 0 : i64, destinationCore = 1 : i64, meshHops = 1 : i64, transferStartNs = 1.020000e+02 : f64, transferFinishNs = 1.050000e+02 : f64, networkLatencyNs = 3.000000e+00 : f64, contentionDelayNs = 1.000000e+00 : f64>
    ],
    sculptor.timing.placement_aware = true,
    sculptor.timing.task_count = 4 : i64,
    sculptor.timing.total_data_bytes = 16 : i64,
    sculptor.timing.total_network_contention_delay_ns = 1.000000e+00 : f64,
    sculptor.timing.total_network_latency_ns = 5.000000e+00 : f64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 1 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 0 : i64, sculptor.runtime.slot = 2 : i64, sculptor.runtime.temp_index = 0 : i64, sculptor.runtime.temp_offset = 0 : i64, sculptor.schedule.logical_array_index = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %tile = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 3 : i64, sculptor.runtime.temp_index = 1 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %mvm_out = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 4 : i64, sculptor.runtime.temp_index = 2 : i64, sculptor.runtime.temp_offset = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>

    %setup = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linear_matrix_tile_0_0", source_layer = "linear_0", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %vector = sculptor.task.create %graph, @task_vector, domain = "digital", task_kind = "digital.vector_tile", task_name = "linear_vector_tile_0", source_layer = "linear_0", source_task_ordinal = 1, inputs[%input], outputs[%tile], deps[] {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.digital_ops = 3 : i64, sculptor.runtime.task_index = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "linear_mvm_0_0", source_layer = "linear_0", source_task_ordinal = 2, inputs[%tile, %array], outputs[%mvm_out], deps[%setup, %vector] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.task_index = 2 : i64, sculptor.schedule.island_id = 0 : i64, sculptor.timing.analog_execute_latency_ns = 1.000000e+02 : f64, sculptor.timing.analog_load_latency_ns = 1.000000e+00 : f64, sculptor.timing.analog_store_latency_ns = 1.000000e+00 : f64, sculptor.timing.critical_path_remaining_ns = 1.020000e+02 : f64, sculptor.timing.dependency_depth = 1 : i64, sculptor.timing.earliest_finish_ns = 1.020000e+02 : f64, sculptor.timing.earliest_start_ns = 0.000000e+00 : f64, sculptor.timing.incoming_data_bytes = 8 : i64, sculptor.timing.intrinsic_latency_ns = 1.020000e+02 : f64, sculptor.timing.is_critical = true, sculptor.timing.outgoing_data_bytes = 8 : i64, sculptor.timing.topological_index = 2 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %bias = sculptor.task.create %graph, @task_bias, domain = "digital", task_kind = "digital.bias_add", task_name = "linear_bias_add", source_layer = "linear_0", source_task_ordinal = 3, inputs[%mvm_out], outputs[%output], deps[%mvm] {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.digital_ops = 2 : i64, sculptor.runtime.task_index = 3 : i64, sculptor.timing.incoming_network_delay_ns = 3.000000e+00 : f64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}

// HELP: --sculptor-export-task-graph-sim-model
// MISSING-OUTPUT: expected non-empty output path for sculptor-export-task-graph-sim-model

// CHECK: "schema_version": 1
// CHECK: "format": "sculptor.task_graph.sim_model"
// CHECK: "graphs": [
// CHECK: "name": "generate_task_graph"
// CHECK: "hardware": {
// CHECK: "topology": "mesh"
// CHECK: "num_cores": 2
// CHECK: "arrays_per_core": 1
// CHECK: "mesh_rows": 1
// CHECK: "mesh_cols": 2
// CHECK: "num_analog_arrays": 2
// CHECK: "resources": [
// CHECK: "kind": "input"
// CHECK: "value_type": "tensor<1x2xf32>"
// CHECK: "byte_size": 8
// CHECK: "kind": "intermediate"
// CHECK: "value_type": "!sculptor.logical.array"
// CHECK: "logical_array_index": 0
// CHECK: "physical_array_id": 0
// CHECK: "local_array_id": 0
// CHECK: "tasks": [
// CHECK: "index": 0
// CHECK: "callee": "task_matrix"
// CHECK: "kind": "sculptor.matrix_setup"
// CHECK: "physical_array_id": 0
// CHECK: "local_array_id": 0
// CHECK: "analog_ops": [
// CHECK: "name": "sculptor.array.set"
// CHECK: "analog_op_counts": [
// CHECK: "name": "sculptor.array.set"
// CHECK: "count": 1
// CHECK: "index": 2
// CHECK: "callee": "task_mvm"
// CHECK: "island_id": 0
// CHECK: "timing": {
// CHECK: "topological_index": 2
// CHECK: "analog_load_latency_ns": 1
// CHECK: "analog_execute_latency_ns": 100
// CHECK: "analog_store_latency_ns": 1
// CHECK: "intrinsic_latency_ns": 102
// CHECK: "is_critical": true
// CHECK: "analog_ops": [
// CHECK: "name": "sculptor.array.load"
// CHECK: "name": "sculptor.array.execute"
// CHECK: "name": "sculptor.array.store"
// CHECK: "analog_op_counts": [
// CHECK: "name": "sculptor.array.load"
// CHECK: "count": 1
// CHECK: "name": "sculptor.array.execute"
// CHECK: "count": 1
// CHECK: "name": "sculptor.array.store"
// CHECK: "count": 1
// CHECK: "dependencies": [
// CHECK: 0
// CHECK: 1
// CHECK: "control_edges": [
// CHECK: "producer_task": 0
// CHECK: "consumer_task": 2
// CHECK: "data_edges": [
// CHECK: "producer_task": 1
// CHECK: "consumer_task": 2
// CHECK: "resource": 3
// CHECK: "byte_size": 8
// CHECK: "source_core": 1
// CHECK: "destination_core": 0
// CHECK: "mesh_distance": 1
// CHECK: "transfer_cost": 8
// CHECK: "inter_core": true
// CHECK: "network_hops": 1
// CHECK: "network_transfer_finish_ns": 2
// CHECK: "network_latency_ns": 2
// CHECK: "network_contention_delay_ns": 0
// CHECK: "summary": {
// CHECK: "task_count": 4
// CHECK: "dependency_count": 3
// CHECK: "inter_core_transfer_bytes": 16
// CHECK: "total_transfer_cost": 16
// CHECK: "total_digital_ops": 5
// CHECK: "num_logical_arrays": 1
// CHECK: "logical_array_to_analog_array": [
// CHECK: 0
// CHECK: "timing": {
// CHECK: "task_count": 4
// CHECK: "execution_edge_count": 3
// CHECK: "execution_depth": 3
// CHECK: "total_data_bytes": 16
// CHECK: "critical_path_ns": 102
// CHECK: "placement_aware": true
// CHECK: "total_network_latency_ns": 5
// CHECK: "total_network_contention_delay_ns": 1
// CHECK: "model": {
// CHECK: "analog_mvm_latency_ns": 100
// CHECK: "analog_io_bits_per_cycle": 256
