// RUN: sculptor-mlir-opt --help | FileCheck %s --check-prefix=HELP
// RUN: sculptor-mlir-opt %s --sculptor-export-task-graph-sim-model 2>&1 | FileCheck %s --check-prefix=MISSING-OUTPUT
// RUN: sculptor-mlir-opt %s --sculptor-export-task-graph-sim-model="output=%t.json" > /dev/null
// RUN: FileCheck %s --input-file=%t.json
// RUN: sculptor-mlir-opt %s --sculptor-lower-golem-to-llvm-shims --sculptor-partition-task-graph-by-core | FileCheck %s --check-prefix=PARTITION-ID

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
    %stored = sculptor.array.store %result {sculptor.tile_physical_shape = [2, 2], sculptor.tile_valid_shape = [2, 2]} : !sculptor.array.result -> tensor<1x2xf32>
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
    sculptor.timing.model = #sculptor.timing_model<costModel = "golem-qemu-v1", costModelRevision = 1 : i64, compilerRevision = "test", timingBoundary = "warm", runtimeTaskPolicy = "lowest-local-task-index", runtimeTransmitPolicy = "overlap-ready-tasks", memoryBackend = "native-untimed", analogMVMLatencyNs = 100 : i64, analogIOBitsPerCycle = 256 : i64, analogIOShared = true, digitalClockGHz = 1.000000e+00 : f64, digitalIssueWidth = 2 : i64, digitalVectorBitsPerCycle = 256 : i64, fixedRuntimeDispatchCycles = 8 : i64, fixedTaskEntryCycles = 4 : i64, fixedTaskExitCycles = 4 : i64, networkLinkBitsPerCycle = 32 : i64, networkHopLatencyCycles = 1 : i64, networkPipelined = true, networkLinkWordBits = 32 : i64, protocolWordsPerRoute = 5 : i64, nicInjectionWordsPerCycle = 1 : i64, rxDmaWordsPerCycle = 1 : i64, routingPolicy = "xy">,
    sculptor.timing.network_edges = [
      #sculptor.network_edge_timing<producerTask = 0 : i64, consumerTask = 2 : i64, sourceCore = 0 : i64, destinationCore = 0 : i64, meshHops = 0 : i64, payloadWords = 0 : i64, protocolWords = 0 : i64, transferStartNs = 0.000000e+00 : f64, injectionStartNs = 0.000000e+00 : f64, injectionFinishNs = 0.000000e+00 : f64, routeArrivalNs = 0.000000e+00 : f64, receiveStartNs = 0.000000e+00 : f64, receiveCompleteNs = 0.000000e+00 : f64, transferFinishNs = 0.000000e+00 : f64, networkLatencyNs = 0.000000e+00 : f64, contentionDelayNs = 0.000000e+00 : f64, nicQueueDelayNs = 0.000000e+00 : f64, linkQueueDelayNs = 0.000000e+00 : f64, receiveQueueDelayNs = 0.000000e+00 : f64, causalParentTask = 0 : i64, causalParentEdge = -1 : i64, causalResource = "same-core">,
      #sculptor.network_edge_timing<producerTask = 1 : i64, consumerTask = 2 : i64, sourceCore = 1 : i64, destinationCore = 0 : i64, meshHops = 1 : i64, payloadWords = 2 : i64, protocolWords = 5 : i64, transferStartNs = 0.000000e+00 : f64, injectionStartNs = 0.000000e+00 : f64, injectionFinishNs = 1.000000e+00 : f64, routeArrivalNs = 1.000000e+00 : f64, receiveStartNs = 1.000000e+00 : f64, receiveCompleteNs = 2.000000e+00 : f64, transferFinishNs = 2.000000e+00 : f64, networkLatencyNs = 2.000000e+00 : f64, contentionDelayNs = 0.000000e+00 : f64, nicQueueDelayNs = 0.000000e+00 : f64, linkQueueDelayNs = 0.000000e+00 : f64, receiveQueueDelayNs = 0.000000e+00 : f64, causalParentTask = 1 : i64, causalParentEdge = -1 : i64, causalResource = "producer">,
      #sculptor.network_edge_timing<producerTask = 2 : i64, consumerTask = 3 : i64, sourceCore = 0 : i64, destinationCore = 1 : i64, meshHops = 1 : i64, payloadWords = 2 : i64, protocolWords = 5 : i64, transferStartNs = 1.020000e+02 : f64, injectionStartNs = 1.030000e+02 : f64, injectionFinishNs = 1.040000e+02 : f64, routeArrivalNs = 1.040000e+02 : f64, receiveStartNs = 1.040000e+02 : f64, receiveCompleteNs = 1.050000e+02 : f64, transferFinishNs = 1.050000e+02 : f64, networkLatencyNs = 3.000000e+00 : f64, contentionDelayNs = 1.000000e+00 : f64, nicQueueDelayNs = 1.000000e+00 : f64, linkQueueDelayNs = 0.000000e+00 : f64, receiveQueueDelayNs = 0.000000e+00 : f64, causalParentTask = 2 : i64, causalParentEdge = -1 : i64, causalResource = "source-nic">
    ],
    sculptor.timing.placement_aware = true,
    sculptor.timing.task_count = 4 : i64,
    sculptor.timing.total_data_bytes = 16 : i64,
    sculptor.timing.sum_edge_network_queue_delay_ns = 1.000000e+00 : f64,
    sculptor.timing.sum_edge_network_service_ns = 4.000000e+00 : f64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 0 : i64, sculptor.schedule.logical_array_index = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %tile = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %mvm_out = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>

    %setup = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linear_matrix_tile_0_0", source_layer = "linear_0", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64, sculptor.timing.local_runtime_index = 0 : i64, sculptor.workload.digital_ops = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %vector = sculptor.task.create %graph, @task_vector, domain = "digital", task_kind = "digital.vector_tile", task_name = "linear_vector_tile_0", source_layer = "linear_0", source_task_ordinal = 1, inputs[%input], outputs[%tile], deps[] {sculptor.runtime.core_id = 1 : i64, sculptor.timing.local_runtime_index = 0 : i64, sculptor.workload.digital_ops = 3 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "linear_mvm_0_0", source_layer = "linear_0", source_task_ordinal = 2, inputs[%tile, %array], outputs[%mvm_out], deps[%setup, %vector] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64, sculptor.schedule.island_id = 0 : i64, sculptor.timing.analog_execute_latency_ns = 1.000000e+02 : f64, sculptor.timing.analog_load_latency_ns = 1.000000e+00 : f64, sculptor.timing.analog_store_latency_ns = 1.000000e+00 : f64, sculptor.timing.critical_path_remaining_ns = 1.020000e+02 : f64, sculptor.timing.dependency_depth = 1 : i64, sculptor.timing.earliest_finish_ns = 1.020000e+02 : f64, sculptor.timing.earliest_start_ns = 0.000000e+00 : f64, sculptor.timing.intrinsic_latency_ns = 1.020000e+02 : f64, sculptor.timing.is_critical = true, sculptor.timing.local_runtime_index = 1 : i64, sculptor.timing.topological_index = 2 : i64, sculptor.workload.analog_execution_count = 1 : i64, sculptor.workload.analog_load_bytes = 8 : i64, sculptor.workload.analog_store_bytes = 8 : i64, sculptor.workload.digital_ops = 0 : i64, sculptor.workload.incoming_data_bytes = 8 : i64, sculptor.workload.outgoing_data_bytes = 8 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %bias = sculptor.task.create %graph, @task_bias, domain = "digital", task_kind = "digital.bias_add", task_name = "linear_bias_add", source_layer = "linear_0", source_task_ordinal = 3, inputs[%mvm_out], outputs[%output], deps[%mvm] {sculptor.runtime.core_id = 1 : i64, sculptor.timing.incoming_network_delay_ns = 3.000000e+00 : f64, sculptor.timing.local_runtime_index = 1 : i64, sculptor.workload.digital_ops = 2 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}

// HELP: --sculptor-export-task-graph-sim-model
// MISSING-OUTPUT: expected non-empty output path for sculptor-export-task-graph-sim-model

// The exporter and deployment partitioner use the same deterministic global
// task identity even though partitioned modules are grouped by core.
// PARTITION-ID: module @core_0
// PARTITION-ID: task_name = "linear_matrix_tile_0_0"{{.*}}sculptor.deployment.global_task_id = 0 : i64
// PARTITION-ID: task_name = "linear_mvm_0_0"{{.*}}sculptor.deployment.global_task_id = 2 : i64
// PARTITION-ID: module @core_1
// PARTITION-ID: task_name = "linear_vector_tile_0"{{.*}}sculptor.deployment.global_task_id = 1 : i64
// PARTITION-ID: task_name = "linear_bias_add"{{.*}}sculptor.deployment.global_task_id = 3 : i64

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
// CHECK: "global_task_id": 0
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
// CHECK: "global_task_id": 2
// CHECK: "callee": "task_mvm"
// CHECK: "island_id": 0
// CHECK: "timing": {
// CHECK: "topological_index": 2
// CHECK: "analog_load_bytes": 8
// CHECK: "analog_execution_count": 1
// CHECK: "analog_store_bytes": 8
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
// CHECK: "sum_edge_network_service_ns": 4
// CHECK: "sum_edge_network_queue_delay_ns": 1
// CHECK: "model": {
// CHECK: "cost_model": "golem-qemu-v1"
// CHECK: "cost_model_revision": 1
// CHECK: "compiler_revision": "test"
// CHECK: "timing_boundary": "warm"
// CHECK: "runtime_task_policy": "lowest-local-task-index"
// CHECK: "runtime_transmit_policy": "overlap-ready-tasks"
// CHECK: "memory_backend": "native-untimed"
// CHECK: "analog_mvm_latency_ns": 100
// CHECK: "analog_io_bits_per_cycle": 256
// CHECK: "network_link_word_bits": 32
// CHECK: "protocol_words_per_route": 5
// CHECK: "nic_injection_words_per_cycle": 1
// CHECK: "rx_dma_words_per_cycle": 1
// CHECK: "routing_policy": "xy"
