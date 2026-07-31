// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-analyze-task-graph-timing | FileCheck %s

module {
  func.func private @produce() -> tensor<1x1xf32> {
    %value = arith.constant dense<1.0> : tensor<1x1xf32>
    return %value : tensor<1x1xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.mesh_cols = 1 : i64,
    sculptor.schedule.mesh_rows = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %output0 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %output1 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %first = sculptor.task.create %graph, @produce, domain = "digital", task_kind = "digital.producer", task_name = "cpu.first", source_layer = "cpu", source_task_ordinal = 0, inputs[], outputs[%output0], deps[] {sculptor.runtime.core_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %second = sculptor.task.create %graph, @produce, domain = "digital", task_kind = "digital.producer", task_name = "cpu.second", source_layer = "cpu", source_task_ordinal = 1, inputs[], outputs[%output1], deps[] {sculptor.runtime.core_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: task_name = "cpu.first"
// CHECK-SAME: sculptor.timing.earliest_start_ns = 0.000000e+00 : f64
// CHECK-SAME: sculptor.timing.local_runtime_index = 0 : i64
// CHECK-LABEL: task_name = "cpu.second"
// CHECK-SAME: sculptor.timing.causal_previous_task = 0 : i64
// CHECK-SAME: sculptor.timing.core_queue_delay_ns = 2.000000e+01 : f64
// CHECK-SAME: sculptor.timing.earliest_start_ns = 2.000000e+01 : f64
// CHECK-SAME: sculptor.timing.local_runtime_index = 1 : i64

// -----

module {
  func.func private @produce() -> tensor<1x64xf32> {
    %value = arith.constant dense<1.0> : tensor<1x64xf32>
    return %value : tensor<1x64xf32>
  }
  func.func private @consume(%arg0: tensor<1x64xf32>) -> tensor<1x64xf32> {
    return %arg0 : tensor<1x64xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.mesh_cols = 3 : i64,
    sculptor.schedule.mesh_rows = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %value0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x64xf32>>
    %output0 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x64xf32>>
    %output1 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x64xf32>>
    %producer0 = sculptor.task.create %graph, @produce, domain = "digital", task_kind = "digital.producer", task_name = "nic.producer0", source_layer = "nic", source_task_ordinal = 0, inputs[], outputs[%value0], deps[] {sculptor.runtime.core_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x64xf32>>) -> !sculptor.task
    %consumer0 = sculptor.task.create %graph, @consume, domain = "digital", task_kind = "digital.consumer", task_name = "nic.consumer0", source_layer = "nic", source_task_ordinal = 1, inputs[%value0], outputs[%output0], deps[] {sculptor.runtime.core_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x64xf32>>, !sculptor.task_resource<tensor<1x64xf32>>) -> !sculptor.task
    %consumer1 = sculptor.task.create %graph, @consume, domain = "digital", task_kind = "digital.consumer", task_name = "nic.consumer1", source_layer = "nic", source_task_ordinal = 2, inputs[%value0], outputs[%output1], deps[] {sculptor.runtime.core_id = 2 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x64xf32>>, !sculptor.task_resource<tensor<1x64xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-DAG: sculptor.timing.sum_nic_queue_delay_ns = 6.900000e+01 : f64
// CHECK-DAG: sculptor.timing.total_payload_words = 128 : i64
// CHECK-DAG: sculptor.timing.total_protocol_words = 10 : i64
// CHECK-DAG: nicQueueDelayNs = 6.900000e+01 : f64
// CHECK-DAG: causalResource = "source-nic:0"

// -----

module {
  func.func private @produce() -> tensor<1x64xf32> {
    %value = arith.constant dense<1.0> : tensor<1x64xf32>
    return %value : tensor<1x64xf32>
  }
  func.func private @consume(%arg0: tensor<1x64xf32>) -> tensor<1x64xf32> {
    return %arg0 : tensor<1x64xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.mesh_cols = 4 : i64,
    sculptor.schedule.mesh_rows = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %value0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x64xf32>>
    %value1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x64xf32>>
    %output0 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x64xf32>>
    %output1 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x64xf32>>
    %producer0 = sculptor.task.create %graph, @produce, domain = "digital", task_kind = "digital.producer", task_name = "link.producer0", source_layer = "link", source_task_ordinal = 0, inputs[], outputs[%value0], deps[] {sculptor.runtime.core_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x64xf32>>) -> !sculptor.task
    %producer1 = sculptor.task.create %graph, @produce, domain = "digital", task_kind = "digital.producer", task_name = "link.producer1", source_layer = "link", source_task_ordinal = 1, inputs[], outputs[%value1], deps[] {sculptor.runtime.core_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x64xf32>>) -> !sculptor.task
    %consumer0 = sculptor.task.create %graph, @consume, domain = "digital", task_kind = "digital.consumer", task_name = "link.consumer0", source_layer = "link", source_task_ordinal = 2, inputs[%value0], outputs[%output0], deps[] {sculptor.runtime.core_id = 3 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x64xf32>>, !sculptor.task_resource<tensor<1x64xf32>>) -> !sculptor.task
    %consumer1 = sculptor.task.create %graph, @consume, domain = "digital", task_kind = "digital.consumer", task_name = "link.consumer1", source_layer = "link", source_task_ordinal = 3, inputs[%value1], outputs[%output1], deps[] {sculptor.runtime.core_id = 3 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x64xf32>>, !sculptor.task_resource<tensor<1x64xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-DAG: sculptor.timing.sum_link_queue_delay_ns = 7.000000e+01 : f64
// CHECK-DAG: linkQueueDelayNs = 7.000000e+01 : f64
// CHECK-DAG: causalResource = "directed-link:1->2"

// -----

module {
  func.func private @produce() -> tensor<1x64xf32> {
    %value = arith.constant dense<1.0> : tensor<1x64xf32>
    return %value : tensor<1x64xf32>
  }
  func.func private @join(%arg0: tensor<1x64xf32>, %arg1: tensor<1x64xf32>) -> tensor<1x64xf32> {
    return %arg0 : tensor<1x64xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.mesh_cols = 3 : i64,
    sculptor.schedule.mesh_rows = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %value0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x64xf32>>
    %value1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x64xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x64xf32>>
    %producer0 = sculptor.task.create %graph, @produce, domain = "digital", task_kind = "digital.producer", task_name = "rx.producer0", source_layer = "rx", source_task_ordinal = 0, inputs[], outputs[%value0], deps[] {sculptor.runtime.core_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x64xf32>>) -> !sculptor.task
    %producer1 = sculptor.task.create %graph, @produce, domain = "digital", task_kind = "digital.producer", task_name = "rx.producer1", source_layer = "rx", source_task_ordinal = 1, inputs[], outputs[%value1], deps[] {sculptor.runtime.core_id = 2 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x64xf32>>) -> !sculptor.task
    %consumer = sculptor.task.create %graph, @join, domain = "digital", task_kind = "digital.join", task_name = "rx.consumer", source_layer = "rx", source_task_ordinal = 2, inputs[%value0, %value1], outputs[%output], deps[] {sculptor.runtime.core_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x64xf32>>, !sculptor.task_resource<tensor<1x64xf32>>, !sculptor.task_resource<tensor<1x64xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-DAG: sculptor.timing.sum_receive_queue_delay_ns = 6.900000e+01 : f64
// CHECK-DAG: receiveQueueDelayNs = 6.900000e+01 : f64
// CHECK-DAG: causalResource = "receive-dma:1"
// CHECK-LABEL: task_name = "rx.consumer"
// CHECK-SAME: sculptor.timing.causal_input_edge = 1 : i64
