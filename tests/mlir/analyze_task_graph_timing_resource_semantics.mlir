// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-analyze-task-graph-timing="digital-issue-width=8 fixed-runtime-dispatch-cycles=0 fixed-task-entry-cycles=0 fixed-task-exit-cycles=0" | FileCheck %s

module {
  func.func private @produce() -> tensor<1x1xf32> {
    %value = arith.constant dense<1.0> : tensor<1x1xf32>
    return %value : tensor<1x1xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.mesh_cols = 2 : i64,
    sculptor.schedule.mesh_rows = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %output0 = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %output1 = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %first = sculptor.task.create %graph, @produce, domain = "digital",
        task_kind = "digital.producer", task_name = "independent.first",
        source_layer = "independent", source_task_ordinal = 0,
        inputs[], outputs[%output0], deps[]
        {sculptor.runtime.core_id = 0 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %second = sculptor.task.create %graph, @produce, domain = "digital",
        task_kind = "digital.producer", task_name = "independent.second",
        source_layer = "independent", source_task_ordinal = 1,
        inputs[], outputs[%output1], deps[]
        {sculptor.runtime.core_id = 1 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// Two independent one-cycle tasks on separate cores overlap.
// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.timing.critical_path_ns = 1.000000e+00 : f64
// CHECK-SAME: sculptor.timing.sum_core_queue_delay_ns = 0.000000e+00 : f64
// CHECK-SAME: sculptor.timing.sum_task_work_ns = 2.000000e+00 : f64

// -----

module {
  func.func private @produce() -> tensor<1x1xf32> {
    %value = arith.constant dense<1.0> : tensor<1x1xf32>
    return %value : tensor<1x1xf32>
  }
  func.func private @consume(%input: tensor<1x1xf32>) -> tensor<1x1xf32> {
    return %input : tensor<1x1xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.mesh_cols = 2 : i64,
    sculptor.schedule.mesh_rows = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %value0 = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %value1 = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %output0 = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %output1 = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %producer0 = sculptor.task.create %graph, @produce, domain = "digital",
        task_kind = "digital.producer", task_name = "opposite.producer0",
        source_layer = "opposite", source_task_ordinal = 0,
        inputs[], outputs[%value0], deps[]
        {sculptor.runtime.core_id = 0 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %producer1 = sculptor.task.create %graph, @produce, domain = "digital",
        task_kind = "digital.producer", task_name = "opposite.producer1",
        source_layer = "opposite", source_task_ordinal = 1,
        inputs[], outputs[%value1], deps[]
        {sculptor.runtime.core_id = 1 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %consumer0 = sculptor.task.create %graph, @consume, domain = "digital",
        task_kind = "digital.consumer", task_name = "opposite.consumer0",
        source_layer = "opposite", source_task_ordinal = 2,
        inputs[%value0], outputs[%output0], deps[]
        {sculptor.runtime.core_id = 1 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %consumer1 = sculptor.task.create %graph, @consume, domain = "digital",
        task_kind = "digital.consumer", task_name = "opposite.consumer1",
        source_layer = "opposite", source_task_ordinal = 3,
        inputs[%value1], outputs[%output1], deps[]
        {sculptor.runtime.core_id = 0 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// Opposite directed channels on the same physical neighbor do not contend.
// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.timing.exposed_contention_ns = 0.000000e+00 : f64
// CHECK-SAME: sculptor.timing.sum_link_queue_delay_ns = 0.000000e+00 : f64
// CHECK-SAME: sculptor.timing.sum_nic_queue_delay_ns = 0.000000e+00 : f64
// CHECK-SAME: sculptor.timing.sum_receive_queue_delay_ns = 0.000000e+00 : f64

// -----

module {
  func.func private @produce() -> tensor<1x1xf32> {
    %value = arith.constant dense<1.0> : tensor<1x1xf32>
    return %value : tensor<1x1xf32>
  }
  func.func private @consume(%input: tensor<1x1xf32>) -> tensor<1x1xf32> {
    return %input : tensor<1x1xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.mesh_cols = 2 : i64,
    sculptor.schedule.mesh_rows = 2 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %value0 = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %value1 = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %output0 = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %output1 = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %producer0 = sculptor.task.create %graph, @produce, domain = "digital",
        task_kind = "digital.producer", task_name = "disjoint.producer0",
        source_layer = "disjoint", source_task_ordinal = 0,
        inputs[], outputs[%value0], deps[]
        {sculptor.runtime.core_id = 0 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %producer1 = sculptor.task.create %graph, @produce, domain = "digital",
        task_kind = "digital.producer", task_name = "disjoint.producer1",
        source_layer = "disjoint", source_task_ordinal = 1,
        inputs[], outputs[%value1], deps[]
        {sculptor.runtime.core_id = 2 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %consumer0 = sculptor.task.create %graph, @consume, domain = "digital",
        task_kind = "digital.consumer", task_name = "disjoint.consumer0",
        source_layer = "disjoint", source_task_ordinal = 2,
        inputs[%value0], outputs[%output0], deps[]
        {sculptor.runtime.core_id = 1 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %consumer1 = sculptor.task.create %graph, @consume, domain = "digital",
        task_kind = "digital.consumer", task_name = "disjoint.consumer1",
        source_layer = "disjoint", source_task_ordinal = 3,
        inputs[%value1], outputs[%output1], deps[]
        {sculptor.runtime.core_id = 3 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// Simultaneous routes over disjoint paths have no queueing.
// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.timing.sum_edge_network_queue_delay_ns = 0.000000e+00 : f64
// CHECK-SAME: sculptor.timing.sum_link_queue_delay_ns = 0.000000e+00 : f64
// CHECK-SAME: sculptor.timing.sum_nic_queue_delay_ns = 0.000000e+00 : f64
// CHECK-SAME: sculptor.timing.sum_receive_queue_delay_ns = 0.000000e+00 : f64

// -----

module {
  func.func private @produce() -> tensor<1x1xf32> {
    %value = arith.constant dense<1.0> : tensor<1x1xf32>
    return %value : tensor<1x1xf32>
  }
  func.func private @consume(%input: tensor<1x1xf32>) -> tensor<1x1xf32> {
    return %input : tensor<1x1xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.mesh_cols = 3 : i64,
    sculptor.schedule.mesh_rows = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %value0 = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %value1 = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %delay_output = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %output0 = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %output1 = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %producer0 = sculptor.task.create %graph, @produce, domain = "digital",
        task_kind = "digital.producer", task_name = "nonoverlap.producer0",
        source_layer = "nonoverlap", source_task_ordinal = 0,
        inputs[], outputs[%value0], deps[]
        {sculptor.runtime.core_id = 0 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %delay = sculptor.task.create %graph, @produce, domain = "digital",
        task_kind = "digital.delay", task_name = "nonoverlap.delay",
        source_layer = "nonoverlap", source_task_ordinal = 1,
        inputs[], outputs[%delay_output], deps[]
        {sculptor.runtime.core_id = 1 : i64,
         sculptor.runtime.digital_ops = 256 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %producer1 = sculptor.task.create %graph, @produce, domain = "digital",
        task_kind = "digital.producer", task_name = "nonoverlap.producer1",
        source_layer = "nonoverlap", source_task_ordinal = 2,
        inputs[], outputs[%value1], deps[%delay]
        {sculptor.runtime.core_id = 1 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>,
           !sculptor.task) -> !sculptor.task
    %consumer0 = sculptor.task.create %graph, @consume, domain = "digital",
        task_kind = "digital.consumer", task_name = "nonoverlap.consumer0",
        source_layer = "nonoverlap", source_task_ordinal = 3,
        inputs[%value0], outputs[%output0], deps[]
        {sculptor.runtime.core_id = 2 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %consumer1 = sculptor.task.create %graph, @consume, domain = "digital",
        task_kind = "digital.consumer", task_name = "nonoverlap.consumer1",
        source_layer = "nonoverlap", source_task_ordinal = 4,
        inputs[%value1], outputs[%output1], deps[]
        {sculptor.runtime.core_id = 2 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// Both routes use directed link 1->2, but their reservations do not overlap.
// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-DAG: sculptor.timing.exposed_contention_ns = 0.000000e+00 : f64
// CHECK-DAG: sculptor.timing.sum_link_queue_delay_ns = 0.000000e+00 : f64
// CHECK-DAG: linkQueueDelayNs = 0.000000e+00 : f64
// CHECK-DAG: linkQueueDelayNs = 0.000000e+00 : f64

// -----

module {
  func.func private @produce() -> tensor<1x1xf32> {
    %value = arith.constant dense<1.0> : tensor<1x1xf32>
    return %value : tensor<1x1xf32>
  }
  func.func private @join(%left: tensor<1x1xf32>,
                          %right: tensor<1x1xf32>) -> tensor<1x1xf32> {
    return %left : tensor<1x1xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.mesh_cols = 4 : i64,
    sculptor.schedule.mesh_rows = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %value_a = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %value_b = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %output = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>
    %producer_a = sculptor.task.create %graph, @produce, domain = "digital",
        task_kind = "digital.producer", task_name = "causal.producer_a",
        source_layer = "causal", source_task_ordinal = 0,
        inputs[], outputs[%value_a], deps[]
        {sculptor.runtime.core_id = 1 : i64,
         sculptor.runtime.digital_ops = 16 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %producer_b = sculptor.task.create %graph, @produce, domain = "digital",
        task_kind = "digital.producer", task_name = "causal.producer_b",
        source_layer = "causal", source_task_ordinal = 1,
        inputs[], outputs[%value_b], deps[]
        {sculptor.runtime.core_id = 3 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %consumer = sculptor.task.create %graph, @join, domain = "digital",
        task_kind = "digital.join", task_name = "causal.consumer",
        source_layer = "causal", source_task_ordinal = 2,
        inputs[%value_a, %value_b], outputs[%output], deps[]
        {sculptor.runtime.core_id = 0 : i64}
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x1xf32>>,
           !sculptor.task_resource<tensor<1x1xf32>>,
           !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// Producer B reserves the shared final directed link first. Producer A finishes
// later, queues behind that reservation, and becomes the consumer's causal input.
// One f32 payload word plus five protocol words crosses each route.
// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.timing.causal_critical_chain = [
// CHECK-SAME: id = 0 : i64, kind = "task", taskIndex = 1 : i64
// CHECK-SAME: id = 1 : i64, kind = "route", taskIndex = -1 : i64, edgeIndex = 1 : i64
// CHECK-SAME: parentEvent = 0 : i64, resource = "producer"
// CHECK-SAME: id = 2 : i64, kind = "route", taskIndex = -1 : i64, edgeIndex = 0 : i64
// CHECK-SAME: parentEvent = 1 : i64, resource = "directed-link:1->0"
// CHECK-SAME: id = 3 : i64, kind = "task", taskIndex = 2 : i64
// CHECK-SAME: parentEvent = 2 : i64, resource = "core:0"
// CHECK-SAME: sculptor.timing.critical_path_ns = 2.400000e+01 : f64
// CHECK-SAME: sculptor.timing.no_contention_makespan_ns = 1.800000e+01 : f64
// CHECK-SAME: sculptor.timing.total_payload_words = 2 : i64
// CHECK-SAME: sculptor.timing.total_protocol_words = 10 : i64
// CHECK-SAME: sculptor.timing.zero_network_makespan_ns = 5.000000e+00 : f64
// CHECK: task_name = "causal.producer_a"
// CHECK-SAME: sculptor.timing.earliest_finish_ns = 2.000000e+00 : f64
// CHECK: task_name = "causal.producer_b"
// CHECK-SAME: sculptor.timing.earliest_finish_ns = 1.000000e+00 : f64
// CHECK: task_name = "causal.consumer"
// CHECK-SAME: sculptor.timing.causal_input_edge = 0 : i64
// CHECK-SAME: sculptor.timing.earliest_start_ns = 2.100000e+01 : f64
