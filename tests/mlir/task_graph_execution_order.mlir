// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing | FileCheck %s

module {
  func.func private @task_identity(%arg0: tensor<1x1xf32>) -> tensor<1x1xf32> {
    return %arg0 : tensor<1x1xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %intermediate = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>

    // Deliberately store the consumer before its data producer. Execution
    // order must come from resource dataflow, not operation storage order.
    %consumer = sculptor.task.create %graph, @task_identity, domain = "digital", task_kind = "digital.consumer", task_name = "consumer", source_layer = "test", source_task_ordinal = 1, inputs[%intermediate], outputs[%output], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    %producer = sculptor.task.create %graph, @task_identity, domain = "digital", task_kind = "digital.producer", task_name = "producer", source_layer = "test", source_task_ordinal = 0, inputs[%input], outputs[%intermediate], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task_resource<tensor<1x1xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.timing.execution_edge_count = 1 : i64
// CHECK-SAME: sculptor.timing.total_data_bytes = 4 : i64
// CHECK: task_name = "consumer"
// CHECK-SAME: sculptor.timing.data_predecessor_count = 1 : i64
// CHECK-SAME: sculptor.timing.topological_index = 1 : i64
// CHECK: task_name = "producer"
// CHECK-SAME: sculptor.timing.data_predecessor_count = 0 : i64
// CHECK-SAME: sculptor.timing.topological_index = 0 : i64
