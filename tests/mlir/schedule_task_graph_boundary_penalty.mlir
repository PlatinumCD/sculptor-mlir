// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=random" | FileCheck %s

module {
  func.func private @task_producer(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @task_consumer(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %tmp = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %producer = sculptor.task.create %graph, @task_producer, domain = "digital", task_kind = "digital.test_producer", task_name = "producer", source_layer = "test", source_task_ordinal = 0, inputs[%input], outputs[%tmp], deps[] {sculptor.runtime.core_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    %consumer = sculptor.task.create %graph, @task_consumer, domain = "digital", task_kind = "digital.test_consumer", task_name = "consumer", source_layer = "test", source_task_ordinal = 1, inputs[%tmp], outputs[%output], deps[%producer] {sculptor.runtime.core_id = 3 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.schedule.boundary_penalty = 4 : i64
// CHECK-SAME: sculptor.schedule.core_transfer_cost = [0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
// CHECK-SAME: sculptor.schedule.graph_score = 20 : i64
// CHECK-SAME: sculptor.schedule.total_transfer_cost = 16 : i64
