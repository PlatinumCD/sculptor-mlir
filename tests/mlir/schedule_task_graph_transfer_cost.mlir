// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=simple-budget" | FileCheck %s

module {
  func.func private @task_step(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32>

  // CHECK: func.func private @generate_task_graph()
  // CHECK-SAME: sculptor.schedule.core_transfer_bytes = [0, 16, 0, 0, 0, 0, 16, 0, 0, 0, 0, 16, 0, 0, 0, 0]
  // CHECK-SAME: sculptor.schedule.core_transfer_cost = [0, 16, 0, 0, 0, 0, 32, 0, 0, 0, 0, 16, 0, 0, 0, 0]
  // CHECK-SAME: sculptor.schedule.inter_core_transfer_bytes = 48 : i64
  // CHECK-SAME: sculptor.schedule.total_transfer_cost = 64 : i64
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %mid0 = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %forward0 = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %mid1 = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %forward1 = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %mid2 = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %forward2 = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>

    %layer0 = sculptor.task.create %graph, @task_step, domain = "digital", task_kind = "digital.vector_tile", task_name = "layer0", source_layer = "layer0", source_task_ordinal = 0, inputs[%input], outputs[%mid0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>) -> !sculptor.task
    %forward0_task = sculptor.task.create %graph, @task_step, domain = "digital", task_kind = "digital.activation", task_name = "forward0", source_layer = "forward", source_task_ordinal = 0, inputs[%mid0], outputs[%forward0], deps[%layer0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task

    %layer1 = sculptor.task.create %graph, @task_step, domain = "digital", task_kind = "digital.vector_tile", task_name = "layer1", source_layer = "layer1", source_task_ordinal = 0, inputs[%forward0], outputs[%mid1], deps[%forward0_task] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task
    %forward1_task = sculptor.task.create %graph, @task_step, domain = "digital", task_kind = "digital.activation", task_name = "forward1", source_layer = "forward", source_task_ordinal = 1, inputs[%mid1], outputs[%forward1], deps[%layer1] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task

    %layer2 = sculptor.task.create %graph, @task_step, domain = "digital", task_kind = "digital.vector_tile", task_name = "layer2", source_layer = "layer2", source_task_ordinal = 0, inputs[%forward1], outputs[%mid2], deps[%forward1_task] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task
    %forward2_task = sculptor.task.create %graph, @task_step, domain = "digital", task_kind = "digital.activation", task_name = "forward2", source_layer = "forward", source_task_ordinal = 2, inputs[%mid2], outputs[%forward2], deps[%layer2] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task

    %layer3 = sculptor.task.create %graph, @task_step, domain = "digital", task_kind = "digital.vector_tile", task_name = "layer3", source_layer = "layer3", source_task_ordinal = 0, inputs[%forward2], outputs[%output], deps[%forward2_task] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}
