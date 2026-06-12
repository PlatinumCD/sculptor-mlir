// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 schedule=simple-budget" 2>&1 | FileCheck %s --check-prefix=ERR

module {
  func.func private @task_step(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %mid = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %forward_out = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>

    %layer0 = sculptor.task.create %graph, @task_step, domain = "digital", task_kind = "digital.vector_tile", task_name = "linear_0_step", source_layer = "linear_0", source_task_ordinal = 0, inputs[%input], outputs[%mid], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    %forward = sculptor.task.create %graph, @task_step, domain = "digital", task_kind = "digital.activation", task_name = "forward_sigmoid", source_layer = "forward", source_task_ordinal = 0, inputs[%mid], outputs[%forward_out], deps[%layer0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %layer1 = sculptor.task.create %graph, @task_step, domain = "digital", task_kind = "digital.vector_tile", task_name = "linear_1_step", source_layer = "linear_1", source_task_ordinal = 0, inputs[%forward_out], outputs[%output], deps[%forward] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}

// ERR: not enough digital cores for task graph source-layer segments: required at least 2 cores, budget provides 1
