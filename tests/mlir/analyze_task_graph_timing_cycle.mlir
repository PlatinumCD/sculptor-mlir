// RUN: not sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing 2>&1 | FileCheck %s

module {
  func.func private @task(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %a = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %b = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %task0 = sculptor.task.create %graph, @task, domain = "digital", task_kind = "digital.compute", task_name = "task0", source_layer = "test", source_task_ordinal = 0, inputs[%b], outputs[%a], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1xf32>>, !sculptor.task_resource<tensor<1xf32>>) -> !sculptor.task
    %task1 = sculptor.task.create %graph, @task, domain = "digital", task_kind = "digital.compute", task_name = "task1", source_layer = "test", source_task_ordinal = 1, inputs[%a], outputs[%b], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1xf32>>, !sculptor.task_resource<tensor<1xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK: error: combined task dependency and resource dataflow graph contains a cycle
