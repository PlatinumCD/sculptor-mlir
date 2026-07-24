// RUN: not sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing 2>&1 | FileCheck %s

#identity = affine_map<(d0) -> (d0)>

module {
  func.func private @task_dynamic(%input: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %size = tensor.dim %input, %c0 : tensor<?xf32>
    %empty = tensor.empty(%size) : tensor<?xf32>
    %result = linalg.generic {
        indexing_maps = [#identity, #identity],
        iterator_types = ["parallel"]}
        ins(%input : tensor<?xf32>) outs(%empty : tensor<?xf32>) {
      ^bb0(%value: f32, %out: f32):
        %sum = arith.addf %value, %one : f32
        linalg.yield %sum : f32
    } -> tensor<?xf32>
    return %result : tensor<?xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<?xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<?xf32>>
    %task = sculptor.task.create %graph, @task_dynamic, domain = "digital", task_kind = "digital.test_dynamic", task_name = "dynamic", source_layer = "test", source_task_ordinal = 0, inputs[%input], outputs[%output], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<?xf32>>, !sculptor.task_resource<tensor<?xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK: error: cannot infer digital operations from a dynamic loop range
