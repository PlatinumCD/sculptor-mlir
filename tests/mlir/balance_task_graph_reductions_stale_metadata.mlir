// RUN: not sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions 2>&1 | FileCheck %s

module {
  func.func private @star_add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %star = sculptor.task.create %graph, @star_add, domain = "digital", task_kind = "digital.reduction", task_name = "stale", source_layer = "reduction", source_task_ordinal = 0, inputs[%input, %input, %input], outputs[%output], deps[] {sculptor.schedule.island_id = 0 : i64, sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK: error: task reduction balancing must run before island construction and placement
