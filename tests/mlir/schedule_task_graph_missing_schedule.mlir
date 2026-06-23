// RUN: not sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=4 arrays-per-core=2" 2>&1 | FileCheck %s

module {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}

// CHECK: expected task graph schedule name
