// RUN: sculptor-mlir-opt %s --sculptor-export-task-graph-sim-model="output=%t.json" 2>&1 | FileCheck %s

module {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}

// CHECK: expected required attr 'sculptor.schedule.num_cores'
