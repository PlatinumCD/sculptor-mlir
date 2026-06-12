// RUN: sculptor-mlir-opt %s --sculptor-export-task-graph-vis 2>&1 | FileCheck %s --check-prefix=MISSING-OUTPUT
// RUN: sculptor-mlir-opt %s --sculptor-export-task-graph-vis="output=%t.bad format=gexf" 2>&1 | FileCheck %s --check-prefix=INVALID-FORMAT

module {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}

// MISSING-OUTPUT: expected non-empty output path for sculptor-export-task-graph-vis
// INVALID-FORMAT: expected sculptor-export-task-graph-vis format to be 'dot' or 'graphml', got 'gexf'
