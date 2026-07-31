// RUN: not sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="digital-issue-width=0" 2>&1 | FileCheck %s --check-prefix=ISSUE
// RUN: not sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="runtime-transmit-policy=serialize-transmit" 2>&1 | FileCheck %s --check-prefix=POLICY
// RUN: not sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing 2>&1 | FileCheck %s --check-prefix=OVERFLOW

module {
  func.func private @task_overflow() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %cmax = arith.constant 9223372036854775807 : index
    scf.for %outer = %c0 to %cmax step %c1 {
      scf.for %inner = %c0 to %c3 step %c1 {
      }
    }
    return
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %task = sculptor.task.create %graph, @task_overflow, domain = "digital",
        task_kind = "digital.overflow", task_name = "overflow",
        source_layer = "validation", source_task_ordinal = 0,
        inputs[], outputs[], deps[]
        : (!sculptor.task_graph) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// ISSUE: error: expected digital issue width to be positive
// POLICY: error: unsupported runtime transmit policy; expected 'overlap-ready-tasks'
// OVERFLOW: error: task cost overflow while counting scf.for iterations
