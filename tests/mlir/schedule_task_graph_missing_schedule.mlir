// RUN: not sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=2" 2>&1 | FileCheck %s
// RUN: not sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" 2>&1 | FileCheck %s --check-prefix=MISSING-TIMING

module {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}

// CHECK: expected task graph schedule name
// MISSING-TIMING: expected pre-placement timing attribute 'sculptor.timing.placement_aware'
// MISSING-TIMING: failed to load pre-placement scheduling timing profile
