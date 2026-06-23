// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=4 arrays-per-core=2 schedule=random" | FileCheck %s

module {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}

// CHECK: sculptor.schedule.arrays_per_core = 2 : i64
// CHECK: sculptor.schedule.num_analog_arrays = 8 : i64
// CHECK: sculptor.schedule.num_cores = 4 : i64
// CHECK-LABEL: func.func private @generate_task_graph
