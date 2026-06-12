// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=4 arrays-per-core=2 topology=mesh mesh-rows=2 mesh-cols=2 schedule=simple-budget" | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: sculptor.schedule.arrays_per_core = 2 : i64
// CHECK-SAME: sculptor.schedule.mesh_cols = 2 : i64
// CHECK-SAME: sculptor.schedule.mesh_rows = 2 : i64
// CHECK-SAME: sculptor.schedule.num_cores = 4 : i64
// CHECK-SAME: sculptor.schedule.topology = "mesh"
module {
  // CHECK: func.func private @generate_task_graph()
  // CHECK-SAME: sculptor.schedule.mesh_cols = 2 : i64
  // CHECK-SAME: sculptor.schedule.mesh_rows = 2 : i64
  // CHECK-SAME: sculptor.schedule.topology = "mesh"
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}
