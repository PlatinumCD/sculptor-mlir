// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=4 arrays-per-core=2" | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: sculptor.schedule.analog_arrays = [0, 1, 2, 3, 4, 5, 6, 7]
// CHECK-SAME: sculptor.schedule.arrays_per_core = 2 : i64
// CHECK-SAME: sculptor.schedule.num_analog_arrays = 8 : i64
// CHECK-SAME: sculptor.schedule.num_cores = 4 : i64
module {
  // CHECK: func.func private @generate_task_graph()
  // CHECK-SAME: sculptor.schedule.analog_arrays = [0, 1, 2, 3, 4, 5, 6, 7]
  // CHECK-SAME: sculptor.schedule.arrays_per_core = 2 : i64
  // CHECK-SAME: sculptor.schedule.dependency_count = 0 : i64
  // CHECK-SAME: sculptor.schedule.logical_array_to_analog_array = []
  // CHECK-SAME: sculptor.schedule.num_analog_arrays = 8 : i64
  // CHECK-SAME: sculptor.schedule.num_cores = 4 : i64
  // CHECK-SAME: sculptor.schedule.num_logical_arrays = 0 : i64
  // CHECK-SAME: sculptor.schedule.task_count = 0 : i64
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}
