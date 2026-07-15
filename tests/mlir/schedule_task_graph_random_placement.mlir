// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=random" | FileCheck %s

module {
  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x2xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x2xf32>
    return %stored : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linear_matrix_tile_0_0", source_layer = "linear_0", source_task_ordinal = 0, inputs[], outputs[%array], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "linear_mvm_0_0", source_layer = "linear_0", source_task_ordinal = 1, inputs[%input, %array], outputs[%output], deps[%setup] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @task_matrix
// CHECK-SAME: sculptor.runtime.core_id = [[CORE:[0-9]+]] : i64
// CHECK-SAME: sculptor.runtime.local_array_id = 0 : i64
// CHECK-SAME: sculptor.runtime.physical_array_id = [[ARRAY:[0-9]+]] : i64
// CHECK-LABEL: func.func private @task_mvm
// CHECK-SAME: sculptor.runtime.core_id = [[CORE]] : i64
// CHECK-SAME: sculptor.runtime.local_array_id = 0 : i64
// CHECK-SAME: sculptor.runtime.physical_array_id = [[ARRAY]] : i64
// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.schedule.logical_array_to_analog_array = {{\[}}[[ARRAY]]{{\]}}
// CHECK-SAME: sculptor.schedule.task_count = 2 : i64
// CHECK: sculptor.task.create {{.*}} @task_matrix
// CHECK-SAME: task_kind = "sculptor.matrix_setup"
// CHECK-SAME: sculptor.runtime.core_id = [[CORE]] : i64
// CHECK-SAME: sculptor.runtime.local_array_id = 0 : i64
// CHECK-SAME: sculptor.runtime.physical_array_id = [[ARRAY]] : i64
// CHECK: sculptor.task.create {{.*}} @task_mvm
// CHECK-SAME: task_kind = "sculptor.mvm"
// CHECK-SAME: sculptor.runtime.core_id = [[CORE]] : i64
// CHECK-SAME: sculptor.runtime.local_array_id = 0 : i64
// CHECK-SAME: sculptor.runtime.physical_array_id = [[ARRAY]] : i64
