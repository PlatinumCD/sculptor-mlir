// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=2 topology=mesh mesh-rows=1 mesh-cols=2 schedule=manhattan-cut" | FileCheck %s

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
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %tmp = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array0 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup0 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer0_matrix", source_layer = "layer0", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer1_matrix", source_layer = "layer1", source_task_ordinal = 0, inputs[], outputs[%array1], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm0 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "layer0_mvm", source_layer = "layer0", source_task_ordinal = 1, inputs[%input, %array0], outputs[%tmp], deps[%setup0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm1 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "layer1_mvm", source_layer = "layer1", source_task_ordinal = 1, inputs[%tmp, %array1], outputs[%output], deps[%setup1, %mvm0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.schedule.graph_score = 0 : i64
// CHECK-SAME: sculptor.schedule.logical_array_to_analog_array = [0, 1]
// CHECK-SAME: sculptor.schedule.total_transfer_cost = 0 : i64
// CHECK: sculptor.task.create {{.*}} @task_matrix
// CHECK-SAME: task_name = "layer0_matrix"
// CHECK-SAME: sculptor.runtime.core_id = 0 : i64
// CHECK-SAME: sculptor.runtime.local_array_id = 0 : i64
// CHECK-SAME: sculptor.runtime.physical_array_id = 0 : i64
// CHECK: sculptor.task.create {{.*}} @task_matrix
// CHECK-SAME: task_name = "layer1_matrix"
// CHECK-SAME: sculptor.runtime.core_id = 0 : i64
// CHECK-SAME: sculptor.runtime.local_array_id = 1 : i64
// CHECK-SAME: sculptor.runtime.physical_array_id = 1 : i64
// CHECK: sculptor.task.create {{.*}} @task_mvm
// CHECK-SAME: task_name = "layer0_mvm"
// CHECK-SAME: sculptor.runtime.core_id = 0 : i64
// CHECK-SAME: sculptor.runtime.local_array_id = 0 : i64
// CHECK-SAME: sculptor.runtime.physical_array_id = 0 : i64
// CHECK: sculptor.task.create {{.*}} @task_mvm
// CHECK-SAME: task_name = "layer1_mvm"
// CHECK-SAME: sculptor.runtime.core_id = 0 : i64
// CHECK-SAME: sculptor.runtime.local_array_id = 1 : i64
// CHECK-SAME: sculptor.runtime.physical_array_id = 1 : i64
