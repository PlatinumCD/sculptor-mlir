// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=4 arrays-per-core=2 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" | FileCheck %s

module {
  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %array0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array2 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array3 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array4 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup0 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer0_matrix", source_layer = "layer0", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer1_matrix", source_layer = "layer1", source_task_ordinal = 0, inputs[], outputs[%array1], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup2 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer2_matrix", source_layer = "layer2", source_task_ordinal = 0, inputs[], outputs[%array2], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup3 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer3_matrix", source_layer = "layer3", source_task_ordinal = 0, inputs[], outputs[%array3], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup4 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer4_matrix", source_layer = "layer4", source_task_ordinal = 0, inputs[], outputs[%array4], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.schedule.logical_array_to_analog_array = [0, 1, 2, 3, 6]
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
// CHECK: sculptor.task.create {{.*}} @task_matrix
// CHECK-SAME: task_name = "layer2_matrix"
// CHECK-SAME: sculptor.runtime.core_id = 1 : i64
// CHECK-SAME: sculptor.runtime.local_array_id = 0 : i64
// CHECK-SAME: sculptor.runtime.physical_array_id = 2 : i64
// CHECK: sculptor.task.create {{.*}} @task_matrix
// CHECK-SAME: task_name = "layer3_matrix"
// CHECK-SAME: sculptor.runtime.core_id = 1 : i64
// CHECK-SAME: sculptor.runtime.local_array_id = 1 : i64
// CHECK-SAME: sculptor.runtime.physical_array_id = 3 : i64
// CHECK: sculptor.task.create {{.*}} @task_matrix
// CHECK-SAME: task_name = "layer4_matrix"
// CHECK-SAME: sculptor.runtime.core_id = 3 : i64
// CHECK-SAME: sculptor.runtime.local_array_id = 0 : i64
// CHECK-SAME: sculptor.runtime.physical_array_id = 6 : i64
