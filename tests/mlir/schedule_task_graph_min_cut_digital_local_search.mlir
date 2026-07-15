// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=snake" --sculptor-fuse-task-graph --sculptor-finalize-task-graph-resources | FileCheck %s

module {
  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm_small(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x1xf32> {
    %empty = tensor.empty() : tensor<1x1xf32>
    return %empty : tensor<1x1xf32>
  }

  func.func private @task_mvm_large(%arg0: tensor<1x25xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> {
    %empty = tensor.empty() : tensor<1x2xf32>
    return %empty : tensor<1x2xf32>
  }

  func.func private @task_bridge(%arg0: tensor<1x1xf32>) -> tensor<1x25xf32> {
    %empty = tensor.empty() : tensor<1x25xf32>
    return %empty : tensor<1x25xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %small = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 4 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %large = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 100 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x25xf32>>
    %array0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup0 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer0_matrix", source_layer = "layer0", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer1_matrix", source_layer = "layer1", source_task_ordinal = 0, inputs[], outputs[%array1], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm0 = sculptor.task.create %graph, @task_mvm_small, domain = "analog", task_kind = "sculptor.mvm", task_name = "layer0_mvm", source_layer = "layer0", source_task_ordinal = 1, inputs[%input, %array0], outputs[%small], deps[%setup0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task) -> !sculptor.task
    %bridge = sculptor.task.create %graph, @task_bridge, domain = "digital", task_kind = "digital.bridge", task_name = "layer0_bridge", source_layer = "layer0", source_task_ordinal = 2, inputs[%small], outputs[%large], deps[%mvm0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task_resource<tensor<1x25xf32>>, !sculptor.task) -> !sculptor.task
    %mvm1 = sculptor.task.create %graph, @task_mvm_large, domain = "analog", task_kind = "sculptor.mvm", task_name = "layer1_mvm", source_layer = "layer1", source_task_ordinal = 1, inputs[%large, %array1], outputs[%output], deps[%setup1, %bridge] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x25xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.schedule.graph_score = 100 : i64
// CHECK-SAME: sculptor.schedule.total_transfer_cost = 100 : i64
// CHECK: sculptor.task.create {{.*}} task_name = "layer0_matrix"
// CHECK-SAME: sculptor.runtime.core_id = 0 : i64
// CHECK: sculptor.task.create {{.*}} task_kind = "mixed.fused"
// CHECK-SAME: task_name = "layer0_same_core_component_core_0_1_2"
// CHECK-SAME: sculptor.runtime.core_id = 0 : i64
// CHECK: sculptor.task.create {{.*}} task_name = "layer1_matrix"
// CHECK-SAME: sculptor.runtime.core_id = 1 : i64
// CHECK: sculptor.task.create {{.*}} task_name = "layer1_mvm"
// CHECK-SAME: sculptor.runtime.core_id = 1 : i64
// CHECK-NOT: task_name = "layer0_bridge"
