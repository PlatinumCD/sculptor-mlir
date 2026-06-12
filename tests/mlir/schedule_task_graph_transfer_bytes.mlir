// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=2 schedule=simple-budget" | FileCheck %s

module {
  func.func private @task_matrix() -> !sculptor.logical.array
  func.func private @task_vector(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32>
  func.func private @task_mvm(%arg0: tensor<1x4xf32>, %arg1: !sculptor.logical.array) -> tensor<1x3xf32>
  func.func private @task_bias(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32>
  func.func private @task_forward(%arg0: tensor<1x3xf32>) -> tensor<1x4xf32>

  // CHECK: func.func private @generate_task_graph()
  // CHECK-SAME: sculptor.schedule.core_transfer_bytes = [0, 20, 16, 0]
  // CHECK-SAME: sculptor.schedule.core_transfer_cost = [0, 20, 16, 0]
  // CHECK-SAME: sculptor.schedule.inter_core_transfer_bytes = 36 : i64
  // CHECK-SAME: sculptor.schedule.total_transfer_cost = 36 : i64
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 12 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>
    %array0 = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %tile0 = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %mvm0_out = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 12 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>
    %layer0_out = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 12 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>
    %forward_out = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %tile1 = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %mvm1_out = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 12 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3xf32>>

    %setup0 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linear_0_matrix", source_layer = "linear_0", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %vector0 = sculptor.task.create %graph, @task_vector, domain = "digital", task_kind = "digital.vector_tile", task_name = "linear_0_vector", source_layer = "linear_0", source_task_ordinal = 1, inputs[%input], outputs[%tile0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>) -> !sculptor.task
    %mvm0 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "linear_0_mvm", source_layer = "linear_0", source_task_ordinal = 2, inputs[%tile0, %array0], outputs[%mvm0_out], deps[%setup0, %vector0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %bias0 = sculptor.task.create %graph, @task_bias, domain = "digital", task_kind = "digital.bias_add", task_name = "linear_0_bias", source_layer = "linear_0", source_task_ordinal = 3, inputs[%mvm0_out], outputs[%layer0_out], deps[%mvm0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task) -> !sculptor.task

    %setup1 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linear_1_matrix", source_layer = "linear_1", source_task_ordinal = 0, inputs[], outputs[%array1], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %forward = sculptor.task.create %graph, @task_forward, domain = "digital", task_kind = "digital.activation", task_name = "forward_activation", source_layer = "forward", source_task_ordinal = 0, inputs[%layer0_out], outputs[%forward_out], deps[%bias0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task
    %vector1 = sculptor.task.create %graph, @task_vector, domain = "digital", task_kind = "digital.vector_tile", task_name = "linear_1_vector", source_layer = "linear_1", source_task_ordinal = 1, inputs[%forward_out], outputs[%tile1], deps[%forward] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task
    %mvm1 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "linear_1_mvm", source_layer = "linear_1", source_task_ordinal = 2, inputs[%tile1, %array1], outputs[%mvm1_out], deps[%setup1, %vector1] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %bias1 = sculptor.task.create %graph, @task_bias, domain = "digital", task_kind = "digital.bias_add", task_name = "linear_1_bias", source_layer = "linear_1", source_task_ordinal = 3, inputs[%mvm1_out], outputs[%output], deps[%mvm1] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task_resource<tensor<1x3xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}
