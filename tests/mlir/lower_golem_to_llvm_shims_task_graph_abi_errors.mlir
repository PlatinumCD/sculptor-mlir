// RUN: not sculptor-mlir-opt %s --sculptor-lower-golem-to-llvm-shims 2>&1 | FileCheck %s

module {
  func.func private @task_matrix() -> !sculptor.logical.array attributes {sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64}
  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> attributes {sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 1 : i64}

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>

    %setup = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix", source_layer = "linear", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // CHECK: expected logical array producer and consumer to share the same physical and local array binding
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm", source_layer = "linear", source_task_ordinal = 1, inputs[%input, %array], outputs[%output], deps[%setup] {sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}
