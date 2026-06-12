// RUN: sculptor-mlir-opt %s --sculptor-export-task-graph-dot="output=%t.dot" > /dev/null
// RUN: FileCheck %s --input-file=%t.dot

module {
  func.func private @task_matrix() -> !sculptor.logical.array
  func.func private @task_vector(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>
  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32>
  func.func private @task_bias(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %tile = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %mvm_out = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>

    %setup = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linear_matrix_tile_0_0", source_layer = "linear_0", source_task_ordinal = 0, inputs[], outputs[%array], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %vector = sculptor.task.create %graph, @task_vector, domain = "digital", task_kind = "digital.vector_tile", task_name = "linear_vector_tile_0", source_layer = "linear_0", source_task_ordinal = 1, inputs[%input], outputs[%tile], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "linear_mvm_0_0", source_layer = "linear_0", source_task_ordinal = 2, inputs[%tile, %array], outputs[%mvm_out], deps[%setup, %vector] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %bias = sculptor.task.create %graph, @task_bias, domain = "digital", task_kind = "digital.bias_add", task_name = "linear_bias_add", source_layer = "linear_0", source_task_ordinal = 3, inputs[%mvm_out], outputs[%output], deps[%mvm] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}

// CHECK: digraph analog_task_graph
// CHECK: subgraph cluster_graph_0
// CHECK: label="@generate_task_graph"
// CHECK: subgraph cluster_0_linear_0
// CHECK: label="linear_0"
// CHECK: label="sculptor.matrix_setup\nlinear_matrix_tile_0_0"
// CHECK-SAME: fillcolor="#fef3c7"
// CHECK: label="digital.vector_tile\nlinear_vector_tile_0"
// CHECK-SAME: fillcolor="#dbeafe"
// CHECK: label="sculptor.mvm\nlinear_mvm_0_0"
// CHECK: label="digital.bias_add\nlinear_bias_add"
// CHECK: task_0_0 -> task_0_2 [style="dotted", color="#b45309"]
// CHECK: task_0_1 -> task_0_2 [style="solid", color="#334155"]
// CHECK: task_0_2 -> task_0_3 [style="solid", color="#334155"]
// CHECK-NOT: source_task_ordinal
// CHECK-NOT: !sculptor.task_resource
