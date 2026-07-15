// RUN: sculptor-mlir-opt %s --sculptor-fuse-task-graph --sculptor-finalize-task-graph-resources | FileCheck %s

module {
  func.func private @task_a(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @task_b(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @task_x(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @task_c(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @task_d(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %a_out = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %b_out = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %x_out = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %c_out = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>

    %a = sculptor.task.create %graph, @task_a, domain = "digital", task_kind = "digital.compute", task_name = "stage_a", source_layer = "layer", source_task_ordinal = 0, inputs[%input], outputs[%a_out], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    %b = sculptor.task.create %graph, @task_b, domain = "digital", task_kind = "digital.compute", task_name = "stage_b", source_layer = "layer", source_task_ordinal = 1, inputs[%a_out], outputs[%b_out], deps[%a] {sculptor.runtime.core_id = 0 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %x = sculptor.task.create %graph, @task_x, domain = "digital", task_kind = "digital.compute", task_name = "stage_x", source_layer = "layer", source_task_ordinal = 2, inputs[%input], outputs[%x_out], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    %c = sculptor.task.create %graph, @task_c, domain = "digital", task_kind = "digital.compute", task_name = "stage_c", source_layer = "layer", source_task_ordinal = 3, inputs[%b_out, %x_out], outputs[%c_out], deps[%b, %x] {sculptor.runtime.core_id = 0 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %d = sculptor.task.create %graph, @task_d, domain = "digital", task_kind = "digital.compute", task_name = "stage_d", source_layer = "layer", source_task_ordinal = 4, inputs[%c_out], outputs[%output], deps[%c] {sculptor.runtime.core_id = 0 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.schedule.dependency_count = 0 : i64
// CHECK-SAME: sculptor.schedule.task_count = 1 : i64
// CHECK: task_kind = "mixed.fused"
// CHECK-SAME: task_name = "layer_same_core_component_core_0_0_4"
// CHECK-SAME: deps[]
// CHECK-NOT: task_name = "stage_b"
// CHECK-NOT: task_name = "stage_x"
// CHECK-NOT: task_name = "stage_c"
// CHECK-NOT: task_name = "stage_d"
// CHECK: return

// CHECK-LABEL: func.func private @task_layer_same_core_component_core_0_0_4_0
// CHECK-SAME: sculptor.task_kind = "mixed.fused"
// CHECK-SAME: sculptor.task_name = "layer_same_core_component_core_0_0_4"
// CHECK-NOT: func.func private @task_b
// CHECK-NOT: func.func private @task_x
// CHECK-NOT: func.func private @task_c
// CHECK-NOT: func.func private @task_d
