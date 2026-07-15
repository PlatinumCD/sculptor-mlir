// RUN: sculptor-mlir-opt %s --sculptor-fuse-task-graph | FileCheck %s

module {
  func.func private @task_a(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @task_b(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %intermediate = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %a = sculptor.task.create %graph, @task_a, domain = "digital", task_kind = "digital.compute", task_name = "island_0_task", source_layer = "layer", source_task_ordinal = 0, inputs[%input], outputs[%intermediate], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    %b = sculptor.task.create %graph, @task_b, domain = "digital", task_kind = "digital.compute", task_name = "island_1_task", source_layer = "layer", source_task_ordinal = 1, inputs[%intermediate], outputs[%output], deps[%a] {sculptor.runtime.core_id = 0 : i64, sculptor.schedule.island_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.schedule.dependency_count = 1 : i64
// CHECK-SAME: sculptor.schedule.task_count = 2 : i64
// CHECK: task_name = "island_0_task"
// CHECK-SAME: sculptor.schedule.island_id = 0 : i64
// CHECK: task_name = "island_1_task"
// CHECK-SAME: sculptor.schedule.island_id = 1 : i64
// CHECK-NOT: task_kind = "mixed.fused"
