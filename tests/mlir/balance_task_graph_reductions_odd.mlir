// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions | FileCheck %s

module {
  func.func private @star_add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %star = sculptor.task.create %graph, @star_add, domain = "digital", task_kind = "digital.reduction", task_name = "odd", source_layer = "reduction", source_task_ordinal = 0, inputs[%input, %input, %input, %input, %input], outputs[%output], deps[] {sculptor.runtime.result_indices = array<i64: 0>, sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-NOT: func.func private @star_add
// CHECK-LABEL: func.func private @generate_task_graph()
// CHECK: task_name = "odd.level0.lane0"
// CHECK-SAME: inputs[%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK: task_name = "odd.level0.lane1"
// CHECK-SAME: inputs[%{{.*}}, %{{.*}}]
// CHECK: task_name = "odd.level1.lane0"
// CHECK-NOT: task_name = "odd.level"
