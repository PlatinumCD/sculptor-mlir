// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions --sculptor-balance-task-graph-reductions | FileCheck %s
// RUN: not sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions="reduction-width=1" 2>&1 | FileCheck %s --check-prefix=INVALID-WIDTH
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions="reduction-width=4" | FileCheck %s --check-prefix=UNCHANGED
// RUN: not sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions="reduction-width=4 require-change=true" 2>&1 | FileCheck %s --check-prefix=REQUIRE-CHANGE
// RUN: python3 %S/../python/test_balanced_reduction_equivalence.py

module {
  func.func private @produce() -> tensor<8xf32> {
    %value = arith.constant dense<1.000000e+00> : tensor<8xf32>
    return %value : tensor<8xf32>
  }

  func.func private @star_add(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @consume(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %leaf0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %leaf1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %leaf2 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %leaf3 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %reduced = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>

    %p0 = sculptor.task.create %graph, @produce, domain = "digital", task_kind = "digital.produce", task_name = "p0", source_layer = "reduction", source_task_ordinal = 0, inputs[], outputs[%leaf0], deps[] {sculptor.runtime.result_indices = array<i64: 0>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    %p1 = sculptor.task.create %graph, @produce, domain = "digital", task_kind = "digital.produce", task_name = "p1", source_layer = "reduction", source_task_ordinal = 1, inputs[], outputs[%leaf1], deps[] {sculptor.runtime.result_indices = array<i64: 0>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    %p2 = sculptor.task.create %graph, @produce, domain = "digital", task_kind = "digital.produce", task_name = "p2", source_layer = "reduction", source_task_ordinal = 2, inputs[], outputs[%leaf2], deps[] {sculptor.runtime.result_indices = array<i64: 0>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    %p3 = sculptor.task.create %graph, @produce, domain = "digital", task_kind = "digital.produce", task_name = "p3", source_layer = "reduction", source_task_ordinal = 3, inputs[], outputs[%leaf3], deps[] {sculptor.runtime.result_indices = array<i64: 0>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    %star = sculptor.task.create %graph, @star_add, domain = "digital", task_kind = "digital.reduction", task_name = "star", source_layer = "reduction", source_task_ordinal = 4, inputs[%leaf0, %leaf1, %leaf2, %leaf3], outputs[%reduced], deps[%p0, %p1, %p2, %p3] {sculptor.runtime.result_indices = array<i64: 0>, sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task, !sculptor.task, !sculptor.task, !sculptor.task) -> !sculptor.task
    %sink = sculptor.task.create %graph, @consume, domain = "digital", task_kind = "digital.consume", task_name = "sink", source_layer = "reduction", source_task_ordinal = 5, inputs[%reduced], outputs[%output], deps[%star] {sculptor.runtime.result_indices = array<i64: 0>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-NOT: func.func private @star_add
// CHECK: func.func private @__sculptor_reduce_add_
// CHECK-SAME: attributes {sculptor.task.reduction_helper = #sculptor.task_reduction<kind = add, reassociate = true>
// CHECK: %[[ADD:.*]] = arith.addf
// CHECK: linalg.yield %[[ADD]]

// CHECK-LABEL: func.func private @generate_task_graph()
// CHECK: task_kind = "digital.reduction", task_name = "star.level0.lane0"
// CHECK-SAME: sculptor.task.reduction_lane = 0 : i64
// CHECK-SAME: sculptor.task.reduction_tree_id = 0 : i64
// CHECK-SAME: sculptor.task.reduction_width = 2 : i64
// CHECK: task_kind = "digital.reduction", task_name = "star.level0.lane1"
// CHECK-SAME: sculptor.task.reduction_lane = 1 : i64
// CHECK: %[[ROOT:.*]] = sculptor.task.create {{.*}} task_name = "star.level1.root"
// CHECK-SAME: outputs[%{{.*}}]
// CHECK-SAME: sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>
// CHECK-NOT: sculptor.task.reduction_lane
// CHECK-NOT: task_name = "star.level"
// CHECK: task_name = "sink"
// CHECK-SAME: deps[%[[ROOT]]]

// INVALID-WIDTH: error: expected task reduction width to be at least two
// REQUIRE-CHANGE: error: expected at least one eligible marked task reduction

// UNCHANGED: func.func private @star_add
// UNCHANGED: task_name = "star"
// UNCHANGED-NOT: sculptor.task.reduction_tree_id
