// RUN: sculptor-mlir-opt %s --sculptor-materialize-tasks | FileCheck %s --implicit-check-not="call @layer" --implicit-check-not="func.func @layer" --implicit-check-not=sculptor.task_graph

module {
  // CHECK-LABEL: func.func @forward
  // CHECK: %[[PREP:.*]] = call @task_layer_prep_0(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[FINISH:.*]] = call @task_layer_finish_1(%[[PREP]]) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK: return %[[FINISH]] : tensor<1x2xf32>
  func.func @forward(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %0 = call @layer(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }

  func.func @layer(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %0 = sculptor.task_region kind = "digital.vector_tile" name = "prep"(%arg0) {
    ^bb0(%input: tensor<1x2xf32>):
      sculptor.yield %input : tensor<1x2xf32>
    } : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %1 = sculptor.task_region kind = "digital.tile_recombine" name = "finish"(%0) {
    ^bb0(%input: tensor<1x2xf32>):
      sculptor.yield %input : tensor<1x2xf32>
    } : (tensor<1x2xf32>) -> tensor<1x2xf32>
    return %1 : tensor<1x2xf32>
  }

  // CHECK-LABEL: func.func private @task_layer_prep_0
  // CHECK-SAME: sculptor.source_layer = "layer"
  // CHECK-SAME: sculptor.source_task_ordinal = 0 : i64
  // CHECK-SAME: sculptor.task_domain = "digital"
  // CHECK-SAME: sculptor.task_kind = "digital.vector_tile"
  // CHECK-SAME: sculptor.task_name = "prep"
  // CHECK: return %arg0 : tensor<1x2xf32>

  // CHECK-LABEL: func.func private @task_layer_finish_1
  // CHECK-SAME: sculptor.source_layer = "layer"
  // CHECK-SAME: sculptor.source_task_ordinal = 1 : i64
  // CHECK-SAME: sculptor.task_domain = "digital"
  // CHECK-SAME: sculptor.task_kind = "digital.tile_recombine"
  // CHECK-SAME: sculptor.task_name = "finish"
  // CHECK: return %arg0 : tensor<1x2xf32>
}
