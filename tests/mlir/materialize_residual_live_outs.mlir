// RUN: sculptor-mlir-opt %s --sculptor-materialize-tasks | FileCheck %s --implicit-check-not=sculptor.task_region --implicit-check-not=sculptor.yield --implicit-check-not="call @branch" --implicit-check-not="call @sink" --implicit-check-not="func.func @branch" --implicit-check-not="func.func @sink"

#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  // CHECK-LABEL: func.func @forward
  // CHECK-NEXT: %[[BRANCH0:.*]] = call @task_branch0_passthrough_0(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: %[[STATE1:.*]]:2 = call @task_forward_activation_0(%arg0, %[[BRANCH0]]) : (tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<1x2xf32>, tensor<1x4xf32>)
  // CHECK-NEXT: %[[BRANCH1:.*]] = call @task_branch1_passthrough_0(%[[STATE1]]#1) : (tensor<1x4xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: %[[STATE2:.*]] = call @task_forward_activation_1(%[[STATE1]]#0, %[[BRANCH1]]) : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: %[[SINK:.*]] = call @task_sink_passthrough_0(%[[STATE2]]) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: return %[[SINK]] : tensor<1x2xf32>
  func.func @forward(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<1x2xf32>
    %0 = call @branch0(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %1 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %0 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %8 = arith.addf %lhs, %rhs : f32
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %2 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%1 : tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = math.tanh %in : f32
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %3 = tensor.pad %2 low[0, 1] high[0, 1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
    } : tensor<1x2xf32> to tensor<1x4xf32>
    %4 = call @branch1(%3) : (tensor<1x4xf32>) -> tensor<1x2xf32>
    %5 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%2, %4 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %8 = arith.addf %lhs, %rhs : f32
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %6 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%5 : tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = math.tanh %in : f32
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %7 = call @sink(%6) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    return %7 : tensor<1x2xf32>
  }

  func.func @branch0(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %0 = sculptor.task_region kind = "digital.bias_add" name = "passthrough"(%arg0) {
    ^bb0(%input: tensor<1x2xf32>):
      sculptor.yield %input : tensor<1x2xf32>
    } : (tensor<1x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }

  func.func @branch1(%arg0: tensor<1x4xf32>) -> tensor<1x2xf32> {
    %0 = sculptor.task_region kind = "digital.bias_add" name = "passthrough"(%arg0) {
    ^bb0(%input: tensor<1x4xf32>):
      %1 = tensor.extract_slice %input[0, 1] [1, 2] [1, 1] : tensor<1x4xf32> to tensor<1x2xf32>
      sculptor.yield %1 : tensor<1x2xf32>
    } : (tensor<1x4xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }

  func.func @sink(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %0 = sculptor.task_region kind = "digital.tile_recombine" name = "passthrough"(%arg0) {
    ^bb0(%input: tensor<1x2xf32>):
      sculptor.yield %input : tensor<1x2xf32>
    } : (tensor<1x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }

  // CHECK-LABEL: func.func private @task_forward_activation_0
  // CHECK-SAME: sculptor.task_kind = "digital.activation"
  // CHECK-SAME: sculptor.task_name = "forward_activation"
  // CHECK: tensor.pad
  // CHECK: return {{.*}} : tensor<1x2xf32>, tensor<1x4xf32>
}
