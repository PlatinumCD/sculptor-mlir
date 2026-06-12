// RUN: sculptor-mlir-opt %s --sculptor-materialize-tasks | FileCheck %s --implicit-check-not=sculptor.task_region --implicit-check-not=sculptor.yield --implicit-check-not="call @source" --implicit-check-not="func.func @source"

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  // CHECK-LABEL: func.func @forward
  // CHECK-NEXT: %[[FIRST:.*]] = call @task_source0_passthrough_0(%arg0) : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  // CHECK-NEXT: %[[RESIDUAL:.*]] = call @task_forward_activation_0(%[[FIRST]]) : (tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32>
  // CHECK-NEXT: %[[SECOND:.*]] = call @task_source1_passthrough_0(%[[RESIDUAL]]) : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  // CHECK-NEXT: return %[[SECOND]] : tensor<1x2x2x2xf32>
  func.func @forward(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %0 = call @source0(%arg0) : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %1 = tensor.empty() : tensor<1x2x4x4xf32>
    %2 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%0 : tensor<1x2x4x4xf32>) outs(%1 : tensor<1x2x4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = math.tanh %in : f32
      linalg.yield %8 : f32
    } -> tensor<1x2x4x4xf32>
    %3 = tensor.empty() : tensor<1x2x2x2xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %5 = tensor.empty() : tensor<2x2xf32>
    %6 = linalg.pooling_nchw_sum
        {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
        ins(%2, %5 : tensor<1x2x4x4xf32>, tensor<2x2xf32>)
        outs(%4 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %7 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%6 : tensor<1x2x2x2xf32>) outs(%3 : tensor<1x2x2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.divf %in, %cst_0 : f32
      linalg.yield %8 : f32
    } -> tensor<1x2x2x2xf32>
    %8 = call @source1(%7) : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %8 : tensor<1x2x2x2xf32>
  }

  func.func @source0(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
    %0 = sculptor.task_region kind = "digital.input_recombine" name = "passthrough"(%arg0) {
    ^bb0(%input: tensor<1x2x4x4xf32>):
      sculptor.yield %input : tensor<1x2x4x4xf32>
    } : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    return %0 : tensor<1x2x4x4xf32>
  }

  func.func @source1(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
    %0 = sculptor.task_region kind = "digital.tile_recombine" name = "passthrough"(%arg0) {
    ^bb0(%input: tensor<1x2x2x2xf32>):
      sculptor.yield %input : tensor<1x2x2x2xf32>
    } : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %0 : tensor<1x2x2x2xf32>
  }

  // CHECK-LABEL: func.func private @task_forward_activation_0
  // CHECK-SAME: sculptor.source_layer = "forward"
  // CHECK-SAME: sculptor.source_task_ordinal = 0 : i64
  // CHECK-SAME: sculptor.task_domain = "digital"
  // CHECK-SAME: sculptor.task_kind = "digital.activation"
  // CHECK-SAME: sculptor.task_name = "forward_activation"
  // CHECK: arith.constant 4.000000e+00 : f32
  // CHECK: math.tanh
  // CHECK: linalg.pooling_nchw_sum
  // CHECK: arith.divf
  // CHECK: return
}
