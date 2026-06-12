// RUN: sculptor-mlir-opt %s --sculptor-materialize-tasks | FileCheck %s --implicit-check-not=sculptor.task_region --implicit-check-not=sculptor.yield --implicit-check-not="call @layer" --implicit-check-not="func.func @layer"

#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  // CHECK-LABEL: func.func @forward
  // CHECK-NEXT: %[[LAYER:.*]] = call @task_layer_passthrough_0(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: %[[SIGMOID:.*]] = call @task_forward_sigmoid_0(%[[LAYER]]) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: return %[[SIGMOID]] : tensor<1x2xf32>
  func.func @forward(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %empty = tensor.empty() : tensor<1x2xf32>
    %0 = call @layer(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %1 = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%0 : tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %neg = arith.negf %in : f32
      %exp = math.exp %neg : f32
      %denom = arith.addf %exp, %cst : f32
      %sigmoid = arith.divf %cst, %denom : f32
      linalg.yield %sigmoid : f32
    } -> tensor<1x2xf32>
    return %1 : tensor<1x2xf32>
  }

  func.func @layer(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %0 = sculptor.task_region kind = "digital.bias_add" name = "layer_passthrough"(%arg0) {
    ^bb0(%input: tensor<1x2xf32>):
      sculptor.yield %input : tensor<1x2xf32>
    } : (tensor<1x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }

  // CHECK-LABEL: func.func private @task_forward_sigmoid_0
  // CHECK-SAME: sculptor.source_layer = "forward"
  // CHECK-SAME: sculptor.source_task_ordinal = 0 : i64
  // CHECK-SAME: sculptor.task_domain = "digital"
  // CHECK-SAME: sculptor.task_kind = "digital.activation"
  // CHECK-SAME: sculptor.task_name = "forward_sigmoid"
  // CHECK: arith.constant 1.000000e+00 : f32
  // CHECK: tensor.empty() : tensor<1x2xf32>
  // CHECK: linalg.generic
  // CHECK: math.exp
  // CHECK: arith.divf
  // CHECK: return
}
