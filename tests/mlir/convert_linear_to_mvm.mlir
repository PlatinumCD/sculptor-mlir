// RUN: sculptor-mlir-opt %s --sculptor-convert-layers | FileCheck %s --implicit-check-not=sculptor.nn.linear --implicit-check-not=sculptor.matrix. --implicit-check-not=sculptor.vector. --implicit-check-not=sculptor.array.

module {
  func.func @forward(
      %arg0: tensor<1x4xf32>,
      %arg1: tensor<1x4xf32>
  ) -> (tensor<1x3xf32>, tensor<1x2xf32>) {
    %0 = call @linear_no_bias(%arg0)
        : (tensor<1x4xf32>) -> tensor<1x3xf32>
    %1 = call @linear_bias(%arg1)
        : (tensor<1x4xf32>) -> tensor<1x2xf32>
    return %0, %1 : tensor<1x3xf32>, tensor<1x2xf32>
  }

  // CHECK-LABEL: func.func @linear_no_bias
  // CHECK: %[[MVM:[0-9]+]] = sculptor.mvm {{.*}} : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<1x3xf32>
  // CHECK: %[[POST:[0-9]+]] = sculptor.task_region kind = "digital.bias_add" name = "linear_bias_add"(%[[MVM]])
  // CHECK: sculptor.yield
  // CHECK: return %[[POST]]
  func.func @linear_no_bias(%arg0: tensor<1x4xf32>)
      -> tensor<1x3xf32>
      attributes {layer_type = "linear"} {
    %w = arith.constant dense<1.000000e+00> : tensor<3x4xf32>
    %0 = sculptor.nn.linear %arg0, %w {has_bias = false}
        : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func @linear_bias
  // CHECK: %[[MVM:[0-9]+]] = sculptor.mvm {{.*}} : (tensor<1x4xf32>, tensor<2x4xf32>) -> tensor<1x2xf32>
  // CHECK: %[[POST:[0-9]+]] = sculptor.task_region kind = "digital.bias_add" name = "linear_bias_add"(%[[MVM]]) {
  // CHECK: arith.constant dense<3.000000e+00> : tensor<2xf32>
  // CHECK: tensor.expand_shape
  // CHECK: %[[ADD:[0-9]+]] = linalg.add
  // CHECK: sculptor.yield %[[ADD]]
  // CHECK: return %[[POST]]
  func.func @linear_bias(%arg0: tensor<1x4xf32>)
      -> tensor<1x2xf32>
      attributes {layer_type = "linear_w_bias"} {
    %w = arith.constant dense<2.000000e+00> : tensor<2x4xf32>
    %b = arith.constant dense<3.000000e+00> : tensor<2xf32>
    %0 = sculptor.nn.linear %arg0, %w, %b {has_bias = true}
        : (tensor<1x4xf32>, tensor<2x4xf32>, tensor<2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }
}
