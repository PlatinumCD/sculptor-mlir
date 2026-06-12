// RUN: sculptor-mlir-opt %s --sculptor-convert-layers | FileCheck %s --implicit-check-not=sculptor.nn.conv1d --implicit-check-not=scf.for --implicit-check-not=memref. --implicit-check-not=bufferization.to_tensor

module {
  func.func @forward(%arg0: tensor<1x1x4xf32>) -> tensor<1x2x3xf32> {
    %0 = call @conv1d_bias(%arg0)
        : (tensor<1x1x4xf32>) -> tensor<1x2x3xf32>
    return %0 : tensor<1x2x3xf32>
  }

  // CHECK-LABEL: func.func @conv1d_bias
  // CHECK: %[[PATCH0:.*]] = sculptor.task_region kind = "digital.conv_patch" name = "conv1d_ow_0"(%arg0)
  // CHECK: tensor.empty() : tensor<1x2xf32>
  // CHECK: tensor.extract
  // CHECK: tensor.insert
  // CHECK: %[[MVM0:.*]] = sculptor.mvm %[[PATCH0]],
  // CHECK-SAME: : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK: sculptor.task_region kind = "digital.bias_add" name = "conv1d_ow_0_bias_add"(%[[MVM0]])
  // CHECK: sculptor.task_region kind = "digital.conv_patch" name = "conv1d_ow_1"(%arg0)
  // CHECK: sculptor.task_region kind = "digital.conv_patch" name = "conv1d_ow_2"(%arg0)
  // CHECK: sculptor.task_region kind = "digital.output_recombine" name = "conv1d_output_recombine"
  // CHECK: tensor.concat dim(2)
  // CHECK: return {{.*}} : tensor<1x2x3xf32>
  func.func @conv1d_bias(%arg0: tensor<1x1x4xf32>)
      -> tensor<1x2x3xf32>
      attributes {layer_type = "conv1d_w_bias"} {
    %w = arith.constant dense<2.000000e+00> : tensor<2x1x2xf32>
    %b = arith.constant dense<3.000000e+00> : tensor<2xf32>
    %0 = sculptor.nn.conv1d %arg0, %w, %b {dilation = [1], has_bias = true, padding = [0], stride = [1]}
        : (tensor<1x1x4xf32>, tensor<2x1x2xf32>, tensor<2xf32>) -> tensor<1x2x3xf32>
    return %0 : tensor<1x2x3xf32>
  }
}
