// RUN: sculptor-mlir-opt %s --sculptor-convert-layers | FileCheck %s --implicit-check-not=sculptor.nn.conv1d --implicit-check-not=sculptor.nn.conv2d --implicit-check-not=sculptor.nn.conv3d --implicit-check-not=sculptor.nn.grouped_conv2d --implicit-check-not=sculptor.matrix. --implicit-check-not=sculptor.vector. --implicit-check-not=sculptor.array.

module {
  func.func @forward(
      %arg0: tensor<1x1x4xf32>,
      %arg1: tensor<1x1x3x3xf32>,
      %arg2: tensor<1x1x3x3x3xf32>,
      %arg3: tensor<1x2x3x3xf32>
  ) -> (
      tensor<1x2x3xf32>,
      tensor<1x2x3xf32>,
      tensor<1x2x2x2xf32>,
      tensor<1x2x2x2xf32>,
      tensor<1x2x2x2xf32>,
      tensor<1x2x2x2xf32>,
      tensor<1x2x2x2x2xf32>,
      tensor<1x2x2x2x2xf32>
  ) {
    %0 = call @conv1d_no_bias(%arg0)
        : (tensor<1x1x4xf32>) -> tensor<1x2x3xf32>
    %1 = call @conv1d_bias(%arg0)
        : (tensor<1x1x4xf32>) -> tensor<1x2x3xf32>
    %2 = call @conv2d_no_bias(%arg1)
        : (tensor<1x1x3x3xf32>) -> tensor<1x2x2x2xf32>
    %3 = call @conv2d_bias(%arg1)
        : (tensor<1x1x3x3xf32>) -> tensor<1x2x2x2xf32>
    %4 = call @grouped_conv2d_no_bias(%arg3)
        : (tensor<1x2x3x3xf32>) -> tensor<1x2x2x2xf32>
    %5 = call @grouped_conv2d_bias(%arg3)
        : (tensor<1x2x3x3xf32>) -> tensor<1x2x2x2xf32>
    %6 = call @conv3d_no_bias(%arg2)
        : (tensor<1x1x3x3x3xf32>) -> tensor<1x2x2x2x2xf32>
    %7 = call @conv3d_bias(%arg2)
        : (tensor<1x1x3x3x3xf32>) -> tensor<1x2x2x2x2xf32>
    return %0, %1, %2, %3, %4, %5, %6, %7
        : tensor<1x2x3xf32>, tensor<1x2x3xf32>,
          tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>,
          tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>,
          tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>
  }

  // CHECK-LABEL: func.func @conv1d_no_bias
  // CHECK: sculptor.mvm {{.*}} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  func.func @conv1d_no_bias(%arg0: tensor<1x1x4xf32>)
      -> tensor<1x2x3xf32>
      attributes {layer_type = "conv1d"} {
    %w = arith.constant dense<1.000000e+00> : tensor<2x1x2xf32>
    %0 = sculptor.nn.conv1d %arg0, %w {dilation = [1], has_bias = false, padding = [0], stride = [1]}
        : (tensor<1x1x4xf32>, tensor<2x1x2xf32>) -> tensor<1x2x3xf32>
    return %0 : tensor<1x2x3xf32>
  }

  // CHECK-LABEL: func.func @conv1d_bias
  // CHECK: sculptor.mvm {{.*}} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  // CHECK: tensor.expand_shape
  // CHECK: linalg.add
  func.func @conv1d_bias(%arg0: tensor<1x1x4xf32>)
      -> tensor<1x2x3xf32>
      attributes {layer_type = "conv1d_w_bias"} {
    %w = arith.constant dense<2.000000e+00> : tensor<2x1x2xf32>
    %b = arith.constant dense<3.000000e+00> : tensor<2xf32>
    %0 = sculptor.nn.conv1d %arg0, %w, %b {dilation = [1], has_bias = true, padding = [0], stride = [1]}
        : (tensor<1x1x4xf32>, tensor<2x1x2xf32>, tensor<2xf32>) -> tensor<1x2x3xf32>
    return %0 : tensor<1x2x3xf32>
  }

  // CHECK-LABEL: func.func @conv2d_no_bias
  // CHECK: sculptor.mvm {{.*}} : (tensor<1x4xf32>, tensor<2x4xf32>) -> tensor<1x2xf32>
  func.func @conv2d_no_bias(%arg0: tensor<1x1x3x3xf32>)
      -> tensor<1x2x2x2xf32>
      attributes {layer_type = "conv2d"} {
    %w = arith.constant dense<1.000000e+00> : tensor<2x1x2x2xf32>
    %0 = sculptor.nn.conv2d %arg0, %w {dilation = [1, 1], has_bias = false, padding = [0, 0], stride = [1, 1]}
        : (tensor<1x1x3x3xf32>, tensor<2x1x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %0 : tensor<1x2x2x2xf32>
  }

  // CHECK-LABEL: func.func @conv2d_bias
  // CHECK: sculptor.mvm {{.*}} : (tensor<1x4xf32>, tensor<2x4xf32>) -> tensor<1x2xf32>
  // CHECK: tensor.expand_shape
  // CHECK: linalg.add
  func.func @conv2d_bias(%arg0: tensor<1x1x3x3xf32>)
      -> tensor<1x2x2x2xf32>
      attributes {layer_type = "conv2d_w_bias"} {
    %w = arith.constant dense<2.000000e+00> : tensor<2x1x2x2xf32>
    %b = arith.constant dense<3.000000e+00> : tensor<2xf32>
    %0 = sculptor.nn.conv2d %arg0, %w, %b {dilation = [1, 1], has_bias = true, padding = [0, 0], stride = [1, 1]}
        : (tensor<1x1x3x3xf32>, tensor<2x1x2x2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
    return %0 : tensor<1x2x2x2xf32>
  }

  // CHECK-LABEL: func.func @grouped_conv2d_no_bias
  // CHECK: sculptor.mvm {{.*}} : (tensor<1x8xf32>, tensor<2x8xf32>) -> tensor<1x2xf32>
  func.func @grouped_conv2d_no_bias(%arg0: tensor<1x2x3x3xf32>)
      -> tensor<1x2x2x2xf32>
      attributes {layer_type = "conv2d_grouped"} {
    %w = arith.constant dense<1.000000e+00> : tensor<2x1x2x2xf32>
    %0 = sculptor.nn.grouped_conv2d %arg0, %w {dilation = [1, 1], groups = 2 : i64, has_bias = false, padding = [0, 0], stride = [1, 1]}
        : (tensor<1x2x3x3xf32>, tensor<2x1x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %0 : tensor<1x2x2x2xf32>
  }

  // CHECK-LABEL: func.func @grouped_conv2d_bias
  // CHECK: sculptor.mvm {{.*}} : (tensor<1x8xf32>, tensor<2x8xf32>) -> tensor<1x2xf32>
  // CHECK: tensor.expand_shape
  // CHECK: linalg.add
  func.func @grouped_conv2d_bias(%arg0: tensor<1x2x3x3xf32>)
      -> tensor<1x2x2x2xf32>
      attributes {layer_type = "conv2d_grouped_w_bias"} {
    %w = arith.constant dense<2.000000e+00> : tensor<2x1x2x2xf32>
    %b = arith.constant dense<3.000000e+00> : tensor<2xf32>
    %0 = sculptor.nn.grouped_conv2d %arg0, %w, %b {dilation = [1, 1], groups = 2 : i64, has_bias = true, padding = [0, 0], stride = [1, 1]}
        : (tensor<1x2x3x3xf32>, tensor<2x1x2x2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
    return %0 : tensor<1x2x2x2xf32>
  }

  // CHECK-LABEL: func.func @conv3d_no_bias
  // CHECK: sculptor.mvm {{.*}} : (tensor<1x8xf32>, tensor<2x8xf32>) -> tensor<1x2xf32>
  func.func @conv3d_no_bias(%arg0: tensor<1x1x3x3x3xf32>)
      -> tensor<1x2x2x2x2xf32>
      attributes {layer_type = "conv3d"} {
    %w = arith.constant dense<1.000000e+00> : tensor<2x1x2x2x2xf32>
    %0 = sculptor.nn.conv3d %arg0, %w {dilation = [1, 1, 1], has_bias = false, padding = [0, 0, 0], stride = [1, 1, 1]}
        : (tensor<1x1x3x3x3xf32>, tensor<2x1x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    return %0 : tensor<1x2x2x2x2xf32>
  }

  // CHECK-LABEL: func.func @conv3d_bias
  // CHECK: sculptor.mvm {{.*}} : (tensor<1x8xf32>, tensor<2x8xf32>) -> tensor<1x2xf32>
  // CHECK: tensor.expand_shape
  // CHECK: linalg.add
  func.func @conv3d_bias(%arg0: tensor<1x1x3x3x3xf32>)
      -> tensor<1x2x2x2x2xf32>
      attributes {layer_type = "conv3d_w_bias"} {
    %w = arith.constant dense<2.000000e+00> : tensor<2x1x2x2x2xf32>
    %b = arith.constant dense<3.000000e+00> : tensor<2xf32>
    %0 = sculptor.nn.conv3d %arg0, %w, %b {dilation = [1, 1, 1], has_bias = true, padding = [0, 0, 0], stride = [1, 1, 1]}
        : (tensor<1x1x3x3x3xf32>, tensor<2x1x2x2x2xf32>, tensor<2xf32>) -> tensor<1x2x2x2x2xf32>
    return %0 : tensor<1x2x2x2x2xf32>
  }
}
