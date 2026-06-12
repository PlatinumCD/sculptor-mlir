// RUN: sculptor-mlir-opt %s --sculptor-extract-layers | FileCheck %s

module {
  // CHECK-LABEL: func.func @forward
  // CHECK: %[[BIAS_CALL:.*]] = call @grucellwbias_0(%arg0, %arg1) : (tensor<1x4xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  // CHECK: %[[NO_BIAS_CALL:.*]] = call @grucell_0(%arg2, %arg3) : (tensor<1x5xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK: return %[[BIAS_CALL]], %[[NO_BIAS_CALL]]
  func.func @forward(
      %x: tensor<1x4xf32>,
      %h: tensor<1x3xf32>,
      %x_no_bias: tensor<1x5xf32>,
      %h_no_bias: tensor<1x2xf32>
  ) -> (tensor<1x3xf32>, tensor<1x2xf32>) {
    %w_ih = arith.constant dense<1.000000e+00> : tensor<9x4xf32>
    %w_hh = arith.constant dense<2.000000e+00> : tensor<9x3xf32>
    %b_ih = arith.constant dense<3.000000e+00> : tensor<9xf32>
    %b_hh = arith.constant dense<4.000000e+00> : tensor<9xf32>
    %0 = sculptor.nn.gru_cell %x, %h, %w_ih, %w_hh, %b_ih, %b_hh
        {has_bias = true}
        : (tensor<1x4xf32>, tensor<1x3xf32>, tensor<9x4xf32>,
           tensor<9x3xf32>, tensor<9xf32>, tensor<9xf32>)
          -> tensor<1x3xf32>

    %nb_w_ih = arith.constant dense<5.000000e+00> : tensor<6x5xf32>
    %nb_w_hh = arith.constant dense<6.000000e+00> : tensor<6x2xf32>
    %1 = sculptor.nn.gru_cell %x_no_bias, %h_no_bias, %nb_w_ih, %nb_w_hh
        {has_bias = false}
        : (tensor<1x5xf32>, tensor<1x2xf32>, tensor<6x5xf32>,
           tensor<6x2xf32>) -> tensor<1x2xf32>

    return %0, %1 : tensor<1x3xf32>, tensor<1x2xf32>
  }

  // CHECK-LABEL: func.func @grucellwbias_0
  // CHECK-SAME: (%arg0: tensor<1x4xf32>, %arg1: tensor<1x3xf32>)
  // CHECK-SAME: attributes {layer_type = "gru_cell_w_bias"}
  // CHECK: arith.constant dense<1.000000e+00> : tensor<9x4xf32>
  // CHECK: arith.constant dense<2.000000e+00> : tensor<9x3xf32>
  // CHECK: arith.constant dense<3.000000e+00> : tensor<9xf32>
  // CHECK: arith.constant dense<4.000000e+00> : tensor<9xf32>
  // CHECK: %[[OUT:.*]] = sculptor.nn.gru_cell %arg0, %arg1
  // CHECK-SAME: has_bias = true
  // CHECK: return %[[OUT]]

  // CHECK-LABEL: func.func @grucell_0
  // CHECK-SAME: (%arg0: tensor<1x5xf32>, %arg1: tensor<1x2xf32>)
  // CHECK-SAME: attributes {layer_type = "gru_cell"}
  // CHECK: arith.constant dense<5.000000e+00> : tensor<6x5xf32>
  // CHECK: arith.constant dense<6.000000e+00> : tensor<6x2xf32>
  // CHECK: %[[NB_OUT:.*]] = sculptor.nn.gru_cell %arg0, %arg1
  // CHECK-SAME: has_bias = false
  // CHECK: return %[[NB_OUT]]
}
