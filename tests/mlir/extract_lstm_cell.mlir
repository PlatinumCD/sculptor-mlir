// RUN: sculptor-mlir-opt %s --sculptor-extract-layers | FileCheck %s

module {
  // CHECK-LABEL: func.func @forward
  // CHECK: %[[BIAS:.*]]:2 = call @lstmcellwbias_0(%arg0, %arg1, %arg2) : (tensor<1x4xf32>, tensor<1x3xf32>, tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<1x3xf32>)
  // CHECK: %[[NO_BIAS:.*]]:2 = call @lstmcell_0(%arg3, %arg4, %arg5) : (tensor<1x5xf32>, tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
  // CHECK: return %[[BIAS]]#0, %[[BIAS]]#1, %[[NO_BIAS]]#0, %[[NO_BIAS]]#1
  func.func @forward(
      %x: tensor<1x4xf32>,
      %h: tensor<1x3xf32>,
      %c: tensor<1x3xf32>,
      %x_no_bias: tensor<1x5xf32>,
      %h_no_bias: tensor<1x2xf32>,
      %c_no_bias: tensor<1x2xf32>
  ) -> (tensor<1x3xf32>, tensor<1x3xf32>,
        tensor<1x2xf32>, tensor<1x2xf32>) {
    %w_ih = arith.constant dense<1.000000e+00> : tensor<12x4xf32>
    %w_hh = arith.constant dense<2.000000e+00> : tensor<12x3xf32>
    %b_ih = arith.constant dense<3.000000e+00> : tensor<12xf32>
    %b_hh = arith.constant dense<4.000000e+00> : tensor<12xf32>
    %h_next, %c_next = sculptor.nn.lstm_cell
        %x, %h, %c, %w_ih, %w_hh, %b_ih, %b_hh
        {has_bias = true}
        : (tensor<1x4xf32>, tensor<1x3xf32>, tensor<1x3xf32>,
           tensor<12x4xf32>, tensor<12x3xf32>, tensor<12xf32>,
           tensor<12xf32>) -> (tensor<1x3xf32>, tensor<1x3xf32>)

    %nb_w_ih = arith.constant dense<5.000000e+00> : tensor<8x5xf32>
    %nb_w_hh = arith.constant dense<6.000000e+00> : tensor<8x2xf32>
    %nb_h_next, %nb_c_next = sculptor.nn.lstm_cell
        %x_no_bias, %h_no_bias, %c_no_bias, %nb_w_ih, %nb_w_hh
        {has_bias = false}
        : (tensor<1x5xf32>, tensor<1x2xf32>, tensor<1x2xf32>,
           tensor<8x5xf32>, tensor<8x2xf32>)
          -> (tensor<1x2xf32>, tensor<1x2xf32>)

    return %h_next, %c_next, %nb_h_next, %nb_c_next
        : tensor<1x3xf32>, tensor<1x3xf32>,
          tensor<1x2xf32>, tensor<1x2xf32>
  }

  // CHECK-LABEL: func.func @lstmcellwbias_0
  // CHECK-SAME: (%arg0: tensor<1x4xf32>, %arg1: tensor<1x3xf32>, %arg2: tensor<1x3xf32>)
  // CHECK-SAME: attributes {layer_type = "lstm_cell_w_bias"}
  // CHECK: arith.constant dense<1.000000e+00> : tensor<12x4xf32>
  // CHECK: arith.constant dense<2.000000e+00> : tensor<12x3xf32>
  // CHECK: arith.constant dense<3.000000e+00> : tensor<12xf32>
  // CHECK: arith.constant dense<4.000000e+00> : tensor<12xf32>
  // CHECK: %[[OUT_H:.*]], %[[OUT_C:.*]] = sculptor.nn.lstm_cell %arg0, %arg1, %arg2
  // CHECK-SAME: has_bias = true
  // CHECK: return %[[OUT_H]], %[[OUT_C]]

  // CHECK-LABEL: func.func @lstmcell_0
  // CHECK-SAME: (%arg0: tensor<1x5xf32>, %arg1: tensor<1x2xf32>, %arg2: tensor<1x2xf32>)
  // CHECK-SAME: attributes {layer_type = "lstm_cell"}
  // CHECK: arith.constant dense<5.000000e+00> : tensor<8x5xf32>
  // CHECK: arith.constant dense<6.000000e+00> : tensor<8x2xf32>
  // CHECK: %[[NB_OUT_H:.*]], %[[NB_OUT_C:.*]] = sculptor.nn.lstm_cell %arg0, %arg1, %arg2
  // CHECK-SAME: has_bias = false
  // CHECK: return %[[NB_OUT_H]], %[[NB_OUT_C]]
}
