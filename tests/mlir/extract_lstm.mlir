// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-extract-layers | FileCheck %s

module {
  // CHECK-LABEL: func.func @forward
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: tensor.collapse_shape
  // CHECK-NOT: tensor.expand_shape
  // CHECK: %[[L0:.*]]:3 = call @lstm_0_0(%arg0, %arg1, %arg2)
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: tensor.collapse_shape
  // CHECK-NOT: tensor.expand_shape
  // CHECK: %[[L1:.*]]:3 = call @lstm_0_1(%[[L0]]#0, %arg1, %arg2)
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: tensor.collapse_shape
  // CHECK-NOT: tensor.expand_shape
  // CHECK: %[[HN:.*]] = tensor.concat dim(0) %[[L0]]#1, %[[L1]]#1
  // CHECK: %[[CN:.*]] = tensor.concat dim(0) %[[L0]]#2, %[[L1]]#2
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: tensor.collapse_shape
  // CHECK-NOT: tensor.expand_shape
  // CHECK: return %[[L1]]#0, %[[HN]], %[[CN]]
  func.func @forward(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
                     %arg2: tensor<2x2x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>) {
    %w_ih_l0 = arith.constant dense<1.000000e+00> : tensor<12x4xf32>
    %w_hh_l0 = arith.constant dense<2.000000e+00> : tensor<12x3xf32>
    %w_ih_l1 = arith.constant dense<3.000000e+00> : tensor<12x3xf32>
    %w_hh_l1 = arith.constant dense<4.000000e+00> : tensor<12x3xf32>
    %output, %hn, %cn = sculptor.nn.lstm %arg0, %arg1, %arg2, recurrent[
        %w_ih_l0, %w_hh_l0, %w_ih_l1, %w_hh_l1]
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>,
           tensor<12x4xf32>, tensor<12x3xf32>, tensor<12x3xf32>,
           tensor<12x3xf32>)
          -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>)
    return %output, %hn, %cn
        : tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>
  }

  // CHECK-LABEL: func.func @lstm_0_0
  // CHECK-SAME: (%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>, %arg2: tensor<2x2x3xf32>)
  // CHECK-SAME: -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>)
  // CHECK-SAME: attributes {layer_type = "lstm"}
  // CHECK: %[[W0_IH:.*]] = arith.constant {{.*}} : tensor<12x4xf32>
  // CHECK: %[[W0_HH:.*]] = arith.constant {{.*}} : tensor<12x3xf32>
  // CHECK: sculptor.nn.lstm_layer %arg0, %arg1, %arg2, %[[W0_IH]], %[[W0_HH]]
  // CHECK-SAME: has_bias = false
  // CHECK-SAME: layer_index = 0 : i64
  // CHECK-SAME: num_layers = 2 : i64

  // CHECK-LABEL: func.func @lstm_0_1
  // CHECK-SAME: (%arg0: tensor<2x3x3xf32>, %arg1: tensor<2x2x3xf32>, %arg2: tensor<2x2x3xf32>)
  // CHECK-SAME: -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>)
  // CHECK-SAME: attributes {layer_type = "lstm"}
  // CHECK: %[[W1_IH:.*]] = arith.constant {{.*}} : tensor<12x3xf32>
  // CHECK: %[[W1_HH:.*]] = arith.constant {{.*}} : tensor<12x3xf32>
  // CHECK: sculptor.nn.lstm_layer %arg0, %arg1, %arg2, %[[W1_IH]], %[[W1_HH]]
  // CHECK-SAME: has_bias = false
  // CHECK-SAME: layer_index = 1 : i64
  // CHECK-SAME: num_layers = 2 : i64
}

// -----

module {
  // CHECK-LABEL: func.func @forward
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: tensor.collapse_shape
  // CHECK-NOT: tensor.expand_shape
  // CHECK: %[[L0:.*]]:3 = call @lstmwbias_0_0(%arg0, %arg1, %arg2)
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: tensor.collapse_shape
  // CHECK-NOT: tensor.expand_shape
  // CHECK: %[[L1:.*]]:3 = call @lstmwbias_0_1(%[[L0]]#0, %arg1, %arg2)
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: tensor.collapse_shape
  // CHECK-NOT: tensor.expand_shape
  // CHECK: %[[HN:.*]] = tensor.concat dim(0) %[[L0]]#1, %[[L1]]#1
  // CHECK: %[[CN:.*]] = tensor.concat dim(0) %[[L0]]#2, %[[L1]]#2
  // CHECK-NOT: tensor.extract_slice
  // CHECK-NOT: tensor.collapse_shape
  // CHECK-NOT: tensor.expand_shape
  // CHECK: return %[[L1]]#0, %[[HN]], %[[CN]]
  func.func @forward(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
                     %arg2: tensor<2x2x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>) {
    %w_ih_l0 = arith.constant dense<1.000000e+00> : tensor<12x4xf32>
    %w_hh_l0 = arith.constant dense<2.000000e+00> : tensor<12x3xf32>
    %b_ih_l0 = arith.constant dense<3.000000e+00> : tensor<12xf32>
    %b_hh_l0 = arith.constant dense<4.000000e+00> : tensor<12xf32>
    %w_ih_l1 = arith.constant dense<5.000000e+00> : tensor<12x3xf32>
    %w_hh_l1 = arith.constant dense<6.000000e+00> : tensor<12x3xf32>
    %b_ih_l1 = arith.constant dense<7.000000e+00> : tensor<12xf32>
    %b_hh_l1 = arith.constant dense<8.000000e+00> : tensor<12xf32>
    %output, %hn, %cn = sculptor.nn.lstm %arg0, %arg1, %arg2, recurrent[
        %w_ih_l0, %w_hh_l0, %b_ih_l0, %b_hh_l0,
        %w_ih_l1, %w_hh_l1, %b_ih_l1, %b_hh_l1]
        {batch_first = true, has_bias = true, hidden_size = 3 : i64,
         num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>,
           tensor<12x4xf32>, tensor<12x3xf32>, tensor<12xf32>,
           tensor<12xf32>, tensor<12x3xf32>, tensor<12x3xf32>,
           tensor<12xf32>, tensor<12xf32>)
          -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>)
    return %output, %hn, %cn
        : tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>
  }

  // CHECK-LABEL: func.func @lstmwbias_0_0
  // CHECK-SAME: (%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>, %arg2: tensor<2x2x3xf32>)
  // CHECK-SAME: -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>)
  // CHECK-SAME: attributes {layer_type = "lstm_w_bias"}
  // CHECK: %[[WB0_IH:.*]] = arith.constant {{.*}} : tensor<12x4xf32>
  // CHECK: %[[WB0_HH:.*]] = arith.constant {{.*}} : tensor<12x3xf32>
  // CHECK: %[[B0_IH:.*]] = arith.constant {{.*}} : tensor<12xf32>
  // CHECK: %[[B0_HH:.*]] = arith.constant {{.*}} : tensor<12xf32>
  // CHECK: sculptor.nn.lstm_layer %arg0, %arg1, %arg2, %[[WB0_IH]], %[[WB0_HH]], %[[B0_IH]], %[[B0_HH]]
  // CHECK-SAME: has_bias = true
  // CHECK-SAME: layer_index = 0 : i64
  // CHECK-SAME: num_layers = 2 : i64

  // CHECK-LABEL: func.func @lstmwbias_0_1
  // CHECK-SAME: (%arg0: tensor<2x3x3xf32>, %arg1: tensor<2x2x3xf32>, %arg2: tensor<2x2x3xf32>)
  // CHECK-SAME: -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>)
  // CHECK-SAME: attributes {layer_type = "lstm_w_bias"}
  // CHECK: %[[WB1_IH:.*]] = arith.constant {{.*}} : tensor<12x3xf32>
  // CHECK: %[[WB1_HH:.*]] = arith.constant {{.*}} : tensor<12x3xf32>
  // CHECK: %[[B1_IH:.*]] = arith.constant {{.*}} : tensor<12xf32>
  // CHECK: %[[B1_HH:.*]] = arith.constant {{.*}} : tensor<12xf32>
  // CHECK: sculptor.nn.lstm_layer %arg0, %arg1, %arg2, %[[WB1_IH]], %[[WB1_HH]], %[[B1_IH]], %[[B1_HH]]
  // CHECK-SAME: has_bias = true
  // CHECK-SAME: layer_index = 1 : i64
  // CHECK-SAME: num_layers = 2 : i64
}
