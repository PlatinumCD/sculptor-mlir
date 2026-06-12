// RUN: sculptor-mlir-opt %s --split-input-file --verify-diagnostics

module {
  func.func @valid_rnn_layer(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
      %arg2: tensor<3x4xf32>, %arg3: tensor<3x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>) {
    %output, %hn = sculptor.nn.rnn_layer %arg0, %arg1, %arg2, %arg3
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         layer_index = 0 : i64, num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<3x4xf32>,
           tensor<3x3xf32>) -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>)
    return %output, %hn : tensor<2x3x3xf32>, tensor<1x2x3xf32>
  }
}

// -----

module {
  func.func @valid_rnn_layer_bias(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
      %arg2: tensor<3x4xf32>, %arg3: tensor<3x3xf32>,
      %arg4: tensor<3xf32>, %arg5: tensor<3xf32>)
      -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>) {
    %output, %hn = sculptor.nn.rnn_layer %arg0, %arg1, %arg2, %arg3, %arg4, %arg5
        {batch_first = true, has_bias = true, hidden_size = 3 : i64,
         layer_index = 1 : i64, num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<3x4xf32>,
           tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>)
          -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>)
    return %output, %hn : tensor<2x3x3xf32>, tensor<1x2x3xf32>
  }
}

// -----

module {
  func.func @invalid_hidden_rank(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3xf32>,
      %arg2: tensor<3x4xf32>, %arg3: tensor<3x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>) {
    // expected-error @+1 {{expected h0 tensor rank to be 3}}
    %output, %hn = sculptor.nn.rnn_layer %arg0, %arg1, %arg2, %arg3
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         layer_index = 0 : i64, num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x3xf32>, tensor<3x4xf32>,
           tensor<3x3xf32>) -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>)
    return %output, %hn : tensor<2x3x3xf32>, tensor<1x2x3xf32>
  }
}

// -----

module {
  func.func @invalid_layer_index(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
      %arg2: tensor<3x4xf32>, %arg3: tensor<3x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>) {
    // expected-error @+1 {{expected layer_index (2) to be within [0, 2)}}
    %output, %hn = sculptor.nn.rnn_layer %arg0, %arg1, %arg2, %arg3
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         layer_index = 2 : i64, num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<3x4xf32>,
           tensor<3x3xf32>) -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>)
    return %output, %hn : tensor<2x3x3xf32>, tensor<1x2x3xf32>
  }
}

// -----

module {
  func.func @invalid_weight_shape(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
      %arg2: tensor<4x4xf32>, %arg3: tensor<3x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>) {
    // expected-error @+1 {{expected w_ih output dimension (4) to match hidden_size (3)}}
    %output, %hn = sculptor.nn.rnn_layer %arg0, %arg1, %arg2, %arg3
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         layer_index = 0 : i64, num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<4x4xf32>,
           tensor<3x3xf32>) -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>)
    return %output, %hn : tensor<2x3x3xf32>, tensor<1x2x3xf32>
  }
}
