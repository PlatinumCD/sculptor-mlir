// RUN: sculptor-mlir-opt %s --split-input-file --verify-diagnostics

module {
  func.func @valid_gru(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
      %arg2: tensor<9x4xf32>, %arg3: tensor<9x3xf32>,
      %arg4: tensor<9x3xf32>, %arg5: tensor<9x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>) {
    %output, %hn = sculptor.nn.gru %arg0, %arg1,
        recurrent[%arg2, %arg3, %arg4, %arg5]
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<9x4xf32>,
           tensor<9x3xf32>, tensor<9x3xf32>, tensor<9x3xf32>)
          -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>)
    return %output, %hn : tensor<2x3x3xf32>, tensor<2x2x3xf32>
  }
}

// -----

module {
  func.func @valid_gru_bias(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
      %arg2: tensor<9x4xf32>, %arg3: tensor<9x3xf32>,
      %arg4: tensor<9xf32>, %arg5: tensor<9xf32>,
      %arg6: tensor<9x3xf32>, %arg7: tensor<9x3xf32>,
      %arg8: tensor<9xf32>, %arg9: tensor<9xf32>)
      -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>) {
    %output, %hn = sculptor.nn.gru %arg0, %arg1,
        recurrent[%arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9]
        {batch_first = true, has_bias = true, hidden_size = 3 : i64,
         num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<9x4xf32>,
           tensor<9x3xf32>, tensor<9xf32>, tensor<9xf32>,
           tensor<9x3xf32>, tensor<9x3xf32>, tensor<9xf32>,
           tensor<9xf32>) -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>)
    return %output, %hn : tensor<2x3x3xf32>, tensor<2x2x3xf32>
  }
}

// -----

module {
  func.func @invalid_hidden_rank(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3xf32>,
      %arg2: tensor<9x4xf32>, %arg3: tensor<9x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>) {
    // expected-error @+1 {{expected h0 tensor rank to be 3}}
    %output, %hn = sculptor.nn.gru %arg0, %arg1,
        recurrent[%arg2, %arg3]
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         num_layers = 1 : i64}
        : (tensor<2x3x4xf32>, tensor<2x3xf32>, tensor<9x4xf32>,
           tensor<9x3xf32>) -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>)
    return %output, %hn : tensor<2x3x3xf32>, tensor<1x2x3xf32>
  }
}

// -----

module {
  func.func @invalid_recurrent_operand_count(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
      %arg2: tensor<9x4xf32>, %arg3: tensor<9x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>) {
    // expected-error @+1 {{expected recurrent operand count (2) to match 4 for 2 layers with has_bias = false}}
    %output, %hn = sculptor.nn.gru %arg0, %arg1,
        recurrent[%arg2, %arg3]
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<9x4xf32>,
           tensor<9x3xf32>) -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>)
    return %output, %hn : tensor<2x3x3xf32>, tensor<2x2x3xf32>
  }
}

// -----

module {
  func.func @invalid_weight_shape(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<1x2x3xf32>,
      %arg2: tensor<8x4xf32>, %arg3: tensor<9x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>) {
    // expected-error @+1 {{expected w_ih output dimension (8) to match 3 * hidden state dimension (9)}}
    %output, %hn = sculptor.nn.gru %arg0, %arg1,
        recurrent[%arg2, %arg3]
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         num_layers = 1 : i64}
        : (tensor<2x3x4xf32>, tensor<1x2x3xf32>, tensor<8x4xf32>,
           tensor<9x3xf32>) -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>)
    return %output, %hn : tensor<2x3x3xf32>, tensor<1x2x3xf32>
  }
}

// -----

module {
  func.func @invalid_bias_shape(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<1x2x3xf32>,
      %arg2: tensor<9x4xf32>, %arg3: tensor<9x3xf32>,
      %arg4: tensor<8xf32>, %arg5: tensor<9xf32>)
      -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>) {
    // expected-error @+1 {{expected b_ih dimension (8) to match 3 * hidden state dimension (9)}}
    %output, %hn = sculptor.nn.gru %arg0, %arg1,
        recurrent[%arg2, %arg3, %arg4, %arg5]
        {batch_first = true, has_bias = true, hidden_size = 3 : i64,
         num_layers = 1 : i64}
        : (tensor<2x3x4xf32>, tensor<1x2x3xf32>, tensor<9x4xf32>,
           tensor<9x3xf32>, tensor<8xf32>, tensor<9xf32>)
          -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>)
    return %output, %hn : tensor<2x3x3xf32>, tensor<1x2x3xf32>
  }
}
