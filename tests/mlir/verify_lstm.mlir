// RUN: sculptor-mlir-opt %s --split-input-file --verify-diagnostics

module {
  func.func @valid_lstm(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
      %arg2: tensor<2x2x3xf32>, %arg3: tensor<12x4xf32>,
      %arg4: tensor<12x3xf32>, %arg5: tensor<12x3xf32>,
      %arg6: tensor<12x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>) {
    %output, %hn, %cn = sculptor.nn.lstm %arg0, %arg1, %arg2,
        recurrent[%arg3, %arg4, %arg5, %arg6]
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>,
           tensor<12x4xf32>, tensor<12x3xf32>, tensor<12x3xf32>,
           tensor<12x3xf32>)
          -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>)
    return %output, %hn, %cn
        : tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>
  }
}

// -----

module {
  func.func @valid_lstm_bias(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
      %arg2: tensor<2x2x3xf32>, %arg3: tensor<12x4xf32>,
      %arg4: tensor<12x3xf32>, %arg5: tensor<12xf32>,
      %arg6: tensor<12xf32>, %arg7: tensor<12x3xf32>,
      %arg8: tensor<12x3xf32>, %arg9: tensor<12xf32>,
      %arg10: tensor<12xf32>)
      -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>) {
    %output, %hn, %cn = sculptor.nn.lstm %arg0, %arg1, %arg2,
        recurrent[%arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10]
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
}

// -----

module {
  func.func @invalid_cell_state_rank(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
      %arg2: tensor<2x3xf32>, %arg3: tensor<12x4xf32>,
      %arg4: tensor<12x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>) {
    // expected-error @+1 {{expected c0 tensor rank to be 3}}
    %output, %hn, %cn = sculptor.nn.lstm %arg0, %arg1, %arg2,
        recurrent[%arg3, %arg4]
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         num_layers = 1 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<2x3xf32>,
           tensor<12x4xf32>, tensor<12x3xf32>)
          -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>)
    return %output, %hn, %cn
        : tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>
  }
}

// -----

module {
  func.func @invalid_recurrent_operand_count(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>,
      %arg2: tensor<2x2x3xf32>, %arg3: tensor<12x4xf32>,
      %arg4: tensor<12x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>) {
    // expected-error @+1 {{expected recurrent operand count (2) to match 4 for 2 layers with has_bias = false}}
    %output, %hn, %cn = sculptor.nn.lstm %arg0, %arg1, %arg2,
        recurrent[%arg3, %arg4]
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         num_layers = 2 : i64}
        : (tensor<2x3x4xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>,
           tensor<12x4xf32>, tensor<12x3xf32>)
          -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>)
    return %output, %hn, %cn
        : tensor<2x3x3xf32>, tensor<2x2x3xf32>, tensor<2x2x3xf32>
  }
}

// -----

module {
  func.func @invalid_weight_shape(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<1x2x3xf32>,
      %arg2: tensor<1x2x3xf32>, %arg3: tensor<11x4xf32>,
      %arg4: tensor<12x3xf32>)
      -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>) {
    // expected-error @+1 {{expected w_ih output dimension (11) to match 4 * hidden state dimension (12)}}
    %output, %hn, %cn = sculptor.nn.lstm %arg0, %arg1, %arg2,
        recurrent[%arg3, %arg4]
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         num_layers = 1 : i64}
        : (tensor<2x3x4xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>,
           tensor<11x4xf32>, tensor<12x3xf32>)
          -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>)
    return %output, %hn, %cn
        : tensor<2x3x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>
  }
}

// -----

module {
  func.func @invalid_bias_shape(
      %arg0: tensor<2x3x4xf32>, %arg1: tensor<1x2x3xf32>,
      %arg2: tensor<1x2x3xf32>, %arg3: tensor<12x4xf32>,
      %arg4: tensor<12x3xf32>, %arg5: tensor<11xf32>,
      %arg6: tensor<12xf32>)
      -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>) {
    // expected-error @+1 {{expected b_ih dimension (11) to match 4 * hidden state dimension (12)}}
    %output, %hn, %cn = sculptor.nn.lstm %arg0, %arg1, %arg2,
        recurrent[%arg3, %arg4, %arg5, %arg6]
        {batch_first = true, has_bias = true, hidden_size = 3 : i64,
         num_layers = 1 : i64}
        : (tensor<2x3x4xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>,
           tensor<12x4xf32>, tensor<12x3xf32>, tensor<11xf32>,
           tensor<12xf32>)
          -> (tensor<2x3x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>)
    return %output, %hn, %cn
        : tensor<2x3x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>
  }
}
