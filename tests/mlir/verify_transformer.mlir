// RUN: sculptor-mlir-opt %s --split-input-file --verify-diagnostics

module {
  func.func @valid_transformer(%arg0: tensor<1x3x4xf32>,
      %arg1: tensor<1x3x4xf32>)
      -> tensor<1x3x4xf32> {
    %qkv_w = arith.constant dense<1.000000e+00> : tensor<12x4xf32>
    %attn_w = arith.constant dense<3.000000e+00> : tensor<4x4xf32>
    %mlp_up_w = arith.constant dense<7.000000e+00> : tensor<8x4xf32>
    %mlp_down_w = arith.constant dense<9.000000e+00> : tensor<4x8xf32>
    %out = sculptor.nn.transformer %arg0, %arg1, parameters[
        %qkv_w, %attn_w, %mlp_up_w, %mlp_down_w,
        %qkv_w, %attn_w, %attn_w, %mlp_up_w, %attn_w, %mlp_up_w,
        %mlp_down_w]
        {activation = "gelu", batch_first = true, causal = false,
         has_final_norm = false, has_layer_norm_affine = false,
         has_projection_bias = false, head_dim = 2 : i64, hidden_size = 4 : i64,
         layer_norm_eps = 1.000000e-05 : f64, mlp_hidden_size = 8 : i64,
         norm_mode = "post", num_decoder_blocks = 1 : i64,
         num_encoder_blocks = 1 : i64, num_heads = 2 : i64}
        : (tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<12x4xf32>,
           tensor<4x4xf32>, tensor<8x4xf32>, tensor<4x8xf32>,
           tensor<12x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>,
           tensor<8x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>,
           tensor<4x8xf32>)
          -> tensor<1x3x4xf32>
    return %out : tensor<1x3x4xf32>
  }
}

// -----

module {
  func.func @valid_transformer_block(%arg0: tensor<1x3x4xf32>)
      -> tensor<1x3x4xf32> {
    %qkv_w = arith.constant dense<1.000000e+00> : tensor<12x4xf32>
    %qkv_b = arith.constant dense<2.000000e+00> : tensor<12xf32>
    %attn_w = arith.constant dense<3.000000e+00> : tensor<4x4xf32>
    %attn_b = arith.constant dense<4.000000e+00> : tensor<4xf32>
    %attn_norm_w = arith.constant dense<5.000000e+00> : tensor<4xf32>
    %attn_norm_b = arith.constant dense<6.000000e+00> : tensor<4xf32>
    %mlp_up_w = arith.constant dense<7.000000e+00> : tensor<8x4xf32>
    %mlp_up_b = arith.constant dense<8.000000e+00> : tensor<8xf32>
    %mlp_down_w = arith.constant dense<9.000000e+00> : tensor<4x8xf32>
    %mlp_down_b = arith.constant dense<1.000000e+01> : tensor<4xf32>
    %mlp_norm_w = arith.constant dense<1.100000e+01> : tensor<4xf32>
    %mlp_norm_b = arith.constant dense<1.200000e+01> : tensor<4xf32>
    %out = sculptor.nn.transformer_block %arg0, memory[],
        qkv[%qkv_w, %qkv_b],
        attn_output[%attn_w, %attn_b],
        attn_norm[%attn_norm_w, %attn_norm_b],
        cross_query[], cross_key_value[], cross_output[], cross_norm[],
        mlp_up[%mlp_up_w, %mlp_up_b], mlp_down[%mlp_down_w, %mlp_down_b],
        mlp_norm[%mlp_norm_w, %mlp_norm_b], final_norm[] block_kind = encoder
        {activation = "gelu", batch_first = true, block_index = 0 : i64,
         causal = false, has_layer_norm_affine = true,
         has_cross_attention = false, has_final_norm = false,
         has_projection_bias = true, head_dim = 2 : i64,
         hidden_size = 4 : i64, layer_norm_eps = 1.000000e-05 : f64,
         mlp_hidden_size = 8 : i64, norm_mode = "post", num_blocks = 1 : i64,
         num_heads = 2 : i64}
        : (tensor<1x3x4xf32>, tensor<12x4xf32>, tensor<12xf32>,
           tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>,
           tensor<8x4xf32>, tensor<8xf32>, tensor<4x8xf32>, tensor<4xf32>,
           tensor<4xf32>, tensor<4xf32>) -> tensor<1x3x4xf32>
    return %out : tensor<1x3x4xf32>
  }
}

// -----

module {
  func.func @valid_decoder_transformer_block(%arg0: tensor<1x3x4xf32>,
      %memory: tensor<1x5x4xf32>) -> tensor<1x3x4xf32> {
    %qkv_w = arith.constant dense<1.000000e+00> : tensor<12x4xf32>
    %qkv_b = arith.constant dense<2.000000e+00> : tensor<12xf32>
    %attn_w = arith.constant dense<3.000000e+00> : tensor<4x4xf32>
    %attn_b = arith.constant dense<4.000000e+00> : tensor<4xf32>
    %attn_norm_w = arith.constant dense<5.000000e+00> : tensor<4xf32>
    %attn_norm_b = arith.constant dense<6.000000e+00> : tensor<4xf32>
    %mlp_up_w = arith.constant dense<7.000000e+00> : tensor<8x4xf32>
    %mlp_up_b = arith.constant dense<8.000000e+00> : tensor<8xf32>
    %mlp_down_w = arith.constant dense<9.000000e+00> : tensor<4x8xf32>
    %mlp_down_b = arith.constant dense<1.000000e+01> : tensor<4xf32>
    %mlp_norm_w = arith.constant dense<1.100000e+01> : tensor<4xf32>
    %mlp_norm_b = arith.constant dense<1.200000e+01> : tensor<4xf32>
    %out = sculptor.nn.transformer_block %arg0, memory[%memory],
        qkv[%qkv_w, %qkv_b],
        attn_output[%attn_w, %attn_b],
        attn_norm[%attn_norm_w, %attn_norm_b],
        cross_query[%attn_w, %attn_b],
        cross_key_value[%mlp_up_w, %mlp_up_b],
        cross_output[%attn_w, %attn_b],
        cross_norm[%attn_norm_w, %attn_norm_b],
        mlp_up[%mlp_up_w, %mlp_up_b], mlp_down[%mlp_down_w, %mlp_down_b],
        mlp_norm[%mlp_norm_w, %mlp_norm_b],
        final_norm[%mlp_norm_w, %mlp_norm_b] block_kind = decoder
        {activation = "gelu", batch_first = true, block_index = 0 : i64,
         causal = false, has_layer_norm_affine = true,
         has_cross_attention = true, has_final_norm = true,
         has_projection_bias = true, head_dim = 2 : i64,
         hidden_size = 4 : i64, layer_norm_eps = 1.000000e-05 : f64,
         mlp_hidden_size = 8 : i64, norm_mode = "post", num_blocks = 1 : i64,
         num_heads = 2 : i64}
        : (tensor<1x3x4xf32>, tensor<1x5x4xf32>,
           tensor<12x4xf32>, tensor<12xf32>,
           tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>,
           tensor<4x4xf32>, tensor<4xf32>, tensor<8x4xf32>, tensor<8xf32>,
           tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>,
           tensor<8x4xf32>, tensor<8xf32>, tensor<4x8xf32>, tensor<4xf32>,
           tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
          -> tensor<1x3x4xf32>
    return %out : tensor<1x3x4xf32>
  }
}

// -----

module {
  func.func @invalid_transformer_parameter_count(%arg0: tensor<1x3x4xf32>,
      %arg1: tensor<1x3x4xf32>)
      -> tensor<1x3x4xf32> {
    %qkv_w = arith.constant dense<1.000000e+00> : tensor<12x4xf32>
    %attn_w = arith.constant dense<3.000000e+00> : tensor<4x4xf32>
    %mlp_up_w = arith.constant dense<7.000000e+00> : tensor<8x4xf32>
    %mlp_down_w = arith.constant dense<9.000000e+00> : tensor<4x8xf32>
    // expected-error @+1 {{expected transformer parameter count (10) to match 11}}
    %out = sculptor.nn.transformer %arg0, %arg1, parameters[
        %qkv_w, %attn_w, %mlp_up_w, %mlp_down_w,
        %qkv_w, %attn_w, %attn_w, %mlp_up_w, %attn_w, %mlp_up_w]
        {activation = "gelu", batch_first = true, causal = false,
         has_final_norm = false, has_layer_norm_affine = false,
         has_projection_bias = false, head_dim = 2 : i64, hidden_size = 4 : i64,
         layer_norm_eps = 1.000000e-05 : f64, mlp_hidden_size = 8 : i64,
         norm_mode = "post", num_decoder_blocks = 1 : i64,
         num_encoder_blocks = 1 : i64, num_heads = 2 : i64}
        : (tensor<1x3x4xf32>, tensor<1x3x4xf32>, tensor<12x4xf32>,
           tensor<4x4xf32>, tensor<8x4xf32>, tensor<4x8xf32>,
           tensor<12x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>,
           tensor<8x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>)
          -> tensor<1x3x4xf32>
    return %out : tensor<1x3x4xf32>
  }
}

// -----

module {
  func.func @invalid_transformer_block_index(%arg0: tensor<1x3x4xf32>)
      -> tensor<1x3x4xf32> {
    %qkv_w = arith.constant dense<1.000000e+00> : tensor<12x4xf32>
    %attn_w = arith.constant dense<3.000000e+00> : tensor<4x4xf32>
    %mlp_up_w = arith.constant dense<7.000000e+00> : tensor<8x4xf32>
    %mlp_down_w = arith.constant dense<9.000000e+00> : tensor<4x8xf32>
    // expected-error @+1 {{expected block_index (2) to be within [0, 2)}}
    %out = sculptor.nn.transformer_block %arg0, memory[], qkv[%qkv_w],
        attn_output[%attn_w], attn_norm[],
        cross_query[], cross_key_value[], cross_output[], cross_norm[],
        mlp_up[%mlp_up_w],
        mlp_down[%mlp_down_w], mlp_norm[], final_norm[] block_kind = encoder
        {activation = "gelu", batch_first = true, block_index = 2 : i64,
         causal = false, has_layer_norm_affine = false,
         has_cross_attention = false, has_final_norm = false,
         has_projection_bias = false, head_dim = 2 : i64,
         hidden_size = 4 : i64, layer_norm_eps = 1.000000e-05 : f64,
         mlp_hidden_size = 8 : i64, norm_mode = "post",
         num_blocks = 2 : i64, num_heads = 2 : i64}
        : (tensor<1x3x4xf32>, tensor<12x4xf32>, tensor<4x4xf32>,
           tensor<8x4xf32>, tensor<4x8xf32>) -> tensor<1x3x4xf32>
    return %out : tensor<1x3x4xf32>
  }
}

// -----

module {
  func.func @invalid_qkv_weight(%arg0: tensor<1x3x4xf32>)
      -> tensor<1x3x4xf32> {
    %qkv_w = arith.constant dense<1.000000e+00> : tensor<11x4xf32>
    %attn_w = arith.constant dense<3.000000e+00> : tensor<4x4xf32>
    %mlp_up_w = arith.constant dense<7.000000e+00> : tensor<8x4xf32>
    %mlp_down_w = arith.constant dense<9.000000e+00> : tensor<4x8xf32>
    // expected-error @+1 {{expected qkv_weight output dimension (11) to match expected output dimension (12)}}
    %out = sculptor.nn.transformer_block %arg0, memory[], qkv[%qkv_w],
        attn_output[%attn_w], attn_norm[],
        cross_query[], cross_key_value[], cross_output[], cross_norm[],
        mlp_up[%mlp_up_w],
        mlp_down[%mlp_down_w], mlp_norm[], final_norm[] block_kind = encoder
        {activation = "gelu", batch_first = true, block_index = 0 : i64,
         causal = false, has_layer_norm_affine = false,
         has_cross_attention = false, has_final_norm = false,
         has_projection_bias = false, head_dim = 2 : i64,
         hidden_size = 4 : i64, layer_norm_eps = 1.000000e-05 : f64,
         mlp_hidden_size = 8 : i64, norm_mode = "post",
         num_blocks = 1 : i64, num_heads = 2 : i64}
        : (tensor<1x3x4xf32>, tensor<11x4xf32>, tensor<4x4xf32>,
           tensor<8x4xf32>, tensor<4x8xf32>) -> tensor<1x3x4xf32>
    return %out : tensor<1x3x4xf32>
  }
}
