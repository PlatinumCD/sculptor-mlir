// RUN: sculptor-mlir-opt %s --sculptor-extract-layers | FileCheck %s --implicit-check-not="sculptor.nn.transformer %" --implicit-check-not='block_kind = "'

module {
  // CHECK-LABEL: func.func @forward
  // CHECK: %[[ENC0:.*]] = call @transformer_encoder_block_0(%arg0)
  // CHECK: %[[ENC1:.*]] = call @transformer_encoder_block_1(%[[ENC0]])
  // CHECK: %[[DEC0:.*]] = call @transformer_decoder_block_0(%arg1, %[[ENC1]])
  // CHECK: %[[DEC1:.*]] = call @transformer_decoder_block_1(%[[DEC0]], %[[ENC1]])
  // CHECK: return %[[DEC1]]
  func.func @forward(%arg0: tensor<1x3x4xf32>,
      %arg1: tensor<1x3x4xf32>) -> tensor<1x3x4xf32> {
    %qkv_w = arith.constant dense<1.000000e+00> : tensor<12x4xf32>
    %attn_w = arith.constant dense<2.000000e+00> : tensor<4x4xf32>
    %mlp_up_w = arith.constant dense<3.000000e+00> : tensor<8x4xf32>
    %mlp_down_w = arith.constant dense<4.000000e+00> : tensor<4x8xf32>
    %out = sculptor.nn.transformer %arg0, %arg1, parameters[
        %qkv_w, %attn_w, %mlp_up_w, %mlp_down_w,
        %qkv_w, %attn_w, %mlp_up_w, %mlp_down_w,
        %qkv_w, %attn_w, %attn_w, %mlp_up_w, %attn_w, %mlp_up_w,
        %mlp_down_w,
        %qkv_w, %attn_w, %attn_w, %mlp_up_w, %attn_w, %mlp_up_w,
        %mlp_down_w]
        {activation = "gelu", batch_first = true, causal = false,
         has_final_norm = true, has_layer_norm_affine = false,
         has_projection_bias = false, head_dim = 2 : i64, hidden_size = 4 : i64,
         layer_norm_eps = 1.000000e-05 : f64, mlp_hidden_size = 8 : i64,
         norm_mode = "post", num_decoder_blocks = 2 : i64,
         num_encoder_blocks = 2 : i64, num_heads = 2 : i64}
        : (tensor<1x3x4xf32>, tensor<1x3x4xf32>,
           tensor<12x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>,
           tensor<4x8xf32>,
           tensor<12x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>,
           tensor<4x8xf32>,
           tensor<12x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>,
           tensor<8x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>,
           tensor<4x8xf32>,
           tensor<12x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>,
           tensor<8x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>,
           tensor<4x8xf32>) -> tensor<1x3x4xf32>
    return %out : tensor<1x3x4xf32>
  }

  // CHECK-LABEL: func.func @transformer_encoder_block_0
  // CHECK-SAME: attributes {layer_type = "transformer_encoder_block"}
  // CHECK: sculptor.nn.transformer_block
  // CHECK-SAME: memory[]
  // CHECK-SAME: final_norm[]
  // CHECK-SAME: block_kind = encoder
  // CHECK-SAME: block_index = 0 : i64
  // CHECK-SAME: has_final_norm = false
  // CHECK-SAME: num_blocks = 2 : i64

  // CHECK-LABEL: func.func @transformer_encoder_block_1
  // CHECK-SAME: attributes {layer_type = "transformer_encoder_block"}
  // CHECK: sculptor.nn.transformer_block
  // CHECK-SAME: final_norm[]
  // CHECK-SAME: block_kind = encoder
  // CHECK-SAME: block_index = 1 : i64
  // CHECK-SAME: has_final_norm = true
  // CHECK-SAME: num_blocks = 2 : i64

  // CHECK-LABEL: func.func @transformer_decoder_block_0
  // CHECK-SAME: (%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x3x4xf32>)
  // CHECK-SAME: attributes {layer_type = "transformer_decoder_block"}
  // CHECK: sculptor.nn.transformer_block %arg0, memory[%arg1]
  // CHECK-SAME: block_kind = decoder
  // CHECK-SAME: block_index = 0 : i64
  // CHECK-SAME: has_final_norm = false

  // CHECK-LABEL: func.func @transformer_decoder_block_1
  // CHECK-SAME: (%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x3x4xf32>)
  // CHECK-SAME: attributes {layer_type = "transformer_decoder_block"}
  // CHECK: sculptor.nn.transformer_block %arg0, memory[%arg1]
  // CHECK-SAME: block_kind = decoder
  // CHECK-SAME: block_index = 1 : i64
  // CHECK-SAME: has_final_norm = true
}
