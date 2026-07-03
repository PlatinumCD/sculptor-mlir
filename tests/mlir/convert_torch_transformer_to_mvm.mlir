// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_1enc_1dec.mlir --sculptor-canonicalize-layers --sculptor-extract-layers --sculptor-convert-layers | FileCheck %s --check-prefix=CONVERT11 --implicit-check-not="sculptor.nn.transformer"
// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_2enc_1dec.mlir --sculptor-canonicalize-layers --sculptor-extract-layers --sculptor-convert-layers | FileCheck %s --check-prefix=CONVERT21 --implicit-check-not="sculptor.nn.transformer"
// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_1enc_2dec.mlir --sculptor-canonicalize-layers --sculptor-extract-layers --sculptor-convert-layers | FileCheck %s --check-prefix=CONVERT12 --implicit-check-not="sculptor.nn.transformer"
// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_2enc_2dec.mlir --sculptor-canonicalize-layers --sculptor-extract-layers --sculptor-convert-layers | FileCheck %s --check-prefix=CONVERT22 --implicit-check-not="sculptor.nn.transformer"
// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_1enc_1dec.mlir --sculptor-canonicalize-layers --sculptor-extract-layers --sculptor-convert-layers | FileCheck %s --check-prefix=COUNT11 --implicit-check-not="sculptor.nn.transformer"
// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_2enc_2dec.mlir --sculptor-canonicalize-layers --sculptor-extract-layers --sculptor-convert-layers | FileCheck %s --check-prefix=COUNT22 --implicit-check-not="sculptor.nn.transformer"
// RUN: sculptor-mlir-opt %s --sculptor-convert-layers | FileCheck %s --check-prefix=DYNAMIC

// COUNT11-COUNT-33: sculptor.mvm
// COUNT22-COUNT-66: sculptor.mvm

// CONVERT11-LABEL: func.func @forward
// CONVERT11: call @transformer_encoder_block_0
// CONVERT11: call @transformer_decoder_block_0
// CONVERT11-LABEL: func.func @transformer_encoder_block_0
// CONVERT11: sculptor.mvm
// CONVERT11: sculptor.task_region kind = "digital.qkv_split" name = "transformer_block_qkv_split"
// CONVERT11: sculptor.task_region kind = "digital.attention_scores" name = "transformer_block_self_attention_scores"
// CONVERT11: sculptor.task_region kind = "digital.attention_softmax" name = "transformer_block_self_attention_softmax"
// CONVERT11: sculptor.task_region kind = "digital.attention_apply" name = "transformer_block_self_attention_apply"
// CONVERT11: sculptor.task_region kind = "digital.head_recombine" name = "transformer_block_self_head_recombine"
// CONVERT11: sculptor.task_region kind = "digital.residual_add" name = "transformer_block_self_attn_residual_add"
// CONVERT11: sculptor.task_region kind = "digital.layer_norm" name = "transformer_block_attn_norm"
// CONVERT11: sculptor.task_region kind = "digital.activation" name = "transformer_block_mlp_gelu"
// CONVERT11: sculptor.task_region kind = "digital.layer_norm" name = "transformer_block_mlp_norm"
// CONVERT11: sculptor.task_region kind = "digital.layer_norm" name = "transformer_block_final_norm"
// CONVERT11-LABEL: func.func @transformer_decoder_block_0
// CONVERT11: sculptor.mvm
// CONVERT11: sculptor.task_region kind = "digital.cross_kv_split" name = "transformer_block_cross_kv_split"
// CONVERT11: sculptor.task_region kind = "digital.attention_scores" name = "transformer_block_cross_attention_scores"
// CONVERT11: sculptor.task_region kind = "digital.attention_apply" name = "transformer_block_cross_attention_apply"
// CONVERT11: sculptor.task_region kind = "digital.residual_add" name = "transformer_block_cross_attn_residual_add"
// CONVERT11: sculptor.task_region kind = "digital.layer_norm" name = "transformer_block_cross_norm"
// CONVERT11: sculptor.task_region kind = "digital.layer_norm" name = "transformer_block_final_norm"

// CONVERT21-LABEL: func.func @transformer_encoder_block_0
// CONVERT21: sculptor.mvm
// CONVERT21-NOT: transformer_block_final_norm
// CONVERT21-LABEL: func.func @transformer_encoder_block_1
// CONVERT21: sculptor.mvm
// CONVERT21: transformer_block_final_norm
// CONVERT21-LABEL: func.func @transformer_decoder_block_0
// CONVERT21: sculptor.mvm
// CONVERT21: digital.cross_kv_split
// CONVERT21: transformer_block_final_norm

// CONVERT12-LABEL: func.func @transformer_encoder_block_0
// CONVERT12: sculptor.mvm
// CONVERT12: transformer_block_final_norm
// CONVERT12-LABEL: func.func @transformer_decoder_block_0
// CONVERT12: sculptor.mvm
// CONVERT12: digital.cross_kv_split
// CONVERT12-NOT: transformer_block_final_norm
// CONVERT12-LABEL: func.func @transformer_decoder_block_1
// CONVERT12: sculptor.mvm
// CONVERT12: digital.cross_kv_split
// CONVERT12: transformer_block_final_norm

// CONVERT22-LABEL: func.func @transformer_encoder_block_0
// CONVERT22: sculptor.mvm
// CONVERT22-NOT: transformer_block_final_norm
// CONVERT22-LABEL: func.func @transformer_encoder_block_1
// CONVERT22: sculptor.mvm
// CONVERT22: transformer_block_final_norm
// CONVERT22-LABEL: func.func @transformer_decoder_block_0
// CONVERT22: sculptor.mvm
// CONVERT22: digital.cross_kv_split
// CONVERT22-NOT: transformer_block_final_norm
// CONVERT22-LABEL: func.func @transformer_decoder_block_1
// CONVERT22: sculptor.mvm
// CONVERT22: digital.cross_kv_split
// CONVERT22: transformer_block_final_norm

module {
  func.func @forward(%arg0: tensor<?x3x4xf32>) -> tensor<?x3x4xf32> {
    %0 = call @transformer_dynamic_block(%arg0)
        : (tensor<?x3x4xf32>) -> tensor<?x3x4xf32>
    return %0 : tensor<?x3x4xf32>
  }

  func.func @transformer_dynamic_block(%arg0: tensor<?x3x4xf32>)
      -> tensor<?x3x4xf32> attributes {layer_type = "transformer_encoder_block"} {
    %qkv_w = arith.constant dense<1.000000e+00> : tensor<12x4xf32>
    %attn_w = arith.constant dense<3.000000e+00> : tensor<4x4xf32>
    %mlp_up_w = arith.constant dense<7.000000e+00> : tensor<8x4xf32>
    %mlp_down_w = arith.constant dense<9.000000e+00> : tensor<4x8xf32>
    %out = sculptor.nn.transformer_block %arg0, memory[],
        qkv[%qkv_w], attn_output[%attn_w], attn_norm[]
        , cross_query[], cross_key_value[], cross_output[], cross_norm[]
        , mlp_up[%mlp_up_w], mlp_down[%mlp_down_w], mlp_norm[]
        , final_norm[] block_kind = encoder
        {activation = "gelu", batch_first = true, block_index = 0 : i64,
         causal = false, has_final_norm = false, has_layer_norm_affine = false,
         has_cross_attention = false, has_projection_bias = false,
         head_dim = 2 : i64, hidden_size = 4 : i64, layer_norm_eps = 1.000000e-05 : f64,
         mlp_hidden_size = 8 : i64, norm_mode = "post", num_blocks = 1 : i64,
         num_heads = 2 : i64}
        : (tensor<?x3x4xf32>, tensor<12x4xf32>, tensor<4x4xf32>,
           tensor<8x4xf32>, tensor<4x8xf32>) -> tensor<?x3x4xf32>
    return %out : tensor<?x3x4xf32>
  }
}

// DYNAMIC-LABEL: func.func @forward
// DYNAMIC: call @transformer_dynamic_block
// DYNAMIC-LABEL: func.func @transformer_dynamic_block
// DYNAMIC: sculptor.nn.transformer_block
// DYNAMIC-NOT: sculptor.mvm
