// RUN: sculptor-mlir-opt %s --sculptor-convert-layers | FileCheck %s

module {
  func.func @forward(%input: tensor<1x3x4xf32>,
                     %memory: tensor<1x5x4xf32>) -> tensor<1x3x4xf32> {
    %result = call @transformer_decoder_block_0(%input, %memory)
        : (tensor<1x3x4xf32>, tensor<1x5x4xf32>) -> tensor<1x3x4xf32>
    return %result : tensor<1x3x4xf32>
  }

  func.func @transformer_decoder_block_0(
      %input: tensor<1x3x4xf32>, %memory: tensor<1x5x4xf32>)
      -> tensor<1x3x4xf32> attributes {layer_type = "transformer_decoder_block"} {
    %qkv = arith.constant dense<1.0> : tensor<12x4xf32>
    %attn = arith.constant dense<1.0> : tensor<4x4xf32>
    %cross_query = arith.constant dense<1.0> : tensor<4x4xf32>
    %cross_kv = arith.constant dense<1.0> : tensor<8x4xf32>
    %cross_output = arith.constant dense<1.0> : tensor<4x4xf32>
    %mlp_up = arith.constant dense<1.0> : tensor<8x4xf32>
    %mlp_down = arith.constant dense<1.0> : tensor<4x8xf32>
    %result = sculptor.nn.transformer_block %input, memory[%memory],
        qkv[%qkv], attn_output[%attn], attn_norm[],
        cross_query[%cross_query], cross_key_value[%cross_kv],
        cross_output[%cross_output], cross_norm[],
        mlp_up[%mlp_up], mlp_down[%mlp_down], mlp_norm[], final_norm[]
        block_kind = decoder
        {activation = "gelu", batch_first = true, block_index = 0 : i64,
         causal = true, has_cross_attention = true, has_final_norm = false,
         has_layer_norm_affine = false, has_projection_bias = false,
         head_dim = 2 : i64, hidden_size = 4 : i64,
         layer_norm_eps = 1.000000e-05 : f64, mlp_hidden_size = 8 : i64,
         norm_mode = "post", num_blocks = 1 : i64, num_heads = 2 : i64}
        : (tensor<1x3x4xf32>, tensor<1x5x4xf32>, tensor<12x4xf32>,
           tensor<4x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>,
           tensor<4x4xf32>, tensor<8x4xf32>, tensor<4x8xf32>)
          -> tensor<1x3x4xf32>
    return %result : tensor<1x3x4xf32>
  }
}

// CHECK-LABEL: func.func @transformer_decoder_block_0
// CHECK: sculptor.task_region kind = "digital.attention_scores" name = "transformer_block_self_attention_scores"
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 3, 2, 2]
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 3, 2, 2]
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK: linalg.index 2
// CHECK: linalg.index 3
// CHECK: arith.cmpi ugt
// CHECK: arith.select
// CHECK-NOT: tensor.extract 
// CHECK-NOT: tensor.insert 
// CHECK: sculptor.task_region kind = "digital.attention_apply" name = "transformer_block_self_attention_apply"
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK: sculptor.task_region kind = "digital.head_recombine" name = "transformer_block_self_head_recombine"
// CHECK: linalg.transpose
// CHECK: tensor.collapse_shape
// CHECK: sculptor.task_region kind = "digital.attention_scores" name = "transformer_block_cross_attention_scores"
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 3, 2, 2]
// CHECK: tensor.expand_shape {{.*}} output_shape [1, 5, 2, 2]
// CHECK: tensor<1x2x3x5xf32>
// CHECK: sculptor.task_region kind = "digital.attention_apply" name = "transformer_block_cross_attention_apply"
// CHECK: tensor<1x2x3x5xf32>
// CHECK: tensor<1x5x2x2xf32>
// CHECK: tensor<1x2x3x2xf32>
