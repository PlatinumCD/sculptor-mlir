// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_1enc_1dec.mlir --sculptor-canonicalize-layers --sculptor-extract-layers | FileCheck %s --check-prefix=EXTRACT11 --implicit-check-not="sculptor.nn.transformer %"
// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_2enc_1dec.mlir --sculptor-canonicalize-layers --sculptor-extract-layers | FileCheck %s --check-prefix=EXTRACT21 --implicit-check-not="sculptor.nn.transformer %"
// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_1enc_2dec.mlir --sculptor-canonicalize-layers --sculptor-extract-layers | FileCheck %s --check-prefix=EXTRACT12 --implicit-check-not="sculptor.nn.transformer %"
// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_2enc_2dec.mlir --sculptor-canonicalize-layers --sculptor-extract-layers | FileCheck %s --check-prefix=EXTRACT22 --implicit-check-not="sculptor.nn.transformer %"

// EXTRACT11-LABEL: func.func @forward
// EXTRACT11: %[[ENC0:[A-Za-z0-9_]+]] = call @transformer_encoder_block_0(%arg0)
// EXTRACT11: %[[DEC0:[A-Za-z0-9_]+]] = call @transformer_decoder_block_0(%arg1, %[[ENC0]])
// EXTRACT11: return %[[DEC0]]
// EXTRACT11-LABEL: func.func @transformer_encoder_block_0
// EXTRACT11-SAME: attributes {layer_type = "transformer_encoder_block"}
// EXTRACT11: sculptor.nn.transformer_block
// EXTRACT11-SAME: memory[]
// EXTRACT11-SAME: final_norm[
// EXTRACT11-SAME: block_kind = encoder
// EXTRACT11-SAME: block_index = 0 : i64
// EXTRACT11-SAME: has_final_norm = true
// EXTRACT11-SAME: num_blocks = 1 : i64
// EXTRACT11-LABEL: func.func @transformer_decoder_block_0
// EXTRACT11-SAME: attributes {layer_type = "transformer_decoder_block"}
// EXTRACT11: sculptor.nn.transformer_block %arg0, memory[%arg1]
// EXTRACT11-SAME: final_norm[
// EXTRACT11-SAME: block_kind = decoder
// EXTRACT11-SAME: block_index = 0 : i64
// EXTRACT11-SAME: has_final_norm = true
// EXTRACT11-SAME: num_blocks = 1 : i64

// EXTRACT21-LABEL: func.func @forward
// EXTRACT21: %[[ENC0:[A-Za-z0-9_]+]] = call @transformer_encoder_block_0(%arg0)
// EXTRACT21: %[[ENC1:[A-Za-z0-9_]+]] = call @transformer_encoder_block_1(%[[ENC0]])
// EXTRACT21: %[[DEC0:[A-Za-z0-9_]+]] = call @transformer_decoder_block_0(%arg1, %[[ENC1]])
// EXTRACT21: return %[[DEC0]]
// EXTRACT21-LABEL: func.func @transformer_encoder_block_0
// EXTRACT21-SAME: attributes {layer_type = "transformer_encoder_block"}
// EXTRACT21: sculptor.nn.transformer_block
// EXTRACT21-SAME: memory[]
// EXTRACT21-SAME: final_norm[]
// EXTRACT21-SAME: block_kind = encoder
// EXTRACT21-SAME: block_index = 0 : i64
// EXTRACT21-SAME: has_final_norm = false
// EXTRACT21-SAME: num_blocks = 2 : i64
// EXTRACT21-LABEL: func.func @transformer_encoder_block_1
// EXTRACT21-SAME: attributes {layer_type = "transformer_encoder_block"}
// EXTRACT21: sculptor.nn.transformer_block
// EXTRACT21-SAME: memory[]
// EXTRACT21-SAME: final_norm[
// EXTRACT21-SAME: block_kind = encoder
// EXTRACT21-SAME: block_index = 1 : i64
// EXTRACT21-SAME: has_final_norm = true
// EXTRACT21-SAME: num_blocks = 2 : i64
// EXTRACT21-LABEL: func.func @transformer_decoder_block_0
// EXTRACT21-SAME: attributes {layer_type = "transformer_decoder_block"}
// EXTRACT21: sculptor.nn.transformer_block %arg0, memory[%arg1]
// EXTRACT21-SAME: final_norm[
// EXTRACT21-SAME: block_kind = decoder
// EXTRACT21-SAME: block_index = 0 : i64
// EXTRACT21-SAME: has_final_norm = true
// EXTRACT21-SAME: num_blocks = 1 : i64

// EXTRACT12-LABEL: func.func @forward
// EXTRACT12: %[[ENC0:[A-Za-z0-9_]+]] = call @transformer_encoder_block_0(%arg0)
// EXTRACT12: %[[DEC0:[A-Za-z0-9_]+]] = call @transformer_decoder_block_0(%arg1, %[[ENC0]])
// EXTRACT12: %[[DEC1:[A-Za-z0-9_]+]] = call @transformer_decoder_block_1(%[[DEC0]], %[[ENC0]])
// EXTRACT12: return %[[DEC1]]
// EXTRACT12-LABEL: func.func @transformer_encoder_block_0
// EXTRACT12-SAME: attributes {layer_type = "transformer_encoder_block"}
// EXTRACT12: sculptor.nn.transformer_block
// EXTRACT12-SAME: memory[]
// EXTRACT12-SAME: final_norm[
// EXTRACT12-SAME: block_kind = encoder
// EXTRACT12-SAME: block_index = 0 : i64
// EXTRACT12-SAME: has_final_norm = true
// EXTRACT12-SAME: num_blocks = 1 : i64
// EXTRACT12-LABEL: func.func @transformer_decoder_block_0
// EXTRACT12-SAME: attributes {layer_type = "transformer_decoder_block"}
// EXTRACT12: sculptor.nn.transformer_block %arg0, memory[%arg1]
// EXTRACT12-SAME: final_norm[]
// EXTRACT12-SAME: block_kind = decoder
// EXTRACT12-SAME: block_index = 0 : i64
// EXTRACT12-SAME: has_final_norm = false
// EXTRACT12-SAME: num_blocks = 2 : i64
// EXTRACT12-LABEL: func.func @transformer_decoder_block_1
// EXTRACT12-SAME: attributes {layer_type = "transformer_decoder_block"}
// EXTRACT12: sculptor.nn.transformer_block %arg0, memory[%arg1]
// EXTRACT12-SAME: final_norm[
// EXTRACT12-SAME: block_kind = decoder
// EXTRACT12-SAME: block_index = 1 : i64
// EXTRACT12-SAME: has_final_norm = true
// EXTRACT12-SAME: num_blocks = 2 : i64

// EXTRACT22-LABEL: func.func @forward
// EXTRACT22: %[[ENC0:[A-Za-z0-9_]+]] = call @transformer_encoder_block_0(%arg0)
// EXTRACT22: %[[ENC1:[A-Za-z0-9_]+]] = call @transformer_encoder_block_1(%[[ENC0]])
// EXTRACT22: %[[DEC0:[A-Za-z0-9_]+]] = call @transformer_decoder_block_0(%arg1, %[[ENC1]])
// EXTRACT22: %[[DEC1:[A-Za-z0-9_]+]] = call @transformer_decoder_block_1(%[[DEC0]], %[[ENC1]])
// EXTRACT22: return %[[DEC1]]
// EXTRACT22-LABEL: func.func @transformer_encoder_block_0
// EXTRACT22-SAME: attributes {layer_type = "transformer_encoder_block"}
// EXTRACT22: sculptor.nn.transformer_block
// EXTRACT22-SAME: memory[]
// EXTRACT22-SAME: final_norm[]
// EXTRACT22-SAME: block_kind = encoder
// EXTRACT22-SAME: block_index = 0 : i64
// EXTRACT22-SAME: has_final_norm = false
// EXTRACT22-SAME: num_blocks = 2 : i64
// EXTRACT22-LABEL: func.func @transformer_encoder_block_1
// EXTRACT22-SAME: attributes {layer_type = "transformer_encoder_block"}
// EXTRACT22: sculptor.nn.transformer_block
// EXTRACT22-SAME: memory[]
// EXTRACT22-SAME: final_norm[
// EXTRACT22-SAME: block_kind = encoder
// EXTRACT22-SAME: block_index = 1 : i64
// EXTRACT22-SAME: has_final_norm = true
// EXTRACT22-SAME: num_blocks = 2 : i64
// EXTRACT22-LABEL: func.func @transformer_decoder_block_0
// EXTRACT22-SAME: attributes {layer_type = "transformer_decoder_block"}
// EXTRACT22: sculptor.nn.transformer_block %arg0, memory[%arg1]
// EXTRACT22-SAME: final_norm[]
// EXTRACT22-SAME: block_kind = decoder
// EXTRACT22-SAME: block_index = 0 : i64
// EXTRACT22-SAME: has_final_norm = false
// EXTRACT22-SAME: num_blocks = 2 : i64
// EXTRACT22-LABEL: func.func @transformer_decoder_block_1
// EXTRACT22-SAME: attributes {layer_type = "transformer_decoder_block"}
// EXTRACT22: sculptor.nn.transformer_block %arg0, memory[%arg1]
// EXTRACT22-SAME: final_norm[
// EXTRACT22-SAME: block_kind = decoder
// EXTRACT22-SAME: block_index = 1 : i64
// EXTRACT22-SAME: has_final_norm = true
// EXTRACT22-SAME: num_blocks = 2 : i64
