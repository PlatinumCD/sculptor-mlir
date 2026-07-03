// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_1enc_1dec.mlir --sculptor-canonicalize-layers | FileCheck %s --check-prefix=CANON11 --implicit-check-not="linalg.matmul" --implicit-check-not="linalg.batch_matmul"
// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_2enc_1dec.mlir --sculptor-canonicalize-layers | FileCheck %s --check-prefix=CANON21 --implicit-check-not="linalg.matmul" --implicit-check-not="linalg.batch_matmul"
// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_1enc_2dec.mlir --sculptor-canonicalize-layers | FileCheck %s --check-prefix=CANON12 --implicit-check-not="linalg.matmul" --implicit-check-not="linalg.batch_matmul"
// RUN: sculptor-mlir-opt %S/fixtures/torch_transformer/torch_transformer_2enc_2dec.mlir --sculptor-canonicalize-layers | FileCheck %s --check-prefix=CANON22 --implicit-check-not="linalg.matmul" --implicit-check-not="linalg.batch_matmul"

// CANON11-LABEL: func.func @forward
// CANON11: %[[STACK:[A-Za-z0-9_]+]] = sculptor.nn.transformer
// CANON11-SAME: activation = "gelu"
// CANON11-SAME: batch_first = true
// CANON11-SAME: head_dim = 2 : i64
// CANON11-SAME: hidden_size = 4 : i64
// CANON11-SAME: mlp_hidden_size = 8 : i64
// CANON11-SAME: num_decoder_blocks = 1 : i64
// CANON11-SAME: num_encoder_blocks = 1 : i64
// CANON11-SAME: num_heads = 2 : i64
// CANON11-NOT: sculptor.nn.transformer
// CANON11: return %[[STACK]]

// CANON21-LABEL: func.func @forward
// CANON21: %[[STACK:[A-Za-z0-9_]+]] = sculptor.nn.transformer
// CANON21-SAME: activation = "gelu"
// CANON21-SAME: batch_first = true
// CANON21-SAME: head_dim = 2 : i64
// CANON21-SAME: hidden_size = 4 : i64
// CANON21-SAME: mlp_hidden_size = 8 : i64
// CANON21-SAME: num_decoder_blocks = 1 : i64
// CANON21-SAME: num_encoder_blocks = 2 : i64
// CANON21-SAME: num_heads = 2 : i64
// CANON21-NOT: sculptor.nn.transformer
// CANON21: return %[[STACK]]

// CANON12-LABEL: func.func @forward
// CANON12: %[[STACK:[A-Za-z0-9_]+]] = sculptor.nn.transformer
// CANON12-SAME: activation = "gelu"
// CANON12-SAME: batch_first = true
// CANON12-SAME: head_dim = 2 : i64
// CANON12-SAME: hidden_size = 4 : i64
// CANON12-SAME: mlp_hidden_size = 8 : i64
// CANON12-SAME: num_decoder_blocks = 2 : i64
// CANON12-SAME: num_encoder_blocks = 1 : i64
// CANON12-SAME: num_heads = 2 : i64
// CANON12-NOT: sculptor.nn.transformer
// CANON12: return %[[STACK]]

// CANON22-LABEL: func.func @forward
// CANON22: %[[STACK:[A-Za-z0-9_]+]] = sculptor.nn.transformer
// CANON22-SAME: activation = "gelu"
// CANON22-SAME: batch_first = true
// CANON22-SAME: head_dim = 2 : i64
// CANON22-SAME: hidden_size = 4 : i64
// CANON22-SAME: mlp_hidden_size = 8 : i64
// CANON22-SAME: num_decoder_blocks = 2 : i64
// CANON22-SAME: num_encoder_blocks = 2 : i64
// CANON22-SAME: num_heads = 2 : i64
// CANON22-NOT: sculptor.nn.transformer
// CANON22: return %[[STACK]]
