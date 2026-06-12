// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=8 array-cols=8" | FileCheck %s --implicit-check-not=sculptor.nn.rnn_layer --implicit-check-not=scf.for --implicit-check-not="sculptor.mvm %"

module {
  func.func @forward(
      %arg0: tensor<1x2x1xf32>,
      %arg1: tensor<1x1x1xf32>
  ) -> (tensor<1x2x1xf32>, tensor<1x1x1xf32>) {
    %0:2 = call @rnn_layer_bias(%arg0, %arg1)
        : (tensor<1x2x1xf32>, tensor<1x1x1xf32>)
          -> (tensor<1x2x1xf32>, tensor<1x1x1xf32>)
    return %0#0, %0#1 : tensor<1x2x1xf32>, tensor<1x1x1xf32>
  }

  // CHECK-LABEL: func.func @rnn_layer_bias
  // CHECK: %[[MATRIX:.*]] = sculptor.task_region kind = "sculptor.matrix_setup" name = "rnn_layer_bias_matrix_tile_0_0"()
  // CHECK: %[[H0:.*]] = sculptor.task_region kind = "digital.hidden_extract" name = "rnn_layer_initial_hidden_extract"(%arg1) {
  // CHECK: tensor.extract_slice
  // CHECK: %[[STEP0:.*]] = sculptor.task_region kind = "digital.timestep_extract" name = "rnn_layer_timestep_extract"(%arg0) {
  // CHECK: arith.constant 0 : index
  // CHECK: tensor.extract_slice
  // CHECK: %[[INPUT0:.*]] = sculptor.task_region kind = "digital.input_recombine" name = "rnn_layer_input_recombine"(%[[STEP0]], %[[H0]]) {
  // CHECK: tensor.concat dim(1)
  // CHECK: %[[VECTOR0:.*]] = sculptor.task_region kind = "digital.vector_tile" name = "rnn_layer_bias_vector_tile_0"(%[[INPUT0]])
  // CHECK: %[[EXEC0:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "rnn_layer_bias_mvm_0_0"(%[[VECTOR0]], %[[MATRIX]])
  // CHECK: %[[RECOMBINE0:.*]] = sculptor.task_region kind = "digital.tile_recombine" name = "rnn_layer_bias_tile_recombine"(%[[EXEC0]])
  // CHECK: %[[BIAS0:.*]] = sculptor.task_region kind = "digital.bias_add" name = "rnn_layer_bias_add"(%[[RECOMBINE0]])
  // CHECK: arith.addf
  // CHECK: %[[TANH0:.*]] = sculptor.task_region kind = "digital.activation" name = "rnn_layer_tanh"(%[[BIAS0]], %[[H0]]) {
  // CHECK: math.tanh
  // CHECK: %[[OUTPUT0:.*]] = sculptor.task_region kind = "digital.output_update" name = "rnn_layer_output_update"(%[[TANH0]]) {
  // CHECK: tensor.empty()
  // CHECK: tensor.insert_slice
  // CHECK: %[[STEP1:.*]] = sculptor.task_region kind = "digital.timestep_extract" name = "rnn_layer_timestep_extract"(%arg0) {
  // CHECK: arith.constant 1 : index
  // CHECK: %[[INPUT1:.*]] = sculptor.task_region kind = "digital.input_recombine" name = "rnn_layer_input_recombine"(%[[STEP1]], %[[TANH0]]) {
  // CHECK: %[[VECTOR1:.*]] = sculptor.task_region kind = "digital.vector_tile" name = "rnn_layer_bias_vector_tile_0"(%[[INPUT1]])
  // CHECK: %[[EXEC1:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "rnn_layer_bias_mvm_0_0"(%[[VECTOR1]], %[[MATRIX]])
  // CHECK: %[[RECOMBINE1:.*]] = sculptor.task_region kind = "digital.tile_recombine" name = "rnn_layer_bias_tile_recombine"(%[[EXEC1]])
  // CHECK: %[[BIAS1:.*]] = sculptor.task_region kind = "digital.bias_add" name = "rnn_layer_bias_add"(%[[RECOMBINE1]])
  // CHECK: %[[TANH1:.*]] = sculptor.task_region kind = "digital.activation" name = "rnn_layer_tanh"(%[[BIAS1]], %[[TANH0]]) {
  // CHECK: %[[OUTPUT1:.*]] = sculptor.task_region kind = "digital.output_update" name = "rnn_layer_output_update"(%[[TANH1]], %[[OUTPUT0]]) {
  // CHECK: tensor.insert_slice
  // CHECK: %[[HIDDEN_OUT:.*]] = sculptor.task_region kind = "digital.hidden_output" name = "rnn_layer_hidden_output"(%[[TANH1]]) {
  // CHECK: tensor.expand_shape
  // CHECK: return %[[OUTPUT1]], %[[HIDDEN_OUT]]
  func.func @rnn_layer_bias(
      %arg0: tensor<1x2x1xf32>,
      %arg1: tensor<1x1x1xf32>
  ) -> (tensor<1x2x1xf32>, tensor<1x1x1xf32>)
      attributes {layer_type = "rnn_w_bias"} {
    %w_ih = arith.constant dense_resource<torch_tensor_1_1_w_ih>
        : tensor<1x1xf32>
    %w_hh = arith.constant dense_resource<torch_tensor_1_1_w_hh>
        : tensor<1x1xf32>
    %b_ih = arith.constant dense<[1.000000e+00]> : tensor<1xf32>
    %b_hh = arith.constant dense<[2.000000e+00]> : tensor<1xf32>
    %output, %hn = sculptor.nn.rnn_layer %arg0, %arg1, %w_ih, %w_hh, %b_ih, %b_hh
        {batch_first = true, has_bias = true, hidden_size = 1 : i64,
         layer_index = 0 : i64, num_layers = 1 : i64}
        : (tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x1xf32>,
           tensor<1x1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x2x1xf32>, tensor<1x1x1xf32>)
    return %output, %hn : tensor<1x2x1xf32>, tensor<1x1x1xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_1_1_w_ih: "0x040000000000803F",
      torch_tensor_1_1_w_hh: "0x0400000000000040"
    }
  }
#-}
