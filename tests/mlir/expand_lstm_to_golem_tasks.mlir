// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=8 array-cols=8" | FileCheck %s --implicit-check-not=sculptor.nn.lstm_layer --implicit-check-not=scf.for --implicit-check-not="sculptor.mvm %"

module {
  func.func @forward(
      %arg0: tensor<1x2x1xf32>,
      %arg1: tensor<1x1x1xf32>,
      %arg2: tensor<1x1x1xf32>
  ) -> (tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>) {
    %0:3 = call @lstm_layer_bias(%arg0, %arg1, %arg2)
        : (tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>)
          -> (tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>)
    return %0#0, %0#1, %0#2
        : tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>
  }

  // CHECK-LABEL: func.func @lstm_layer_bias
  // CHECK: %[[MATRIX:.*]] = sculptor.task_region kind = "sculptor.matrix_setup" name = "lstm_layer_bias_matrix_tile_0_0"()
  // CHECK: %[[H0:.*]] = sculptor.task_region kind = "digital.hidden_extract" name = "lstm_layer_initial_hidden_extract"(%arg1) {
  // CHECK: tensor.extract_slice
  // CHECK: %[[C0:.*]] = sculptor.task_region kind = "digital.cell_extract" name = "lstm_layer_initial_cell_extract"(%arg2) {
  // CHECK: tensor.extract_slice
  // CHECK: %[[STEP0:.*]] = sculptor.task_region kind = "digital.timestep_extract" name = "lstm_layer_timestep_extract"(%arg0) {
  // CHECK: arith.constant 0 : index
  // CHECK: tensor.extract_slice
  // CHECK: %[[INPUT0:.*]] = sculptor.task_region kind = "digital.input_recombine" name = "lstm_layer_input_recombine"(%[[STEP0]], %[[H0]]) {
  // CHECK: tensor.concat dim(1)
  // CHECK: %[[VECTOR0:.*]] = sculptor.task_region kind = "digital.vector_tile" name = "lstm_layer_bias_vector_tile_0"(%[[INPUT0]])
  // CHECK: %[[EXEC0:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "lstm_layer_bias_mvm_0_0"(%[[VECTOR0]], %[[MATRIX]])
  // CHECK: %[[RECOMBINE0:.*]] = sculptor.task_region kind = "digital.tile_recombine" name = "lstm_layer_bias_tile_recombine"(%[[EXEC0]])
  // CHECK: %[[BIAS0:.*]] = sculptor.task_region kind = "digital.bias_add" name = "lstm_layer_bias_add"(%[[RECOMBINE0]]) {
  // CHECK: arith.addf
  // CHECK: %[[SPLIT0:.*]]:4 = sculptor.task_region kind = "digital.gate_split" name = "lstm_layer_gate_split"(%[[BIAS0]]) {
  // CHECK: tensor.extract_slice
  // CHECK: %[[ACT0:.*]]:4 = sculptor.task_region kind = "digital.activation" name = "lstm_layer_gate_activation"(%[[SPLIT0]]#0, %[[SPLIT0]]#1, %[[SPLIT0]]#2, %[[SPLIT0]]#3) {
  // CHECK: math.exp
  // CHECK: math.tanh
  // CHECK: %[[CELL0:.*]] = sculptor.task_region kind = "digital.cell_update" name = "lstm_layer_cell_update"(%[[ACT0]]#1, %[[C0]], %[[ACT0]]#0, %[[ACT0]]#2) {
  // CHECK: arith.addf
  // CHECK: %[[HIDDEN0:.*]] = sculptor.task_region kind = "digital.hidden_update" name = "lstm_layer_hidden_update"(%[[ACT0]]#3, %[[CELL0]], %[[H0]]) {
  // CHECK: math.tanh
  // CHECK: arith.mulf
  // CHECK: %[[OUTPUT0:.*]] = sculptor.task_region kind = "digital.output_update" name = "lstm_layer_output_update"(%[[HIDDEN0]]) {
  // CHECK: tensor.empty()
  // CHECK: tensor.insert_slice
  // CHECK: %[[STEP1:.*]] = sculptor.task_region kind = "digital.timestep_extract" name = "lstm_layer_timestep_extract"(%arg0) {
  // CHECK: arith.constant 1 : index
  // CHECK: %[[INPUT1:.*]] = sculptor.task_region kind = "digital.input_recombine" name = "lstm_layer_input_recombine"(%[[STEP1]], %[[HIDDEN0]]) {
  // CHECK: %[[VECTOR1:.*]] = sculptor.task_region kind = "digital.vector_tile" name = "lstm_layer_bias_vector_tile_0"(%[[INPUT1]])
  // CHECK: %[[EXEC1:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "lstm_layer_bias_mvm_0_0"(%[[VECTOR1]], %[[MATRIX]])
  // CHECK: %[[RECOMBINE1:.*]] = sculptor.task_region kind = "digital.tile_recombine" name = "lstm_layer_bias_tile_recombine"(%[[EXEC1]])
  // CHECK: %[[BIAS1:.*]] = sculptor.task_region kind = "digital.bias_add" name = "lstm_layer_bias_add"(%[[RECOMBINE1]]) {
  // CHECK: %[[SPLIT1:.*]]:4 = sculptor.task_region kind = "digital.gate_split" name = "lstm_layer_gate_split"(%[[BIAS1]]) {
  // CHECK: %[[ACT1:.*]]:4 = sculptor.task_region kind = "digital.activation" name = "lstm_layer_gate_activation"(%[[SPLIT1]]#0, %[[SPLIT1]]#1, %[[SPLIT1]]#2, %[[SPLIT1]]#3) {
  // CHECK: %[[CELL1:.*]] = sculptor.task_region kind = "digital.cell_update" name = "lstm_layer_cell_update"(%[[ACT1]]#1, %[[CELL0]], %[[ACT1]]#0, %[[ACT1]]#2) {
  // CHECK: %[[HIDDEN1:.*]] = sculptor.task_region kind = "digital.hidden_update" name = "lstm_layer_hidden_update"(%[[ACT1]]#3, %[[CELL1]], %[[HIDDEN0]]) {
  // CHECK: %[[OUTPUT1:.*]] = sculptor.task_region kind = "digital.output_update" name = "lstm_layer_output_update"(%[[HIDDEN1]], %[[OUTPUT0]]) {
  // CHECK: %[[HIDDEN_OUT:.*]] = sculptor.task_region kind = "digital.hidden_output" name = "lstm_layer_hidden_output"(%[[HIDDEN1]]) {
  // CHECK: tensor.expand_shape
  // CHECK: %[[CELL_OUT:.*]] = sculptor.task_region kind = "digital.cell_output" name = "lstm_layer_cell_output"(%[[CELL1]]) {
  // CHECK: tensor.expand_shape
  // CHECK: return %[[OUTPUT1]], %[[HIDDEN_OUT]], %[[CELL_OUT]]
  func.func @lstm_layer_bias(
      %arg0: tensor<1x2x1xf32>,
      %arg1: tensor<1x1x1xf32>,
      %arg2: tensor<1x1x1xf32>
  ) -> (tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>)
      attributes {layer_type = "lstm_w_bias"} {
    %w_ih = arith.constant dense_resource<torch_tensor_4_1_w_ih>
        : tensor<4x1xf32>
    %w_hh = arith.constant dense_resource<torch_tensor_4_1_w_hh>
        : tensor<4x1xf32>
    %b_ih = arith.constant dense<[
      1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00
    ]> : tensor<4xf32>
    %b_hh = arith.constant dense<[
      5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00
    ]> : tensor<4xf32>
    %output, %hn, %cn = sculptor.nn.lstm_layer
        %arg0, %arg1, %arg2, %w_ih, %w_hh, %b_ih, %b_hh
        {batch_first = true, has_bias = true, hidden_size = 1 : i64,
         layer_index = 0 : i64, num_layers = 1 : i64}
        : (tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>,
           tensor<4x1xf32>, tensor<4x1xf32>, tensor<4xf32>,
           tensor<4xf32>)
          -> (tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>)
    return %output, %hn, %cn
        : tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_4_1_w_ih: "0x040000000000803F0000803F0000803F0000803F",
      torch_tensor_4_1_w_hh: "0x0400000000000040000000400000004000000040"
    }
  }
#-}
