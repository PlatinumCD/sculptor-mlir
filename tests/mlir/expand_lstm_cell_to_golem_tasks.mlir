// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=8 array-cols=8" | FileCheck %s --implicit-check-not=sculptor.nn.lstm_cell --implicit-check-not="sculptor.mvm %"

module {
  func.func @forward(
      %arg0: tensor<1x1xf32>,
      %arg1: tensor<1x1xf32>,
      %arg2: tensor<1x1xf32>
  ) -> (tensor<1x1xf32>, tensor<1x1xf32>) {
    %0:2 = call @lstm_cell_bias(%arg0, %arg1, %arg2)
        : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>)
          -> (tensor<1x1xf32>, tensor<1x1xf32>)
    return %0#0, %0#1 : tensor<1x1xf32>, tensor<1x1xf32>
  }

  // CHECK-LABEL: func.func @lstm_cell_bias
  // CHECK: %[[MATRIX:.*]] = sculptor.task_region kind = "sculptor.matrix_setup" name = "lstm_cell_bias_matrix_tile_0_0"()
  // CHECK: %[[INPUT:.*]] = sculptor.task_region kind = "digital.input_recombine" name = "lstm_cell_input_recombine"(%arg0, %arg1) {
  // CHECK: tensor.concat dim(1)
  // CHECK: %[[VECTOR:.*]] = sculptor.task_region kind = "digital.vector_tile" name = "lstm_cell_bias_vector_tile_0"(%[[INPUT]])
  // CHECK: %[[EXEC:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "lstm_cell_bias_mvm_0_0"(%[[VECTOR]], %[[MATRIX]])
  // CHECK: %[[RECOMBINE:.*]] = sculptor.task_region kind = "digital.tile_recombine" name = "lstm_cell_bias_tile_recombine"(%[[EXEC]])
  // CHECK: %[[BIAS:.*]] = sculptor.task_region kind = "digital.bias_add" name = "lstm_cell_bias_add"(%[[RECOMBINE]]) {
  // CHECK: linalg.add
  // CHECK: %[[SPLIT:.*]]:4 = sculptor.task_region kind = "digital.gate_split" name = "lstm_cell_gate_split"(%[[BIAS]]) {
  // CHECK: tensor.extract_slice
  // CHECK: %[[ACT:.*]]:4 = sculptor.task_region kind = "digital.activation" name = "lstm_cell_gate_activation"(%[[SPLIT]]#0, %[[SPLIT]]#1, %[[SPLIT]]#2, %[[SPLIT]]#3) {
  // CHECK: math.exp
  // CHECK: math.tanh
  // CHECK: %[[CELL:.*]] = sculptor.task_region kind = "digital.cell_update" name = "lstm_cell_cell_update"(%[[ACT]]#1, %arg2, %[[ACT]]#0, %[[ACT]]#2) {
  // CHECK: arith.addf
  // CHECK: %[[HIDDEN:.*]] = sculptor.task_region kind = "digital.hidden_update" name = "lstm_cell_hidden_update"(%[[ACT]]#3, %[[CELL]]) {
  // CHECK: math.tanh
  // CHECK: arith.mulf
  // CHECK: return %[[HIDDEN]], %[[CELL]] : tensor<1x1xf32>, tensor<1x1xf32>
  func.func @lstm_cell_bias(
      %arg0: tensor<1x1xf32>,
      %arg1: tensor<1x1xf32>,
      %arg2: tensor<1x1xf32>
  ) -> (tensor<1x1xf32>, tensor<1x1xf32>)
      attributes {layer_type = "lstm_cell_w_bias"} {
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
    %h, %c = sculptor.nn.lstm_cell
        %arg0, %arg1, %arg2, %w_ih, %w_hh, %b_ih, %b_hh
        {has_bias = true}
        : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>,
           tensor<4x1xf32>, tensor<4x1xf32>, tensor<4xf32>,
           tensor<4xf32>) -> (tensor<1x1xf32>, tensor<1x1xf32>)
    return %h, %c : tensor<1x1xf32>, tensor<1x1xf32>
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
