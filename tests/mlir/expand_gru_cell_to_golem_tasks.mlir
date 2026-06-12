// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=8 array-cols=8" | FileCheck %s --implicit-check-not=sculptor.nn.gru_cell --implicit-check-not="sculptor.mvm %"

module {
  func.func @forward(
      %arg0: tensor<1x1xf32>,
      %arg1: tensor<1x1xf32>
  ) -> tensor<1x1xf32> {
    %0 = call @gru_cell_bias(%arg0, %arg1)
        : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }

  // CHECK-LABEL: func.func @gru_cell_bias
  // CHECK: %[[MATRIX:.*]] = sculptor.task_region kind = "sculptor.matrix_setup" name = "gru_cell_bias_matrix_tile_0_0"()
  // CHECK: %[[INPUT:.*]] = sculptor.task_region kind = "digital.input_recombine" name = "gru_cell_input_recombine"(%arg0, %arg1) {
  // CHECK: tensor.concat dim(1)
  // CHECK: %[[VECTOR:.*]] = sculptor.task_region kind = "digital.vector_tile" name = "gru_cell_bias_vector_tile_0"(%[[INPUT]])
  // CHECK: %[[EXEC:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "gru_cell_bias_mvm_0_0"(%[[VECTOR]], %[[MATRIX]])
  // CHECK: %[[RECOMBINE:.*]] = sculptor.task_region kind = "digital.tile_recombine" name = "gru_cell_bias_tile_recombine"(%[[EXEC]])
  // CHECK: %[[BIAS:.*]] = sculptor.task_region kind = "digital.bias_add" name = "gru_cell_bias_add"(%[[RECOMBINE]]) {
  // CHECK: linalg.add
  // CHECK: %[[SPLIT:.*]]:4 = sculptor.task_region kind = "digital.gate_split" name = "gru_cell_gate_split"(%[[BIAS]]) {
  // CHECK: tensor.extract_slice
  // CHECK: %[[ACT:.*]]:2 = sculptor.task_region kind = "digital.activation" name = "gru_cell_gate_activation"(%[[SPLIT]]#0, %[[SPLIT]]#1) {
  // CHECK: math.exp
  // CHECK: %[[CANDIDATE:.*]] = sculptor.task_region kind = "digital.candidate_update" name = "gru_cell_candidate_update"(%[[ACT]]#0, %[[SPLIT]]#2, %[[SPLIT]]#3) {
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: math.tanh
  // CHECK: %[[HIDDEN:.*]] = sculptor.task_region kind = "digital.hidden_update" name = "gru_cell_hidden_update"(%[[CANDIDATE]], %[[ACT]]#1, %arg1) {
  // CHECK: arith.subf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: return %[[HIDDEN]] : tensor<1x1xf32>
  func.func @gru_cell_bias(
      %arg0: tensor<1x1xf32>,
      %arg1: tensor<1x1xf32>
  ) -> tensor<1x1xf32>
      attributes {layer_type = "gru_cell_w_bias"} {
    %w_ih = arith.constant dense_resource<torch_tensor_3_1_w_ih>
        : tensor<3x1xf32>
    %w_hh = arith.constant dense_resource<torch_tensor_3_1_w_hh>
        : tensor<3x1xf32>
    %b_ih = arith.constant dense<[
      1.000000e+00, 2.000000e+00, 3.000000e+00
    ]> : tensor<3xf32>
    %b_hh = arith.constant dense<[
      4.000000e+00, 5.000000e+00, 6.000000e+00
    ]> : tensor<3xf32>
    %0 = sculptor.nn.gru_cell %arg0, %arg1, %w_ih, %w_hh, %b_ih, %b_hh
        {has_bias = true}
        : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<3x1xf32>,
           tensor<3x1xf32>, tensor<3xf32>, tensor<3xf32>)
          -> tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_1_w_ih: "0x040000000000803F0000803F0000803F",
      torch_tensor_3_1_w_hh: "0x04000000000000400000004000000040"
    }
  }
#-}
