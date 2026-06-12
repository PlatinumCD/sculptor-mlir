// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=8 array-cols=8" | FileCheck %s --implicit-check-not=sculptor.nn.rnn_cell --implicit-check-not="sculptor.mvm %"

module {
  func.func @forward(
      %arg0: tensor<1x4xf32>,
      %arg1: tensor<1x3xf32>
  ) -> tensor<1x3xf32> {
    %0 = call @rnn_cell_bias(%arg0, %arg1)
        : (tensor<1x4xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func @rnn_cell_bias
  // CHECK: %[[MATRIX:.*]] = sculptor.task_region kind = "sculptor.matrix_setup" name = "rnn_cell_bias_matrix_tile_0_0"()
  // CHECK: %[[INPUT:.*]] = sculptor.task_region kind = "digital.input_recombine" name = "rnn_cell_input_recombine"(%arg0, %arg1) {
  // CHECK: tensor.concat dim(1)
  // CHECK: %[[VECTOR:.*]] = sculptor.task_region kind = "digital.vector_tile" name = "rnn_cell_bias_vector_tile_0"(%[[INPUT]])
  // CHECK: %[[EXEC:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "rnn_cell_bias_mvm_0_0"(%[[VECTOR]], %[[MATRIX]])
  // CHECK: %[[RECOMBINE:.*]] = sculptor.task_region kind = "digital.tile_recombine" name = "rnn_cell_bias_tile_recombine"(%[[EXEC]])
  // CHECK: %[[BIAS:.*]] = sculptor.task_region kind = "digital.bias_add" name = "rnn_cell_bias_add"(%[[RECOMBINE]]) {
  // CHECK: linalg.add
  // CHECK: %[[ACTIVATION:.*]] = sculptor.task_region kind = "digital.activation" name = "rnn_cell_tanh"(%[[BIAS]]) {
  // CHECK: math.tanh
  // CHECK: return %[[ACTIVATION]] : tensor<1x3xf32>
  func.func @rnn_cell_bias(%arg0: tensor<1x4xf32>, %arg1: tensor<1x3xf32>)
      -> tensor<1x3xf32>
      attributes {layer_type = "rnn_cell_w_bias"} {
    %w_ih = arith.constant dense_resource<torch_tensor_3_4_torch.float32>
        : tensor<3x4xf32>
    %w_hh = arith.constant dense_resource<torch_tensor_3_3_torch.float32>
        : tensor<3x3xf32>
    %b_ih = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]>
        : tensor<3xf32>
    %b_hh = arith.constant dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]>
        : tensor<3xf32>
    %0 = sculptor.nn.rnn_cell %arg0, %arg1, %w_ih, %w_hh, %b_ih, %b_hh
        {activation = "tanh", has_bias = true}
        : (tensor<1x4xf32>, tensor<1x3xf32>, tensor<3x4xf32>,
           tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>)
          -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D",
      torch_tensor_3_3_torch.float32: "0x040000000AD7A33B0AD7233C8FC2753C0AD7A33CCDCCCC3C8FC2F53C295C0F3D0AD7233DEC51383D"
    }
  }
#-}
