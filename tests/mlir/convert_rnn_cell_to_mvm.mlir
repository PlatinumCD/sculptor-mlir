// RUN: sculptor-mlir-opt %s --sculptor-convert-layers | FileCheck %s --implicit-check-not=sculptor.nn.rnn_cell --implicit-check-not=sculptor.matrix. --implicit-check-not=sculptor.vector. --implicit-check-not=sculptor.array.

module {
  func.func @forward(
      %arg0: tensor<1x4xf32>,
      %arg1: tensor<1x3xf32>,
      %arg2: tensor<1x4xf32>,
      %arg3: tensor<1x3xf32>,
      %arg4: tensor<1x3xf32>,
      %arg5: tensor<1x3xf32>
  ) -> (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) {
    %0 = call @rnn_cell_no_bias(%arg0, %arg1)
        : (tensor<1x4xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %1 = call @rnn_cell_bias(%arg2, %arg3)
        : (tensor<1x4xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %2 = call @rnn_cell_equal_dims_bias(%arg4, %arg5)
        : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    return %0, %1, %2 : tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func @rnn_cell_no_bias
  // CHECK: %[[WEIGHT:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<3x7xf32>
  // CHECK: %[[INPUT:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "rnn_cell_input_recombine"(%arg0, %arg1) {
  // CHECK: ^bb0(%[[CELL_INPUT:.*]]: tensor<1x4xf32>, %[[CELL_H:.*]]: tensor<1x3xf32>):
  // CHECK: %[[CONCAT:[A-Za-z0-9_]+]] = tensor.concat dim(1) %[[CELL_INPUT]], %[[CELL_H]] : (tensor<1x4xf32>, tensor<1x3xf32>) -> tensor<1x7xf32>
  // CHECK: sculptor.yield %[[CONCAT]] : tensor<1x7xf32>
  // CHECK: %[[MVM:[A-Za-z0-9_]+]] = sculptor.mvm %[[INPUT]], %[[WEIGHT]] : (tensor<1x7xf32>, tensor<3x7xf32>) -> tensor<1x3xf32>
  // CHECK: %[[BIAS:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "rnn_cell_bias_add"(%[[MVM]]) {
  // CHECK: ^bb0(%[[BIAS_ARG:.*]]: tensor<1x3xf32>):
  // CHECK: sculptor.yield %[[BIAS_ARG]] : tensor<1x3xf32>
  // CHECK: %[[TANH:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.activation" name = "rnn_cell_tanh"(%[[BIAS]]) {
  // CHECK: math.tanh
  // CHECK: sculptor.yield
  // CHECK: return %[[TANH]]
  func.func @rnn_cell_no_bias(%arg0: tensor<1x4xf32>, %arg1: tensor<1x3xf32>)
      -> tensor<1x3xf32>
      attributes {layer_type = "rnn_cell"} {
    %w_ih = arith.constant dense<1.000000e+00> : tensor<3x4xf32>
    %w_hh = arith.constant dense<2.000000e+00> : tensor<3x3xf32>
    %0 = sculptor.nn.rnn_cell %arg0, %arg1, %w_ih, %w_hh {activation = "tanh", has_bias = false}
        : (tensor<1x4xf32>, tensor<1x3xf32>, tensor<3x4xf32>, tensor<3x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func @rnn_cell_bias
  // CHECK: %[[WEIGHT:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<3x7xf32>
  // CHECK: %[[INPUT:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "rnn_cell_input_recombine"(%arg0, %arg1) {
  // CHECK: ^bb0(%[[CELL_INPUT:.*]]: tensor<1x4xf32>, %[[CELL_H:.*]]: tensor<1x3xf32>):
  // CHECK: %[[CONCAT:[A-Za-z0-9_]+]] = tensor.concat dim(1) %[[CELL_INPUT]], %[[CELL_H]] : (tensor<1x4xf32>, tensor<1x3xf32>) -> tensor<1x7xf32>
  // CHECK: sculptor.yield %[[CONCAT]] : tensor<1x7xf32>
  // CHECK: %[[MVM:[A-Za-z0-9_]+]] = sculptor.mvm %[[INPUT]], %[[WEIGHT]] : (tensor<1x7xf32>, tensor<3x7xf32>) -> tensor<1x3xf32>
  // CHECK: %[[BIAS_REGION:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "rnn_cell_bias_add"(%[[MVM]]) {
  // CHECK: %[[BIAS:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<3xf32>
  // CHECK: tensor.expand_shape %[[BIAS]]
  // CHECK: %[[ADD:[A-Za-z0-9_]+]] = linalg.add
  // CHECK: sculptor.yield %[[ADD]]
  // CHECK: %[[TANH:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.activation" name = "rnn_cell_tanh"(%[[BIAS_REGION]]) {
  // CHECK: math.tanh
  // CHECK: sculptor.yield
  // CHECK: return %[[TANH]]
  func.func @rnn_cell_bias(%arg0: tensor<1x4xf32>, %arg1: tensor<1x3xf32>)
      -> tensor<1x3xf32>
      attributes {layer_type = "rnn_cell_w_bias"} {
    %w_ih = arith.constant dense<1.000000e+00> : tensor<3x4xf32>
    %w_hh = arith.constant dense<2.000000e+00> : tensor<3x3xf32>
    %b_ih = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>
    %b_hh = arith.constant dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf32>
    %0 = sculptor.nn.rnn_cell %arg0, %arg1, %w_ih, %w_hh, %b_ih, %b_hh {activation = "tanh", has_bias = true}
        : (tensor<1x4xf32>, tensor<1x3xf32>, tensor<3x4xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func @rnn_cell_equal_dims_bias
  // CHECK: %[[WEIGHT:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<3x6xf32>
  // CHECK: %[[INPUT:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "rnn_cell_input_recombine"(%arg0, %arg1) {
  // CHECK: ^bb0(%[[CELL_INPUT:.*]]: tensor<1x3xf32>, %[[CELL_H:.*]]: tensor<1x3xf32>):
  // CHECK: %[[CONCAT:[A-Za-z0-9_]+]] = tensor.concat dim(1) %[[CELL_INPUT]], %[[CELL_H]] : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x6xf32>
  // CHECK: sculptor.yield %[[CONCAT]] : tensor<1x6xf32>
  // CHECK: %[[MVM:[A-Za-z0-9_]+]] = sculptor.mvm %[[INPUT]], %[[WEIGHT]] : (tensor<1x6xf32>, tensor<3x6xf32>) -> tensor<1x3xf32>
  // CHECK: %[[BIAS_REGION:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "rnn_cell_bias_add"(%[[MVM]]) {
  // CHECK: %[[BIAS:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<3xf32>
  // CHECK: tensor.expand_shape %[[BIAS]]
  // CHECK: %[[ADD:[A-Za-z0-9_]+]] = linalg.add
  // CHECK: sculptor.yield %[[ADD]]
  // CHECK: %[[TANH:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.activation" name = "rnn_cell_tanh"(%[[BIAS_REGION]]) {
  // CHECK: math.tanh
  // CHECK: sculptor.yield
  // CHECK: return %[[TANH]]
  func.func @rnn_cell_equal_dims_bias(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>)
      -> tensor<1x3xf32>
      attributes {layer_type = "rnn_cell_w_bias"} {
    %w_ih = arith.constant dense<1.000000e+00> : tensor<3x3xf32>
    %w_hh = arith.constant dense<2.000000e+00> : tensor<3x3xf32>
    %b_ih = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>
    %b_hh = arith.constant dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf32>
    %0 = sculptor.nn.rnn_cell %arg0, %arg1, %w_ih, %w_hh, %b_ih, %b_hh {activation = "tanh", has_bias = true}
        : (tensor<1x3xf32>, tensor<1x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}
