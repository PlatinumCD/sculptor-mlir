// RUN: sculptor-mlir-opt %s --sculptor-convert-layers | FileCheck %s --implicit-check-not=sculptor.nn.rnn_layer --implicit-check-not=scf.for --implicit-check-not=sculptor.matrix. --implicit-check-not=sculptor.vector. --implicit-check-not=sculptor.array.

module {
  // CHECK-LABEL: func.func @forward
  // CHECK: call @rnn_0_0(%arg0, %arg1)
  // CHECK: call @rnnwbias_0_1(%arg2, %arg1)
  func.func @forward(
      %arg0: tensor<1x2x4xf32>,
      %arg1: tensor<2x1x3xf32>,
      %arg2: tensor<1x2x3xf32>
  ) -> (tensor<1x2x3xf32>, tensor<1x1x3xf32>,
        tensor<1x2x3xf32>, tensor<1x1x3xf32>) {
    %0:2 = call @rnn_0_0(%arg0, %arg1)
        : (tensor<1x2x4xf32>, tensor<2x1x3xf32>)
          -> (tensor<1x2x3xf32>, tensor<1x1x3xf32>)
    %1:2 = call @rnnwbias_0_1(%arg2, %arg1)
        : (tensor<1x2x3xf32>, tensor<2x1x3xf32>)
          -> (tensor<1x2x3xf32>, tensor<1x1x3xf32>)
    return %0#0, %0#1, %1#0, %1#1
        : tensor<1x2x3xf32>, tensor<1x1x3xf32>,
          tensor<1x2x3xf32>, tensor<1x1x3xf32>
  }

  // CHECK-LABEL: func.func @rnn_0_0
  // CHECK: %[[WEIGHT:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<3x7xf32>
  // CHECK: %[[H0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_extract" name = "rnn_layer_initial_hidden_extract"(%arg1) {
  // CHECK: tensor.extract_slice
  // CHECK: tensor.collapse_shape
  // CHECK: %[[STEP0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.timestep_extract" name = "rnn_layer_timestep_extract"(%arg0) {
  // CHECK: arith.constant 0 : index
  // CHECK: tensor.extract_slice
  // CHECK: tensor.collapse_shape
  // CHECK: %[[INPUT0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "rnn_layer_input_recombine"(%[[STEP0]], %[[H0]]) {
  // CHECK: tensor.concat dim(1)
  // CHECK: %[[MVM0:[A-Za-z0-9_]+]] = sculptor.mvm %[[INPUT0]], %[[WEIGHT]] : (tensor<1x7xf32>, tensor<3x7xf32>) -> tensor<1x3xf32>
  // CHECK: %[[BIAS0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "rnn_layer_bias_add"(%[[MVM0]]) {
  // CHECK: sculptor.yield
  // CHECK: %[[TANH0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.activation" name = "rnn_layer_tanh"(%[[BIAS0]], %[[H0]]) {
  // CHECK: math.tanh
  // CHECK: %[[OUTPUT0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.output_update" name = "rnn_layer_output_update"(%[[TANH0]]) {
  // CHECK: tensor.empty()
  // CHECK: tensor.insert_slice
  // CHECK: %[[STEP1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.timestep_extract" name = "rnn_layer_timestep_extract"(%arg0) {
  // CHECK: arith.constant 1 : index
  // CHECK: %[[INPUT1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "rnn_layer_input_recombine"(%[[STEP1]], %[[TANH0]]) {
  // CHECK: %[[MVM1:[A-Za-z0-9_]+]] = sculptor.mvm %[[INPUT1]], %[[WEIGHT]] : (tensor<1x7xf32>, tensor<3x7xf32>) -> tensor<1x3xf32>
  // CHECK: %[[BIAS1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "rnn_layer_bias_add"(%[[MVM1]]) {
  // CHECK: %[[TANH1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.activation" name = "rnn_layer_tanh"(%[[BIAS1]], %[[TANH0]]) {
  // CHECK: %[[OUTPUT1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.output_update" name = "rnn_layer_output_update"(%[[TANH1]], %[[OUTPUT0]]) {
  // CHECK: tensor.insert_slice
  // CHECK: %[[HIDDEN_OUT:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_output" name = "rnn_layer_hidden_output"(%[[TANH1]]) {
  // CHECK: tensor.expand_shape
  // CHECK: return %[[OUTPUT1]], %[[HIDDEN_OUT]]
  func.func @rnn_0_0(%arg0: tensor<1x2x4xf32>, %arg1: tensor<2x1x3xf32>)
      -> (tensor<1x2x3xf32>, tensor<1x1x3xf32>)
      attributes {layer_type = "rnn"} {
    %w_ih = arith.constant dense<1.000000e+00> : tensor<3x4xf32>
    %w_hh = arith.constant dense<2.000000e+00> : tensor<3x3xf32>
    %output, %hn = sculptor.nn.rnn_layer %arg0, %arg1, %w_ih, %w_hh
        {batch_first = true, has_bias = false, hidden_size = 3 : i64,
         layer_index = 0 : i64, num_layers = 2 : i64}
        : (tensor<1x2x4xf32>, tensor<2x1x3xf32>, tensor<3x4xf32>,
           tensor<3x3xf32>) -> (tensor<1x2x3xf32>, tensor<1x1x3xf32>)
    return %output, %hn : tensor<1x2x3xf32>, tensor<1x1x3xf32>
  }

  // CHECK-LABEL: func.func @rnnwbias_0_1
  // CHECK: %[[WEIGHT:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<3x6xf32>
  // CHECK: %[[H0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_extract" name = "rnn_layer_initial_hidden_extract"(%arg1) {
  // CHECK: tensor.extract_slice
  // CHECK: tensor.collapse_shape
  // CHECK: %[[STEP0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.timestep_extract" name = "rnn_layer_timestep_extract"(%arg0) {
  // CHECK: arith.constant 0 : index
  // CHECK: tensor.extract_slice
  // CHECK: tensor.collapse_shape
  // CHECK: %[[INPUT0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "rnn_layer_input_recombine"(%[[STEP0]], %[[H0]]) {
  // CHECK: tensor.concat dim(1)
  // CHECK: %[[MVM0:[A-Za-z0-9_]+]] = sculptor.mvm %[[INPUT0]], %[[WEIGHT]] : (tensor<1x6xf32>, tensor<3x6xf32>) -> tensor<1x3xf32>
  // CHECK: %[[BIAS0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "rnn_layer_bias_add"(%[[MVM0]]) {
  // CHECK: arith.constant {{.*}} : tensor<3xf32>
  // CHECK: tensor.expand_shape
  // CHECK: arith.addf
  // CHECK: sculptor.yield
  // CHECK: %[[TANH0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.activation" name = "rnn_layer_tanh"(%[[BIAS0]], %[[H0]]) {
  // CHECK: math.tanh
  // CHECK: %[[OUTPUT0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.output_update" name = "rnn_layer_output_update"(%[[TANH0]]) {
  // CHECK: tensor.empty()
  // CHECK: tensor.insert_slice
  // CHECK: %[[STEP1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.timestep_extract" name = "rnn_layer_timestep_extract"(%arg0) {
  // CHECK: arith.constant 1 : index
  // CHECK: %[[INPUT1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "rnn_layer_input_recombine"(%[[STEP1]], %[[TANH0]]) {
  // CHECK: %[[MVM1:[A-Za-z0-9_]+]] = sculptor.mvm %[[INPUT1]], %[[WEIGHT]] : (tensor<1x6xf32>, tensor<3x6xf32>) -> tensor<1x3xf32>
  // CHECK: %[[BIAS1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "rnn_layer_bias_add"(%[[MVM1]]) {
  // CHECK: %[[TANH1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.activation" name = "rnn_layer_tanh"(%[[BIAS1]], %[[TANH0]]) {
  // CHECK: %[[OUTPUT1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.output_update" name = "rnn_layer_output_update"(%[[TANH1]], %[[OUTPUT0]]) {
  // CHECK: tensor.insert_slice
  // CHECK: %[[HIDDEN_OUT:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_output" name = "rnn_layer_hidden_output"(%[[TANH1]]) {
  // CHECK: tensor.expand_shape
  // CHECK: return %[[OUTPUT1]], %[[HIDDEN_OUT]]
  func.func @rnnwbias_0_1(%arg0: tensor<1x2x3xf32>,
                          %arg1: tensor<2x1x3xf32>)
      -> (tensor<1x2x3xf32>, tensor<1x1x3xf32>)
      attributes {layer_type = "rnn_w_bias"} {
    %w_ih = arith.constant dense<3.000000e+00> : tensor<3x3xf32>
    %w_hh = arith.constant dense<4.000000e+00> : tensor<3x3xf32>
    %b_ih = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>
    %b_hh = arith.constant dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf32>
    %output, %hn = sculptor.nn.rnn_layer %arg0, %arg1, %w_ih, %w_hh, %b_ih, %b_hh
        {batch_first = true, has_bias = true, hidden_size = 3 : i64,
         layer_index = 1 : i64, num_layers = 2 : i64}
        : (tensor<1x2x3xf32>, tensor<2x1x3xf32>, tensor<3x3xf32>,
           tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>)
          -> (tensor<1x2x3xf32>, tensor<1x1x3xf32>)
    return %output, %hn : tensor<1x2x3xf32>, tensor<1x1x3xf32>
  }
}
