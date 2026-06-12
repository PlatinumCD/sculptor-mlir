// RUN: sculptor-mlir-opt %s --sculptor-convert-layers | FileCheck %s --implicit-check-not=sculptor.nn.gru_layer --implicit-check-not=scf.for --implicit-check-not=sculptor.matrix. --implicit-check-not=sculptor.vector. --implicit-check-not=sculptor.array. --implicit-check-not=golem

module {
  // CHECK-LABEL: func.func @forward
  // CHECK: call @gru_0_0(%arg0, %arg1)
  // CHECK: call @gruwbias_0_1(%arg2, %arg1)
  func.func @forward(
      %arg0: tensor<1x2x5xf32>,
      %arg1: tensor<2x1x2xf32>,
      %arg2: tensor<1x2x3xf32>
  ) -> (tensor<1x2x2xf32>, tensor<1x1x2xf32>,
        tensor<1x2x2xf32>, tensor<1x1x2xf32>) {
    %0:2 = call @gru_0_0(%arg0, %arg1)
        : (tensor<1x2x5xf32>, tensor<2x1x2xf32>)
          -> (tensor<1x2x2xf32>, tensor<1x1x2xf32>)
    %1:2 = call @gruwbias_0_1(%arg2, %arg1)
        : (tensor<1x2x3xf32>, tensor<2x1x2xf32>)
          -> (tensor<1x2x2xf32>, tensor<1x1x2xf32>)
    return %0#0, %0#1, %1#0, %1#1
        : tensor<1x2x2xf32>, tensor<1x1x2xf32>,
          tensor<1x2x2xf32>, tensor<1x1x2xf32>
  }

  // CHECK-LABEL: func.func @gru_0_0
  // CHECK: %[[WEIGHT:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<8x7xf32>
  // CHECK: %[[H0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_extract" name = "gru_layer_initial_hidden_extract"(%arg1) {
  // CHECK: tensor.extract_slice
  // CHECK: tensor.collapse_shape
  // CHECK: %[[STEP0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.timestep_extract" name = "gru_layer_timestep_extract"(%arg0) {
  // CHECK: arith.constant 0 : index
  // CHECK: tensor.extract_slice
  // CHECK: tensor.collapse_shape
  // CHECK: %[[INPUT0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "gru_layer_input_recombine"(%[[STEP0]], %[[H0]]) {
  // CHECK: tensor.concat dim(1)
  // CHECK: %[[MVM0:[A-Za-z0-9_]+]] = sculptor.mvm %[[INPUT0]], %[[WEIGHT]] : (tensor<1x7xf32>, tensor<8x7xf32>) -> tensor<1x8xf32>
  // CHECK: %[[BIAS0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "gru_layer_bias_add"(%[[MVM0]]) {
  // CHECK: sculptor.yield
  // CHECK: %[[SPLIT0:[A-Za-z0-9_]+]]:4 = sculptor.task_region kind = "digital.gate_split" name = "gru_layer_gate_split"(%[[BIAS0]]) {
  // CHECK: tensor.extract_slice {{.*}}[0, 0] [1, 2] [1, 1]
  // CHECK: tensor.extract_slice {{.*}}[0, 2] [1, 2] [1, 1]
  // CHECK: tensor.extract_slice {{.*}}[0, 4] [1, 2] [1, 1]
  // CHECK: tensor.extract_slice {{.*}}[0, 6] [1, 2] [1, 1]
  // CHECK: %[[ACT0:[A-Za-z0-9_]+]]:2 = sculptor.task_region kind = "digital.activation" name = "gru_layer_gate_activation"(%[[SPLIT0]]#0, %[[SPLIT0]]#1) {
  // CHECK: math.exp
  // CHECK: %[[CANDIDATE0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.candidate_update" name = "gru_layer_candidate_update"(%[[ACT0]]#0, %[[SPLIT0]]#2, %[[SPLIT0]]#3) {
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: math.tanh
  // CHECK: %[[HIDDEN0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_update" name = "gru_layer_hidden_update"(%[[CANDIDATE0]], %[[ACT0]]#1, %[[H0]]) {
  // CHECK: arith.subf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: %[[OUTPUT0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.output_update" name = "gru_layer_output_update"(%[[HIDDEN0]]) {
  // CHECK: tensor.empty()
  // CHECK: tensor.insert_slice
  // CHECK: %[[STEP1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.timestep_extract" name = "gru_layer_timestep_extract"(%arg0) {
  // CHECK: arith.constant 1 : index
  // CHECK: %[[INPUT1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "gru_layer_input_recombine"(%[[STEP1]], %[[HIDDEN0]]) {
  // CHECK: %[[MVM1:[A-Za-z0-9_]+]] = sculptor.mvm %[[INPUT1]], %[[WEIGHT]] : (tensor<1x7xf32>, tensor<8x7xf32>) -> tensor<1x8xf32>
  // CHECK: %[[BIAS1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "gru_layer_bias_add"(%[[MVM1]]) {
  // CHECK: %[[SPLIT1:[A-Za-z0-9_]+]]:4 = sculptor.task_region kind = "digital.gate_split" name = "gru_layer_gate_split"(%[[BIAS1]]) {
  // CHECK: %[[ACT1:[A-Za-z0-9_]+]]:2 = sculptor.task_region kind = "digital.activation" name = "gru_layer_gate_activation"(%[[SPLIT1]]#0, %[[SPLIT1]]#1) {
  // CHECK: %[[CANDIDATE1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.candidate_update" name = "gru_layer_candidate_update"(%[[ACT1]]#0, %[[SPLIT1]]#2, %[[SPLIT1]]#3) {
  // CHECK: %[[HIDDEN1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_update" name = "gru_layer_hidden_update"(%[[CANDIDATE1]], %[[ACT1]]#1, %[[HIDDEN0]]) {
  // CHECK: %[[OUTPUT1:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.output_update" name = "gru_layer_output_update"(%[[HIDDEN1]], %[[OUTPUT0]]) {
  // CHECK: tensor.insert_slice
  // CHECK: %[[HIDDEN_OUT:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_output" name = "gru_layer_hidden_output"(%[[HIDDEN1]]) {
  // CHECK: tensor.expand_shape
  // CHECK: return %[[OUTPUT1]], %[[HIDDEN_OUT]]
  func.func @gru_0_0(%arg0: tensor<1x2x5xf32>,
                     %arg1: tensor<2x1x2xf32>)
      -> (tensor<1x2x2xf32>, tensor<1x1x2xf32>)
      attributes {layer_type = "gru"} {
    %w_ih = arith.constant dense<1.000000e+00> : tensor<6x5xf32>
    %w_hh = arith.constant dense<2.000000e+00> : tensor<6x2xf32>
    %output, %hn = sculptor.nn.gru_layer %arg0, %arg1, %w_ih, %w_hh
        {batch_first = true, has_bias = false, hidden_size = 2 : i64,
         layer_index = 0 : i64, num_layers = 2 : i64}
        : (tensor<1x2x5xf32>, tensor<2x1x2xf32>, tensor<6x5xf32>,
           tensor<6x2xf32>)
          -> (tensor<1x2x2xf32>, tensor<1x1x2xf32>)
    return %output, %hn
        : tensor<1x2x2xf32>, tensor<1x1x2xf32>
  }

  // CHECK-LABEL: func.func @gruwbias_0_1
  // CHECK: %[[WEIGHT:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<8x5xf32>
  // CHECK: %[[H0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_extract" name = "gru_layer_initial_hidden_extract"(%arg1) {
  // CHECK: %[[STEP0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.timestep_extract" name = "gru_layer_timestep_extract"(%arg0) {
  // CHECK: %[[INPUT0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "gru_layer_input_recombine"(%[[STEP0]], %[[H0]]) {
  // CHECK: %[[MVM0:[A-Za-z0-9_]+]] = sculptor.mvm %[[INPUT0]], %[[WEIGHT]] : (tensor<1x5xf32>, tensor<8x5xf32>) -> tensor<1x8xf32>
  // CHECK: %[[BIAS0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "gru_layer_bias_add"(%[[MVM0]]) {
  // CHECK: arith.constant {{.*}} : tensor<8xf32>
  // CHECK: tensor.expand_shape
  // CHECK: arith.addf
  // CHECK: %[[SPLIT0:[A-Za-z0-9_]+]]:4 = sculptor.task_region kind = "digital.gate_split" name = "gru_layer_gate_split"(%[[BIAS0]]) {
  // CHECK: %[[ACT0:[A-Za-z0-9_]+]]:2 = sculptor.task_region kind = "digital.activation" name = "gru_layer_gate_activation"(%[[SPLIT0]]#0, %[[SPLIT0]]#1) {
  // CHECK: %[[CANDIDATE0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.candidate_update" name = "gru_layer_candidate_update"(%[[ACT0]]#0, %[[SPLIT0]]#2, %[[SPLIT0]]#3) {
  // CHECK: %[[HIDDEN0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_update" name = "gru_layer_hidden_update"(%[[CANDIDATE0]], %[[ACT0]]#1, %[[H0]]) {
  // CHECK: %[[OUTPUT0:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.output_update" name = "gru_layer_output_update"(%[[HIDDEN0]]) {
  // CHECK: tensor.empty()
  // CHECK: %[[HIDDEN_OUT:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_output" name = "gru_layer_hidden_output"
  // CHECK: return {{.*}}%[[HIDDEN_OUT]]
  func.func @gruwbias_0_1(%arg0: tensor<1x2x3xf32>,
                          %arg1: tensor<2x1x2xf32>)
      -> (tensor<1x2x2xf32>, tensor<1x1x2xf32>)
      attributes {layer_type = "gru_w_bias"} {
    %w_ih = arith.constant dense<3.000000e+00> : tensor<6x3xf32>
    %w_hh = arith.constant dense<4.000000e+00> : tensor<6x2xf32>
    %b_ih = arith.constant dense<[
      1.000000e+00, 2.000000e+00, 3.000000e+00,
      4.000000e+00, 5.000000e+00, 6.000000e+00
    ]> : tensor<6xf32>
    %b_hh = arith.constant dense<[
      7.000000e+00, 8.000000e+00, 9.000000e+00,
      1.000000e+01, 1.100000e+01, 1.200000e+01
    ]> : tensor<6xf32>
    %output, %hn = sculptor.nn.gru_layer
        %arg0, %arg1, %w_ih, %w_hh, %b_ih, %b_hh
        {batch_first = true, has_bias = true, hidden_size = 2 : i64,
         layer_index = 1 : i64, num_layers = 2 : i64}
        : (tensor<1x2x3xf32>, tensor<2x1x2xf32>, tensor<6x3xf32>,
           tensor<6x2xf32>, tensor<6xf32>, tensor<6xf32>)
          -> (tensor<1x2x2xf32>, tensor<1x1x2xf32>)
    return %output, %hn
        : tensor<1x2x2xf32>, tensor<1x1x2xf32>
  }
}
