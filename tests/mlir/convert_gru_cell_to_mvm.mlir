// RUN: sculptor-mlir-opt %s --sculptor-convert-layers | FileCheck %s --implicit-check-not=sculptor.nn.gru_cell --implicit-check-not=sculptor.matrix. --implicit-check-not=sculptor.vector. --implicit-check-not=sculptor.array.

module {
  func.func @forward(
      %arg0: tensor<1x5xf32>,
      %arg1: tensor<1x2xf32>,
      %arg2: tensor<1x4xf32>,
      %arg3: tensor<1x3xf32>,
      %arg4: tensor<1x3xf32>,
      %arg5: tensor<1x3xf32>
  ) -> (tensor<1x2xf32>, tensor<1x3xf32>, tensor<1x3xf32>) {
    %0 = call @gru_cell_no_bias(%arg0, %arg1)
        : (tensor<1x5xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    %1 = call @gru_cell_bias(%arg2, %arg3)
        : (tensor<1x4xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %2 = call @gru_cell_equal_dims_bias(%arg4, %arg5)
        : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    return %0, %1, %2 : tensor<1x2xf32>, tensor<1x3xf32>, tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func @gru_cell_no_bias
  // CHECK: %[[WEIGHT:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<8x7xf32>
  // CHECK: %[[INPUT:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "gru_cell_input_recombine"(%arg0, %arg1) {
  // CHECK: ^bb0(%[[CELL_INPUT:.*]]: tensor<1x5xf32>, %[[CELL_H:.*]]: tensor<1x2xf32>):
  // CHECK: %[[CONCAT:[A-Za-z0-9_]+]] = tensor.concat dim(1) %[[CELL_INPUT]], %[[CELL_H]] : (tensor<1x5xf32>, tensor<1x2xf32>) -> tensor<1x7xf32>
  // CHECK: sculptor.yield %[[CONCAT]] : tensor<1x7xf32>
  // CHECK: %[[MVM:[A-Za-z0-9_]+]] = sculptor.mvm %[[INPUT]], %[[WEIGHT]] : (tensor<1x7xf32>, tensor<8x7xf32>) -> tensor<1x8xf32>
  // CHECK: %[[BIAS:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "gru_cell_bias_add"(%[[MVM]]) {
  // CHECK: ^bb0(%[[BIAS_ARG:.*]]: tensor<1x8xf32>):
  // CHECK: sculptor.yield %[[BIAS_ARG]] : tensor<1x8xf32>
  // CHECK: %[[SPLIT:[A-Za-z0-9_]+]]:4 = sculptor.task_region kind = "digital.gate_split" name = "gru_cell_gate_split"(%[[BIAS]]) {
  // CHECK: tensor.extract_slice {{.*}}[0, 0] [1, 2] [1, 1]
  // CHECK: tensor.extract_slice {{.*}}[0, 2] [1, 2] [1, 1]
  // CHECK: tensor.extract_slice {{.*}}[0, 4] [1, 2] [1, 1]
  // CHECK: tensor.extract_slice {{.*}}[0, 6] [1, 2] [1, 1]
  // CHECK: sculptor.yield
  // CHECK: %[[ACT:[A-Za-z0-9_]+]]:2 = sculptor.task_region kind = "digital.activation" name = "gru_cell_gate_activation"(%[[SPLIT]]#0, %[[SPLIT]]#1) {
  // CHECK: math.exp
  // CHECK: sculptor.yield
  // CHECK: %[[CANDIDATE:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.candidate_update" name = "gru_cell_candidate_update"(%[[ACT]]#0, %[[SPLIT]]#2, %[[SPLIT]]#3) {
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: math.tanh
  // CHECK: sculptor.yield
  // CHECK: %[[HIDDEN:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_update" name = "gru_cell_hidden_update"(%[[CANDIDATE]], %[[ACT]]#1, %arg1) {
  // CHECK: arith.subf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: sculptor.yield
  // CHECK: return %[[HIDDEN]] : tensor<1x2xf32>
  func.func @gru_cell_no_bias(%arg0: tensor<1x5xf32>, %arg1: tensor<1x2xf32>)
      -> tensor<1x2xf32>
      attributes {layer_type = "gru_cell"} {
    %w_ih = arith.constant dense<1.000000e+00> : tensor<6x5xf32>
    %w_hh = arith.constant dense<2.000000e+00> : tensor<6x2xf32>
    %0 = sculptor.nn.gru_cell %arg0, %arg1, %w_ih, %w_hh
        {has_bias = false}
        : (tensor<1x5xf32>, tensor<1x2xf32>, tensor<6x5xf32>,
           tensor<6x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }

  // CHECK-LABEL: func.func @gru_cell_bias
  // CHECK: %[[BIAS_WEIGHT:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<12x7xf32>
  // CHECK: %[[BIAS_INPUT:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "gru_cell_input_recombine"(%arg0, %arg1) {
  // CHECK: ^bb0(%[[CELL_INPUT:.*]]: tensor<1x4xf32>, %[[CELL_H:.*]]: tensor<1x3xf32>):
  // CHECK: %[[CONCAT:[A-Za-z0-9_]+]] = tensor.concat dim(1) %[[CELL_INPUT]], %[[CELL_H]] : (tensor<1x4xf32>, tensor<1x3xf32>) -> tensor<1x7xf32>
  // CHECK: sculptor.yield %[[CONCAT]] : tensor<1x7xf32>
  // CHECK: %[[BIAS_MVM:[A-Za-z0-9_]+]] = sculptor.mvm %[[BIAS_INPUT]], %[[BIAS_WEIGHT]] : (tensor<1x7xf32>, tensor<12x7xf32>) -> tensor<1x12xf32>
  // CHECK: %[[BIAS_REGION:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "gru_cell_bias_add"(%[[BIAS_MVM]]) {
  // CHECK: %[[BIAS:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<12xf32>
  // CHECK: tensor.expand_shape %[[BIAS]]
  // CHECK: %[[PRE:[A-Za-z0-9_]+]] = linalg.add
  // CHECK: sculptor.yield %[[PRE]]
  // CHECK: %[[SPLIT:[A-Za-z0-9_]+]]:4 = sculptor.task_region kind = "digital.gate_split" name = "gru_cell_gate_split"(%[[BIAS_REGION]]) {
  // CHECK: tensor.extract_slice {{.*}}[0, 0] [1, 3] [1, 1]
  // CHECK: tensor.extract_slice {{.*}}[0, 3] [1, 3] [1, 1]
  // CHECK: tensor.extract_slice {{.*}}[0, 6] [1, 3] [1, 1]
  // CHECK: tensor.extract_slice {{.*}}[0, 9] [1, 3] [1, 1]
  // CHECK: sculptor.yield
  // CHECK: %[[ACT:[A-Za-z0-9_]+]]:2 = sculptor.task_region kind = "digital.activation" name = "gru_cell_gate_activation"(%[[SPLIT]]#0, %[[SPLIT]]#1) {
  // CHECK: math.exp
  // CHECK: sculptor.yield
  // CHECK: %[[CANDIDATE:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.candidate_update" name = "gru_cell_candidate_update"(%[[ACT]]#0, %[[SPLIT]]#2, %[[SPLIT]]#3) {
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: math.tanh
  // CHECK: sculptor.yield
  // CHECK: %[[HIDDEN:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_update" name = "gru_cell_hidden_update"(%[[CANDIDATE]], %[[ACT]]#1, %arg1) {
  // CHECK: arith.subf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: sculptor.yield
  // CHECK: return %[[HIDDEN]] : tensor<1x3xf32>
  func.func @gru_cell_bias(%arg0: tensor<1x4xf32>, %arg1: tensor<1x3xf32>)
      -> tensor<1x3xf32>
      attributes {layer_type = "gru_cell_w_bias"} {
    %w_ih = arith.constant dense<1.000000e+00> : tensor<9x4xf32>
    %w_hh = arith.constant dense<2.000000e+00> : tensor<9x3xf32>
    %b_ih = arith.constant dense<[
      1.000000e+00, 2.000000e+00, 3.000000e+00,
      4.000000e+00, 5.000000e+00, 6.000000e+00,
      7.000000e+00, 8.000000e+00, 9.000000e+00
    ]> : tensor<9xf32>
    %b_hh = arith.constant dense<[
      1.000000e+01, 1.100000e+01, 1.200000e+01,
      1.300000e+01, 1.400000e+01, 1.500000e+01,
      1.600000e+01, 1.700000e+01, 1.800000e+01
    ]> : tensor<9xf32>
    %0 = sculptor.nn.gru_cell %arg0, %arg1, %w_ih, %w_hh, %b_ih, %b_hh
        {has_bias = true}
        : (tensor<1x4xf32>, tensor<1x3xf32>, tensor<9x4xf32>,
           tensor<9x3xf32>, tensor<9xf32>, tensor<9xf32>)
          -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func @gru_cell_equal_dims_bias
  // CHECK: %[[EQ_WEIGHT:[A-Za-z0-9_]+]] = arith.constant {{.*}} : tensor<12x6xf32>
  // CHECK: %[[EQ_INPUT:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.input_recombine" name = "gru_cell_input_recombine"(%arg0, %arg1) {
  // CHECK: tensor.concat dim(1)
  // CHECK: %[[EQ_MVM:[A-Za-z0-9_]+]] = sculptor.mvm %[[EQ_INPUT]], %[[EQ_WEIGHT]] : (tensor<1x6xf32>, tensor<12x6xf32>) -> tensor<1x12xf32>
  // CHECK: %[[EQ_BIAS:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.bias_add" name = "gru_cell_bias_add"(%[[EQ_MVM]]) {
  // CHECK: linalg.add
  // CHECK: %[[EQ_SPLIT:[A-Za-z0-9_]+]]:4 = sculptor.task_region kind = "digital.gate_split" name = "gru_cell_gate_split"(%[[EQ_BIAS]]) {
  // CHECK: %[[EQ_ACT:[A-Za-z0-9_]+]]:2 = sculptor.task_region kind = "digital.activation" name = "gru_cell_gate_activation"(%[[EQ_SPLIT]]#0, %[[EQ_SPLIT]]#1) {
  // CHECK: %[[EQ_CANDIDATE:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.candidate_update" name = "gru_cell_candidate_update"(%[[EQ_ACT]]#0, %[[EQ_SPLIT]]#2, %[[EQ_SPLIT]]#3) {
  // CHECK: %[[EQ_HIDDEN:[A-Za-z0-9_]+]] = sculptor.task_region kind = "digital.hidden_update" name = "gru_cell_hidden_update"(%[[EQ_CANDIDATE]], %[[EQ_ACT]]#1, %arg1) {
  // CHECK: return %[[EQ_HIDDEN]] : tensor<1x3xf32>
  func.func @gru_cell_equal_dims_bias(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>)
      -> tensor<1x3xf32>
      attributes {layer_type = "gru_cell_w_bias"} {
    %w_ih = arith.constant dense<1.000000e+00> : tensor<9x3xf32>
    %w_hh = arith.constant dense<2.000000e+00> : tensor<9x3xf32>
    %b_ih = arith.constant dense<3.000000e+00> : tensor<9xf32>
    %b_hh = arith.constant dense<4.000000e+00> : tensor<9xf32>
    %0 = sculptor.nn.gru_cell %arg0, %arg1, %w_ih, %w_hh, %b_ih, %b_hh
        {has_bias = true}
        : (tensor<1x3xf32>, tensor<1x3xf32>, tensor<9x3xf32>,
           tensor<9x3xf32>, tensor<9xf32>, tensor<9xf32>)
          -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}
