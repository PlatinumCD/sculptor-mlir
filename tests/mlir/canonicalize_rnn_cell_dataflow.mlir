// RUN: sculptor-mlir-opt %s --sculptor-canonicalize-layers | FileCheck %s --implicit-check-not=sculptor.nn.linear --implicit-check-not=tensor.empty

#map = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>

module {
  // CHECK-LABEL: func.func @forward
  // CHECK: %[[CELL0:[0-9]+]] = sculptor.nn.rnn_cell %arg0, %arg1
  // CHECK-SAME: activation = "tanh"
  // CHECK-SAME: has_bias = true
  // CHECK: %[[CELL1:[0-9]+]] = sculptor.nn.rnn_cell %[[CELL0]], %arg2
  // CHECK-SAME: activation = "tanh"
  // CHECK-SAME: has_bias = true
  // CHECK: return %[[CELL1]]
  func.func @forward(
      %x: tensor<1x4xf32>,
      %h0: tensor<1x3xf32>,
      %h1: tensor<1x3xf32>
  ) -> tensor<1x3xf32> {
    %zero = arith.constant 0.000000e+00 : f32

    %w0_hh = arith.constant dense<1.000000e+00> : tensor<3x3xf32>
    %w0_hh_empty = tensor.empty() : tensor<3x3xf32>
    %w0_hh_t = linalg.transpose
        ins(%w0_hh : tensor<3x3xf32>)
        outs(%w0_hh_empty : tensor<3x3xf32>) permutation = [1, 0]
    %w0_ih = arith.constant dense<2.000000e+00> : tensor<3x4xf32>
    %w0_ih_empty = tensor.empty() : tensor<4x3xf32>
    %w0_ih_t = linalg.transpose
        ins(%w0_ih : tensor<3x4xf32>)
        outs(%w0_ih_empty : tensor<4x3xf32>) permutation = [1, 0]
    %out0_empty = tensor.empty() : tensor<1x3xf32>
    %fill0 = linalg.fill ins(%zero : f32)
        outs(%out0_empty : tensor<1x3xf32>) -> tensor<1x3xf32>
    %hh0 = linalg.matmul
        ins(%h0, %w0_hh_t : tensor<1x3xf32>, tensor<3x3xf32>)
        outs(%fill0 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %b0_hh = arith.constant dense<3.000000e+00> : tensor<3xf32>
    %hh0_biased = linalg.generic {
        indexing_maps = [#map, #bias, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hh0, %b0_hh : tensor<1x3xf32>, tensor<3xf32>)
        outs(%out0_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %bias_value: f32, %out: f32):
      %sum = arith.addf %in, %bias_value : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %ih0 = linalg.matmul
        ins(%x, %w0_ih_t : tensor<1x4xf32>, tensor<4x3xf32>)
        outs(%fill0 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %b0_ih = arith.constant dense<4.000000e+00> : tensor<3xf32>
    %ih0_biased = linalg.generic {
        indexing_maps = [#map, #bias, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ih0, %b0_ih : tensor<1x3xf32>, tensor<3xf32>)
        outs(%out0_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %bias_value: f32, %out: f32):
      %sum = arith.addf %in, %bias_value : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %pre0 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hh0_biased, %ih0_biased : tensor<1x3xf32>, tensor<1x3xf32>)
        outs(%out0_empty : tensor<1x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %cell0 = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%pre0 : tensor<1x3xf32>)
        outs(%out0_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %tanh = math.tanh %in : f32
      linalg.yield %tanh : f32
    } -> tensor<1x3xf32>

    %w1_hh = arith.constant dense<5.000000e+00> : tensor<3x3xf32>
    %w1_hh_empty = tensor.empty() : tensor<3x3xf32>
    %w1_hh_t = linalg.transpose
        ins(%w1_hh : tensor<3x3xf32>)
        outs(%w1_hh_empty : tensor<3x3xf32>) permutation = [1, 0]
    %w1_ih = arith.constant dense<6.000000e+00> : tensor<3x3xf32>
    %w1_ih_empty = tensor.empty() : tensor<3x3xf32>
    %w1_ih_t = linalg.transpose
        ins(%w1_ih : tensor<3x3xf32>)
        outs(%w1_ih_empty : tensor<3x3xf32>) permutation = [1, 0]
    %out1_empty = tensor.empty() : tensor<1x3xf32>
    %fill1 = linalg.fill ins(%zero : f32)
        outs(%out1_empty : tensor<1x3xf32>) -> tensor<1x3xf32>
    %hh1 = linalg.matmul
        ins(%h1, %w1_hh_t : tensor<1x3xf32>, tensor<3x3xf32>)
        outs(%fill1 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %b1_hh = arith.constant dense<7.000000e+00> : tensor<3xf32>
    %hh1_biased = linalg.generic {
        indexing_maps = [#map, #bias, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hh1, %b1_hh : tensor<1x3xf32>, tensor<3xf32>)
        outs(%out1_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %bias_value: f32, %out: f32):
      %sum = arith.addf %in, %bias_value : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %ih1 = linalg.matmul
        ins(%cell0, %w1_ih_t : tensor<1x3xf32>, tensor<3x3xf32>)
        outs(%fill1 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %b1_ih = arith.constant dense<8.000000e+00> : tensor<3xf32>
    %ih1_biased = linalg.generic {
        indexing_maps = [#map, #bias, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ih1, %b1_ih : tensor<1x3xf32>, tensor<3xf32>)
        outs(%out1_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %bias_value: f32, %out: f32):
      %sum = arith.addf %in, %bias_value : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %pre1 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hh1_biased, %ih1_biased : tensor<1x3xf32>, tensor<1x3xf32>)
        outs(%out1_empty : tensor<1x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %cell1 = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%pre1 : tensor<1x3xf32>)
        outs(%out1_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %tanh = math.tanh %in : f32
      linalg.yield %tanh : f32
    } -> tensor<1x3xf32>

    return %cell1 : tensor<1x3xf32>
  }
}
