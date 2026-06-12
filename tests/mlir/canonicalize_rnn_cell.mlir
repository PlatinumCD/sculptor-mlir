// RUN: sculptor-mlir-opt %s --sculptor-canonicalize-layers | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-canonicalize-layers --sculptor-extract-layers | FileCheck %s --check-prefix=PIPE

#map = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>

module {
  // CHECK-LABEL: func.func @forward
  // CHECK: sculptor.nn.rnn_cell
  // CHECK-SAME: activation = "tanh"
  // CHECK-SAME: has_bias = true
  // CHECK-NOT: tensor.empty
  // CHECK: sculptor.nn.rnn_cell
  // CHECK-SAME: activation = "tanh"
  // CHECK-SAME: has_bias = false
  // CHECK-NOT: sculptor.nn.rnn_cell
  // CHECK: math.tanh
  // PIPE-LABEL: func.func @forward
  // PIPE: call @rnncellwbias_0
  // PIPE: call @rnncell_0
  // PIPE-LABEL: func.func @rnncellwbias_0
  // PIPE-SAME: attributes {layer_type = "rnn_cell_w_bias"}
  // PIPE: sculptor.nn.rnn_cell
  // PIPE-LABEL: func.func @rnncell_0
  // PIPE-SAME: attributes {layer_type = "rnn_cell"}
  // PIPE: sculptor.nn.rnn_cell
  func.func @forward(
      %x: tensor<1x4xf32>,
      %h: tensor<1x3xf32>,
      %x_no_bias: tensor<1x5xf32>,
      %h_no_bias: tensor<1x2xf32>,
      %ambiguous_x: tensor<1x3xf32>,
      %ambiguous_h: tensor<1x3xf32>
  ) -> (tensor<1x3xf32>, tensor<1x2xf32>, tensor<1x3xf32>) {
    %zero = arith.constant 0.000000e+00 : f32

    %w_hh = arith.constant dense<1.000000e+00> : tensor<3x3xf32>
    %w_hh_empty = tensor.empty() : tensor<3x3xf32>
    %w_hh_t = linalg.transpose
        ins(%w_hh : tensor<3x3xf32>)
        outs(%w_hh_empty : tensor<3x3xf32>) permutation = [1, 0]
    %w_ih = arith.constant dense<2.000000e+00> : tensor<3x4xf32>
    %w_ih_empty = tensor.empty() : tensor<4x3xf32>
    %w_ih_t = linalg.transpose
        ins(%w_ih : tensor<3x4xf32>)
        outs(%w_ih_empty : tensor<4x3xf32>) permutation = [1, 0]
    %out_empty = tensor.empty() : tensor<1x3xf32>
    %fill = linalg.fill ins(%zero : f32)
        outs(%out_empty : tensor<1x3xf32>) -> tensor<1x3xf32>
    %hh = linalg.matmul
        ins(%h, %w_hh_t : tensor<1x3xf32>, tensor<3x3xf32>)
        outs(%fill : tensor<1x3xf32>) -> tensor<1x3xf32>
    %b_hh = arith.constant dense<3.000000e+00> : tensor<3xf32>
    %hh_biased = linalg.generic {
        indexing_maps = [#map, #bias, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hh, %b_hh : tensor<1x3xf32>, tensor<3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %bias: f32, %out: f32):
      %sum = arith.addf %in, %bias : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %ih = linalg.matmul
        ins(%x, %w_ih_t : tensor<1x4xf32>, tensor<4x3xf32>)
        outs(%fill : tensor<1x3xf32>) -> tensor<1x3xf32>
    %b_ih = arith.constant dense<4.000000e+00> : tensor<3xf32>
    %ih_biased = linalg.generic {
        indexing_maps = [#map, #bias, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ih, %b_ih : tensor<1x3xf32>, tensor<3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %bias: f32, %out: f32):
      %sum = arith.addf %in, %bias : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %pre = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hh_biased, %ih_biased : tensor<1x3xf32>, tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %biased = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%pre : tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %tanh = math.tanh %in : f32
      linalg.yield %tanh : f32
    } -> tensor<1x3xf32>

    %nb_w_ih = arith.constant dense<5.000000e+00> : tensor<2x5xf32>
    %nb_w_ih_empty = tensor.empty() : tensor<5x2xf32>
    %nb_w_ih_t = linalg.transpose
        ins(%nb_w_ih : tensor<2x5xf32>)
        outs(%nb_w_ih_empty : tensor<5x2xf32>) permutation = [1, 0]
    %nb_w_hh = arith.constant dense<6.000000e+00> : tensor<2x2xf32>
    %nb_w_hh_empty = tensor.empty() : tensor<2x2xf32>
    %nb_w_hh_t = linalg.transpose
        ins(%nb_w_hh : tensor<2x2xf32>)
        outs(%nb_w_hh_empty : tensor<2x2xf32>) permutation = [1, 0]
    %nb_out_empty = tensor.empty() : tensor<1x2xf32>
    %nb_fill = linalg.fill ins(%zero : f32)
        outs(%nb_out_empty : tensor<1x2xf32>) -> tensor<1x2xf32>
    %nb_ih = linalg.matmul
        ins(%x_no_bias, %nb_w_ih_t : tensor<1x5xf32>, tensor<5x2xf32>)
        outs(%nb_fill : tensor<1x2xf32>) -> tensor<1x2xf32>
    %nb_hh = linalg.matmul
        ins(%h_no_bias, %nb_w_hh_t : tensor<1x2xf32>, tensor<2x2xf32>)
        outs(%nb_fill : tensor<1x2xf32>) -> tensor<1x2xf32>
    %nb_pre = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%nb_ih, %nb_hh : tensor<1x2xf32>, tensor<1x2xf32>)
        outs(%nb_out_empty : tensor<1x2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x2xf32>
    %no_bias = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%nb_pre : tensor<1x2xf32>)
        outs(%nb_out_empty : tensor<1x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %tanh = math.tanh %in : f32
      linalg.yield %tanh : f32
    } -> tensor<1x2xf32>

    %amb_w0 = arith.constant dense<7.000000e+00> : tensor<3x3xf32>
    %amb_w0_empty = tensor.empty() : tensor<3x3xf32>
    %amb_w0_t = linalg.transpose
        ins(%amb_w0 : tensor<3x3xf32>)
        outs(%amb_w0_empty : tensor<3x3xf32>) permutation = [1, 0]
    %amb_w1 = arith.constant dense<8.000000e+00> : tensor<3x3xf32>
    %amb_w1_empty = tensor.empty() : tensor<3x3xf32>
    %amb_w1_t = linalg.transpose
        ins(%amb_w1 : tensor<3x3xf32>)
        outs(%amb_w1_empty : tensor<3x3xf32>) permutation = [1, 0]
    %amb_out_empty = tensor.empty() : tensor<1x3xf32>
    %amb_fill = linalg.fill ins(%zero : f32)
        outs(%amb_out_empty : tensor<1x3xf32>) -> tensor<1x3xf32>
    %amb0 = linalg.matmul
        ins(%ambiguous_x, %amb_w0_t : tensor<1x3xf32>, tensor<3x3xf32>)
        outs(%amb_fill : tensor<1x3xf32>) -> tensor<1x3xf32>
    %amb1 = linalg.matmul
        ins(%ambiguous_h, %amb_w1_t : tensor<1x3xf32>, tensor<3x3xf32>)
        outs(%amb_fill : tensor<1x3xf32>) -> tensor<1x3xf32>
    %amb_pre = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%amb0, %amb1 : tensor<1x3xf32>, tensor<1x3xf32>)
        outs(%amb_out_empty : tensor<1x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %ambiguous = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%amb_pre : tensor<1x3xf32>)
        outs(%amb_out_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %tanh = math.tanh %in : f32
      linalg.yield %tanh : f32
    } -> tensor<1x3xf32>

    return %biased, %no_bias, %ambiguous
        : tensor<1x3xf32>, tensor<1x2xf32>, tensor<1x3xf32>
  }
}
