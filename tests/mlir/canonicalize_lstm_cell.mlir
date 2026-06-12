// RUN: sculptor-mlir-opt %s --sculptor-canonicalize-layers | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-canonicalize-layers --sculptor-extract-layers | FileCheck %s --check-prefix=PIPE

#map = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>
#vec = affine_map<(d0) -> (d0)>
#row = affine_map<(d0, d1) -> (d0, 0)>
#col = affine_map<(d0, d1) -> (0, d1)>

module {
  // CHECK-LABEL: func.func @forward
  // CHECK: %[[H:.*]], %[[C:.*]] = sculptor.nn.lstm_cell
  // CHECK-SAME: {has_bias = true}
  // CHECK-NOT: linalg.
  // CHECK: return %[[H]], %[[C]]
  // PIPE-LABEL: func.func @forward
  // PIPE: %[[CALL:.*]]:2 = call @lstmcellwbias_0(%arg0, %arg1, %arg2)
  // PIPE: return %[[CALL]]#0, %[[CALL]]#1
  // PIPE-LABEL: func.func @lstmcellwbias_0
  // PIPE-SAME: attributes {layer_type = "lstm_cell_w_bias"}
  // PIPE: sculptor.nn.lstm_cell
  // PIPE-SAME: has_bias = true
  // PIPE: return
  func.func @forward(
      %x: tensor<1x2xf32>,
      %h: tensor<1x1xf32>,
      %c_prev: tensor<1x1xf32>
  ) -> (tensor<1x1xf32>, tensor<1x1xf32>) {
    %zero = arith.constant 0.000000e+00 : f32
    %one = arith.constant 1.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64

    %w_hh = arith.constant dense<2.000000e+00> : tensor<4x1xf32>
    %w_hh_empty = tensor.empty() : tensor<1x4xf32>
    %w_hh_t = linalg.transpose
        ins(%w_hh : tensor<4x1xf32>)
        outs(%w_hh_empty : tensor<1x4xf32>) permutation = [1, 0]
    %gate_empty = tensor.empty() : tensor<1x4xf32>
    %gate_fill = linalg.fill ins(%zero : f32)
        outs(%gate_empty : tensor<1x4xf32>) -> tensor<1x4xf32>
    %hh = linalg.matmul
        ins(%h, %w_hh_t : tensor<1x1xf32>, tensor<1x4xf32>)
        outs(%gate_fill : tensor<1x4xf32>) -> tensor<1x4xf32>
    %b_hh = arith.constant dense<3.000000e+00> : tensor<4xf32>
    %hh_biased = linalg.generic {
        indexing_maps = [#map, #bias, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hh, %b_hh : tensor<1x4xf32>, tensor<4xf32>)
        outs(%gate_empty : tensor<1x4xf32>) {
    ^bb0(%in: f32, %bias_in: f32, %out: f32):
      %sum = arith.addf %in, %bias_in : f32
      linalg.yield %sum : f32
    } -> tensor<1x4xf32>

    %w_ih = arith.constant dense<1.000000e+00> : tensor<4x2xf32>
    %w_ih_empty = tensor.empty() : tensor<2x4xf32>
    %w_ih_t = linalg.transpose
        ins(%w_ih : tensor<4x2xf32>)
        outs(%w_ih_empty : tensor<2x4xf32>) permutation = [1, 0]
    %ih = linalg.matmul
        ins(%x, %w_ih_t : tensor<1x2xf32>, tensor<2x4xf32>)
        outs(%gate_fill : tensor<1x4xf32>) -> tensor<1x4xf32>
    %b_ih = arith.constant dense<4.000000e+00> : tensor<4xf32>
    %ih_biased = linalg.generic {
        indexing_maps = [#map, #bias, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ih, %b_ih : tensor<1x4xf32>, tensor<4xf32>)
        outs(%gate_empty : tensor<1x4xf32>) {
    ^bb0(%in: f32, %bias_in: f32, %out: f32):
      %sum = arith.addf %in, %bias_in : f32
      linalg.yield %sum : f32
    } -> tensor<1x4xf32>

    %pre = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hh_biased, %ih_biased : tensor<1x4xf32>, tensor<1x4xf32>)
        outs(%gate_empty : tensor<1x4xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x4xf32>
    %collapsed = tensor.collapse_shape %pre [[0, 1]]
        : tensor<1x4xf32> into tensor<4xf32>

    %zero_index_empty = tensor.empty() : tensor<1xi64>
    %zero_index = linalg.generic {
        indexing_maps = [#vec], iterator_types = ["parallel"]}
        outs(%zero_index_empty : tensor<1xi64>) {
    ^bb0(%out: i64):
      linalg.yield %c0_i64 : i64
    } -> tensor<1xi64>
    %zero_index_exp = tensor.expand_shape %zero_index [[0, 1]]
        output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    %base_empty = tensor.empty() : tensor<1x1xi64>
    %base = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%zero_index_exp : tensor<1x1xi64>)
        outs(%base_empty : tensor<1x1xi64>) {
    ^bb0(%in: i64, %out: i64):
      %scaled = arith.muli %in, %c4_i64 : i64
      linalg.yield %scaled : i64
    } -> tensor<1x1xi64>
    %range_empty = tensor.empty() : tensor<1xi64>
    %range = linalg.generic {
        indexing_maps = [#vec], iterator_types = ["parallel"]}
        outs(%range_empty : tensor<1xi64>) {
    ^bb0(%out: i64):
      %idx = linalg.index 0 : index
      %cast = arith.index_cast %idx : index to i64
      linalg.yield %cast : i64
    } -> tensor<1xi64>
    %range_exp = tensor.expand_shape %range [[0, 1]]
        output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    %indices_empty = tensor.empty() : tensor<1x1xi64>
    %indices = linalg.generic {
        indexing_maps = [#row, #col, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%base, %range_exp : tensor<1x1xi64>, tensor<1x1xi64>)
        outs(%indices_empty : tensor<1x1xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %out: i64):
      %sum = arith.addi %lhs, %rhs : i64
      linalg.yield %sum : i64
    } -> tensor<1x1xi64>
    %indices_flat = tensor.collapse_shape %indices [[0, 1]]
        : tensor<1x1xi64> into tensor<1xi64>

    %gate_slice_empty = tensor.empty() : tensor<1xf32>
    %input_slice = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%indices_flat : tensor<1xi64>)
        outs(%gate_slice_empty : tensor<1xf32>) {
    ^bb0(%idx_in: i64, %out: f32):
      %is_neg = arith.cmpi slt, %idx_in, %c0_i64 : i64
      %wrapped = arith.addi %idx_in, %c4_i64 : i64
      %idx = arith.select %is_neg, %wrapped, %idx_in : i64
      %idx_cast = arith.index_cast %idx : i64 to index
      %value = tensor.extract %collapsed[%idx_cast] : tensor<4xf32>
      linalg.yield %value : f32
    } -> tensor<1xf32>
    %input_exp = tensor.expand_shape %input_slice [[0, 1]]
        output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>

    %forget_indices = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%indices_flat : tensor<1xi64>)
        outs(%range_empty : tensor<1xi64>) {
    ^bb0(%idx_in: i64, %out: i64):
      %idx = arith.addi %idx_in, %c1_i64 : i64
      linalg.yield %idx : i64
    } -> tensor<1xi64>
    %forget_slice = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%forget_indices : tensor<1xi64>)
        outs(%gate_slice_empty : tensor<1xf32>) {
    ^bb0(%idx_in: i64, %out: f32):
      %is_neg = arith.cmpi slt, %idx_in, %c0_i64 : i64
      %wrapped = arith.addi %idx_in, %c4_i64 : i64
      %idx = arith.select %is_neg, %wrapped, %idx_in : i64
      %idx_cast = arith.index_cast %idx : i64 to index
      %value = tensor.extract %collapsed[%idx_cast] : tensor<4xf32>
      linalg.yield %value : f32
    } -> tensor<1xf32>
    %forget_exp = tensor.expand_shape %forget_slice [[0, 1]]
        output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>

    %candidate_indices = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%indices_flat : tensor<1xi64>)
        outs(%range_empty : tensor<1xi64>) {
    ^bb0(%idx_in: i64, %out: i64):
      %idx = arith.addi %idx_in, %c2_i64 : i64
      linalg.yield %idx : i64
    } -> tensor<1xi64>
    %candidate_slice = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%candidate_indices : tensor<1xi64>)
        outs(%gate_slice_empty : tensor<1xf32>) {
    ^bb0(%idx_in: i64, %out: f32):
      %is_neg = arith.cmpi slt, %idx_in, %c0_i64 : i64
      %wrapped = arith.addi %idx_in, %c4_i64 : i64
      %idx = arith.select %is_neg, %wrapped, %idx_in : i64
      %idx_cast = arith.index_cast %idx : i64 to index
      %value = tensor.extract %collapsed[%idx_cast] : tensor<4xf32>
      linalg.yield %value : f32
    } -> tensor<1xf32>
    %candidate_exp = tensor.expand_shape %candidate_slice [[0, 1]]
        output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>

    %output_indices = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%indices_flat : tensor<1xi64>)
        outs(%range_empty : tensor<1xi64>) {
    ^bb0(%idx_in: i64, %out: i64):
      %idx = arith.addi %idx_in, %c3_i64 : i64
      linalg.yield %idx : i64
    } -> tensor<1xi64>
    %output_slice = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%output_indices : tensor<1xi64>)
        outs(%gate_slice_empty : tensor<1xf32>) {
    ^bb0(%idx_in: i64, %out: f32):
      %is_neg = arith.cmpi slt, %idx_in, %c0_i64 : i64
      %wrapped = arith.addi %idx_in, %c4_i64 : i64
      %idx = arith.select %is_neg, %wrapped, %idx_in : i64
      %idx_cast = arith.index_cast %idx : i64 to index
      %value = tensor.extract %collapsed[%idx_cast] : tensor<4xf32>
      linalg.yield %value : f32
    } -> tensor<1xf32>
    %output_exp = tensor.expand_shape %output_slice [[0, 1]]
        output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>

    %state_empty = tensor.empty() : tensor<1x1xf32>
    %input_gate = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%input_exp : tensor<1x1xf32>) outs(%state_empty : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %neg = arith.negf %in : f32
      %exp = math.exp %neg : f32
      %sum = arith.addf %exp, %one : f32
      %sigmoid = arith.divf %one, %sum : f32
      linalg.yield %sigmoid : f32
    } -> tensor<1x1xf32>
    %forget_gate = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%forget_exp : tensor<1x1xf32>) outs(%state_empty : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %neg = arith.negf %in : f32
      %exp = math.exp %neg : f32
      %sum = arith.addf %exp, %one : f32
      %sigmoid = arith.divf %one, %sum : f32
      linalg.yield %sigmoid : f32
    } -> tensor<1x1xf32>
    %candidate_gate = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%candidate_exp : tensor<1x1xf32>)
        outs(%state_empty : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %tanh = math.tanh %in : f32
      linalg.yield %tanh : f32
    } -> tensor<1x1xf32>
    %output_gate = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%output_exp : tensor<1x1xf32>) outs(%state_empty : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %neg = arith.negf %in : f32
      %exp = math.exp %neg : f32
      %sum = arith.addf %exp, %one : f32
      %sigmoid = arith.divf %one, %sum : f32
      linalg.yield %sigmoid : f32
    } -> tensor<1x1xf32>

    %forget_cell = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%forget_gate, %c_prev : tensor<1x1xf32>, tensor<1x1xf32>)
        outs(%state_empty : tensor<1x1xf32>) {
    ^bb0(%gate: f32, %cell: f32, %out: f32):
      %mul = arith.mulf %gate, %cell : f32
      linalg.yield %mul : f32
    } -> tensor<1x1xf32>
    %input_candidate = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%input_gate, %candidate_gate : tensor<1x1xf32>, tensor<1x1xf32>)
        outs(%state_empty : tensor<1x1xf32>) {
    ^bb0(%gate: f32, %candidate: f32, %out: f32):
      %mul = arith.mulf %gate, %candidate : f32
      linalg.yield %mul : f32
    } -> tensor<1x1xf32>
    %c_next = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%forget_cell, %input_candidate : tensor<1x1xf32>, tensor<1x1xf32>)
        outs(%state_empty : tensor<1x1xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x1xf32>
    %c_tanh = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%c_next : tensor<1x1xf32>) outs(%state_empty : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %tanh = math.tanh %in : f32
      linalg.yield %tanh : f32
    } -> tensor<1x1xf32>
    %h_next = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%output_gate, %c_tanh : tensor<1x1xf32>, tensor<1x1xf32>)
        outs(%state_empty : tensor<1x1xf32>) {
    ^bb0(%gate: f32, %cell: f32, %out: f32):
      %mul = arith.mulf %gate, %cell : f32
      linalg.yield %mul : f32
    } -> tensor<1x1xf32>

    return %h_next, %c_next : tensor<1x1xf32>, tensor<1x1xf32>
  }
}
