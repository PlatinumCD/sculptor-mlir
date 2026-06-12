// RUN: sculptor-mlir-opt %s --sculptor-canonicalize-layers | FileCheck %s --implicit-check-not=linalg. --implicit-check-not=tensor.empty
// RUN: sculptor-mlir-opt %s --sculptor-canonicalize-layers --sculptor-extract-layers | FileCheck %s --check-prefix=PIPE

#vec = affine_map<(d0) -> (d0)>
#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
#col = affine_map<(d0, d1) -> (0, d1)>

module {
  // CHECK-LABEL: func.func @forward
  // CHECK: %[[OUT:.*]] = sculptor.nn.gru_cell %arg0, %arg1
  // CHECK-SAME: {has_bias = false}
  // CHECK: return %[[OUT]]
  // PIPE-LABEL: func.func @forward
  // PIPE: %[[CALL:.*]] = call @grucell_0(%arg0, %arg1) : (tensor<1x4xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  // PIPE: return %[[CALL]]
  func.func @forward(%arg0: tensor<1x4xf32>, %arg1: tensor<1x3xf32>)
      -> tensor<1x3xf32> {
    %w_ih = arith.constant dense<1.000000e+00> : tensor<9x4xf32>
    %zero_i64 = arith.constant 0 : i64
    %zero = arith.constant 0.000000e+00 : f32
    %one = arith.constant 1.000000e+00 : f32
    %w_hh = arith.constant dense<2.000000e+00> : tensor<9x3xf32>
    %gate_width = arith.constant 9 : i64
    %hidden_size = arith.constant 3 : i64
    %new_gate_offset = arith.constant 6 : i64

    %w_ih_t_empty = tensor.empty() : tensor<4x9xf32>
    %w_ih_t = linalg.transpose
        ins(%w_ih : tensor<9x4xf32>)
        outs(%w_ih_t_empty : tensor<4x9xf32>) permutation = [1, 0]
    %pre_empty = tensor.empty() : tensor<1x9xf32>
    %pre_fill = linalg.fill ins(%zero : f32)
        outs(%pre_empty : tensor<1x9xf32>) -> tensor<1x9xf32>
    %ih = linalg.matmul
        ins(%arg0, %w_ih_t : tensor<1x4xf32>, tensor<4x9xf32>)
        outs(%pre_fill : tensor<1x9xf32>) -> tensor<1x9xf32>
    %ih_flat = tensor.collapse_shape %ih [[0, 1]]
        : tensor<1x9xf32> into tensor<9xf32>

    %zero_index_empty = tensor.empty() : tensor<1xi64>
    %zero_index = linalg.generic {
        indexing_maps = [#vec], iterator_types = ["parallel"]}
        outs(%zero_index_empty : tensor<1xi64>) {
    ^bb0(%out: i64):
      linalg.yield %zero_i64 : i64
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
      %scaled = arith.muli %in, %gate_width : i64
      linalg.yield %scaled : i64
    } -> tensor<1x1xi64>
    %range_empty = tensor.empty() : tensor<3xi64>
    %range = linalg.generic {
        indexing_maps = [#vec], iterator_types = ["parallel"]}
        outs(%range_empty : tensor<3xi64>) {
    ^bb0(%out: i64):
      %idx = linalg.index 0 : index
      %cast = arith.index_cast %idx : index to i64
      linalg.yield %cast : i64
    } -> tensor<3xi64>
    %range_exp = tensor.expand_shape %range [[0, 1]]
        output_shape [1, 3] : tensor<3xi64> into tensor<1x3xi64>
    %indices_empty = tensor.empty() : tensor<1x3xi64>
    %indices = linalg.generic {
        indexing_maps = [#row, #col, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%base, %range_exp : tensor<1x1xi64>, tensor<1x3xi64>)
        outs(%indices_empty : tensor<1x3xi64>) {
    ^bb0(%lhs: i64, %rhs: i64, %out: i64):
      %sum = arith.addi %lhs, %rhs : i64
      linalg.yield %sum : i64
    } -> tensor<1x3xi64>
    %indices_flat = tensor.collapse_shape %indices [[0, 1]]
        : tensor<1x3xi64> into tensor<3xi64>

    %gate_empty = tensor.empty() : tensor<3xf32>
    %ih_reset = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%indices_flat : tensor<3xi64>)
        outs(%gate_empty : tensor<3xf32>) {
    ^bb0(%idx_in: i64, %out: f32):
      %is_neg = arith.cmpi slt, %idx_in, %zero_i64 : i64
      %wrapped = arith.addi %idx_in, %gate_width : i64
      %idx = arith.select %is_neg, %wrapped, %idx_in : i64
      %idx_cast = arith.index_cast %idx : i64 to index
      %value = tensor.extract %ih_flat[%idx_cast] : tensor<9xf32>
      linalg.yield %value : f32
    } -> tensor<3xf32>
    %ih_reset_exp = tensor.expand_shape %ih_reset [[0, 1]]
        output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %update_indices = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%indices_flat : tensor<3xi64>)
        outs(%range_empty : tensor<3xi64>) {
    ^bb0(%idx_in: i64, %out: i64):
      %idx = arith.addi %idx_in, %hidden_size : i64
      linalg.yield %idx : i64
    } -> tensor<3xi64>
    %ih_update = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%update_indices : tensor<3xi64>)
        outs(%gate_empty : tensor<3xf32>) {
    ^bb0(%idx_in: i64, %out: f32):
      %is_neg = arith.cmpi slt, %idx_in, %zero_i64 : i64
      %wrapped = arith.addi %idx_in, %gate_width : i64
      %idx = arith.select %is_neg, %wrapped, %idx_in : i64
      %idx_cast = arith.index_cast %idx : i64 to index
      %value = tensor.extract %ih_flat[%idx_cast] : tensor<9xf32>
      linalg.yield %value : f32
    } -> tensor<3xf32>
    %ih_update_exp = tensor.expand_shape %ih_update [[0, 1]]
        output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %new_indices = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%indices_flat : tensor<3xi64>)
        outs(%range_empty : tensor<3xi64>) {
    ^bb0(%idx_in: i64, %out: i64):
      %idx = arith.addi %idx_in, %new_gate_offset : i64
      linalg.yield %idx : i64
    } -> tensor<3xi64>
    %ih_new = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%new_indices : tensor<3xi64>)
        outs(%gate_empty : tensor<3xf32>) {
    ^bb0(%idx_in: i64, %out: f32):
      %is_neg = arith.cmpi slt, %idx_in, %zero_i64 : i64
      %wrapped = arith.addi %idx_in, %gate_width : i64
      %idx = arith.select %is_neg, %wrapped, %idx_in : i64
      %idx_cast = arith.index_cast %idx : i64 to index
      %value = tensor.extract %ih_flat[%idx_cast] : tensor<9xf32>
      linalg.yield %value : f32
    } -> tensor<3xf32>
    %ih_new_exp = tensor.expand_shape %ih_new [[0, 1]]
        output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>

    %w_hh_t_empty = tensor.empty() : tensor<3x9xf32>
    %w_hh_t = linalg.transpose
        ins(%w_hh : tensor<9x3xf32>)
        outs(%w_hh_t_empty : tensor<3x9xf32>) permutation = [1, 0]
    %hh = linalg.matmul
        ins(%arg1, %w_hh_t : tensor<1x3xf32>, tensor<3x9xf32>)
        outs(%pre_fill : tensor<1x9xf32>) -> tensor<1x9xf32>
    %hh_flat = tensor.collapse_shape %hh [[0, 1]]
        : tensor<1x9xf32> into tensor<9xf32>
    %hh_reset = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%indices_flat : tensor<3xi64>)
        outs(%gate_empty : tensor<3xf32>) {
    ^bb0(%idx_in: i64, %out: f32):
      %is_neg = arith.cmpi slt, %idx_in, %zero_i64 : i64
      %wrapped = arith.addi %idx_in, %gate_width : i64
      %idx = arith.select %is_neg, %wrapped, %idx_in : i64
      %idx_cast = arith.index_cast %idx : i64 to index
      %value = tensor.extract %hh_flat[%idx_cast] : tensor<9xf32>
      linalg.yield %value : f32
    } -> tensor<3xf32>
    %hh_reset_exp = tensor.expand_shape %hh_reset [[0, 1]]
        output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %hh_update = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%update_indices : tensor<3xi64>)
        outs(%gate_empty : tensor<3xf32>) {
    ^bb0(%idx_in: i64, %out: f32):
      %is_neg = arith.cmpi slt, %idx_in, %zero_i64 : i64
      %wrapped = arith.addi %idx_in, %gate_width : i64
      %idx = arith.select %is_neg, %wrapped, %idx_in : i64
      %idx_cast = arith.index_cast %idx : i64 to index
      %value = tensor.extract %hh_flat[%idx_cast] : tensor<9xf32>
      linalg.yield %value : f32
    } -> tensor<3xf32>
    %hh_update_exp = tensor.expand_shape %hh_update [[0, 1]]
        output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %hh_new = linalg.generic {
        indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
        ins(%new_indices : tensor<3xi64>)
        outs(%gate_empty : tensor<3xf32>) {
    ^bb0(%idx_in: i64, %out: f32):
      %is_neg = arith.cmpi slt, %idx_in, %zero_i64 : i64
      %wrapped = arith.addi %idx_in, %gate_width : i64
      %idx = arith.select %is_neg, %wrapped, %idx_in : i64
      %idx_cast = arith.index_cast %idx : i64 to index
      %value = tensor.extract %hh_flat[%idx_cast] : tensor<9xf32>
      linalg.yield %value : f32
    } -> tensor<3xf32>
    %hh_new_exp = tensor.expand_shape %hh_new [[0, 1]]
        output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>

    %out_empty = tensor.empty() : tensor<1x3xf32>
    %reset_pre = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hh_reset_exp, %ih_reset_exp : tensor<1x3xf32>, tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %reset = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%reset_pre : tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %neg = arith.negf %in : f32
      %exp = math.exp %neg : f32
      %denom = arith.addf %exp, %one : f32
      %sigmoid = arith.divf %one, %denom : f32
      linalg.yield %sigmoid : f32
    } -> tensor<1x3xf32>
    %update_pre = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hh_update_exp, %ih_update_exp : tensor<1x3xf32>, tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %update = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%update_pre : tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %neg = arith.negf %in : f32
      %exp = math.exp %neg : f32
      %denom = arith.addf %exp, %one : f32
      %sigmoid = arith.divf %one, %denom : f32
      linalg.yield %sigmoid : f32
    } -> tensor<1x3xf32>
    %reset_new = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hh_new_exp, %reset : tensor<1x3xf32>, tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %mul = arith.mulf %lhs, %rhs : f32
      linalg.yield %mul : f32
    } -> tensor<1x3xf32>
    %candidate_pre = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%ih_new_exp, %reset_new : tensor<1x3xf32>, tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>
    %candidate = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%candidate_pre : tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %tanh = math.tanh %in : f32
      linalg.yield %tanh : f32
    } -> tensor<1x3xf32>
    %hidden_minus_candidate = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg1, %candidate : tensor<1x3xf32>, tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%hidden: f32, %candidate_in: f32, %out: f32):
      %sub = arith.subf %hidden, %candidate_in : f32
      linalg.yield %sub : f32
    } -> tensor<1x3xf32>
    %update_delta = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%hidden_minus_candidate, %update : tensor<1x3xf32>, tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %mul = arith.mulf %lhs, %rhs : f32
      linalg.yield %mul : f32
    } -> tensor<1x3xf32>
    %result = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%update_delta, %candidate : tensor<1x3xf32>, tensor<1x3xf32>)
        outs(%out_empty : tensor<1x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<1x3xf32>

    return %result : tensor<1x3xf32>
  }

  // PIPE-LABEL: func.func @grucell_0
  // PIPE-SAME: (%arg0: tensor<1x4xf32>, %arg1: tensor<1x3xf32>)
  // PIPE-SAME: attributes {layer_type = "gru_cell"}
  // PIPE: sculptor.nn.gru_cell %arg0, %arg1
  // PIPE-SAME: has_bias = false
  // PIPE: return
}
