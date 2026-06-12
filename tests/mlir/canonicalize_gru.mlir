#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#map4 = affine_map<(d0, d1) -> (0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
#map7 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map9 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
#map10 = affine_map<(d0, d1, d2) -> (d1, d2)>
// RUN: sculptor-mlir-opt %s --sculptor-canonicalize-layers | FileCheck %s --implicit-check-not=linalg. --implicit-check-not=tensor.empty

// CHECK-LABEL: func.func @forward
// CHECK: %[[W0_IH:.*]] = arith.constant {{.*}} : tensor<9x4xf32>
// CHECK: %[[B0_IH:.*]] = arith.constant {{.*}} : tensor<9xf32>
// CHECK: %[[W0_HH:.*]] = arith.constant {{.*}} : tensor<9x3xf32>
// CHECK: %[[B0_HH:.*]] = arith.constant {{.*}} : tensor<9xf32>
// CHECK: %[[W1_IH:.*]] = arith.constant {{.*}} : tensor<9x3xf32>
// CHECK: %[[B1_IH:.*]] = arith.constant {{.*}} : tensor<9xf32>
// CHECK: %[[W1_HH:.*]] = arith.constant {{.*}} : tensor<9x3xf32>
// CHECK: %[[B1_HH:.*]] = arith.constant {{.*}} : tensor<9xf32>
// CHECK: %[[OUT:.*]], %[[HN:.*]] = sculptor.nn.gru %arg0, %arg1, recurrent[%[[W0_IH]], %[[W0_HH]], %[[B0_IH]], %[[B0_HH]], %[[W1_IH]], %[[W1_HH]], %[[B1_IH]], %[[B1_HH]]]
// CHECK-SAME: batch_first = true
// CHECK-SAME: has_bias = true
// CHECK-SAME: hidden_size = 3 : i64
// CHECK-SAME: num_layers = 2 : i64
// CHECK: return %[[OUT]], %[[HN]]

module {
  func.func @forward(%arg0: tensor<1x3x4xf32>, %arg1: tensor<2x1x3xf32>) -> (tensor<1x3x3xf32>, tensor<2x1x3xf32>) {
    %cst = arith.constant dense_resource<torch_tensor_9_4_torch.float32> : tensor<9x4xf32>
    %c3_i64 = arith.constant 3 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant dense_resource<torch_tensor_9_torch.float32> : tensor<9xf32>
    %c9_i64 = arith.constant 9 : i64
    %cst_3 = arith.constant dense_resource<torch_tensor_9_3_torch.float32> : tensor<9x3xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_9_torch.float32_1> : tensor<9xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_9_3_torch.float32_1> : tensor<9x3xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_9_torch.float32_2> : tensor<9xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_9_3_torch.float32_2> : tensor<9x3xf32>
    %cst_8 = arith.constant dense_resource<torch_tensor_9_torch.float32_3> : tensor<9xf32>
    %c6_i64 = arith.constant 6 : i64
    %c12_i64 = arith.constant 12 : i64
    %c15_i64 = arith.constant 15 : i64
    %c18_i64 = arith.constant 18 : i64
    %c21_i64 = arith.constant 21 : i64
    %c24_i64 = arith.constant 24 : i64
    %extracted_slice = tensor.extract_slice %arg1[0, 0, 0] [1, 1, 3] [1, 1, 1] : tensor<2x1x3xf32> to tensor<1x1x3xf32>
    %extracted_slice_9 = tensor.extract_slice %arg1[1, 0, 0] [1, 1, 3] [1, 1, 1] : tensor<2x1x3xf32> to tensor<1x1x3xf32>
    %0 = tensor.empty() : tensor<3x1x4xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<1x3x4xf32>) outs(%0 : tensor<3x1x4xf32>) permutation = [1, 0, 2] 
    %collapsed = tensor.collapse_shape %transposed [[0], [1, 2]] : tensor<3x1x4xf32> into tensor<3x4xf32>
    %1 = tensor.empty() : tensor<4x9xf32>
    %transposed_10 = linalg.transpose ins(%cst : tensor<9x4xf32>) outs(%1 : tensor<4x9xf32>) permutation = [1, 0] 
    %2 = tensor.empty() : tensor<3x9xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<3x9xf32>) -> tensor<3x9xf32>
    %4 = linalg.matmul ins(%collapsed, %transposed_10 : tensor<3x4xf32>, tensor<4x9xf32>) outs(%3 : tensor<3x9xf32>) -> tensor<3x9xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_2 : tensor<3x9xf32>, tensor<9xf32>) outs(%2 : tensor<3x9xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<3x9xf32>
    %expanded = tensor.expand_shape %5 [[0], [1, 2]] output_shape [3, 1, 9] : tensor<3x9xf32> into tensor<3x1x9xf32>
    %extracted_slice_11 = tensor.extract_slice %expanded[0, 0, 0] [1, 1, 9] [1, 1, 1] : tensor<3x1x9xf32> to tensor<1x1x9xf32>
    %extracted_slice_12 = tensor.extract_slice %expanded[1, 0, 0] [1, 1, 9] [1, 1, 1] : tensor<3x1x9xf32> to tensor<1x1x9xf32>
    %extracted_slice_13 = tensor.extract_slice %expanded[2, 0, 0] [1, 1, 9] [1, 1, 1] : tensor<3x1x9xf32> to tensor<1x1x9xf32>
    %collapsed_14 = tensor.collapse_shape %extracted_slice_11 [[0, 1, 2]] : tensor<1x1x9xf32> into tensor<9xf32>
    %6 = tensor.empty() : tensor<1xi64>
    %7 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%6 : tensor<1xi64>) {
    ^bb0(%out: i64):
      linalg.yield %c0_i64 : i64
    } -> tensor<1xi64>
    %expanded_15 = tensor.expand_shape %7 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    %8 = tensor.empty() : tensor<1x1xi64>
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_15 : tensor<1x1xi64>) outs(%8 : tensor<1x1xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.muli %in, %c9_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<1x1xi64>
    %10 = tensor.empty() : tensor<3xi64>
    %11 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%10 : tensor<3xi64>) {
    ^bb0(%out: i64):
      %144 = linalg.index 0 : index
      %145 = arith.index_cast %144 : index to i64
      linalg.yield %145 : i64
    } -> tensor<3xi64>
    %expanded_16 = tensor.expand_shape %11 [[0, 1]] output_shape [1, 3] : tensor<3xi64> into tensor<1x3xi64>
    %12 = tensor.empty() : tensor<1x3xi64>
    %13 = linalg.generic {indexing_maps = [#map3, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %expanded_16 : tensor<1x1xi64>, tensor<1x3xi64>) outs(%12 : tensor<1x3xi64>) {
    ^bb0(%in: i64, %in_88: i64, %out: i64):
      %144 = arith.addi %in, %in_88 : i64
      linalg.yield %144 : i64
    } -> tensor<1x3xi64>
    %collapsed_17 = tensor.collapse_shape %13 [[0, 1]] : tensor<1x3xi64> into tensor<3xi64>
    %14 = tensor.empty() : tensor<3xf32>
    %15 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_17 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_14[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_18 = tensor.expand_shape %15 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %16 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_17 : tensor<3xi64>) outs(%10 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.addi %in, %c3_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<3xi64>
    %17 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%16 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_14[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_19 = tensor.expand_shape %17 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %18 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_17 : tensor<3xi64>) outs(%10 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.addi %in, %c6_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<3xi64>
    %19 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%18 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_14[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_20 = tensor.expand_shape %19 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %collapsed_21 = tensor.collapse_shape %extracted_slice [[0], [1, 2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %transposed_22 = linalg.transpose ins(%cst_3 : tensor<9x3xf32>) outs(%2 : tensor<3x9xf32>) permutation = [1, 0] 
    %20 = tensor.empty() : tensor<1x9xf32>
    %21 = linalg.fill ins(%cst_0 : f32) outs(%20 : tensor<1x9xf32>) -> tensor<1x9xf32>
    %22 = linalg.matmul ins(%collapsed_21, %transposed_22 : tensor<1x3xf32>, tensor<3x9xf32>) outs(%21 : tensor<1x9xf32>) -> tensor<1x9xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%22, %cst_4 : tensor<1x9xf32>, tensor<9xf32>) outs(%20 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x9xf32>
    %collapsed_23 = tensor.collapse_shape %23 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %expanded_24 = tensor.expand_shape %7 [[0, 1, 2]] output_shape [1, 1, 1] : tensor<1xi64> into tensor<1x1x1xi64>
    %24 = tensor.empty() : tensor<1x1x1xi64>
    %25 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_24 : tensor<1x1x1xi64>) outs(%24 : tensor<1x1x1xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.muli %in, %c9_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<1x1x1xi64>
    %26 = linalg.generic {indexing_maps = [#map6, #map7, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %25 : tensor<1x1x1xi64>, tensor<1x1x1xi64>) outs(%24 : tensor<1x1x1xi64>) {
    ^bb0(%in: i64, %in_88: i64, %out: i64):
      %144 = arith.addi %in, %in_88 : i64
      linalg.yield %144 : i64
    } -> tensor<1x1x1xi64>
    %expanded_25 = tensor.expand_shape %11 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xi64> into tensor<1x1x3xi64>
    %27 = tensor.empty() : tensor<1x1x3xi64>
    %28 = linalg.generic {indexing_maps = [#map8, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %expanded_25 : tensor<1x1x1xi64>, tensor<1x1x3xi64>) outs(%27 : tensor<1x1x3xi64>) {
    ^bb0(%in: i64, %in_88: i64, %out: i64):
      %144 = arith.addi %in, %in_88 : i64
      linalg.yield %144 : i64
    } -> tensor<1x1x3xi64>
    %collapsed_26 = tensor.collapse_shape %28 [[0, 1, 2]] : tensor<1x1x3xi64> into tensor<3xi64>
    %29 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_26 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_23[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_27 = tensor.expand_shape %29 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %30 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_26 : tensor<3xi64>) outs(%10 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.addi %in, %c3_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<3xi64>
    %31 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%30 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_23[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_28 = tensor.expand_shape %31 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %32 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_26 : tensor<3xi64>) outs(%10 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.addi %in, %c6_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<3xi64>
    %33 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%32 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_23[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_29 = tensor.expand_shape %33 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %34 = tensor.empty() : tensor<1x1x3xf32>
    %35 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_27, %expanded_18 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %36 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %37 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_28, %expanded_19 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %38 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %39 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_29, %36 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %40 = linalg.generic {indexing_maps = [#map10, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_20, %39 : tensor<1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %41 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = math.tanh %in : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %42 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %41 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.subf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %43 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %38 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %44 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %41 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %collapsed_30 = tensor.collapse_shape %extracted_slice_12 [[0, 1, 2]] : tensor<1x1x9xf32> into tensor<9xf32>
    %45 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_17 : tensor<3xi64>) outs(%10 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.addi %in, %c9_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<3xi64>
    %46 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%45 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_30[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_31 = tensor.expand_shape %46 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %47 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_17 : tensor<3xi64>) outs(%10 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.addi %in, %c12_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<3xi64>
    %48 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%47 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_30[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_32 = tensor.expand_shape %48 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %49 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_17 : tensor<3xi64>) outs(%10 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.addi %in, %c15_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<3xi64>
    %50 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%49 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_30[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_33 = tensor.expand_shape %50 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %collapsed_34 = tensor.collapse_shape %44 [[0], [1, 2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %51 = linalg.matmul ins(%collapsed_34, %transposed_22 : tensor<1x3xf32>, tensor<3x9xf32>) outs(%21 : tensor<1x9xf32>) -> tensor<1x9xf32>
    %52 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%51, %cst_4 : tensor<1x9xf32>, tensor<9xf32>) outs(%20 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x9xf32>
    %collapsed_35 = tensor.collapse_shape %52 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %53 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_26 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_35[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_36 = tensor.expand_shape %53 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %54 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%30 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_35[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_37 = tensor.expand_shape %54 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %55 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%32 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_35[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_38 = tensor.expand_shape %55 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %56 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_36, %expanded_31 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %57 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%56 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %58 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_37, %expanded_32 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %59 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%58 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %60 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_38, %57 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %61 = linalg.generic {indexing_maps = [#map10, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_33, %60 : tensor<1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %62 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = math.tanh %in : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %63 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %62 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.subf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %64 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63, %59 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %65 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%64, %62 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %collapsed_39 = tensor.collapse_shape %extracted_slice_13 [[0, 1, 2]] : tensor<1x1x9xf32> into tensor<9xf32>
    %66 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_17 : tensor<3xi64>) outs(%10 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.addi %in, %c18_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<3xi64>
    %67 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%66 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_39[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_40 = tensor.expand_shape %67 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %68 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_17 : tensor<3xi64>) outs(%10 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.addi %in, %c21_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<3xi64>
    %69 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%68 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_39[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_41 = tensor.expand_shape %69 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %70 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_17 : tensor<3xi64>) outs(%10 : tensor<3xi64>) {
    ^bb0(%in: i64, %out: i64):
      %144 = arith.addi %in, %c24_i64 : i64
      linalg.yield %144 : i64
    } -> tensor<3xi64>
    %71 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%70 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_39[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_42 = tensor.expand_shape %71 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %collapsed_43 = tensor.collapse_shape %65 [[0], [1, 2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %72 = linalg.matmul ins(%collapsed_43, %transposed_22 : tensor<1x3xf32>, tensor<3x9xf32>) outs(%21 : tensor<1x9xf32>) -> tensor<1x9xf32>
    %73 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%72, %cst_4 : tensor<1x9xf32>, tensor<9xf32>) outs(%20 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x9xf32>
    %collapsed_44 = tensor.collapse_shape %73 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %74 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_26 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_44[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_45 = tensor.expand_shape %74 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %75 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%30 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_44[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_46 = tensor.expand_shape %75 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %76 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%32 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_44[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_47 = tensor.expand_shape %76 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %77 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_45, %expanded_40 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %78 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %79 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_46, %expanded_41 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %80 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %81 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_47, %78 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %82 = linalg.generic {indexing_maps = [#map10, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_42, %81 : tensor<1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %83 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = math.tanh %in : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %84 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%65, %83 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.subf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %85 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %80 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %86 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %83 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %concat = tensor.concat dim(0) %44, %65, %86 : (tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>) -> tensor<3x1x3xf32>
    %collapsed_48 = tensor.collapse_shape %86 [[0, 1], [2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %collapsed_49 = tensor.collapse_shape %concat [[0], [1, 2]] : tensor<3x1x3xf32> into tensor<3x3xf32>
    %transposed_50 = linalg.transpose ins(%cst_5 : tensor<9x3xf32>) outs(%2 : tensor<3x9xf32>) permutation = [1, 0] 
    %87 = linalg.matmul ins(%collapsed_49, %transposed_50 : tensor<3x3xf32>, tensor<3x9xf32>) outs(%3 : tensor<3x9xf32>) -> tensor<3x9xf32>
    %88 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%87, %cst_6 : tensor<3x9xf32>, tensor<9xf32>) outs(%2 : tensor<3x9xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<3x9xf32>
    %expanded_51 = tensor.expand_shape %88 [[0], [1, 2]] output_shape [3, 1, 9] : tensor<3x9xf32> into tensor<3x1x9xf32>
    %extracted_slice_52 = tensor.extract_slice %expanded_51[0, 0, 0] [1, 1, 9] [1, 1, 1] : tensor<3x1x9xf32> to tensor<1x1x9xf32>
    %extracted_slice_53 = tensor.extract_slice %expanded_51[1, 0, 0] [1, 1, 9] [1, 1, 1] : tensor<3x1x9xf32> to tensor<1x1x9xf32>
    %extracted_slice_54 = tensor.extract_slice %expanded_51[2, 0, 0] [1, 1, 9] [1, 1, 1] : tensor<3x1x9xf32> to tensor<1x1x9xf32>
    %collapsed_55 = tensor.collapse_shape %extracted_slice_52 [[0, 1, 2]] : tensor<1x1x9xf32> into tensor<9xf32>
    %89 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_17 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_55[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_56 = tensor.expand_shape %89 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %90 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%16 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_55[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_57 = tensor.expand_shape %90 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %91 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%18 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_55[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_58 = tensor.expand_shape %91 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %collapsed_59 = tensor.collapse_shape %extracted_slice_9 [[0], [1, 2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %transposed_60 = linalg.transpose ins(%cst_7 : tensor<9x3xf32>) outs(%2 : tensor<3x9xf32>) permutation = [1, 0] 
    %92 = linalg.matmul ins(%collapsed_59, %transposed_60 : tensor<1x3xf32>, tensor<3x9xf32>) outs(%21 : tensor<1x9xf32>) -> tensor<1x9xf32>
    %93 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%92, %cst_8 : tensor<1x9xf32>, tensor<9xf32>) outs(%20 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x9xf32>
    %collapsed_61 = tensor.collapse_shape %93 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %94 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_26 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_61[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_62 = tensor.expand_shape %94 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %95 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%30 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_61[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_63 = tensor.expand_shape %95 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %96 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%32 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_61[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_64 = tensor.expand_shape %96 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %97 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_62, %expanded_56 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %98 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%97 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %99 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_63, %expanded_57 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %100 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%99 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %101 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_64, %98 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %102 = linalg.generic {indexing_maps = [#map10, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_58, %101 : tensor<1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %103 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = math.tanh %in : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %104 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_9, %103 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.subf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %105 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%104, %100 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %106 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%105, %103 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %collapsed_65 = tensor.collapse_shape %extracted_slice_53 [[0, 1, 2]] : tensor<1x1x9xf32> into tensor<9xf32>
    %107 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%45 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_65[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_66 = tensor.expand_shape %107 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %108 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%47 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_65[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_67 = tensor.expand_shape %108 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %109 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%49 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_65[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_68 = tensor.expand_shape %109 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %collapsed_69 = tensor.collapse_shape %106 [[0], [1, 2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %110 = linalg.matmul ins(%collapsed_69, %transposed_60 : tensor<1x3xf32>, tensor<3x9xf32>) outs(%21 : tensor<1x9xf32>) -> tensor<1x9xf32>
    %111 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%110, %cst_8 : tensor<1x9xf32>, tensor<9xf32>) outs(%20 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x9xf32>
    %collapsed_70 = tensor.collapse_shape %111 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %112 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_26 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_70[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_71 = tensor.expand_shape %112 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %113 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%30 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_70[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_72 = tensor.expand_shape %113 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %114 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%32 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_70[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_73 = tensor.expand_shape %114 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %115 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_71, %expanded_66 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %116 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %117 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_72, %expanded_67 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %118 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %119 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_73, %116 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %120 = linalg.generic {indexing_maps = [#map10, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_68, %119 : tensor<1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %121 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%120 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = math.tanh %in : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %122 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106, %121 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.subf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %123 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%122, %118 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %124 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123, %121 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %collapsed_74 = tensor.collapse_shape %extracted_slice_54 [[0, 1, 2]] : tensor<1x1x9xf32> into tensor<9xf32>
    %125 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%66 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_74[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_75 = tensor.expand_shape %125 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %126 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%68 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_74[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_76 = tensor.expand_shape %126 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %127 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%70 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_74[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_77 = tensor.expand_shape %127 [[0, 1]] output_shape [1, 3] : tensor<3xf32> into tensor<1x3xf32>
    %collapsed_78 = tensor.collapse_shape %124 [[0], [1, 2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %128 = linalg.matmul ins(%collapsed_78, %transposed_60 : tensor<1x3xf32>, tensor<3x9xf32>) outs(%21 : tensor<1x9xf32>) -> tensor<1x9xf32>
    %129 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%128, %cst_8 : tensor<1x9xf32>, tensor<9xf32>) outs(%20 : tensor<1x9xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x9xf32>
    %collapsed_79 = tensor.collapse_shape %129 [[0, 1]] : tensor<1x9xf32> into tensor<9xf32>
    %130 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%collapsed_26 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_79[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_80 = tensor.expand_shape %130 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %131 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%30 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_79[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_81 = tensor.expand_shape %131 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %132 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%32 : tensor<3xi64>) outs(%14 : tensor<3xf32>) {
    ^bb0(%in: i64, %out: f32):
      %144 = arith.cmpi slt, %in, %c0_i64 : i64
      %145 = arith.addi %in, %c9_i64 : i64
      %146 = arith.select %144, %145, %in : i64
      %147 = arith.index_cast %146 : i64 to index
      %extracted = tensor.extract %collapsed_79[%147] : tensor<9xf32>
      linalg.yield %extracted : f32
    } -> tensor<3xf32>
    %expanded_82 = tensor.expand_shape %132 [[0, 1, 2]] output_shape [1, 1, 3] : tensor<3xf32> into tensor<1x1x3xf32>
    %133 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_80, %expanded_75 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %134 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%133 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %135 = linalg.generic {indexing_maps = [#map5, #map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_81, %expanded_76 : tensor<1x1x3xf32>, tensor<1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %136 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%135 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = arith.negf %in : f32
      %145 = math.exp %144 : f32
      %146 = arith.addf %145, %cst_1 : f32
      %147 = arith.divf %cst_1, %146 : f32
      linalg.yield %147 : f32
    } -> tensor<1x1x3xf32>
    %137 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_82, %134 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %138 = linalg.generic {indexing_maps = [#map10, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_77, %137 : tensor<1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %139 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%138 : tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %144 = math.tanh %in : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %140 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%124, %139 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.subf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %141 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%140, %136 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.mulf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %142 = linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%141, %139 : tensor<1x1x3xf32>, tensor<1x1x3xf32>) outs(%34 : tensor<1x1x3xf32>) {
    ^bb0(%in: f32, %in_88: f32, %out: f32):
      %144 = arith.addf %in, %in_88 : f32
      linalg.yield %144 : f32
    } -> tensor<1x1x3xf32>
    %concat_83 = tensor.concat dim(0) %106, %124, %142 : (tensor<1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x1x3xf32>) -> tensor<3x1x3xf32>
    %collapsed_84 = tensor.collapse_shape %142 [[0, 1], [2]] : tensor<1x1x3xf32> into tensor<1x3xf32>
    %143 = tensor.empty() : tensor<1x3x3xf32>
    %transposed_85 = linalg.transpose ins(%concat_83 : tensor<3x1x3xf32>) outs(%143 : tensor<1x3x3xf32>) permutation = [1, 0, 2] 
    %concat_86 = tensor.concat dim(0) %collapsed_48, %collapsed_84 : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
    %expanded_87 = tensor.expand_shape %concat_86 [[0], [1, 2]] output_shape [2, 1, 3] : tensor<2x3xf32> into tensor<2x1x3xf32>
    return %transposed_85, %expanded_87 : tensor<1x3x3xf32>, tensor<2x1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_9_4_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53DB81E053E295C0F3E9A99193E0AD7233E7B142E3EEC51383E5C8F423ECDCC4C3E3D0A573EAE47613E1F856B3E8FC2753E0000803EB81E853E713D8A3E295C8F3EE17A943E9A99993E52B89E3E0AD7A33EC3F5A83E7B14AE3E3333B33EEC51B83E",
      torch_tensor_9_torch.float32: "0x040000000AD7A33C0AD7233D8FC2753D0AD7A33DCDCCCC3D8FC2F53D295C0F3E0AD7233EEC51383E",
      torch_tensor_9_3_torch.float32: "0x040000000AD7A33B0AD7233C8FC2753C0AD7A33CCDCCCC3C8FC2F53C295C0F3D0AD7233DEC51383DCDCC4C3DAE47613D8FC2753DB81E853D295C8F3D9A99993D0AD7A33D7B14AE3DEC51B83D5C8FC23DCDCCCC3D3D0AD73DAE47E13D1F85EB3D8FC2F53D0000003EB81E053E713D0A3E",
      torch_tensor_9_torch.float32_1: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83D",
      torch_tensor_9_3_torch.float32_1: "0x040000006F12033C6F12833CA69BC43C6F12033D0AD7233DA69B443D4260653D6F12833DBC74933D0AD7A33D5839B43DA69BC43DF4FDD43D4260E53D8FC2F53D6F12033E96430B3EBC74133EE3A51B3E0AD7233E31082C3E5839343E7F6A3C3EA69B443ECDCC4C3EF4FD543E1B2F5D3E",
      torch_tensor_9_torch.float32_2: "0x040000008988883C8988083DCDCC4C3D8988883DABAAAA3DCDCCCC3DEFEEEE3D8988083E9A99193E",
      torch_tensor_9_3_torch.float32_2: "0x04000000B4A2913BB4A2113C0E745A3CB4A2913C610BB63C0E74DA3CBBDCFE3CB4A2113D0AD7233D610B363DB73F483D0E745A3D64A86C3DBBDC7E3D8988883DB4A2913DDFBC9A3D0AD7A33D35F1AC3D610BB63D8C25BF3DB73FC83DE259D13D0E74DA3D398EE33D64A8EC3D8FC2F53D",
      torch_tensor_9_torch.float32_3: "0x0400000009F2143C09F2943C0E6BDF3C09F2143D8C2E3A3D0E6B5F3DC853823D09F2943D4A90A73D"
    }
  }
#-}
