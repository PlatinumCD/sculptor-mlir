// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-canonicalize-layers | FileCheck %s --implicit-check-not=linalg. --implicit-check-not=tensor.empty

// CHECK-LABEL: func.func @forward
// CHECK: %[[NB_W0_IH:.*]] = arith.constant {{.*}} : tensor<3x4xf32>
// CHECK: %[[NB_W0_HH:.*]] = arith.constant {{.*}} : tensor<3x3xf32>
// CHECK: %[[NB_W1_IH:.*]] = arith.constant {{.*}} : tensor<3x3xf32>
// CHECK: %[[NB_W1_HH:.*]] = arith.constant {{.*}} : tensor<3x3xf32>
// CHECK: %[[NB_OUT:.*]], %[[NB_HN:.*]] = sculptor.nn.rnn %arg0, %arg1, recurrent[%[[NB_W0_IH]], %[[NB_W0_HH]], %[[NB_W1_IH]], %[[NB_W1_HH]]]
// CHECK-SAME: batch_first = true
// CHECK-SAME: has_bias = false
// CHECK-SAME: hidden_size = 3 : i64
// CHECK-SAME: num_layers = 2 : i64
// CHECK: return %[[NB_OUT]], %[[NB_HN]]
#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @forward(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>) -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>) {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_3_3_torch.float32> : tensor<3x3xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_3_3_torch.float32_1> : tensor<3x3xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_3_3_torch.float32_2> : tensor<3x3xf32>
    %extracted_slice = tensor.extract_slice %arg1[0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<2x2x3xf32> to tensor<1x2x3xf32>
    %extracted_slice_4 = tensor.extract_slice %arg1[1, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<2x2x3xf32> to tensor<1x2x3xf32>
    %0 = tensor.empty() : tensor<3x2x4xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<2x3x4xf32>) outs(%0 : tensor<3x2x4xf32>) permutation = [1, 0, 2] 
    %1 = tensor.empty() : tensor<4x3xf32>
    %transposed_5 = linalg.transpose ins(%cst : tensor<3x4xf32>) outs(%1 : tensor<4x3xf32>) permutation = [1, 0] 
    %2 = tensor.empty() : tensor<3x4x3xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_5 : tensor<4x3xf32>) outs(%2 : tensor<3x4x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x4x3xf32>
    %4 = tensor.empty() : tensor<3x2x3xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<3x2x3xf32>) -> tensor<3x2x3xf32>
    %6 = linalg.batch_matmul ins(%transposed, %3 : tensor<3x2x4xf32>, tensor<3x4x3xf32>) outs(%5 : tensor<3x2x3xf32>) -> tensor<3x2x3xf32>
    %extracted_slice_6 = tensor.extract_slice %6[0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %extracted_slice_7 = tensor.extract_slice %6[1, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %extracted_slice_8 = tensor.extract_slice %6[2, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %collapsed = tensor.collapse_shape %extracted_slice_6 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %collapsed_9 = tensor.collapse_shape %extracted_slice_7 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %collapsed_10 = tensor.collapse_shape %extracted_slice_8 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %7 = tensor.empty() : tensor<3x3xf32>
    %transposed_11 = linalg.transpose ins(%cst_1 : tensor<3x3xf32>) outs(%7 : tensor<3x3xf32>) permutation = [1, 0] 
    %collapsed_12 = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %8 = tensor.empty() : tensor<2x3xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %10 = linalg.matmul ins(%collapsed_12, %transposed_11 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%9 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %expanded = tensor.expand_shape %10 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %11 = tensor.empty() : tensor<1x2x3xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %collapsed : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_39: f32, %out: f32):
      %33 = arith.addf %in, %in_39 : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %13 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12 : tensor<1x2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %33 = math.tanh %in : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %collapsed_13 = tensor.collapse_shape %13 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %14 = linalg.matmul ins(%collapsed_13, %transposed_11 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%9 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %expanded_14 = tensor.expand_shape %14 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %15 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_14, %collapsed_9 : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_39: f32, %out: f32):
      %33 = arith.addf %in, %in_39 : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %16 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15 : tensor<1x2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %33 = math.tanh %in : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %collapsed_15 = tensor.collapse_shape %16 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %17 = linalg.matmul ins(%collapsed_15, %transposed_11 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%9 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %expanded_16 = tensor.expand_shape %17 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_16, %collapsed_10 : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_39: f32, %out: f32):
      %33 = arith.addf %in, %in_39 : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %19 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18 : tensor<1x2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %33 = math.tanh %in : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %concat = tensor.concat dim(0) %13, %16, %19 : (tensor<1x2x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<3x2x3xf32>
    %collapsed_17 = tensor.collapse_shape %19 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %transposed_18 = linalg.transpose ins(%cst_2 : tensor<3x3xf32>) outs(%7 : tensor<3x3xf32>) permutation = [1, 0] 
    %collapsed_19 = tensor.collapse_shape %concat [[0, 1], [2]] : tensor<3x2x3xf32> into tensor<6x3xf32>
    %20 = tensor.empty() : tensor<6x3xf32>
    %21 = linalg.fill ins(%cst_0 : f32) outs(%20 : tensor<6x3xf32>) -> tensor<6x3xf32>
    %22 = linalg.matmul ins(%collapsed_19, %transposed_18 : tensor<6x3xf32>, tensor<3x3xf32>) outs(%21 : tensor<6x3xf32>) -> tensor<6x3xf32>
    %expanded_20 = tensor.expand_shape %22 [[0, 1], [2]] output_shape [3, 2, 3] : tensor<6x3xf32> into tensor<3x2x3xf32>
    %extracted_slice_21 = tensor.extract_slice %expanded_20[0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %extracted_slice_22 = tensor.extract_slice %expanded_20[1, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %extracted_slice_23 = tensor.extract_slice %expanded_20[2, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %collapsed_24 = tensor.collapse_shape %extracted_slice_21 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %collapsed_25 = tensor.collapse_shape %extracted_slice_22 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %collapsed_26 = tensor.collapse_shape %extracted_slice_23 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %transposed_27 = linalg.transpose ins(%cst_3 : tensor<3x3xf32>) outs(%7 : tensor<3x3xf32>) permutation = [1, 0] 
    %collapsed_28 = tensor.collapse_shape %extracted_slice_4 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %23 = linalg.matmul ins(%collapsed_28, %transposed_27 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%9 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %expanded_29 = tensor.expand_shape %23 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %24 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_29, %collapsed_24 : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_39: f32, %out: f32):
      %33 = arith.addf %in, %in_39 : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %25 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24 : tensor<1x2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %33 = math.tanh %in : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %collapsed_30 = tensor.collapse_shape %25 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %26 = linalg.matmul ins(%collapsed_30, %transposed_27 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%9 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %expanded_31 = tensor.expand_shape %26 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %27 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_31, %collapsed_25 : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_39: f32, %out: f32):
      %33 = arith.addf %in, %in_39 : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %28 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27 : tensor<1x2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %33 = math.tanh %in : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %collapsed_32 = tensor.collapse_shape %28 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %29 = linalg.matmul ins(%collapsed_32, %transposed_27 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%9 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %expanded_33 = tensor.expand_shape %29 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %30 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_33, %collapsed_26 : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_39: f32, %out: f32):
      %33 = arith.addf %in, %in_39 : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %31 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30 : tensor<1x2x3xf32>) outs(%11 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %33 = math.tanh %in : f32
      linalg.yield %33 : f32
    } -> tensor<1x2x3xf32>
    %concat_34 = tensor.concat dim(0) %25, %28, %31 : (tensor<1x2x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<3x2x3xf32>
    %collapsed_35 = tensor.collapse_shape %31 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %32 = tensor.empty() : tensor<2x3x3xf32>
    %transposed_36 = linalg.transpose ins(%concat_34 : tensor<3x2x3xf32>) outs(%32 : tensor<2x3x3xf32>) permutation = [1, 0, 2] 
    %concat_37 = tensor.concat dim(0) %collapsed_17, %collapsed_35 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>
    %expanded_38 = tensor.expand_shape %concat_37 [[0, 1], [2]] output_shape [2, 2, 3] : tensor<4x3xf32> into tensor<2x2x3xf32>
    return %transposed_36, %expanded_38 : tensor<2x3x3xf32>, tensor<2x2x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D",
      torch_tensor_3_3_torch.float32: "0x040000000AD7A33B0AD7233C8FC2753C0AD7A33CCDCCCC3C8FC2F53C295C0F3D0AD7233DEC51383D",
      torch_tensor_3_3_torch.float32_1: "0x040000006F12033C6F12833CA69BC43C6F12033D0AD7233DA69B443D4260653D6F12833DBC74933D",
      torch_tensor_3_3_torch.float32_2: "0x04000000B4A2913BB4A2113C0E745A3CB4A2913C610BB63C0E74DA3CBBDCFE3CB4A2913D0AD7233D"
    }
  }
#-}

// -----

// CHECK-LABEL: func.func @forward
// CHECK: %[[B_W0_IH:.*]] = arith.constant {{.*}} : tensor<3x4xf32>
// CHECK: %[[B_B0_IH:.*]] = arith.constant {{.*}} : tensor<3xf32>
// CHECK: %[[B_W0_HH:.*]] = arith.constant {{.*}} : tensor<3x3xf32>
// CHECK: %[[B_B0_HH:.*]] = arith.constant {{.*}} : tensor<3xf32>
// CHECK: %[[B_W1_IH:.*]] = arith.constant {{.*}} : tensor<3x3xf32>
// CHECK: %[[B_B1_IH:.*]] = arith.constant {{.*}} : tensor<3xf32>
// CHECK: %[[B_W1_HH:.*]] = arith.constant {{.*}} : tensor<3x3xf32>
// CHECK: %[[B_B1_HH:.*]] = arith.constant {{.*}} : tensor<3xf32>
// CHECK: %[[B_OUT:.*]], %[[B_HN:.*]] = sculptor.nn.rnn %arg0, %arg1, recurrent[%[[B_W0_IH]], %[[B_W0_HH]], %[[B_B0_IH]], %[[B_B0_HH]], %[[B_W1_IH]], %[[B_W1_HH]], %[[B_B1_IH]], %[[B_B1_HH]]]
// CHECK-SAME: batch_first = true
// CHECK-SAME: has_bias = true
// CHECK-SAME: hidden_size = 3 : i64
// CHECK-SAME: num_layers = 2 : i64
// CHECK: return %[[B_OUT]], %[[B_HN]]
#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @forward(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x3xf32>) -> (tensor<2x3x3xf32>, tensor<2x2x3xf32>) {
    %cst = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_3_3_torch.float32> : tensor<3x3xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_3_torch.float32_1> : tensor<3xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_3_3_torch.float32_1> : tensor<3x3xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_3_torch.float32_2> : tensor<3xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_3_3_torch.float32_2> : tensor<3x3xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_3_torch.float32_3> : tensor<3xf32>
    %extracted_slice = tensor.extract_slice %arg1[0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<2x2x3xf32> to tensor<1x2x3xf32>
    %extracted_slice_8 = tensor.extract_slice %arg1[1, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<2x2x3xf32> to tensor<1x2x3xf32>
    %0 = tensor.empty() : tensor<3x2x4xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<2x3x4xf32>) outs(%0 : tensor<3x2x4xf32>) permutation = [1, 0, 2] 
    %1 = tensor.empty() : tensor<4x3xf32>
    %transposed_9 = linalg.transpose ins(%cst : tensor<3x4xf32>) outs(%1 : tensor<4x3xf32>) permutation = [1, 0] 
    %2 = tensor.empty() : tensor<3x4x3xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_9 : tensor<4x3xf32>) outs(%2 : tensor<3x4x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x4x3xf32>
    %4 = tensor.empty() : tensor<3x2x3xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<3x2x3xf32>) -> tensor<3x2x3xf32>
    %6 = linalg.batch_matmul ins(%transposed, %3 : tensor<3x2x4xf32>, tensor<3x4x3xf32>) outs(%5 : tensor<3x2x3xf32>) -> tensor<3x2x3xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %cst_1 : tensor<3x2x3xf32>, tensor<3xf32>) outs(%4 : tensor<3x2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<3x2x3xf32>
    %extracted_slice_10 = tensor.extract_slice %7[0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %extracted_slice_11 = tensor.extract_slice %7[1, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %extracted_slice_12 = tensor.extract_slice %7[2, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %collapsed = tensor.collapse_shape %extracted_slice_10 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %collapsed_13 = tensor.collapse_shape %extracted_slice_11 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %collapsed_14 = tensor.collapse_shape %extracted_slice_12 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %collapsed_15 = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %8 = tensor.empty() : tensor<3x3xf32>
    %transposed_16 = linalg.transpose ins(%cst_2 : tensor<3x3xf32>) outs(%8 : tensor<3x3xf32>) permutation = [1, 0] 
    %9 = tensor.empty() : tensor<2x3xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %11 = linalg.matmul ins(%collapsed_15, %transposed_16 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%10 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %12 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%11, %cst_3 : tensor<2x3xf32>, tensor<3xf32>) outs(%9 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<2x3xf32>
    %expanded = tensor.expand_shape %12 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %13 = tensor.empty() : tensor<1x2x3xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %collapsed : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %15 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14 : tensor<1x2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = math.tanh %in : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %collapsed_17 = tensor.collapse_shape %15 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %16 = linalg.matmul ins(%collapsed_17, %transposed_16 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%10 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %17 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%16, %cst_3 : tensor<2x3xf32>, tensor<3xf32>) outs(%9 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<2x3xf32>
    %expanded_18 = tensor.expand_shape %17 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_18, %collapsed_13 : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %19 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18 : tensor<1x2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = math.tanh %in : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %collapsed_19 = tensor.collapse_shape %19 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %20 = linalg.matmul ins(%collapsed_19, %transposed_16 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%10 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %21 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%20, %cst_3 : tensor<2x3xf32>, tensor<3xf32>) outs(%9 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<2x3xf32>
    %expanded_20 = tensor.expand_shape %21 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %22 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_20, %collapsed_14 : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %23 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22 : tensor<1x2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = math.tanh %in : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %concat = tensor.concat dim(0) %15, %19, %23 : (tensor<1x2x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<3x2x3xf32>
    %collapsed_21 = tensor.collapse_shape %23 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %collapsed_22 = tensor.collapse_shape %concat [[0, 1], [2]] : tensor<3x2x3xf32> into tensor<6x3xf32>
    %transposed_23 = linalg.transpose ins(%cst_4 : tensor<3x3xf32>) outs(%8 : tensor<3x3xf32>) permutation = [1, 0] 
    %24 = tensor.empty() : tensor<6x3xf32>
    %25 = linalg.fill ins(%cst_0 : f32) outs(%24 : tensor<6x3xf32>) -> tensor<6x3xf32>
    %26 = linalg.matmul ins(%collapsed_22, %transposed_23 : tensor<6x3xf32>, tensor<3x3xf32>) outs(%25 : tensor<6x3xf32>) -> tensor<6x3xf32>
    %27 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%26, %cst_5 : tensor<6x3xf32>, tensor<3xf32>) outs(%24 : tensor<6x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<6x3xf32>
    %expanded_24 = tensor.expand_shape %27 [[0, 1], [2]] output_shape [3, 2, 3] : tensor<6x3xf32> into tensor<3x2x3xf32>
    %extracted_slice_25 = tensor.extract_slice %expanded_24[0, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %extracted_slice_26 = tensor.extract_slice %expanded_24[1, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %extracted_slice_27 = tensor.extract_slice %expanded_24[2, 0, 0] [1, 2, 3] [1, 1, 1] : tensor<3x2x3xf32> to tensor<1x2x3xf32>
    %collapsed_28 = tensor.collapse_shape %extracted_slice_25 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %collapsed_29 = tensor.collapse_shape %extracted_slice_26 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %collapsed_30 = tensor.collapse_shape %extracted_slice_27 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %collapsed_31 = tensor.collapse_shape %extracted_slice_8 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %transposed_32 = linalg.transpose ins(%cst_6 : tensor<3x3xf32>) outs(%8 : tensor<3x3xf32>) permutation = [1, 0] 
    %28 = linalg.matmul ins(%collapsed_31, %transposed_32 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%10 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %29 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%28, %cst_7 : tensor<2x3xf32>, tensor<3xf32>) outs(%9 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<2x3xf32>
    %expanded_33 = tensor.expand_shape %29 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %30 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_33, %collapsed_28 : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %31 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30 : tensor<1x2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = math.tanh %in : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %collapsed_34 = tensor.collapse_shape %31 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %32 = linalg.matmul ins(%collapsed_34, %transposed_32 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%10 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %33 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%32, %cst_7 : tensor<2x3xf32>, tensor<3xf32>) outs(%9 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<2x3xf32>
    %expanded_35 = tensor.expand_shape %33 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %34 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_35, %collapsed_29 : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %35 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34 : tensor<1x2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = math.tanh %in : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %collapsed_36 = tensor.collapse_shape %35 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %36 = linalg.matmul ins(%collapsed_36, %transposed_32 : tensor<2x3xf32>, tensor<3x3xf32>) outs(%10 : tensor<2x3xf32>) -> tensor<2x3xf32>
    %37 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%36, %cst_7 : tensor<2x3xf32>, tensor<3xf32>) outs(%9 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<2x3xf32>
    %expanded_37 = tensor.expand_shape %37 [[0, 1], [2]] output_shape [1, 2, 3] : tensor<2x3xf32> into tensor<1x2x3xf32>
    %38 = linalg.generic {indexing_maps = [#map1, #map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_37, %collapsed_30 : tensor<1x2x3xf32>, tensor<2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %in_43: f32, %out: f32):
      %41 = arith.addf %in, %in_43 : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %39 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38 : tensor<1x2x3xf32>) outs(%13 : tensor<1x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %41 = math.tanh %in : f32
      linalg.yield %41 : f32
    } -> tensor<1x2x3xf32>
    %concat_38 = tensor.concat dim(0) %31, %35, %39 : (tensor<1x2x3xf32>, tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<3x2x3xf32>
    %collapsed_39 = tensor.collapse_shape %39 [[0, 1], [2]] : tensor<1x2x3xf32> into tensor<2x3xf32>
    %40 = tensor.empty() : tensor<2x3x3xf32>
    %transposed_40 = linalg.transpose ins(%concat_38 : tensor<3x2x3xf32>) outs(%40 : tensor<2x3x3xf32>) permutation = [1, 0, 2] 
    %concat_41 = tensor.concat dim(0) %collapsed_21, %collapsed_39 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>
    %expanded_42 = tensor.expand_shape %concat_41 [[0, 1], [2]] output_shape [2, 2, 3] : tensor<4x3xf32> into tensor<2x2x3xf32>
    return %transposed_40, %expanded_42 : tensor<2x3x3xf32>, tensor<2x2x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D",
      torch_tensor_3_torch.float32: "0x040000000AD7A33C0AD7233D8FC2753D",
      torch_tensor_3_3_torch.float32: "0x040000000AD7A33B0AD7233C8FC2753C0AD7A33CCDCCCC3C8FC2F53C295C0F3D0AD7233DEC51383D",
      torch_tensor_3_torch.float32_1: "0x040000000AD7233C0AD7A33C8FC2F53C",
      torch_tensor_3_3_torch.float32_1: "0x040000006F12033C6F12833CA69BC43C6F12033D0AD7233DA69B443D4260653D6F12833DBC74933D",
      torch_tensor_3_torch.float32_2: "0x040000008988883C8988083DCDCC4C3D",
      torch_tensor_3_3_torch.float32_2: "0x04000000B4A2913BB4A2113C0E745A3CB4A2913C610BB63C0E74DA3CBBDCFE3CB4A2113D0AD7233D",
      torch_tensor_3_torch.float32_3: "0x0400000009F2143C09F2943C0E6BDF3C"
    }
  }
#-}
