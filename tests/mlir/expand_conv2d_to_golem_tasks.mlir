// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=12 array-cols=12" | FileCheck %s --implicit-check-not=sculptor.nn.conv2d --implicit-check-not="sculptor.mvm %" --implicit-check-not=scf.for --implicit-check-not=memref. --implicit-check-not=bufferization.to_tensor

module {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32> {
    %0 = call @conv2d_bias(%arg0)
        : (tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32>
    return %0 : tensor<1x1x3x3xf32>
  }

  // CHECK-LABEL: func.func @conv2d_bias
  // CHECK: sculptor.task_region kind = "sculptor.matrix_setup" name = "conv2d_bias_matrix_tile_0_0"()
  // CHECK: %[[PATCH0:.*]] = sculptor.task_region kind = "digital.conv_patch" name = "conv2d_oh_0_ow_0"(%arg0)
  // CHECK: tensor.empty() : tensor<1x9xf32>
  // CHECK: tensor.extract
  // CHECK: tensor.insert
  // CHECK: %[[VECTOR0:.*]] = sculptor.task_region kind = "digital.vector_tile" name = "conv2d_bias_vector_tile_0"(%[[PATCH0]])
  // CHECK: %[[EXEC0:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "conv2d_bias_mvm_0_0"(%[[VECTOR0]],
  // CHECK: %[[RECOMBINE0:.*]] = sculptor.task_region kind = "digital.tile_recombine" name = "conv2d_bias_tile_recombine"(%[[EXEC0]])
  // CHECK: sculptor.task_region kind = "digital.bias_add" name = "conv2d_oh_0_ow_0_bias_add"(%[[RECOMBINE0]])
  // CHECK: sculptor.task_region kind = "digital.conv_patch" name = "conv2d_oh_2_ow_2"(%arg0)
  // CHECK: sculptor.task_region kind = "digital.output_recombine" name = "conv2d_output_recombine"
  // CHECK: tensor.expand_shape
  // CHECK: tensor.concat dim(3)
  // CHECK: tensor.concat dim(2)
  // CHECK: return {{.*}} : tensor<1x1x3x3xf32>
  func.func @conv2d_bias(%arg0: tensor<1x1x5x5xf32>)
      -> tensor<1x1x3x3xf32>
      attributes {layer_type = "conv2d_w_bias"} {
    %w = arith.constant dense_resource<torch_tensor_1_1_3_3_torch.float32> : tensor<1x1x3x3xf32>
    %b = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %0 = sculptor.nn.conv2d %arg0, %w, %b {dilation = [1, 1], has_bias = true, padding = [0, 0], stride = [1, 1]}
        : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>) -> tensor<1x1x3x3xf32>
    return %0 : tensor<1x1x3x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_1_1_3_3_torch.float32: "0x040000000AD7A33DEC51B83D9A99193E9A99193FAE47613E0AD7A33DEC51B83D0000003F14AE473F"
    }
  }
#-}
