// RUN: sculptor-mlir-opt %s --sculptor-expand-mvm-to-golem="array-rows=2 array-cols=3" | FileCheck %s --implicit-check-not=sculptor.mvm

module {
  // CHECK-LABEL: func.func @tile_set_load
  // CHECK-NOT: arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
  // CHECK: %[[A00:.*]] = sculptor.task_region kind = "sculptor.matrix_setup" name = "tile_set_load_matrix_tile_0_0"() {
  // CHECK: %[[T00:.*]] = arith.constant dense_resource<torch_tensor_3_4_torch.float32__tile_0_0> : tensor<2x3xf32>
  // CHECK: %[[SET00:.*]] = sculptor.array.set %[[T00]] : tensor<2x3xf32> -> !sculptor.logical.array
  // CHECK: sculptor.yield %[[SET00]] : !sculptor.logical.array
  // CHECK: } {sculptor.source_resource = "torch_tensor_3_4_torch.float32", sculptor.tile = [0, 0], sculptor.tile_grid = [2, 2], sculptor.tile_physical_shape = [2, 3], sculptor.tile_valid_shape = [2, 3]} : () -> !sculptor.logical.array
  // CHECK: %[[A01:.*]] = sculptor.task_region kind = "sculptor.matrix_setup" name = "tile_set_load_matrix_tile_0_1"() {
  // CHECK: arith.constant dense_resource<torch_tensor_3_4_torch.float32__tile_0_1> : tensor<2x3xf32>
  // CHECK: sculptor.array.set {{.*}} : tensor<2x3xf32> -> !sculptor.logical.array
  // CHECK: sculptor.yield {{.*}} : !sculptor.logical.array
  // CHECK: } {sculptor.source_resource = "torch_tensor_3_4_torch.float32", sculptor.tile = [0, 1], sculptor.tile_grid = [2, 2], sculptor.tile_physical_shape = [2, 3], sculptor.tile_valid_shape = [2, 1]} : () -> !sculptor.logical.array
  // CHECK: %[[A10:.*]] = sculptor.task_region kind = "sculptor.matrix_setup" name = "tile_set_load_matrix_tile_1_0"() {
  // CHECK: arith.constant dense_resource<torch_tensor_3_4_torch.float32__tile_1_0> : tensor<2x3xf32>
  // CHECK: sculptor.array.set {{.*}} : tensor<2x3xf32> -> !sculptor.logical.array
  // CHECK: sculptor.yield {{.*}} : !sculptor.logical.array
  // CHECK: } {sculptor.source_resource = "torch_tensor_3_4_torch.float32", sculptor.tile = [1, 0], sculptor.tile_grid = [2, 2], sculptor.tile_physical_shape = [2, 3], sculptor.tile_valid_shape = [1, 3]} : () -> !sculptor.logical.array
  // CHECK: %[[A11:.*]] = sculptor.task_region kind = "sculptor.matrix_setup" name = "tile_set_load_matrix_tile_1_1"() {
  // CHECK: arith.constant dense_resource<torch_tensor_3_4_torch.float32__tile_1_1> : tensor<2x3xf32>
  // CHECK: sculptor.array.set {{.*}} : tensor<2x3xf32> -> !sculptor.logical.array
  // CHECK: sculptor.yield {{.*}} : !sculptor.logical.array
  // CHECK: } {sculptor.source_resource = "torch_tensor_3_4_torch.float32", sculptor.tile = [1, 1], sculptor.tile_grid = [2, 2], sculptor.tile_physical_shape = [2, 3], sculptor.tile_valid_shape = [1, 1]} : () -> !sculptor.logical.array
  // CHECK: %[[V0:.*]] = sculptor.task_region kind = "digital.vector_tile" name = "tile_set_load_vector_tile_0"(%arg0) {
  // CHECK: ^bb0(%[[PREP0_ARG:.*]]: tensor<1x4xf32>):
  // CHECK: %[[V0_SLICE:.*]] = tensor.extract_slice %[[PREP0_ARG]][0, 0] [1, 3] [1, 1] : tensor<1x4xf32> to tensor<1x3xf32>
  // CHECK: sculptor.yield %[[V0_SLICE]] : tensor<1x3xf32>
  // CHECK: } {sculptor.vector_tile = 0 : i64, sculptor.vector_tile_grid = 2 : i64, sculptor.vector_tile_physical_cols = 3 : i64, sculptor.vector_tile_valid_cols = 3 : i64} : (tensor<1x4xf32>) -> tensor<1x3xf32>
  // CHECK: %[[V1:.*]] = sculptor.task_region kind = "digital.vector_tile" name = "tile_set_load_vector_tile_1"(%arg0) {
  // CHECK: ^bb0(%[[PREP1_ARG:.*]]: tensor<1x4xf32>):
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<1x3xf32>
  // CHECK: %[[V1SRC:.*]] = tensor.extract_slice %[[PREP1_ARG]][0, 3] [1, 1] [1, 1] : tensor<1x4xf32> to tensor<1x1xf32>
  // CHECK: %[[V1_PADDED:.*]] = tensor.insert_slice %[[V1SRC]] into %[[ZERO]][0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x3xf32>
  // CHECK: sculptor.yield %[[V1_PADDED]] : tensor<1x3xf32>
  // CHECK: } {sculptor.vector_tile = 1 : i64, sculptor.vector_tile_grid = 2 : i64, sculptor.vector_tile_physical_cols = 3 : i64, sculptor.vector_tile_valid_cols = 1 : i64} : (tensor<1x4xf32>) -> tensor<1x3xf32>
  // CHECK: %[[X00:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "tile_set_load_mvm_0_0"(%[[V0]], %[[A00]]) {
  // CHECK: ^bb0(%[[XV00:.*]]: tensor<1x3xf32>, %[[XA00:.*]]: !sculptor.logical.array):
  // CHECK: sculptor.array.load %[[XV00]], %[[XA00]] : tensor<1x3xf32>, !sculptor.logical.array
  // CHECK: %[[E00:.*]] = sculptor.array.execute %[[XA00]] : !sculptor.logical.array -> !sculptor.array.result
  // CHECK: %[[S00:.*]] = sculptor.array.store %[[E00]] : !sculptor.array.result -> tensor<1x2xf32>
  // CHECK: sculptor.yield %[[S00]] : tensor<1x2xf32>
  // CHECK: } {sculptor.source_resource = "torch_tensor_3_4_torch.float32", sculptor.tile = [0, 0], sculptor.tile_grid = [2, 2], sculptor.tile_physical_shape = [2, 3], sculptor.tile_valid_shape = [2, 3], sculptor.vector_tile = 0 : i64, sculptor.vector_tile_grid = 2 : i64, sculptor.vector_tile_physical_cols = 3 : i64, sculptor.vector_tile_valid_cols = 3 : i64} : (tensor<1x3xf32>, !sculptor.logical.array) -> tensor<1x2xf32>
  // CHECK: %[[X10:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "tile_set_load_mvm_1_0"(%[[V0]], %[[A10]]) {
  // CHECK: sculptor.array.load {{.*}} : tensor<1x3xf32>, !sculptor.logical.array
  // CHECK: sculptor.array.store {{.*}} : !sculptor.array.result -> tensor<1x1xf32>
  // CHECK: sculptor.yield {{.*}} : tensor<1x1xf32>
  // CHECK: %[[X01:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "tile_set_load_mvm_0_1"(%[[V1]], %[[A01]]) {
  // CHECK: sculptor.array.load {{.*}} : tensor<1x3xf32>, !sculptor.logical.array
  // CHECK: sculptor.yield {{.*}} : tensor<1x2xf32>
  // CHECK: %[[X11:.*]] = sculptor.task_region kind = "sculptor.mvm" name = "tile_set_load_mvm_1_1"(%[[V1]], %[[A11]]) {
  // CHECK: sculptor.array.load {{.*}} : tensor<1x3xf32>, !sculptor.logical.array
  // CHECK: sculptor.array.store {{.*}} : !sculptor.array.result -> tensor<1x1xf32>
  // CHECK: sculptor.yield {{.*}} : tensor<1x1xf32>
  // CHECK: %[[RECOMBINE:.*]] = sculptor.task_region kind = "digital.tile_recombine" name = "tile_set_load_tile_recombine"(%[[X00]], %[[X01]], %[[X10]], %[[X11]]) {
  // CHECK: ^bb0(%[[R00:.*]]: tensor<1x2xf32>, %[[R01:.*]]: tensor<1x2xf32>, %[[R10:.*]]: tensor<1x1xf32>, %[[R11:.*]]: tensor<1x1xf32>):
  // CHECK: %[[INIT0:.*]] = tensor.empty() : tensor<1x2xf32>
  // CHECK: %[[ROW0:.*]] = linalg.add ins(%[[R00]], %[[R01]] : tensor<1x2xf32>, tensor<1x2xf32>) outs(%[[INIT0]] : tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK: %[[INIT1:.*]] = tensor.empty() : tensor<1x1xf32>
  // CHECK: %[[ROW1:.*]] = linalg.add ins(%[[R10]], %[[R11]] : tensor<1x1xf32>, tensor<1x1xf32>) outs(%[[INIT1]] : tensor<1x1xf32>) -> tensor<1x1xf32>
  // CHECK: %[[RESULT:.*]] = tensor.concat dim(1) %[[ROW0]], %[[ROW1]] : (tensor<1x2xf32>, tensor<1x1xf32>) -> tensor<1x3xf32>
  // CHECK-NOT: tensor.extract_slice
  // CHECK: sculptor.yield %[[RESULT]] : tensor<1x3xf32>
  // CHECK: return %[[RECOMBINE]] : tensor<1x3xf32>
  func.func @tile_set_load(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %weight = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %result = sculptor.mvm %arg0, %weight
        : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<1x3xf32>
    return %result : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D"
    }
  }
#-}
