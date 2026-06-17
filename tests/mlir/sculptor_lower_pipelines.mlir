// RUN: not sculptor-mlir-opt %s --sculptor-lower-to-golem="array-rows=2 array-cols=3" --sculptor-lower-golem-to-task-graph="cores=2 arrays-per-core=2" 2>&1 | FileCheck %s

module {
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = call @tile_set_load(%arg0)
        : (tensor<1x4xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }

  func.func @tile_set_load(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %weight = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %result = sculptor.mvm %arg0, %weight
        : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<1x3xf32>
    return %result : tensor<1x3xf32>
  }
}

// CHECK: no task graph schedulers are registered

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D"
    }
  }
#-}
