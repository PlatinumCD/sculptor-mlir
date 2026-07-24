// RUN: sculptor-mlir-opt %s --sculptor-expand-mvm-to-golem="array-rows=2 array-cols=3" | FileCheck %s --implicit-check-not=sculptor.mvm --implicit-check-not="dense_resource<cache_weight> : tensor<3x4xf32>"

module {
  func.func @repeated_weight(%vector: tensor<1x4xf32>)
      -> (tensor<1x3xf32>, tensor<1x3xf32>) {
    %weight = arith.constant dense_resource<cache_weight> : tensor<3x4xf32>
    %first = sculptor.mvm %vector, %weight
        : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<1x3xf32>
    %second = sculptor.mvm %vector, %weight
        : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<1x3xf32>
    return %first, %second : tensor<1x3xf32>, tensor<1x3xf32>
  }
}

// One 2x2 tile grid is shared by both MVMs.
// CHECK-COUNT-4: sculptor.task_region kind = "sculptor.matrix_setup"
// CHECK-COUNT-2: sculptor.task_region kind = "digital.vector_tile"
// CHECK-COUNT-4: sculptor.task_region kind = "sculptor.mvm"
// CHECK: sculptor.task_region kind = "digital.tile_recombine"
// CHECK-COUNT-2: sculptor.task_region kind = "digital.vector_tile"
// CHECK-COUNT-4: sculptor.task_region kind = "sculptor.mvm"
// CHECK: sculptor.task_region kind = "digital.tile_recombine"

{-#
  dialect_resources: {
    builtin: {
      cache_weight: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D"
    }
  }
#-}
