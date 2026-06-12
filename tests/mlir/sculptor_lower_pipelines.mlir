// RUN: sculptor-mlir-opt %s --pass-pipeline='builtin.module(sculptor-lower-to-golem{array-rows=2 array-cols=3},sculptor-lower-golem-to-task-graph{cores=2 arrays-per-core=2 schedule=simple-budget})' | FileCheck %s --implicit-check-not=sculptor.array.

module {
  // CHECK: func.func private @golem_analog_mvm_store
  // CHECK: func.func private @golem_analog_mvm_compute
  // CHECK: func.func private @golem_analog_mvm_load
  // CHECK: func.func private @golem_analog_mvm_set

  // CHECK-LABEL: func.func @forward
  // CHECK: call @task_tile_set_load_matrix_tile_0_0_0
  // CHECK: call @task_tile_set_load_tile_recombine_10
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = call @tile_set_load(%arg0)
        : (tensor<1x4xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func private @task_tile_set_load_mvm_0_0_6
  // CHECK-SAME: sculptor.runtime.core_id = 0 : i64
  // CHECK: call @golem_analog_mvm_load
  // CHECK: call @golem_analog_mvm_compute
  // CHECK: call @golem_analog_mvm_store

  // CHECK-LABEL: func.func private @generate_task_graph()
  // CHECK-SAME: sculptor.schedule.num_cores = 2 : i64
  // CHECK-SAME: sculptor.schedule.num_logical_arrays = 4 : i64
  // CHECK: sculptor.task_graph.create
  // CHECK: sculptor.task.create
  // CHECK: task_kind = "sculptor.mvm"
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
