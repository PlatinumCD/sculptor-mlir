// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=12 array-cols=12" | FileCheck %s --implicit-check-not=sculptor.nn.conv2d --implicit-check-not="sculptor.mvm %" --implicit-check-not=sculptor.mvm_sequence --implicit-check-not=memref. --implicit-check-not=bufferization.to_tensor
// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=1 array-cols=4" | FileCheck %s --check-prefix=MULTI --implicit-check-not=sculptor.nn.conv2d --implicit-check-not="sculptor.mvm %" --implicit-check-not=sculptor.mvm_sequence
// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=12 array-cols=12" --sculptor-materialize-tasks --sculptor-assemble-task-graph | FileCheck %s --check-prefix=GRAPH
// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=12 array-cols=12" --sculptor-materialize-tasks --sculptor-assemble-task-graph --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing="analog-mvm-latency-ns=100 analog-io-bits-per-cycle=256 digital-clock-ghz=1 digital-issue-width=2 digital-vector-bits-per-cycle=256 network-hop-latency-cycles=1 network-link-bits-per-cycle=32" | FileCheck %s --check-prefix=TIMING
// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=12 array-cols=12" --sculptor-materialize-tasks --sculptor-assemble-task-graph --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=snake" --sculptor-lower-scheduled-mvm-to-digital --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=DIGITAL --implicit-check-not="!sculptor.logical.array" --implicit-check-not="sculptor.array."

module {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32> {
    %0 = call @conv2d_bias(%arg0)
        : (tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32>
    return %0 : tensor<1x1x3x3xf32>
  }

  // CHECK-LABEL: func.func @conv2d_bias
  // CHECK: sculptor.task_region kind = "sculptor.matrix_setup" name = "conv2d_bias_matrix_tile_0_0"()
  // CHECK: %[[PATCHES:.*]] = sculptor.task_region kind = "digital.conv_patch" name = "conv2d_patch_sequence"(%arg0)
  // CHECK: tensor.empty() : tensor<9x9xf32>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: tensor.extract
  // CHECK: tensor.insert
  // CHECK: %[[EXEC:.*]] = sculptor.task_region kind = "sculptor.conv_tile_mvm" name = "conv2d_bias_mvm_0_0"(%[[PATCHES]],
  // CHECK: tensor.empty() : tensor<9x1xf32>
  // CHECK: scf.for
  // CHECK: tensor.extract_slice
  // CHECK: sculptor.array.load
  // CHECK: sculptor.array.execute
  // CHECK: sculptor.array.store
  // CHECK: tensor.insert_slice
  // CHECK: %[[RECOMBINE:.*]] = sculptor.task_region kind = "digital.tile_recombine" name = "conv2d_bias_tile_recombine"(%[[EXEC]])
  // CHECK: sculptor.task_region kind = "digital.bias_add" name = "conv2d_output_assembly"(%[[RECOMBINE]])
  // CHECK: tensor.empty() : tensor<1x1x3x3xf32>
  // CHECK: scf.for
  // CHECK: arith.addf
  // CHECK: tensor.insert
  // CHECK: return {{.*}} : tensor<1x1x3x3xf32>
  // MULTI-LABEL: func.func @conv2d_bias
  // MULTI-COUNT-3: sculptor.task_region kind = "sculptor.matrix_setup"
  // MULTI: %[[PATCHES:.*]] = sculptor.task_region kind = "digital.conv_patch" name = "conv2d_patch_sequence"
  // MULTI-COUNT-3: sculptor.task_region kind = "sculptor.conv_tile_mvm"
  // MULTI: sculptor.task_region kind = "digital.tile_recombine"
  // MULTI-COUNT-2: linalg.add
  // MULTI: sculptor.task_region kind = "digital.bias_add" name = "conv2d_output_assembly"
  // MULTI: return {{.*}} : tensor<1x1x3x3xf32>
  // GRAPH-LABEL: func.func private @generate_task_graph
  // GRAPH: sculptor.task.create {{.*}} task_kind = "sculptor.conv_tile_mvm"
  // GRAPH-SAME: sculptor.runtime.analog_execution_count = 9 : i64
  // GRAPH: sculptor.task.create {{.*}} task_kind = "digital.bias_add"
  // GRAPH-SAME: sculptor.runtime.digital_ops = 9 : i64
  // TIMING: sculptor.task.create {{.*}} task_kind = "sculptor.conv_tile_mvm"
  // TIMING-SAME: sculptor.runtime.analog_execution_count = 9 : i64
  // TIMING-SAME: sculptor.timing.analog_execute_latency_ns = 9.000000e+02 : f64
  // DIGITAL-LABEL: func.func private @task_conv2d_bias_mvm_0_0_2(%arg0: tensor<9x9xf32>) -> tensor<9x1xf32>
  // DIGITAL-SAME: sculptor.runtime.digital_ops = 2592 : i64
  // DIGITAL-SAME: sculptor.task_domain = "digital"
  // DIGITAL-SAME: sculptor.task_kind = "digital.matmul"
  // DIGITAL: tensor.extract_slice %arg0[0, 0] [9, 9] [1, 1]
  // DIGITAL: arith.constant dense<0.000000e+00> : tensor<9x12xf32>
  // DIGITAL: tensor.insert_slice {{.*}} into {{.*}}[0, 0] [9, 9] [1, 1]
  // DIGITAL: arith.constant {{.*}} : tensor<12x12xf32>
  // DIGITAL: linalg.matmul_transpose_b
  // DIGITAL-SAME: tensor<9x12xf32>, tensor<12x12xf32>
  // DIGITAL: tensor.extract_slice {{.*}}[0, 0] [9, 1] [1, 1]
  // DIGITAL: sculptor.task.create {{.*}} domain = "digital", task_kind = "digital.matmul"
  // DIGITAL-SAME: inputs[%{{[0-9]+}}], outputs[%{{[0-9]+}}], deps[%{{[0-9]+}}, %{{[0-9]+}}]
  // DIGITAL-SAME: sculptor.runtime.digital_ops = 2592 : i64
  // DIGITAL-SAME: sculptor.schedule.island_id = 0 : i64
  // DIGITAL-SAME: sculptor.timing.analog_execute_latency_ns = 0.000000e+00 : f64
  // DIGITAL-SAME: sculptor.timing.digital_ops = 2592 : i64
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
