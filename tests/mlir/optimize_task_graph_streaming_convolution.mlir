// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" --sculptor-materialize-tasks --sculptor-assemble-task-graph --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=1 arrays-per-core=3 topology=mesh mesh-rows=1 mesh-cols=1 schedule=snake" --sculptor-optimize-task-graph="patterns=streaming-convolution require-change=true" --sculptor-fuse-task-graph --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=STREAM --implicit-check-not="tensor<9x9xf32>" --implicit-check-not="task_kind = \"digital.conv_patch\"" --implicit-check-not="task_kind = \"sculptor.conv_tile_mvm\"" --implicit-check-not="task_kind = \"digital.tile_recombine\"" --implicit-check-not="task_kind = \"digital.bias_add\""
// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" --sculptor-materialize-tasks --sculptor-assemble-task-graph --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=1 arrays-per-core=3 topology=mesh mesh-rows=1 mesh-cols=1 schedule=snake" --sculptor-optimize-task-graph --sculptor-fuse-task-graph --sculptor-analyze-task-graph-timing --sculptor-lower-golem-to-llvm-shims --sculptor-finalize-task-graph-resources | FileCheck %s --check-prefix=SHIM --implicit-check-not="!sculptor.logical.array" --implicit-check-not="sculptor.array." --implicit-check-not="memref<1x1x1xf32>"
// RUN: sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" --sculptor-materialize-tasks --sculptor-assemble-task-graph --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=1 arrays-per-core=3 topology=mesh mesh-rows=1 mesh-cols=1 schedule=snake" --sculptor-optimize-task-graph --sculptor-fuse-task-graph --sculptor-analyze-task-graph-timing --sculptor-lower-golem-to-llvm-shims --sculptor-partition-task-graph-by-core --sculptor-extract-core-module="core-id=0" --sculptor-finalize-task-graph-resources -o %t.core.mlir
// RUN: sculptor-mlir-opt %t.core.mlir --canonicalize --cse --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation-pipeline --convert-bufferization-to-memref --optimize-allocation-liveness -o %t.deallocated.mlir
// RUN: sculptor-mlir-opt %t.deallocated.mlir | FileCheck %s --check-prefix=DEALLOC
// RUN: sculptor-mlir-opt %t.core.mlir --canonicalize --cse --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --convert-bufferization-to-memref --buffer-hoisting --buffer-loop-hoisting --buffer-deallocation-pipeline --optimize-allocation-liveness | FileCheck %s --check-prefix=HOIST
// RUN: sculptor-mlir-opt %t.deallocated.mlir --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --convert-math-to-llvm --expand-strided-metadata --lower-affine --convert-arith-to-llvm --convert-index-to-llvm --convert-cf-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o %t.llvm.mlir
// RUN: sculptor-mlir-opt %t.llvm.mlir --sculptor-emit-golem-tile-abi --sculptor-finalize-golem-intrinsics | mlir-translate --mlir-to-llvmir > /dev/null
// RUN: not sculptor-mlir-opt %s --sculptor-convert-layers --sculptor-expand-mvm-to-golem="array-rows=4 array-cols=4" --sculptor-materialize-tasks --sculptor-assemble-task-graph --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=snake" --sculptor-optimize-task-graph="require-change=true" 2>&1 | FileCheck %s --check-prefix=DISTRIBUTED

module {
  func.func @forward(%arg0: tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32> {
    %0 = call @conv2d_bias(%arg0)
        : (tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32>
    return %0 : tensor<1x1x3x3xf32>
  }

  func.func @conv2d_bias(%arg0: tensor<1x1x5x5xf32>)
      -> tensor<1x1x3x3xf32>
      attributes {layer_type = "conv2d_w_bias"} {
    %w = arith.constant dense_resource<streaming_conv_weight> : tensor<1x1x3x3xf32>
    %b = arith.constant dense<1.000000e+00> : tensor<1xf32>
    %0 = sculptor.nn.conv2d %arg0, %w, %b {dilation = [1, 1], has_bias = true, padding = [0, 0], stride = [1, 1]}
        : (tensor<1x1x5x5xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>) -> tensor<1x1x3x3xf32>
    return %0 : tensor<1x1x3x3xf32>
  }
}

// STREAM: sculptor.schedule.task_count = 4 : i64
// STREAM-SAME: sculptor.schedule.total_digital_ops = 108 : i64
// STREAM-COUNT-3: task_kind = "sculptor.matrix_setup"
// STREAM: sculptor.task.create
// STREAM-SAME: task_kind = "mixed.streaming_conv_mvm"
// STREAM-SAME: inputs[{{[^]]+}}]
// STREAM-SAME: deps[{{[^]]+}}]
// STREAM-SAME: sculptor.runtime.analog_execution_counts = [9, 9, 9]
// STREAM-SAME: sculptor.runtime.analog_load_bytes = 432 : i64
// STREAM-SAME: sculptor.runtime.analog_load_bytes_per_array = [144, 144, 144]
// STREAM-SAME: sculptor.runtime.analog_store_bytes = 432 : i64
// STREAM-SAME: sculptor.runtime.analog_store_bytes_per_array = [144, 144, 144]
// STREAM-SAME: sculptor.runtime.array_bindings = [{input_index = 1 : i64, local_array_id = 0 : i64, physical_array_id = 0 : i64}, {input_index = 2 : i64, local_array_id = 1 : i64, physical_array_id = 1 : i64}, {input_index = 3 : i64, local_array_id = 2 : i64, physical_array_id = 2 : i64}]
// The aggregate phase counters expose all 27 executions, while the 936 ns
// pipeline span models the three independent arrays overlapping. A serial sum
// of the phase totals would be 2728 ns.
// STREAM-SAME: sculptor.timing.analog_execute_latency_ns = 2.700000e+03 : f64
// STREAM-SAME: sculptor.timing.analog_load_latency_ns = 1.400000e+01 : f64
// STREAM-SAME: sculptor.timing.analog_pipeline_latency_ns = 9.360000e+02 : f64
// STREAM-SAME: sculptor.timing.analog_store_latency_ns = 1.400000e+01 : f64
// STREAM-SAME: sculptor.timing.intrinsic_latency_ns = 1.647000e+03 : f64
// STREAM-LABEL: func.func private @task_conv2d_bias_streaming_conv_mvm(
// STREAM: sculptor.array.load
// STREAM-SAME: sculptor.runtime.local_array_id = 0 : i64
// STREAM: sculptor.array.execute
// STREAM-SAME: sculptor.runtime.local_array_id = 0 : i64
// STREAM: sculptor.array.load
// STREAM-SAME: sculptor.runtime.local_array_id = 1 : i64
// STREAM: sculptor.array.execute
// STREAM-SAME: sculptor.runtime.local_array_id = 1 : i64
// STREAM: sculptor.array.load
// STREAM-SAME: sculptor.runtime.local_array_id = 2 : i64
// STREAM: sculptor.array.execute
// STREAM-SAME: sculptor.runtime.local_array_id = 2 : i64
// STREAM: sculptor.array.store
// STREAM-SAME: sculptor.runtime.local_array_id = 0 : i64
// STREAM-SAME: sculptor.tile_physical_shape = [4, 4]
// STREAM-SAME: sculptor.tile_valid_shape = [1, 4]
// STREAM: sculptor.array.store
// STREAM-SAME: sculptor.runtime.local_array_id = 1 : i64
// STREAM-SAME: sculptor.tile_physical_shape = [4, 4]
// STREAM-SAME: sculptor.tile_valid_shape = [1, 4]
// STREAM: sculptor.array.store
// STREAM-SAME: sculptor.runtime.local_array_id = 2 : i64
// STREAM-SAME: sculptor.tile_physical_shape = [4, 4]
// STREAM-SAME: sculptor.tile_valid_shape = [1, 1]

// SHIM: sculptor.task.create {{.*}} task_kind = "mixed.streaming_conv_mvm"
// SHIM-SAME: inputs[{{[^]]+}}]
// SHIM-SAME: outputs[{{[^]]+}}]
// SHIM-SAME: sculptor.runtime.input_slots = [0]
// SHIM-SAME: sculptor.runtime.output_slots = [1]
// SHIM-LABEL: func.func private @task_conv2d_bias_streaming_conv_mvm(%arg0: tensor<1x1x5x5xf32>)
// SHIM: %[[ID0:.*]] = arith.constant 0 : i32
// SHIM: call @golem_analog_mvm_load({{.*}}, %[[ID0]])
// SHIM: %[[ID0C:.*]] = arith.constant 0 : i32
// SHIM: call @golem_analog_mvm_compute(%[[ID0C]])
// SHIM: %[[ID1:.*]] = arith.constant 1 : i32
// SHIM: call @golem_analog_mvm_load({{.*}}, %[[ID1]])
// SHIM: %[[ID1C:.*]] = arith.constant 1 : i32
// SHIM: call @golem_analog_mvm_compute(%[[ID1C]])
// SHIM: %[[ID2:.*]] = arith.constant 2 : i32
// SHIM: call @golem_analog_mvm_load({{.*}}, %[[ID2]])
// SHIM: %[[ID2C:.*]] = arith.constant 2 : i32
// SHIM: call @golem_analog_mvm_compute(%[[ID2C]])
// SHIM: memref.alloc() {alignment = 64 : i64} : memref<1x1x4xf32>
// SHIM: call @golem_analog_mvm_store
// SHIM: memref.alloc() {alignment = 64 : i64} : memref<1x1x4xf32>
// SHIM: call @golem_analog_mvm_store
// SHIM: memref.alloc() {alignment = 64 : i64} : memref<1x1x4xf32>
// SHIM: call @golem_analog_mvm_store

// DEALLOC-LABEL: func.func private @task_conv2d_bias_matrix_tile_0_0_0()
// DEALLOC: %[[SETUP:.*]] = memref.alloc() {{.*}} : memref<4x4xf32>
// DEALLOC: call @golem_analog_mvm_set(%[[SETUP]],
// DEALLOC-NEXT: memref.dealloc %[[SETUP]] : memref<4x4xf32>
// DEALLOC-LABEL: func.func private @task_conv2d_bias_streaming_conv_mvm(
// DEALLOC: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<1x1x3x3xf32>
// DEALLOC: scf.for
// DEALLOC: %[[PATCH:.*]] = memref.alloc() {{.*}} : memref<1x4xf32>
// DEALLOC: func.call @golem_analog_mvm_load(%[[PATCH]],
// DEALLOC-NEXT: memref.dealloc %[[PATCH]] : memref<1x4xf32>
// DEALLOC: %[[LOGICAL:.*]] = memref.alloc() : memref<1x1xf32>
// DEALLOC: %[[SCRATCH:.*]] = memref.alloc() {{.*}} : memref<1x1x4xf32>
// DEALLOC: func.call @golem_analog_mvm_store
// DEALLOC: memref.dealloc %[[SCRATCH]] : memref<1x1x4xf32>
// DEALLOC: memref.dealloc %[[LOGICAL]] : memref<1x1xf32>
// DEALLOC-NOT: memref.dealloc %[[OUTPUT]] : memref<1x1x3x3xf32>
// DEALLOC: return %[[OUTPUT]] : memref<1x1x3x3xf32>

// HOIST-LABEL: func.func private @task_conv2d_bias_streaming_conv_mvm(
// HOIST: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<1x1x3x3xf32>
// HOIST: %[[PATCH:.*]] = memref.alloc() {{.*}} : memref<1x4xf32>
// HOIST: %[[SCRATCH:.*]] = memref.alloc() {{.*}} : memref<1x1x4xf32>
// HOIST: scf.for
// HOIST: func.call @golem_analog_mvm_load(%[[PATCH]],
// HOIST: func.call @golem_analog_mvm_store
// HOIST-NOT: memref.dealloc %[[PATCH]] : memref<1x4xf32>
// HOIST: }
// HOIST: memref.dealloc %[[SCRATCH]] : memref<1x1x4xf32>
// HOIST: memref.dealloc %[[PATCH]] : memref<1x4xf32>
// HOIST-NOT: memref.dealloc %[[OUTPUT]] : memref<1x1x3x3xf32>
// HOIST: return %[[OUTPUT]] : memref<1x1x3x3xf32>

// DISTRIBUTED: error: no selected task-graph optimization pattern applied

{-#
  dialect_resources: {
    builtin: {
      streaming_conv_weight: "0x040000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041"
    }
  }
#-}
