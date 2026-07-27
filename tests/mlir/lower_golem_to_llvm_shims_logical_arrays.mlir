// RUN: sculptor-mlir-opt %s --sculptor-lower-golem-to-llvm-shims | FileCheck %s --check-prefixes=CHECK,GRAPH --implicit-check-not=sculptor.array. --implicit-check-not=!sculptor.logical.array --implicit-check-not=memref<1x1x1xf32>
// RUN: sculptor-mlir-opt %s --sculptor-lower-golem-to-llvm-shims --sculptor-finalize-task-graph-resources | FileCheck %s --check-prefix=FINAL --implicit-check-not=!sculptor.logical.array
// RUN: sculptor-mlir-opt %s --sculptor-finalize-task-graph-resources --sculptor-lower-golem-to-llvm-shims | FileCheck %s --check-prefix=FINAL --implicit-check-not=!sculptor.logical.array

module {
  // CHECK: func.func private @golem_analog_mvm_store(memref<?x?x?xf32>, i32)
  // CHECK: func.func private @golem_analog_mvm_compute(i32)
  // CHECK: func.func private @golem_analog_mvm_load(memref<1x2xf32>, i32)
  // CHECK: func.func private @golem_analog_mvm_set(memref<2x2xf32>, i32)

  // CHECK-LABEL: func.func @forward(
  // CHECK-SAME: %[[ARG0:.*]]: tensor<1x2xf32>
  // CHECK-SAME: ) -> tensor<1x2xf32>
  func.func @forward(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %array = call @task_matrix7() : () -> !sculptor.logical.array
    // CHECK: call @task_matrix7() : () -> ()
    // CHECK: %[[OUT:.*]] = call @task_mvm7(%[[ARG0]]) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %out = call @task_mvm7(%arg0, %array) : (tensor<1x2xf32>, !sculptor.logical.array) -> tensor<1x2xf32>
    // CHECK: return %[[OUT]] : tensor<1x2xf32>
    return %out : tensor<1x2xf32>
  }

  // CHECK-LABEL: func.func private @task_matrix7()
  // CHECK-SAME: sculptor.runtime.core_id = 1 : i64
  // CHECK-SAME: sculptor.runtime.local_array_id = 2 : i64
  // CHECK-SAME: sculptor.runtime.physical_array_id = 7 : i64
  func.func private @task_matrix7() -> !sculptor.logical.array attributes {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.local_array_id = 2 : i64, sculptor.runtime.physical_array_id = 7 : i64} {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    // CHECK: %[[SET_ID:.*]] = arith.constant 2 : i32
    // CHECK: call @golem_analog_mvm_set({{.*}}, %[[SET_ID]]) : (memref<2x2xf32>, i32) -> ()
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    // CHECK: return
    return %array : !sculptor.logical.array
  }

  // CHECK-LABEL: func.func private @task_mvm7(
  // CHECK-SAME: %[[VECTOR:.*]]: tensor<1x2xf32>
  // CHECK-SAME: ) -> tensor<1x2xf32>
  // CHECK-SAME: sculptor.runtime.core_id = 1 : i64
  // CHECK-SAME: sculptor.runtime.local_array_id = 2 : i64
  // CHECK-SAME: sculptor.runtime.physical_array_id = 7 : i64
  func.func private @task_mvm7(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> attributes {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.local_array_id = 2 : i64, sculptor.runtime.physical_array_id = 7 : i64} {
    // CHECK: %[[LOAD_ID:.*]] = arith.constant 2 : i32
    // CHECK: call @golem_analog_mvm_load({{.*}}, %[[LOAD_ID]]) : (memref<1x2xf32>, i32) -> ()
    sculptor.array.load %arg0, %arg1 : tensor<1x2xf32>, !sculptor.logical.array
    // CHECK: %[[EXEC_ID:.*]] = arith.constant 2 : i32
    // CHECK: call @golem_analog_mvm_compute(%[[EXEC_ID]]) : (i32) -> ()
    %exec = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    // CHECK: %[[STORE_ID:.*]] = arith.constant 2 : i32
    // CHECK: %[[STORE_SCRATCH:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x1x2xf32>
    // CHECK: %[[STORE_DYNAMIC:.*]] = memref.cast %[[STORE_SCRATCH]] : memref<1x1x2xf32> to memref<?x?x?xf32>
    // CHECK: call @golem_analog_mvm_store(%[[STORE_DYNAMIC]], %[[STORE_ID]]) : (memref<?x?x?xf32>, i32) -> ()
    %out = sculptor.array.store %exec {sculptor.tile_physical_shape = [2, 2], sculptor.tile_valid_shape = [2, 2]} : !sculptor.array.result -> tensor<1x2xf32>
    // CHECK: return
    return %out : tensor<1x2xf32>
  }

  // A second output width must reuse the same dynamically shaped store shim.
  // CHECK-LABEL: func.func private @task_mvm_partial(
  // CHECK: %[[PARTIAL_RESULT:.*]] = memref.alloc() : memref<1x1xf32>
  // CHECK: %[[PARTIAL_SCRATCH:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x1x4xf32>
  // CHECK: %[[PARTIAL_DYNAMIC:.*]] = memref.cast %[[PARTIAL_SCRATCH]] : memref<1x1x4xf32> to memref<?x?x?xf32>
  // CHECK: call @golem_analog_mvm_store(%[[PARTIAL_DYNAMIC]], {{.*}}) : (memref<?x?x?xf32>, i32) -> ()
  // CHECK: %[[PARTIAL_ZERO:.*]] = arith.constant 0 : index
  // CHECK: %[[PARTIAL_STEP:.*]] = arith.constant 1 : index
  // CHECK: %[[PARTIAL_VALID_ROWS:.*]] = arith.constant 1 : index
  // CHECK: scf.for {{.*}} = %[[PARTIAL_ZERO]] to %[[PARTIAL_VALID_ROWS]] step %[[PARTIAL_STEP]]
  // CHECK: memref.store {{.*}}, %[[PARTIAL_RESULT]]
  // CHECK: bufferization.to_tensor %[[PARTIAL_RESULT]]
  func.func private @task_mvm_partial(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x1xf32> attributes {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.local_array_id = 2 : i64, sculptor.runtime.physical_array_id = 7 : i64} {
    sculptor.array.load %arg0, %arg1 : tensor<1x2xf32>, !sculptor.logical.array
    %exec = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %out = sculptor.array.store %exec {sculptor.tile_physical_shape = [4, 4], sculptor.tile_valid_shape = [1, 3]} : !sculptor.array.result -> tensor<1x1xf32>
    return %out : tensor<1x1xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {sculptor.schedule.logical_array_to_analog_array = [7], sculptor.schedule.num_logical_arrays = 1 : i64} {
    // GRAPH-LABEL: func.func private @generate_task_graph()
    // GRAPH-SAME: sculptor.schedule.logical_array_to_analog_array = [7]
    // GRAPH-SAME: sculptor.schedule.num_logical_arrays = 1 : i64
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    // GRAPH: %[[INPUT:.*]] = sculptor.task_graph.input
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    // GRAPH: %[[OUTPUT:.*]] = sculptor.task_graph.output
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array7 = sculptor.task_graph.intermediate %graph {sculptor.schedule.logical_array_index = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>

    // GRAPH: %[[SETUP:.*]] = sculptor.task.create
    // GRAPH-SAME: @task_matrix7
    // GRAPH-SAME: inputs[], outputs[], deps[]
    // GRAPH-SAME: sculptor.runtime.local_array_id = 2 : i64
    // GRAPH-SAME: sculptor.runtime.physical_array_id = 7 : i64
    %setup7 = sculptor.task.create %graph, @task_matrix7, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix_7", source_layer = "linear_7", source_task_ordinal = 0, inputs[], outputs[%array7], deps[] {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.local_array_id = 2 : i64, sculptor.runtime.physical_array_id = 7 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // GRAPH: sculptor.task.create
    // GRAPH-SAME: @task_mvm7
    // GRAPH-SAME: inputs[%[[INPUT]]], outputs[%[[OUTPUT]]], deps[%[[SETUP]]]
    // GRAPH-SAME: sculptor.runtime.local_array_id = 2 : i64
    // GRAPH-SAME: sculptor.runtime.physical_array_id = 7 : i64
    %mvm7 = sculptor.task.create %graph, @task_mvm7, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm_7", source_layer = "linear_7", source_task_ordinal = 1, inputs[%input, %array7], outputs[%output], deps[] {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.local_array_id = 2 : i64, sculptor.runtime.physical_array_id = 7 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }

  // FINAL-LABEL: func.func private @generate_task_graph()
  // FINAL-SAME: sculptor.runtime.input_slots = [0]
  // FINAL-SAME: sculptor.runtime.output_slots = [1]
  // FINAL-SAME: sculptor.runtime.resource_count = 2 : i64
  // FINAL-SAME: sculptor.runtime.temp_count = 0 : i64
  // FINAL-SAME: sculptor.schedule.logical_array_to_analog_array = [7]
  // FINAL-SAME: sculptor.schedule.num_logical_arrays = 1 : i64
  // FINAL: task_name = "matrix_7"
  // FINAL-SAME: sculptor.runtime.input_slots = []
  // FINAL-SAME: sculptor.runtime.output_slots = []
  // FINAL: task_name = "mvm_7"
  // FINAL-SAME: sculptor.runtime.input_slots = [0]
  // FINAL-SAME: sculptor.runtime.output_slots = [1]
}
