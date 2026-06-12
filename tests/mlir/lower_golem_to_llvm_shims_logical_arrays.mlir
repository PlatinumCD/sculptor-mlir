// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=5 schedule=simple-budget" --sculptor-lower-golem-to-llvm-shims | FileCheck %s --implicit-check-not=sculptor.array.

module {
  // CHECK: func.func private @golem_analog_mvm_store(memref<1x1x2xf32>, i32)
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
  func.func private @task_matrix7() -> !sculptor.logical.array {
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
  func.func private @task_mvm7(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> {
    // CHECK: %[[LOAD_ID:.*]] = arith.constant 2 : i32
    // CHECK: call @golem_analog_mvm_load({{.*}}, %[[LOAD_ID]]) : (memref<1x2xf32>, i32) -> ()
    sculptor.array.load %arg0, %arg1 : tensor<1x2xf32>, !sculptor.logical.array
    // CHECK: %[[EXEC_ID:.*]] = arith.constant 2 : i32
    // CHECK: call @golem_analog_mvm_compute(%[[EXEC_ID]]) : (i32) -> ()
    %exec = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    // CHECK: %[[STORE_ID:.*]] = arith.constant 2 : i32
    // CHECK: call @golem_analog_mvm_store({{.*}}, %[[STORE_ID]]) : (memref<1x1x2xf32>, i32) -> ()
    %out = sculptor.array.store %exec : !sculptor.array.result -> tensor<1x2xf32>
    // CHECK: return
    return %out : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array0 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array2 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array3 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array4 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array5 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array6 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array7 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>

    %setup7 = sculptor.task.create %graph, @task_matrix7, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix_7", source_layer = "linear_7", source_task_ordinal = 0, inputs[], outputs[%array7], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm7 = sculptor.task.create %graph, @task_mvm7, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm_7", source_layer = "linear_7", source_task_ordinal = 1, inputs[%input, %array7], outputs[%output], deps[%setup7] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}
