// RUN: sculptor-mlir-opt %s --sculptor-assemble-task-graph | FileCheck %s --implicit-check-not=sculptor.assembly --implicit-check-not=sculptor.runtime.byte_size --implicit-check-not=sculptor.runtime.input_slots --implicit-check-not=sculptor.runtime.output_slots --implicit-check-not=sculptor.runtime.resource_count --implicit-check-not=sculptor.runtime.slot --implicit-check-not=sculptor.runtime.task_index --implicit-check-not=sculptor.runtime.temp_ --implicit-check-not=sculptor.runtime.workspace_size

module {
  func.func @forward(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %array = call @task_matrix() : () -> !sculptor.logical.array
    %mvm = call @task_mvm(%arg0, %array) : (tensor<1x2xf32>, !sculptor.logical.array) -> tensor<1x2xf32>
    %result = call @task_finish(%mvm) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
  }

  func.func private @task_matrix() -> !sculptor.logical.array attributes {
    sculptor.source_layer = "layer",
    sculptor.source_task_ordinal = 0 : i64,
    sculptor.task_domain = "analog",
    sculptor.task_kind = "sculptor.matrix_setup",
    sculptor.task_name = "matrix"
  } {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> attributes {
    sculptor.source_layer = "layer",
    sculptor.source_task_ordinal = 1 : i64,
    sculptor.task_domain = "analog",
    sculptor.task_kind = "sculptor.mvm",
    sculptor.task_name = "mvm",
    weight_dependencies = ["task_matrix"]
  } {
    sculptor.array.load %arg0, %arg1 : tensor<1x2xf32>, !sculptor.logical.array
    %execution = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %result = sculptor.array.store %execution : !sculptor.array.result -> tensor<1x2xf32>
    return %result : tensor<1x2xf32>
  }

  func.func private @task_finish(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> attributes {
    sculptor.source_layer = "layer",
    sculptor.source_task_ordinal = 2 : i64,
    sculptor.task_domain = "digital",
    sculptor.task_kind = "digital.compute",
    sculptor.task_name = "finish"
  } {
    return %arg0 : tensor<1x2xf32>
  }
}

// CHECK-LABEL: func.func private @generate_task_graph()
// CHECK: %[[GRAPH:.*]] = sculptor.task_graph.create
// CHECK-NEXT: %[[INPUT:.*]] = sculptor.task_graph.input %[[GRAPH]]
// CHECK-NEXT: %[[OUTPUT:.*]] = sculptor.task_graph.output %[[GRAPH]]
// CHECK-NEXT: %[[ARRAY:.*]] = sculptor.task_graph.intermediate %[[GRAPH]]
// CHECK-NEXT: %[[MVM_OUT:.*]] = sculptor.task_graph.intermediate %[[GRAPH]]
// CHECK-NEXT: %[[PERSISTENT:.*]] = sculptor.task_graph.persistent %[[GRAPH]]
// CHECK-NEXT: %[[SETUP:.*]] = sculptor.task.create %[[GRAPH]], @task_matrix
// CHECK-SAME: inputs[]
// CHECK-SAME: outputs[%[[ARRAY]]]
// CHECK-SAME: deps[]
// CHECK-NEXT: %[[MVM:.*]] = sculptor.task.create %[[GRAPH]], @task_mvm
// CHECK-SAME: inputs[%[[INPUT]], %[[ARRAY]], %[[PERSISTENT]]]
// CHECK-SAME: outputs[%[[MVM_OUT]]]
// CHECK-SAME: deps[%[[SETUP]]]
// CHECK-NEXT: %[[FINISH:.*]] = sculptor.task.create %[[GRAPH]], @task_finish
// CHECK-SAME: inputs[%[[MVM_OUT]]]
// CHECK-SAME: outputs[%[[OUTPUT]]]
// CHECK-SAME: deps[%[[MVM]]]
// CHECK-NEXT: return %[[GRAPH]] : !sculptor.task_graph
