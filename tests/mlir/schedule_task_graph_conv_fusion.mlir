// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=random" | FileCheck %s

module {
  func.func @forward(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %dead = call @task_dead(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    return %dead : tensor<1x2xf32>
  }

  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_conv_patch(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @task_vector_tile(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x2xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x2xf32>
    return %stored : tensor<1x2xf32>
  }

  func.func private @task_tile_recombine(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @task_bias_add(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @task_dead(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> attributes {sculptor.task_kind = "digital.dead"} {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [0],
    sculptor.runtime.output_slots = [1],
    sculptor.runtime.resource_count = 7 : i64,
    sculptor.runtime.temp_base_slot = 2 : i64,
    sculptor.runtime.temp_count = 5 : i64,
    sculptor.runtime.temp_offsets = [0, 0, 8, 16, 24],
    sculptor.runtime.workspace_size = 32 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 1 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 0 : i64, sculptor.runtime.slot = 2 : i64, sculptor.runtime.temp_index = 0 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %patch = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 3 : i64, sculptor.runtime.temp_index = 1 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %vector = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 4 : i64, sculptor.runtime.temp_index = 2 : i64, sculptor.runtime.temp_offset = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %mvm_out = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 5 : i64, sculptor.runtime.temp_index = 3 : i64, sculptor.runtime.temp_offset = 16 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %recombined = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 6 : i64, sculptor.runtime.temp_index = 4 : i64, sculptor.runtime.temp_offset = 24 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>

    %setup = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "conv2dwbias_0_matrix_tile_0_0", source_layer = "conv2dwbias_0", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.runtime.input_slots = [], sculptor.runtime.output_slots = [2], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %conv = sculptor.task.create %graph, @task_conv_patch, domain = "digital", task_kind = "digital.conv_patch", task_name = "conv2d_oh_0_ow_0", source_layer = "conv2dwbias_0", source_task_ordinal = 1, inputs[%input], outputs[%patch], deps[] {sculptor.runtime.input_slots = [0], sculptor.runtime.output_slots = [3], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    %vector_task = sculptor.task.create %graph, @task_vector_tile, domain = "digital", task_kind = "digital.vector_tile", task_name = "conv2dwbias_0_vector_tile_0", source_layer = "conv2dwbias_0", source_task_ordinal = 2, inputs[%patch], outputs[%vector], deps[%conv] {sculptor.runtime.input_slots = [3], sculptor.runtime.output_slots = [4], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 2 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "conv2dwbias_0_mvm_0_0", source_layer = "conv2dwbias_0", source_task_ordinal = 3, inputs[%vector, %array], outputs[%mvm_out], deps[%setup, %vector_task] {sculptor.runtime.input_slots = [4, 2], sculptor.runtime.output_slots = [5], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 3 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %tile = sculptor.task.create %graph, @task_tile_recombine, domain = "digital", task_kind = "digital.tile_recombine", task_name = "conv2dwbias_0_tile_recombine", source_layer = "conv2dwbias_0", source_task_ordinal = 4, inputs[%mvm_out], outputs[%recombined], deps[%mvm] {sculptor.runtime.input_slots = [5], sculptor.runtime.output_slots = [6], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 4 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %bias = sculptor.task.create %graph, @task_bias_add, domain = "digital", task_kind = "digital.bias_add", task_name = "conv2d_oh_0_ow_0_bias_add", source_layer = "conv2dwbias_0", source_task_ordinal = 5, inputs[%recombined], outputs[%output], deps[%tile] {sculptor.runtime.input_slots = [6], sculptor.runtime.output_slots = [1], sculptor.runtime.result_indices = [0], sculptor.runtime.task_index = 5 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @task_conv2dwbias_0_conv_tile_mvm_oh_0_ow_0_1
// CHECK-SAME: (%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32>
// CHECK-SAME: sculptor.task_kind = "sculptor.conv_tile_mvm"
// CHECK: sculptor.array.load
// CHECK: sculptor.array.execute
// CHECK: sculptor.array.store
// CHECK: return {{.*}} : tensor<1x2xf32>
// CHECK-NOT: func.func private @task_conv_patch
// CHECK-NOT: func.func private @task_vector_tile
// CHECK-NOT: func.func private @task_mvm
// CHECK-NOT: func.func private @task_tile_recombine
// CHECK-NOT: func.func private @task_bias_add
// CHECK-NOT: func.func private @task_dead
// CHECK-NOT: func.func @forward
// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-SAME: sculptor.runtime.resource_count = 3 : i64
// CHECK-SAME: sculptor.runtime.temp_count = 1 : i64
// CHECK-SAME: sculptor.schedule.dependency_count = 1 : i64
// CHECK-SAME: sculptor.schedule.task_count = 2 : i64
// CHECK: %[[GRAPH:.*]] = sculptor.task_graph.create
// CHECK: %[[INPUT:.*]] = sculptor.task_graph.input %[[GRAPH]]
// CHECK: %[[OUTPUT:.*]] = sculptor.task_graph.output %[[GRAPH]]
// CHECK: %[[ARRAY:.*]] = sculptor.task_graph.temporary %[[GRAPH]]
// CHECK-NOT: sculptor.task_graph.temporary
// CHECK: %[[SETUP:.*]] = sculptor.task.create %[[GRAPH]], @task_matrix
// CHECK-SAME: task_kind = "sculptor.matrix_setup"
// CHECK: sculptor.task.create %[[GRAPH]], @task_conv2dwbias_0_conv_tile_mvm_oh_0_ow_0_1
// CHECK-SAME: task_kind = "sculptor.conv_tile_mvm"
// CHECK-SAME: inputs[%[[INPUT]], %[[ARRAY]]]
// CHECK-SAME: outputs[%[[OUTPUT]]]
// CHECK-SAME: deps[%[[SETUP]]]
// CHECK-SAME: sculptor.runtime.output_slots = [1]
// CHECK-SAME: sculptor.runtime.result_indices = [0]
// CHECK-NOT: task_kind = "digital.conv_patch"
// CHECK-NOT: task_kind = "digital.vector_tile"
// CHECK-NOT: task_kind = "sculptor.mvm"
// CHECK-NOT: task_kind = "digital.tile_recombine"
// CHECK-NOT: task_kind = "digital.bias_add"
// CHECK: return %[[GRAPH]] : !sculptor.task_graph
