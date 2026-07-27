// RUN: sculptor-mlir-opt %s --sculptor-emit-golem-tile-abi | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-emit-golem-tile-abi --sculptor-finalize-golem-intrinsics | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

// CHECK: llvm.mlir.global internal constant @__golem_tile_resources
// CHECK: llvm.func @golem_tile_resource_count
// CHECK: llvm.mlir.constant(6 : i32) : i32
// CHECK: llvm.mlir.global internal constant @__golem_tile_resource_dimensions
// CHECK: llvm.func @golem_tile_resource_dimension_count
// CHECK: llvm.mlir.constant(9 : i32) : i32
// CHECK: llvm.func @golem_tile_workspace_size
// CHECK: llvm.mlir.constant(36 : i64) : i64
// CHECK: llvm.mlir.global internal constant @__golem_tile_task_bindings
// CHECK: llvm.func @golem_tile_task_binding_count
// CHECK: llvm.mlir.constant(3 : i32) : i32
// CHECK: llvm.mlir.global internal constant @__golem_tile_task_binding_data
// CHECK: llvm.func @golem_tile_task_binding_data_count
// CHECK: llvm.mlir.constant(11 : i32) : i32
// CHECK-NOT: sculptor.
//
// LLVM: @__golem_tile_resources = internal constant [6 x
// LLVM-SAME: { i32 100, i32 0, i32 0, i32 0, i32 1, i32 4, i32 0, i32 2, i64 192, i64 0 }
// LLVM-SAME: { i32 101, i32 10, i32 1, i32 4, i32 1, i32 1, i32 4, i32 1, i64 16, i64 0 }
// LLVM-SAME: { i32 102, i32 0, i32 2, i32 2, i32 1, i32 1, i32 5, i32 1, i64 8, i64 16 }
// LLVM-SAME: { i32 103, i32 0, i32 3, i32 2, i32 1, i32 1, i32 6, i32 1, i64 8, i64 24 }
// LLVM-SAME: { i32 104, i32 11, i32 4, i32 5, i32 1, i32 1, i32 7, i32 1, i64 4, i64 32 }
// LLVM-SAME: { i32 105, i32 0, i32 5, i32 1, i32 1, i32 1, i32 8, i32 2, i64 4, i64 0 }
// LLVM: @__golem_tile_resource_dimensions = internal constant [9 x i64] [i64 1, i64 3, i64 4, i64 4, i64 4, i64 2, i64 2, i64 1, i64 1]
// LLVM: @__golem_tile_task_bindings = internal constant [3 x
// LLVM-SAME: { i32 1, i32 0, i32 2, i32 2, i32 2, i32 4, i32 0, i32 0 }
// LLVM-SAME: { i32 2, i32 4, i32 1, i32 5, i32 1, i32 6, i32 1, i32 0 }
// LLVM-SAME: { i32 3, i32 7, i32 1, i32 8, i32 1, i32 9, i32 2, i32 0 }
// LLVM: @__golem_tile_task_binding_data = internal constant [11 x i32] [i32 0, i32 1, i32 2, i32 3, i32 2, i32 4, i32 1, i32 3, i32 5, i32 1, i32 2]
// LLVM-LABEL: define i64 @golem_tile_workspace_size()
// LLVM: ret i64 36

module attributes {
  sculptor.deployment.incoming_routes = [
    #sculptor.deployment_route<
      id = 10 : i64,
      sourceCore = 6 : i64,
      sourceTask = 9 : i64,
      sourceOutput = 0 : i64,
      destinationCore = 7 : i64,
      destinationTask = 1 : i64,
      destinationInput = 1 : i64,
      resourceId = 101 : i64,
      byteSize = 16 : i64
    >
  ],
  sculptor.deployment.model_inputs = [{
    global_resource_id = 100 : i64,
    input_index = 0 : i64,
    owner_core = 7 : i64
  }],
  sculptor.deployment.model_outputs = [{
    global_resource_id = 105 : i64,
    output_index = 0 : i64,
    owner_core = 7 : i64
  }],
  sculptor.deployment.outgoing_routes = [
    #sculptor.deployment_route<
      id = 11 : i64,
      sourceCore = 7 : i64,
      sourceTask = 2 : i64,
      sourceOutput = 0 : i64,
      destinationCore = 8 : i64,
      destinationTask = 20 : i64,
      destinationInput = 0 : i64,
      resourceId = 104 : i64,
      byteSize = 4 : i64
    >
  ],
  sculptor.runtime.core_id = 7 : i64
} {
  llvm.func @setup()
  llvm.func @multi_input_output(
    !llvm.ptr, !llvm.ptr, i64,
    i64, i64, i64, i64,
    i64, i64, i64, i64,
    !llvm.ptr, !llvm.ptr, i64, i64, i64
  ) -> !llvm.struct<(
    struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>,
    struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  )>
  llvm.func @route_output_task(
    !llvm.ptr, !llvm.ptr, i64, i64, i64
  ) -> !llvm.struct<(
    ptr, ptr, i64, array<1 x i64>, array<1 x i64>
  )>
  llvm.func @model_output_task(
    !llvm.ptr, !llvm.ptr, i64, i64, i64
  ) -> !llvm.struct<(
    ptr, ptr, i64, array<1 x i64>, array<1 x i64>
  )>

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [0],
    sculptor.runtime.output_slots = [5],
    sculptor.runtime.resource_count = 6 : i64,
    sculptor.runtime.route_input_slots = [1],
    sculptor.runtime.route_output_slots = [4],
    sculptor.runtime.workspace_size = 36 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %model_input = sculptor.task_graph.input %graph {
      sculptor.deployment.global_resource_id = 100 : i64,
      sculptor.runtime.byte_size = 192 : i64,
      sculptor.runtime.slot = 0 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x3x4x4xf32>>
    %route_input = sculptor.task_graph.route_input %graph {
      sculptor.deployment.global_resource_id = 101 : i64,
      sculptor.deployment.route_id = 10 : i64,
      sculptor.runtime.byte_size = 16 : i64,
      sculptor.runtime.slot = 1 : i64,
      sculptor.runtime.temp_offset = 0 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<4xf32>>
    %intermediate0 = sculptor.task_graph.intermediate %graph {
      sculptor.deployment.global_resource_id = 102 : i64,
      sculptor.runtime.byte_size = 8 : i64,
      sculptor.runtime.slot = 2 : i64,
      sculptor.runtime.temp_offset = 16 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<2xf32>>
    %intermediate1 = sculptor.task_graph.intermediate %graph {
      sculptor.deployment.global_resource_id = 103 : i64,
      sculptor.runtime.byte_size = 8 : i64,
      sculptor.runtime.slot = 3 : i64,
      sculptor.runtime.temp_offset = 24 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<2xf32>>
    %route_output = sculptor.task_graph.route_output %graph {
      sculptor.deployment.global_resource_id = 104 : i64,
      sculptor.deployment.route_id = 11 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 4 : i64,
      sculptor.runtime.temp_offset = 32 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1xf32>>
    %model_output = sculptor.task_graph.output %graph {
      sculptor.deployment.global_resource_id = 105 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 5 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1xf32>>

    %boot = sculptor.task.create
      %graph, @setup,
      domain = "analog",
      task_kind = "sculptor.matrix_setup",
      task_name = "setup",
      source_layer = "plan",
      source_task_ordinal = 0,
      inputs[],
      outputs[],
      deps[] {
        sculptor.deployment.global_task_id = 0 : i64,
        sculptor.runtime.core_id = 7 : i64,
        sculptor.runtime.input_slots = [],
        sculptor.runtime.local_array_id = 0 : i64,
        sculptor.runtime.output_slots = [],
        sculptor.runtime.physical_array_id = 28 : i64,
        sculptor.runtime.task_index = 0 : i64
      } : (!sculptor.task_graph) -> !sculptor.task
    %task1 = sculptor.task.create
      %graph, @multi_input_output,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "multi_input_output",
      source_layer = "plan",
      source_task_ordinal = 1,
      inputs[%model_input, %route_input],
      outputs[%intermediate0, %intermediate1],
      deps[%boot] {
        sculptor.deployment.global_task_id = 1 : i64,
        sculptor.runtime.core_id = 7 : i64,
        sculptor.runtime.input_slots = [0, 1],
        sculptor.runtime.output_slots = [2, 3],
        sculptor.runtime.result_indices = [0, 1],
        sculptor.runtime.task_index = 1 : i64
      } : (
        !sculptor.task_graph,
        !sculptor.task_resource<tensor<1x3x4x4xf32>>,
        !sculptor.task_resource<tensor<4xf32>>,
        !sculptor.task_resource<tensor<2xf32>>,
        !sculptor.task_resource<tensor<2xf32>>,
        !sculptor.task
      ) -> !sculptor.task
    %task2 = sculptor.task.create
      %graph, @route_output_task,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "route_output",
      source_layer = "plan",
      source_task_ordinal = 2,
      inputs[%intermediate0],
      outputs[%route_output],
      deps[%boot, %task1] {
        sculptor.deployment.global_task_id = 2 : i64,
        sculptor.runtime.core_id = 7 : i64,
        sculptor.runtime.input_slots = [2],
        sculptor.runtime.output_slots = [4],
        sculptor.runtime.result_indices = [0],
        sculptor.runtime.task_index = 2 : i64
      } : (
        !sculptor.task_graph,
        !sculptor.task_resource<tensor<2xf32>>,
        !sculptor.task_resource<tensor<1xf32>>,
        !sculptor.task,
        !sculptor.task
      ) -> !sculptor.task
    %task3 = sculptor.task.create
      %graph, @model_output_task,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "model_output",
      source_layer = "plan",
      source_task_ordinal = 3,
      inputs[%intermediate1],
      outputs[%model_output],
      deps[%boot, %task1, %task2] {
        sculptor.deployment.global_task_id = 3 : i64,
        sculptor.runtime.core_id = 7 : i64,
        sculptor.runtime.input_slots = [3],
        sculptor.runtime.output_slots = [5],
        sculptor.runtime.result_indices = [0],
        sculptor.runtime.task_index = 3 : i64
      } : (
        !sculptor.task_graph,
        !sculptor.task_resource<tensor<2xf32>>,
        !sculptor.task_resource<tensor<1xf32>>,
        !sculptor.task,
        !sculptor.task,
        !sculptor.task
      ) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}
