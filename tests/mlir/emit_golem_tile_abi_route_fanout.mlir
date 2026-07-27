// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-emit-golem-tile-abi | FileCheck %s
// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-emit-golem-tile-abi --sculptor-finalize-golem-intrinsics | mlir-translate --split-input-file --mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

// CHECK-LABEL: module {
// CHECK: llvm.mlir.global internal constant @__golem_tile_incoming_routes
// CHECK: llvm.mlir.constant(63 : i32) : i32
// CHECK: llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.mlir.constant(64 : i32) : i32
// CHECK: llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.func @golem_tile_incoming_route_count
// CHECK: llvm.mlir.constant(2 : i32) : i32
// CHECK-NOT: sculptor.
//
// LLVM: @__golem_tile_incoming_routes = internal constant
// LLVM-SAME: i32 63, i32 0, i32 100, i32 0, i32 8, i32 183, i32 0, i32 123, i32 0, i64 4
// LLVM-SAME: i32 64, i32 0, i32 100, i32 0, i32 8, i32 184, i32 0, i32 123, i32 1, i64 4

module attributes {
  sculptor.deployment.incoming_routes = [
    #sculptor.deployment_route<
      id = 63 : i64,
      sourceCore = 0 : i64,
      sourceTask = 100 : i64,
      sourceOutput = 0 : i64,
      destinationCore = 8 : i64,
      destinationTask = 183 : i64,
      destinationInput = 0 : i64,
      resourceId = 123 : i64,
      byteSize = 4 : i64
    >,
    #sculptor.deployment_route<
      id = 64 : i64,
      sourceCore = 0 : i64,
      sourceTask = 100 : i64,
      sourceOutput = 0 : i64,
      destinationCore = 8 : i64,
      destinationTask = 184 : i64,
      destinationInput = 0 : i64,
      resourceId = 123 : i64,
      byteSize = 4 : i64
    >
  ],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 8 : i64
} {
  llvm.func @consume_183(!llvm.ptr, !llvm.ptr, i64, i64, i64)
  llvm.func @consume_184(!llvm.ptr, !llvm.ptr, i64, i64, i64)

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [],
    sculptor.runtime.output_slots = [],
    sculptor.runtime.resource_count = 2 : i64,
    sculptor.runtime.route_input_slots = [0, 1],
    sculptor.runtime.route_output_slots = [],
    sculptor.runtime.workspace_size = 8 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %route63 = sculptor.task_graph.route_input %graph {
      sculptor.deployment.global_resource_id = 123 : i64,
      sculptor.deployment.route_id = 63 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 0 : i64,
      sculptor.runtime.temp_offset = 0 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %route64 = sculptor.task_graph.route_input %graph {
      sculptor.deployment.global_resource_id = 123 : i64,
      sculptor.deployment.route_id = 64 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 1 : i64,
      sculptor.runtime.temp_offset = 4 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %task183 = sculptor.task.create
      %graph, @consume_183,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "consume_183",
      source_layer = "fanout",
      source_task_ordinal = 0,
      inputs[%route63],
      outputs[],
      deps[] {
        sculptor.deployment.global_task_id = 183 : i64,
        sculptor.runtime.core_id = 8 : i64,
        sculptor.runtime.input_slots = [0],
        sculptor.runtime.output_slots = [],
        sculptor.runtime.task_index = 0 : i64
      } : (
        !sculptor.task_graph,
        !sculptor.task_resource<tensor<1xf32>>
      ) -> !sculptor.task
    %task184 = sculptor.task.create
      %graph, @consume_184,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "consume_184",
      source_layer = "fanout",
      source_task_ordinal = 1,
      inputs[%route64],
      outputs[],
      deps[] {
        sculptor.deployment.global_task_id = 184 : i64,
        sculptor.runtime.core_id = 8 : i64,
        sculptor.runtime.input_slots = [1],
        sculptor.runtime.output_slots = [],
        sculptor.runtime.task_index = 1 : i64
      } : (
        !sculptor.task_graph,
        !sculptor.task_resource<tensor<1xf32>>
      ) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }

  // Keep the same logical payload ID while giving each outgoing transfer its
  // own route identity and local buffer.
}

// -----

// CHECK-LABEL: module {
// CHECK: llvm.mlir.global internal constant @__golem_tile_outgoing_routes
// CHECK: llvm.func @golem_tile_outgoing_route_count
// CHECK: llvm.mlir.constant(2 : i32) : i32
// CHECK-NOT: sculptor.
//
// LLVM: @__golem_tile_outgoing_routes = internal constant
// LLVM-SAME: i32 63, i32 8, i32 100, i32 0, i32 9, i32 183, i32 0, i32 123, i32 0, i64 4
// LLVM-SAME: i32 64, i32 8, i32 100, i32 1, i32 9, i32 184, i32 0, i32 123, i32 1, i64 4

module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [
    #sculptor.deployment_route<
      id = 63 : i64,
      sourceCore = 8 : i64,
      sourceTask = 100 : i64,
      sourceOutput = 0 : i64,
      destinationCore = 9 : i64,
      destinationTask = 183 : i64,
      destinationInput = 0 : i64,
      resourceId = 123 : i64,
      byteSize = 4 : i64
    >,
    #sculptor.deployment_route<
      id = 64 : i64,
      sourceCore = 8 : i64,
      sourceTask = 100 : i64,
      sourceOutput = 1 : i64,
      destinationCore = 9 : i64,
      destinationTask = 184 : i64,
      destinationInput = 0 : i64,
      resourceId = 123 : i64,
      byteSize = 4 : i64
    >
  ],
  sculptor.runtime.core_id = 8 : i64
} {
  llvm.func @produce_100() -> !llvm.struct<(
    struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>,
    struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  )>

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [],
    sculptor.runtime.output_slots = [],
    sculptor.runtime.resource_count = 2 : i64,
    sculptor.runtime.route_input_slots = [],
    sculptor.runtime.route_output_slots = [0, 1],
    sculptor.runtime.workspace_size = 8 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %route63 = sculptor.task_graph.route_output %graph {
      sculptor.deployment.global_resource_id = 123 : i64,
      sculptor.deployment.route_id = 63 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 0 : i64,
      sculptor.runtime.temp_offset = 0 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %route64 = sculptor.task_graph.route_output %graph {
      sculptor.deployment.global_resource_id = 123 : i64,
      sculptor.deployment.route_id = 64 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 1 : i64,
      sculptor.runtime.temp_offset = 4 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %task100 = sculptor.task.create
      %graph, @produce_100,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "produce_100",
      source_layer = "fanout",
      source_task_ordinal = 0,
      inputs[],
      outputs[%route63, %route64],
      deps[] {
        sculptor.deployment.global_task_id = 100 : i64,
        sculptor.runtime.core_id = 8 : i64,
        sculptor.runtime.input_slots = [],
        sculptor.runtime.output_slots = [0, 1],
        sculptor.runtime.result_indices = [0, 1],
        sculptor.runtime.task_index = 0 : i64
      } : (
        !sculptor.task_graph,
        !sculptor.task_resource<tensor<1xf32>>,
        !sculptor.task_resource<tensor<1xf32>>
      ) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}
