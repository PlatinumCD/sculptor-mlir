// RUN: sculptor-mlir-opt %s --sculptor-emit-golem-tile-abi | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-emit-golem-tile-abi --sculptor-finalize-golem-intrinsics | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

// CHECK: llvm.func @golem_tile_abi_features
// CHECK: llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.func @golem_tile_scratchpad_required_bytes
// CHECK: llvm.mlir.constant(128 : i64) : i64
// CHECK: llvm.mlir.global internal constant @__golem_tile_dma_descriptors
// CHECK: llvm.func @golem_tile_dma_descriptor_count
// CHECK: llvm.mlir.constant(2 : i32) : i32
// CHECK: llvm.mlir.global internal constant @__golem_tile_resources
// CHECK-NOT: sculptor.

// LLVM: @__golem_tile_dma_descriptors = internal constant [2 x
// LLVM-SAME: { i32 0, i32 0, i32 0, i32 -1, i64 0, i64 32, i32 0, i32 1, i32 10, i32 1, i32 0, i32 1, i64 0 }
// LLVM-SAME: { i32 1, i32 1, i32 1, i32 -1, i64 64, i64 16, i32 1, i32 3, i32 7, i32 1, i32 1, i32 0, i64 0 }
// LLVM: @__golem_tile_resources = internal constant [2 x
// LLVM-SAME: { i32 10, i32 0, i32 0, i32 0, i32 1, i32 2, i32 0, i32 4, i64 32, i64 0 }
// LLVM-SAME: { i32 11, i32 0, i32 1, i32 1, i32 1, i32 2, i32 2, i32 4, i64 16, i64 64 }

module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [{
    global_resource_id = 10 : i64,
    input_index = 0 : i64,
    owner_core = 0 : i64
  }],
  sculptor.deployment.model_outputs = [{
    global_resource_id = 11 : i64,
    output_index = 0 : i64,
    owner_core = 0 : i64
  }],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [0],
    sculptor.runtime.output_slots = [1],
    sculptor.runtime.resource_count = 2 : i64,
    sculptor.runtime.route_input_slots = [],
    sculptor.runtime.route_output_slots = [],
    sculptor.runtime.scratchpad_abi_version = 2 : i32,
    sculptor.runtime.scratchpad_dma_descriptors = [
      {
        byte_size = 32 : i64,
        completion_token_id = 0 : i32,
        descriptor_id = 0 : i32,
        destination_storage = 1 : i32,
        direction = 0 : i32,
        flags = 1 : i32,
        global_resource_id = 10 : i64,
        reserved = 0 : i64,
        route_id = 4294967295 : i64,
        scratchpad_offset = 0 : i64,
        source_storage = 0 : i32,
        trigger_id = 10 : i64,
        trigger_kind = 1 : i32
      },
      {
        byte_size = 16 : i64,
        completion_token_id = 1 : i32,
        descriptor_id = 1 : i32,
        destination_storage = 0 : i32,
        direction = 1 : i32,
        flags = 1 : i32,
        global_resource_id = 11 : i64,
        reserved = 0 : i64,
        route_id = 4294967295 : i64,
        scratchpad_offset = 64 : i64,
        source_storage = 1 : i32,
        trigger_id = 7 : i64,
        trigger_kind = 3 : i32
      }
    ],
    sculptor.runtime.scratchpad_feature_bits = 1 : i32,
    sculptor.runtime.scratchpad_required_bytes = 128 : i64,
    sculptor.runtime.workspace_size = 0 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {
      sculptor.deployment.global_resource_id = 10 : i64,
      sculptor.runtime.byte_size = 32 : i64,
      sculptor.runtime.scratchpad_offset = 0 : i64,
      sculptor.runtime.slot = 0 : i64,
      sculptor.runtime.storage_class = "scratchpad"
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x8xf32>>
    %output = sculptor.task_graph.output %graph {
      sculptor.deployment.global_resource_id = 11 : i64,
      sculptor.runtime.byte_size = 16 : i64,
      sculptor.runtime.scratchpad_offset = 64 : i64,
      sculptor.runtime.slot = 1 : i64,
      sculptor.runtime.storage_class = "scratchpad"
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x4xf32>>
    return %graph : !sculptor.task_graph
  }
}
