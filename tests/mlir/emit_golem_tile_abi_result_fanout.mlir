// RUN: sculptor-mlir-opt %s --sculptor-emit-golem-tile-abi | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-emit-golem-tile-abi --sculptor-finalize-golem-intrinsics | mlir-translate --mlir-to-llvmir | FileCheck %s --check-prefix=LLVM

// CHECK-LABEL: llvm.func internal @__golem_tile_execute_task_1
// CHECK: llvm.call @fanout_00
// CHECK: llvm.intr.memcpy
// CHECK-NOT: llvm.intr.memcpy
// CHECK-NOT: llvm.call @free
// CHECK-LABEL: llvm.func internal @__golem_tile_execute_task_2
// CHECK: llvm.call @fanout_001
// CHECK: llvm.intr.memcpy
// CHECK-NOT: llvm.intr.memcpy
// CHECK-NOT: llvm.call @free
// CHECK-LABEL: llvm.func internal @__golem_tile_execute_task_3
// CHECK: llvm.call @fanout_000
// CHECK: llvm.intr.memcpy
// CHECK: llvm.intr.memcpy
// CHECK-NOT: llvm.intr.memcpy
// CHECK-NOT: llvm.call @free
// CHECK: llvm.func @golem_tile_core_id
// CHECK-NOT: sculptor.

// LLVM-LABEL: define internal i32 @__golem_tile_execute_task_1
// LLVM: call void @fanout_00
// LLVM-LABEL: define internal i32 @__golem_tile_execute_task_2
// LLVM: call void @fanout_001
// LLVM-LABEL: define internal i32 @__golem_tile_execute_task_3
// LLVM: call void @fanout_000

module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  llvm.func @fanout_00(
    !llvm.ptr, !llvm.ptr, i64, i64, i64
  )
  llvm.func @fanout_001(
    !llvm.ptr, !llvm.ptr, i64, i64, i64,
    !llvm.ptr, !llvm.ptr, i64, i64, i64
  )
  llvm.func @fanout_000(
    !llvm.ptr, !llvm.ptr, i64, i64, i64
  )

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [],
    sculptor.runtime.output_slots = [],
    sculptor.runtime.resource_count = 8 : i64,
    sculptor.runtime.route_input_slots = [],
    sculptor.runtime.route_output_slots = [],
    sculptor.runtime.workspace_size = 32 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %r0 = sculptor.task_graph.intermediate %graph {
      sculptor.deployment.global_resource_id = 0 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 0 : i64,
      sculptor.runtime.temp_offset = 0 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %r1 = sculptor.task_graph.intermediate %graph {
      sculptor.deployment.global_resource_id = 1 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 1 : i64,
      sculptor.runtime.temp_offset = 4 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %r2 = sculptor.task_graph.intermediate %graph {
      sculptor.deployment.global_resource_id = 2 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 2 : i64,
      sculptor.runtime.temp_offset = 8 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %r3 = sculptor.task_graph.intermediate %graph {
      sculptor.deployment.global_resource_id = 3 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 3 : i64,
      sculptor.runtime.temp_offset = 12 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %r4 = sculptor.task_graph.intermediate %graph {
      sculptor.deployment.global_resource_id = 4 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 4 : i64,
      sculptor.runtime.temp_offset = 16 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %r5 = sculptor.task_graph.intermediate %graph {
      sculptor.deployment.global_resource_id = 5 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 5 : i64,
      sculptor.runtime.temp_offset = 20 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %r6 = sculptor.task_graph.intermediate %graph {
      sculptor.deployment.global_resource_id = 6 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 6 : i64,
      sculptor.runtime.temp_offset = 24 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %r7 = sculptor.task_graph.intermediate %graph {
      sculptor.deployment.global_resource_id = 7 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 7 : i64,
      sculptor.runtime.temp_offset = 28 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>

    %task1 = sculptor.task.create
      %graph, @fanout_00,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "fanout_00",
      source_layer = "fanout",
      source_task_ordinal = 0,
      inputs[],
      outputs[%r0, %r1],
      deps[] {
        sculptor.deployment.global_task_id = 1 : i64,
        sculptor.runtime.core_id = 0 : i64,
        sculptor.runtime.input_slots = [],
        sculptor.runtime.output_slots = [0, 1],
        sculptor.runtime.result_indices = [0, 0],
        sculptor.runtime.task_index = 0 : i64
      } : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1xf32>>,
           !sculptor.task_resource<tensor<1xf32>>) -> !sculptor.task
    %task2 = sculptor.task.create
      %graph, @fanout_001,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "fanout_001",
      source_layer = "fanout",
      source_task_ordinal = 1,
      inputs[],
      outputs[%r2, %r3, %r4],
      deps[] {
        sculptor.deployment.global_task_id = 2 : i64,
        sculptor.runtime.core_id = 0 : i64,
        sculptor.runtime.input_slots = [],
        sculptor.runtime.output_slots = [2, 3, 4],
        sculptor.runtime.result_indices = [0, 0, 1],
        sculptor.runtime.task_index = 1 : i64
      } : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1xf32>>,
           !sculptor.task_resource<tensor<1xf32>>,
           !sculptor.task_resource<tensor<1xf32>>) -> !sculptor.task
    %task3 = sculptor.task.create
      %graph, @fanout_000,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "fanout_000",
      source_layer = "fanout",
      source_task_ordinal = 2,
      inputs[],
      outputs[%r5, %r6, %r7],
      deps[] {
        sculptor.deployment.global_task_id = 3 : i64,
        sculptor.runtime.core_id = 0 : i64,
        sculptor.runtime.input_slots = [],
        sculptor.runtime.output_slots = [5, 6, 7],
        sculptor.runtime.result_indices = [0, 0, 0],
        sculptor.runtime.task_index = 2 : i64
      } : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1xf32>>,
           !sculptor.task_resource<tensor<1xf32>>,
           !sculptor.task_resource<tensor<1xf32>>) -> !sculptor.task
    func.return %graph : !sculptor.task_graph
  }
}
