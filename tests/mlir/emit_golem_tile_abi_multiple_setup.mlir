// RUN: sculptor-mlir-opt %s --sculptor-emit-golem-tile-abi | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-emit-golem-tile-abi | mlir-translate --mlir-to-llvmir > /dev/null
//
// CHECK-LABEL: llvm.func @setup_array_2()
// CHECK: llvm.mlir.constant(2 : i32)
// CHECK: llvm.call @use_array
// CHECK-LABEL: llvm.func @setup_array_0()
// CHECK: llvm.mlir.constant(0 : i32)
// CHECK: llvm.call @use_array
// CHECK-LABEL: llvm.func @setup_array_1()
// CHECK: llvm.mlir.constant(1 : i32)
// CHECK: llvm.call @use_array
//
// The graph order is 12, 10, 11. Boot ordering must instead follow local
// task indexes 0, 1, 2 and therefore produce 10, 11, 12.
// CHECK-LABEL: llvm.mlir.global internal constant @__golem_tile_boot_tasks()
// CHECK: llvm.mlir.constant(10 : i32)
// CHECK: llvm.mlir.addressof @__golem_tile_execute_task_10
// CHECK: llvm.mlir.constant(11 : i32)
// CHECK: llvm.mlir.addressof @__golem_tile_execute_task_11
// CHECK: llvm.mlir.constant(12 : i32)
// CHECK: llvm.mlir.addressof @__golem_tile_execute_task_12
// CHECK-LABEL: llvm.func @golem_tile_boot_task_count()
// CHECK: llvm.mlir.constant(3 : i32)
// CHECK-LABEL: llvm.func @golem_tile_dispatch_tasks()
// CHECK: llvm.mlir.zero : !llvm.ptr
// CHECK-LABEL: llvm.func @golem_tile_dispatch_task_count()
// CHECK: llvm.mlir.constant(0 : i32)
// CHECK-NOT: sculptor.

module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 7 : i64
} {
  llvm.func @use_array(i32)

  llvm.func @setup_array_2() attributes {
    sculptor.runtime.core_id = 7 : i64,
    sculptor.runtime.local_array_id = 2 : i64,
    sculptor.runtime.physical_array_id = 30 : i64
  } {
    %array = llvm.mlir.constant(2 : i32) : i32
    llvm.call @use_array(%array) : (i32) -> ()
    llvm.return
  }

  llvm.func @setup_array_0() attributes {
    sculptor.runtime.core_id = 7 : i64,
    sculptor.runtime.local_array_id = 0 : i64,
    sculptor.runtime.physical_array_id = 28 : i64
  } {
    %array = llvm.mlir.constant(0 : i32) : i32
    llvm.call @use_array(%array) : (i32) -> ()
    llvm.return
  }

  llvm.func @setup_array_1() attributes {
    sculptor.runtime.core_id = 7 : i64,
    sculptor.runtime.local_array_id = 1 : i64,
    sculptor.runtime.physical_array_id = 29 : i64
  } {
    %array = llvm.mlir.constant(1 : i32) : i32
    llvm.call @use_array(%array) : (i32) -> ()
    llvm.return
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [],
    sculptor.runtime.output_slots = [],
    sculptor.runtime.resource_count = 0 : i64,
    sculptor.runtime.route_input_slots = [],
    sculptor.runtime.route_output_slots = [],
    sculptor.runtime.workspace_size = 0 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %setup2 = sculptor.task.create
      %graph, @setup_array_2,
      domain = "analog",
      task_kind = "sculptor.matrix_setup",
      task_name = "setup_2",
      source_layer = "linear_2",
      source_task_ordinal = 0,
      inputs[],
      outputs[],
      deps[] {
        sculptor.deployment.global_task_id = 12 : i64,
        sculptor.runtime.core_id = 7 : i64,
        sculptor.runtime.input_slots = [],
        sculptor.runtime.local_array_id = 2 : i64,
        sculptor.runtime.output_slots = [],
        sculptor.runtime.physical_array_id = 30 : i64,
        sculptor.runtime.task_index = 2 : i64
      } : (!sculptor.task_graph) -> !sculptor.task
    %setup0 = sculptor.task.create
      %graph, @setup_array_0,
      domain = "analog",
      task_kind = "sculptor.matrix_setup",
      task_name = "setup_0",
      source_layer = "linear_0",
      source_task_ordinal = 0,
      inputs[],
      outputs[],
      deps[] {
        sculptor.deployment.global_task_id = 10 : i64,
        sculptor.runtime.core_id = 7 : i64,
        sculptor.runtime.input_slots = [],
        sculptor.runtime.local_array_id = 0 : i64,
        sculptor.runtime.output_slots = [],
        sculptor.runtime.physical_array_id = 28 : i64,
        sculptor.runtime.task_index = 0 : i64
      } : (!sculptor.task_graph) -> !sculptor.task
    %setup1 = sculptor.task.create
      %graph, @setup_array_1,
      domain = "analog",
      task_kind = "sculptor.matrix_setup",
      task_name = "setup_1",
      source_layer = "linear_1",
      source_task_ordinal = 0,
      inputs[],
      outputs[],
      deps[] {
        sculptor.deployment.global_task_id = 11 : i64,
        sculptor.runtime.core_id = 7 : i64,
        sculptor.runtime.input_slots = [],
        sculptor.runtime.local_array_id = 1 : i64,
        sculptor.runtime.output_slots = [],
        sculptor.runtime.physical_array_id = 29 : i64,
        sculptor.runtime.task_index = 1 : i64
      } : (!sculptor.task_graph) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}
