// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-emit-golem-tile-abi | FileCheck %s
// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-emit-golem-tile-abi --sculptor-finalize-golem-intrinsics | mlir-translate --split-input-file --mlir-to-llvmir > /dev/null
// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-emit-golem-tile-abi --sculptor-finalize-golem-intrinsics | mlir-translate --split-input-file --mlir-to-llvmir | FileCheck %s --check-prefix=LLVM
//
// LLVM: @__golem_tile_boot_tasks = internal constant {{.*}}i32 0, ptr @__golem_tile_execute_task_0, i32 0, i32 0
// LLVM: @__golem_tile_dispatch_tasks = internal constant {{.*}}i32 1, ptr @__golem_tile_execute_task_1, i32 1, i32 1
// LLVM: @__golem_tile_outgoing_routes = internal constant {{.*}}i32 0, i32 0, i32 1, i32 0, i32 1, i32 3, i32 0, i32 2, i32 1, i64 12
// LLVM: @__golem_tile_model_inputs = internal constant {{.*}}i32 0, i32 0, i32 0, i32 0, i64 16
// LLVM: @__golem_tile_boot_tasks = internal constant {{.*}}i32 2, ptr @__golem_tile_execute_task_2, i32 0, i32 0
// LLVM: @__golem_tile_dispatch_tasks = internal constant {{.*}}i32 3, ptr @__golem_tile_execute_task_3, i32 1, i32 1
// LLVM: @__golem_tile_incoming_routes = internal constant {{.*}}i32 0, i32 0, i32 1, i32 0, i32 1, i32 3, i32 0, i32 2, i32 0, i64 12
// LLVM: @__golem_tile_model_outputs = internal constant {{.*}}i32 0, i32 1, i32 1, i32 1, i64 8
//
// CHECK-LABEL: module {
// CHECK: llvm.func @task_setup_0()
// CHECK: llvm.func @task_mvm_0
// CHECK-NOT: task_setup_1
// CHECK-NOT: task_mvm_1
// CHECK: llvm.func internal @__golem_tile_execute_task_0
// CHECK: llvm.call @task_setup_0()
// CHECK: llvm.func internal @__golem_tile_execute_task_1
// CHECK: llvm.call @task_mvm_0
// CHECK-NOT: llvm.intr.memcpy
// CHECK-NOT: llvm.call @free
// CHECK: llvm.func @golem_tile_core_id()
// CHECK: llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.mlir.global internal constant @__golem_tile_boot_tasks()
// CHECK: llvm.mlir.addressof @__golem_tile_execute_task_0
// CHECK: llvm.func @golem_tile_boot_task_count()
// CHECK: llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.mlir.global internal constant @__golem_tile_dispatch_tasks()
// CHECK: llvm.mlir.addressof @__golem_tile_execute_task_1
// CHECK: llvm.func @golem_tile_dispatch_task_count()
// CHECK: llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.func @golem_tile_incoming_route_count()
// CHECK: llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.mlir.global internal constant @__golem_tile_outgoing_routes()
// CHECK: llvm.func @golem_tile_outgoing_route_count()
// CHECK: llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.mlir.global internal constant @__golem_tile_model_inputs()
// CHECK-NOT: sculptor.

module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [{
    global_resource_id = 0 : i64,
    input_index = 0 : i64,
    owner_core = 0 : i64
  }],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [
    #sculptor.deployment_route<
      id = 0 : i64,
      sourceCore = 0 : i64,
      sourceTask = 1 : i64,
      sourceOutput = 0 : i64,
      destinationCore = 1 : i64,
      destinationTask = 3 : i64,
      destinationInput = 0 : i64,
      resourceId = 2 : i64,
      byteSize = 12 : i64
    >
  ],
  sculptor.runtime.core_id = 0 : i64
} {
  llvm.func @task_setup_0() attributes {
    sculptor.runtime.core_id = 0 : i64,
    sculptor.runtime.local_array_id = 0 : i64,
    sculptor.runtime.physical_array_id = 0 : i64
  }
  llvm.func @task_mvm_0(
    !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64,
    !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64
  ) attributes {
    sculptor.runtime.core_id = 0 : i64,
    sculptor.runtime.local_array_id = 0 : i64,
    sculptor.runtime.physical_array_id = 0 : i64
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [0],
    sculptor.runtime.output_slots = [],
    sculptor.runtime.resource_count = 2 : i64,
    sculptor.runtime.route_input_slots = [],
    sculptor.runtime.route_output_slots = [1],
    sculptor.runtime.workspace_size = 12 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {
      sculptor.deployment.global_resource_id = 0 : i64,
      sculptor.runtime.byte_size = 16 : i64,
      sculptor.runtime.slot = 0 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x4xf32>>
    %route = sculptor.task_graph.route_output %graph {
      sculptor.deployment.global_resource_id = 2 : i64,
      sculptor.deployment.route_id = 0 : i64,
      sculptor.runtime.byte_size = 12 : i64,
      sculptor.runtime.slot = 1 : i64,
      sculptor.runtime.temp_offset = 0 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x3xf32>>
    %setup = sculptor.task.create
      %graph, @task_setup_0,
      domain = "analog",
      task_kind = "sculptor.matrix_setup",
      task_name = "setup_0",
      source_layer = "linear_0",
      source_task_ordinal = 0,
      inputs[],
      outputs[],
      deps[] {
        sculptor.deployment.global_task_id = 0 : i64,
        sculptor.runtime.core_id = 0 : i64,
        sculptor.runtime.input_slots = [],
        sculptor.runtime.local_array_id = 0 : i64,
        sculptor.runtime.output_slots = [],
        sculptor.runtime.physical_array_id = 0 : i64,
        sculptor.runtime.task_index = 0 : i64
      } : (!sculptor.task_graph) -> !sculptor.task
    %compute = sculptor.task.create
      %graph, @task_mvm_0,
      domain = "analog",
      task_kind = "sculptor.mvm",
      task_name = "mvm_0",
      source_layer = "linear_0",
      source_task_ordinal = 1,
      inputs[%input],
      outputs[%route],
      deps[%setup] {
        sculptor.deployment.global_task_id = 1 : i64,
        sculptor.runtime.core_id = 0 : i64,
        sculptor.runtime.input_slots = [0],
        sculptor.runtime.local_array_id = 0 : i64,
        sculptor.runtime.output_slots = [1],
        sculptor.runtime.physical_array_id = 0 : i64,
        sculptor.runtime.result_indices = [0],
        sculptor.runtime.task_index = 1 : i64
      } : (
        !sculptor.task_graph,
        !sculptor.task_resource<tensor<1x4xf32>>,
        !sculptor.task_resource<tensor<1x3xf32>>,
        !sculptor.task
      ) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

// CHECK-LABEL: module {
// CHECK: llvm.func @task_setup_1()
// CHECK: llvm.func @task_mvm_1
// CHECK-NOT: task_setup_0
// CHECK-NOT: task_mvm_0
// CHECK: llvm.func internal @__golem_tile_execute_task_2
// CHECK: llvm.call @task_setup_1()
// CHECK: llvm.func internal @__golem_tile_execute_task_3
// CHECK: llvm.call @task_mvm_1
// CHECK: llvm.func @golem_tile_core_id()
// CHECK: llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.mlir.global internal constant @__golem_tile_boot_tasks()
// CHECK: llvm.mlir.addressof @__golem_tile_execute_task_2
// CHECK: llvm.func @golem_tile_boot_task_count()
// CHECK: llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.mlir.global internal constant @__golem_tile_dispatch_tasks()
// CHECK: llvm.mlir.addressof @__golem_tile_execute_task_3
// CHECK: llvm.func @golem_tile_dispatch_task_count()
// CHECK: llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.mlir.global internal constant @__golem_tile_incoming_routes()
// CHECK: llvm.func @golem_tile_incoming_route_count()
// CHECK: llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.func @golem_tile_outgoing_route_count()
// CHECK: llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.mlir.global internal constant @__golem_tile_model_outputs()
// CHECK-NOT: sculptor.

module attributes {
  sculptor.deployment.incoming_routes = [
    #sculptor.deployment_route<
      id = 0 : i64,
      sourceCore = 0 : i64,
      sourceTask = 1 : i64,
      sourceOutput = 0 : i64,
      destinationCore = 1 : i64,
      destinationTask = 3 : i64,
      destinationInput = 0 : i64,
      resourceId = 2 : i64,
      byteSize = 12 : i64
    >
  ],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [{
    global_resource_id = 1 : i64,
    output_index = 0 : i64,
    owner_core = 1 : i64
  }],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 1 : i64
} {
  llvm.func @task_setup_1() attributes {
    sculptor.runtime.core_id = 1 : i64,
    sculptor.runtime.local_array_id = 0 : i64,
    sculptor.runtime.physical_array_id = 1 : i64
  }
  llvm.func @task_mvm_1(
    !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64
  ) -> !llvm.struct<(
    ptr, ptr, i64, array<2 x i64>, array<2 x i64>
  )> attributes {
    sculptor.runtime.core_id = 1 : i64,
    sculptor.runtime.local_array_id = 0 : i64,
    sculptor.runtime.physical_array_id = 1 : i64
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [],
    sculptor.runtime.output_slots = [1],
    sculptor.runtime.resource_count = 2 : i64,
    sculptor.runtime.route_input_slots = [0],
    sculptor.runtime.route_output_slots = [],
    sculptor.runtime.workspace_size = 12 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %route = sculptor.task_graph.route_input %graph {
      sculptor.deployment.global_resource_id = 2 : i64,
      sculptor.deployment.route_id = 0 : i64,
      sculptor.runtime.byte_size = 12 : i64,
      sculptor.runtime.slot = 0 : i64,
      sculptor.runtime.temp_offset = 0 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x3xf32>>
    %output = sculptor.task_graph.output %graph {
      sculptor.deployment.global_resource_id = 1 : i64,
      sculptor.runtime.byte_size = 8 : i64,
      sculptor.runtime.slot = 1 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x2xf32>>
    %setup = sculptor.task.create
      %graph, @task_setup_1,
      domain = "analog",
      task_kind = "sculptor.matrix_setup",
      task_name = "setup_1",
      source_layer = "linear_1",
      source_task_ordinal = 0,
      inputs[],
      outputs[],
      deps[] {
        sculptor.deployment.global_task_id = 2 : i64,
        sculptor.runtime.core_id = 1 : i64,
        sculptor.runtime.input_slots = [],
        sculptor.runtime.local_array_id = 0 : i64,
        sculptor.runtime.output_slots = [],
        sculptor.runtime.physical_array_id = 1 : i64,
        sculptor.runtime.task_index = 0 : i64
      } : (!sculptor.task_graph) -> !sculptor.task
    %compute = sculptor.task.create
      %graph, @task_mvm_1,
      domain = "analog",
      task_kind = "sculptor.mvm",
      task_name = "mvm_1",
      source_layer = "linear_1",
      source_task_ordinal = 1,
      inputs[%route],
      outputs[%output],
      deps[%setup] {
        sculptor.deployment.global_task_id = 3 : i64,
        sculptor.runtime.core_id = 1 : i64,
        sculptor.runtime.input_slots = [0],
        sculptor.runtime.local_array_id = 0 : i64,
        sculptor.runtime.output_slots = [1],
        sculptor.runtime.physical_array_id = 1 : i64,
        sculptor.runtime.result_indices = [0],
        sculptor.runtime.task_index = 1 : i64
      } : (
        !sculptor.task_graph,
        !sculptor.task_resource<tensor<1x3xf32>>,
        !sculptor.task_resource<tensor<1x2xf32>>,
        !sculptor.task
      ) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}
