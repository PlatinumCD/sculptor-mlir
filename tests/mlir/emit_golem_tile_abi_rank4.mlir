// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-emit-golem-tile-abi | FileCheck %s
// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-emit-golem-tile-abi --sculptor-finalize-golem-intrinsics | mlir-translate --split-input-file --mlir-to-llvmir > /dev/null

// CHECK-LABEL: module {
// CHECK: llvm.func @rank4_model_task(
// CHECK-SAME: !llvm.ptr, !llvm.ptr, i64,
// CHECK-SAME: i64, i64, i64, i64,
// CHECK-SAME: i64, i64, i64, i64
// CHECK-SAME: ) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK: llvm.func internal @__golem_tile_execute_task_0
// CHECK: llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG: llvm.mlir.constant(48 : i64) : i64
// CHECK-DAG: llvm.mlir.constant(16 : i64) : i64
// CHECK-DAG: llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG: llvm.extractvalue {{.*}}[3, 0]
// CHECK-DAG: llvm.extractvalue {{.*}}[3, 1]
// CHECK-DAG: llvm.extractvalue {{.*}}[3, 2]
// CHECK-DAG: llvm.extractvalue {{.*}}[3, 3]
// CHECK-DAG: llvm.extractvalue {{.*}}[4, 0]
// CHECK-DAG: llvm.extractvalue {{.*}}[4, 1]
// CHECK-DAG: llvm.extractvalue {{.*}}[4, 2]
// CHECK-DAG: llvm.extractvalue {{.*}}[4, 3]
// CHECK: llvm.call @rank4_model_task
// CHECK: llvm.extractvalue {{.*}}[3, 3]
// CHECK: llvm.extractvalue {{.*}}[4, 3]
// CHECK-NOT: sculptor.

module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [{
    global_resource_id = 0 : i64,
    input_index = 0 : i64,
    owner_core = 0 : i64
  }],
  sculptor.deployment.model_outputs = [{
    global_resource_id = 1 : i64,
    output_index = 0 : i64,
    owner_core = 0 : i64
  }],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  llvm.func @rank4_model_task(
    !llvm.ptr, !llvm.ptr, i64,
    i64, i64, i64, i64,
    i64, i64, i64, i64
  ) -> !llvm.struct<(
    ptr, ptr, i64, array<4 x i64>, array<4 x i64>
  )>

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [0],
    sculptor.runtime.output_slots = [1],
    sculptor.runtime.resource_count = 2 : i64,
    sculptor.runtime.route_input_slots = [],
    sculptor.runtime.route_output_slots = [],
    sculptor.runtime.workspace_size = 0 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {
      sculptor.deployment.global_resource_id = 0 : i64,
      sculptor.runtime.byte_size = 192 : i64,
      sculptor.runtime.slot = 0 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x3x4x4xf32>>
    %output = sculptor.task_graph.output %graph {
      sculptor.deployment.global_resource_id = 1 : i64,
      sculptor.runtime.byte_size = 192 : i64,
      sculptor.runtime.slot = 1 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x3x4x4xf32>>
    %task = sculptor.task.create
      %graph, @rank4_model_task,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "rank4_model",
      source_layer = "rank4",
      source_task_ordinal = 0,
      inputs[%input],
      outputs[%output],
      deps[] {
        sculptor.deployment.global_task_id = 0 : i64,
        sculptor.runtime.core_id = 0 : i64,
        sculptor.runtime.input_slots = [0],
        sculptor.runtime.output_slots = [1],
        sculptor.runtime.result_indices = [0],
        sculptor.runtime.task_index = 0 : i64
      } : (
        !sculptor.task_graph,
        !sculptor.task_resource<tensor<1x3x4x4xf32>>,
        !sculptor.task_resource<tensor<1x3x4x4xf32>>
      ) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

// CHECK-LABEL: module {
// CHECK: llvm.func @rank0_compat_task(!llvm.ptr, !llvm.ptr, i64)
// CHECK-SAME: -> !llvm.struct<(ptr, ptr, i64, array<0 x i64>, array<0 x i64>)>
// CHECK: llvm.func internal @__golem_tile_execute_task_30
// CHECK: llvm.call @rank0_compat_task
// CHECK-NOT: sculptor.

module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [{
    global_resource_id = 30 : i64,
    input_index = 0 : i64,
    owner_core = 3 : i64
  }],
  sculptor.deployment.model_outputs = [{
    global_resource_id = 31 : i64,
    output_index = 0 : i64,
    owner_core = 3 : i64
  }],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 3 : i64
} {
  llvm.func @rank0_compat_task(
    !llvm.ptr, !llvm.ptr, i64
  ) -> !llvm.struct<(
    ptr, ptr, i64, array<0 x i64>, array<0 x i64>
  )>

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [0],
    sculptor.runtime.output_slots = [1],
    sculptor.runtime.resource_count = 2 : i64,
    sculptor.runtime.route_input_slots = [],
    sculptor.runtime.route_output_slots = [],
    sculptor.runtime.workspace_size = 0 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {
      sculptor.deployment.global_resource_id = 30 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 0 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<f32>>
    %output = sculptor.task_graph.output %graph {
      sculptor.deployment.global_resource_id = 31 : i64,
      sculptor.runtime.byte_size = 4 : i64,
      sculptor.runtime.slot = 1 : i64
    } : !sculptor.task_graph -> !sculptor.task_resource<tensor<f32>>
    %task = sculptor.task.create
      %graph, @rank0_compat_task,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "rank0_compat",
      source_layer = "rank0",
      source_task_ordinal = 0,
      inputs[%input],
      outputs[%output],
      deps[] {
        sculptor.deployment.global_task_id = 30 : i64,
        sculptor.runtime.core_id = 3 : i64,
        sculptor.runtime.input_slots = [0],
        sculptor.runtime.output_slots = [1],
        sculptor.runtime.result_indices = [0],
        sculptor.runtime.task_index = 0 : i64
      } : (
        !sculptor.task_graph,
        !sculptor.task_resource<tensor<f32>>,
        !sculptor.task_resource<tensor<f32>>
      ) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

// CHECK-LABEL: module {
// CHECK: llvm.func @rank4_route_task(
// CHECK: llvm.func internal @__golem_tile_execute_task_20
// CHECK: llvm.mlir.constant(4 : i64) : i64
// CHECK: llvm.call @rank4_route_task
// CHECK: llvm.mlir.global internal constant @__golem_tile_incoming_routes
// CHECK: llvm.mlir.global internal constant @__golem_tile_outgoing_routes
// CHECK-NOT: sculptor.

module attributes {
  sculptor.deployment.incoming_routes = [
    #sculptor.deployment_route<
      id = 7 : i64,
      sourceCore = 0 : i64,
      sourceTask = 10 : i64,
      sourceOutput = 0 : i64,
      destinationCore = 1 : i64,
      destinationTask = 20 : i64,
      destinationInput = 0 : i64,
      resourceId = 100 : i64,
      byteSize = 192 : i64
    >
  ],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [
    #sculptor.deployment_route<
      id = 8 : i64,
      sourceCore = 1 : i64,
      sourceTask = 20 : i64,
      sourceOutput = 0 : i64,
      destinationCore = 2 : i64,
      destinationTask = 30 : i64,
      destinationInput = 0 : i64,
      resourceId = 101 : i64,
      byteSize = 192 : i64
    >
  ],
  sculptor.runtime.core_id = 1 : i64
} {
  llvm.func @rank4_route_task(
    !llvm.ptr, !llvm.ptr, i64,
    i64, i64, i64, i64,
    i64, i64, i64, i64
  ) -> !llvm.struct<(
    ptr, ptr, i64, array<4 x i64>, array<4 x i64>
  )>

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.input_slots = [],
    sculptor.runtime.output_slots = [],
    sculptor.runtime.resource_count = 2 : i64,
    sculptor.runtime.route_input_slots = [0],
    sculptor.runtime.route_output_slots = [1],
    sculptor.runtime.workspace_size = 384 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.route_input %graph {
      sculptor.deployment.global_resource_id = 100 : i64,
      sculptor.deployment.route_id = 7 : i64,
      sculptor.runtime.byte_size = 192 : i64,
      sculptor.runtime.slot = 0 : i64,
      sculptor.runtime.temp_offset = 0 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x3x4x4xf32>>
    %output = sculptor.task_graph.route_output %graph {
      sculptor.deployment.global_resource_id = 101 : i64,
      sculptor.deployment.route_id = 8 : i64,
      sculptor.runtime.byte_size = 192 : i64,
      sculptor.runtime.slot = 1 : i64,
      sculptor.runtime.temp_offset = 192 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x3x4x4xf32>>
    %task = sculptor.task.create
      %graph, @rank4_route_task,
      domain = "digital",
      task_kind = "digital.compute",
      task_name = "rank4_route",
      source_layer = "rank4",
      source_task_ordinal = 0,
      inputs[%input],
      outputs[%output],
      deps[] {
        sculptor.deployment.global_task_id = 20 : i64,
        sculptor.runtime.core_id = 1 : i64,
        sculptor.runtime.input_slots = [0],
        sculptor.runtime.output_slots = [1],
        sculptor.runtime.result_indices = [0],
        sculptor.runtime.task_index = 0 : i64
      } : (
        !sculptor.task_graph,
        !sculptor.task_resource<tensor<1x3x4x4xf32>>,
        !sculptor.task_resource<tensor<1x3x4x4xf32>>
      ) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}
