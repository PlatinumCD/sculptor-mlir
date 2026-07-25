// RUN: sculptor-mlir-opt %s --sculptor-finalize-task-graph-resources | FileCheck %s

module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 4 : i64
} {
  func.func private @route_task(
      %arg0: tensor<1x3xf32>,
      %arg1: tensor<1x2xf32>)
      -> (tensor<1x4xf32>, tensor<1x1xf32>)

  // CHECK-LABEL: func.func private @generate_task_graph()
  // CHECK-SAME: sculptor.runtime.input_slots = []
  // CHECK-SAME: sculptor.runtime.output_slots = []
  // CHECK-SAME: sculptor.runtime.resource_count = 4 : i64
  // Route IDs 2 and 9 map to slots 1 and 0.
  // CHECK-SAME: sculptor.runtime.route_input_slots = [1, 0]
  // Route IDs 3 and 8 map to slots 3 and 2.
  // CHECK-SAME: sculptor.runtime.route_output_slots = [3, 2]
  // CHECK-SAME: sculptor.runtime.temp_count = 0 : i64
  // CHECK-SAME: sculptor.runtime.workspace_size = 60 : i64
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph

    // CHECK: sculptor.task_graph.route_input
    // CHECK-SAME: sculptor.deployment.global_resource_id = 109 : i64
    // CHECK-SAME: sculptor.deployment.route_id = 9 : i64
    // CHECK-SAME: sculptor.runtime.byte_size = 12 : i64
    // CHECK-SAME: sculptor.runtime.slot = 0 : i64
    // CHECK-SAME: sculptor.runtime.temp_offset = 48 : i64
    %in9 = sculptor.task_graph.route_input %graph {
      sculptor.deployment.global_resource_id = 109 : i64,
      sculptor.deployment.route_id = 9 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x3xf32>>

    // CHECK: sculptor.task_graph.route_input
    // CHECK-SAME: sculptor.deployment.global_resource_id = 102 : i64
    // CHECK-SAME: sculptor.deployment.route_id = 2 : i64
    // CHECK-SAME: sculptor.runtime.byte_size = 8 : i64
    // CHECK-SAME: sculptor.runtime.slot = 1 : i64
    // CHECK-SAME: sculptor.runtime.temp_offset = 0 : i64
    %in2 = sculptor.task_graph.route_input %graph {
      sculptor.deployment.global_resource_id = 102 : i64,
      sculptor.deployment.route_id = 2 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x2xf32>>

    // CHECK: sculptor.task_graph.route_output
    // CHECK-SAME: sculptor.deployment.global_resource_id = 108 : i64
    // CHECK-SAME: sculptor.deployment.route_id = 8 : i64
    // CHECK-SAME: sculptor.runtime.byte_size = 16 : i64
    // CHECK-SAME: sculptor.runtime.slot = 2 : i64
    // CHECK-SAME: sculptor.runtime.temp_offset = 32 : i64
    %out8 = sculptor.task_graph.route_output %graph {
      sculptor.deployment.global_resource_id = 108 : i64,
      sculptor.deployment.route_id = 8 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x4xf32>>

    // CHECK: sculptor.task_graph.route_output
    // CHECK-SAME: sculptor.deployment.global_resource_id = 103 : i64
    // CHECK-SAME: sculptor.deployment.route_id = 3 : i64
    // CHECK-SAME: sculptor.runtime.byte_size = 4 : i64
    // CHECK-SAME: sculptor.runtime.slot = 3 : i64
    // CHECK-SAME: sculptor.runtime.temp_offset = 16 : i64
    %out3 = sculptor.task_graph.route_output %graph {
      sculptor.deployment.global_resource_id = 103 : i64,
      sculptor.deployment.route_id = 3 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x1xf32>>

    // CHECK: sculptor.task.create
    // CHECK-SAME: inputs[
    // CHECK-SAME: outputs[
    // CHECK-SAME: deps[]
    // CHECK-SAME: sculptor.deployment.global_task_id = 27 : i64
    // CHECK-SAME: sculptor.runtime.input_slots = [0, 1]
    // CHECK-SAME: sculptor.runtime.output_slots = [2, 3]
    // CHECK-SAME: sculptor.runtime.task_index = 0 : i64
    %task = sculptor.task.create %graph, @route_task,
        domain = "digital",
        task_kind = "digital.route",
        task_name = "route_task",
        source_layer = "route_fixture",
        source_task_ordinal = 0,
        inputs[%in9, %in2],
        outputs[%out8, %out3],
        deps[] {
          sculptor.deployment.global_task_id = 27 : i64,
          sculptor.runtime.core_id = 4 : i64
        } : (!sculptor.task_graph,
             !sculptor.task_resource<tensor<1x3xf32>>,
             !sculptor.task_resource<tensor<1x2xf32>>,
             !sculptor.task_resource<tensor<1x4xf32>>,
             !sculptor.task_resource<tensor<1x1xf32>>)
            -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}
