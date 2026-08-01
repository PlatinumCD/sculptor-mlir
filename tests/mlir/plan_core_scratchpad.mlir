// RUN: sculptor-mlir-opt %s --sculptor-plan-core-scratchpad="bytes=256 alignment=64" --sculptor-finalize-task-graph-resources | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-plan-core-scratchpad="local-memory=workspace" | FileCheck %s --check-prefix=NOOP

module attributes {sculptor.runtime.core_id = 3 : i64} {
  func.func private @first(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32>
  func.func private @second(%arg0: tensor<1x8xf32>) -> tensor<1x4xf32>

  // CHECK-LABEL: func.func private @generate_task_graph()
  // CHECK-SAME: sculptor.runtime.scratchpad_abi_version = 2 : i32
  // CHECK-SAME: sculptor.runtime.scratchpad_alignment = 64 : i64
  // CHECK-SAME: sculptor.runtime.scratchpad_dma_descriptors = [
  // CHECK-SAME: direction = 0 : i32
  // CHECK-SAME: direction = 1 : i32
  // CHECK-SAME: sculptor.runtime.scratchpad_required_bytes = 128 : i64
  // CHECK-SAME: sculptor.runtime.workspace_size = 0 : i64
  // NOOP-LABEL: func.func private @generate_task_graph()
  // NOOP-NOT: sculptor.runtime.scratchpad_
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    // CHECK: sculptor.task_graph.input
    // CHECK-SAME: sculptor.runtime.scratchpad_offset = 0 : i64
    // CHECK-SAME: sculptor.runtime.storage_class = "scratchpad"
    %input = sculptor.task_graph.input %graph {
      sculptor.deployment.global_resource_id = 0 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x8xf32>>
    // CHECK: sculptor.task_graph.intermediate
    // CHECK-SAME: sculptor.runtime.scratchpad_offset = 64 : i64
    // CHECK-SAME: sculptor.runtime.storage_class = "scratchpad"
    // CHECK-NOT: sculptor.runtime.temp_offset
    %middle = sculptor.task_graph.intermediate %graph {
      sculptor.deployment.global_resource_id = 1 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x8xf32>>
    // CHECK: sculptor.task_graph.output
    // CHECK-SAME: sculptor.runtime.scratchpad_offset = 0 : i64
    // CHECK-SAME: sculptor.runtime.storage_class = "scratchpad"
    %output = sculptor.task_graph.output %graph {
      sculptor.deployment.global_resource_id = 2 : i64
    } : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x4xf32>>

    %task0 = sculptor.task.create %graph, @first,
      domain = "digital", task_kind = "digital.compute",
      task_name = "first", source_layer = "scratchpad",
      source_task_ordinal = 0, inputs[%input], outputs[%middle], deps[] {
        sculptor.deployment.global_task_id = 10 : i64,
        sculptor.runtime.core_id = 3 : i64
      } : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x8xf32>>,
           !sculptor.task_resource<tensor<1x8xf32>>) -> !sculptor.task
    %task1 = sculptor.task.create %graph, @second,
      domain = "digital", task_kind = "digital.compute",
      task_name = "second", source_layer = "scratchpad",
      source_task_ordinal = 1, inputs[%middle], outputs[%output], deps[%task0] {
        sculptor.deployment.global_task_id = 11 : i64,
        sculptor.runtime.core_id = 3 : i64
      } : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x8xf32>>,
           !sculptor.task_resource<tensor<1x4xf32>>,
           !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}
