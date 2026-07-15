// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=random" | FileCheck %s

module {
  func.func private @task_producer(%scalar: f16, %buffer: memref<2x3xf32>)
      -> (f16, memref<2x3xf32>) {
    return %scalar, %buffer : f16, memref<2x3xf32>
  }

  func.func private @task_consumer(%scalar: f16, %buffer: memref<2x3xf32>)
      -> (f16, memref<2x3xf32>) {
    return %scalar, %buffer : f16, memref<2x3xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input_scalar = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<f16>
    %input_buffer = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<memref<2x3xf32>>
    %output_scalar = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<f16>
    %output_buffer = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<memref<2x3xf32>>
    %scalar = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 999 : i64} : !sculptor.task_graph -> !sculptor.task_resource<f16>
    %buffer = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<memref<2x3xf32>>
    %producer = sculptor.task.create %graph, @task_producer, domain = "digital", task_kind = "digital.producer", task_name = "producer", source_layer = "test", source_task_ordinal = 0, inputs[%input_scalar, %input_buffer], outputs[%scalar, %buffer], deps[] {sculptor.runtime.core_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<f16>, !sculptor.task_resource<memref<2x3xf32>>, !sculptor.task_resource<f16>, !sculptor.task_resource<memref<2x3xf32>>) -> !sculptor.task
    %consumer = sculptor.task.create %graph, @task_consumer, domain = "digital", task_kind = "digital.consumer", task_name = "consumer", source_layer = "test", source_task_ordinal = 1, inputs[%scalar, %buffer], outputs[%output_scalar, %output_buffer], deps[%producer] {sculptor.runtime.core_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<f16>, !sculptor.task_resource<memref<2x3xf32>>, !sculptor.task_resource<f16>, !sculptor.task_resource<memref<2x3xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph()
// CHECK-SAME: sculptor.schedule.graph_score = 26 : i64
// CHECK-SAME: sculptor.schedule.inter_core_transfer_bytes = 26 : i64
// CHECK-SAME: sculptor.schedule.total_transfer_cost = 26 : i64
