// RUN: sculptor-mlir-opt %s --sculptor-assemble-task-graph | FileCheck %s --implicit-check-not=sculptor.assembly --implicit-check-not=sculptor.runtime.byte_size --implicit-check-not=sculptor.runtime.input_slots --implicit-check-not=sculptor.runtime.output_slots --implicit-check-not=sculptor.runtime.resource_count --implicit-check-not=sculptor.runtime.slot --implicit-check-not=sculptor.runtime.task_index --implicit-check-not=sculptor.runtime.temp_ --implicit-check-not=sculptor.runtime.workspace_size

module {
  func.func @forward(%scalar: f16, %buffer: memref<2x3xf32>)
      -> (f16, memref<2x3xf32>) {
    %0:2 = call @task_producer(%scalar, %buffer)
        : (f16, memref<2x3xf32>) -> (f16, memref<2x3xf32>)
    %1:2 = call @task_consumer(%0#0, %0#1)
        : (f16, memref<2x3xf32>) -> (f16, memref<2x3xf32>)
    return %1#0, %1#1 : f16, memref<2x3xf32>
  }

  func.func private @task_producer(%scalar: f16, %buffer: memref<2x3xf32>)
      -> (f16, memref<2x3xf32>) attributes {
    sculptor.source_layer = "test",
    sculptor.source_task_ordinal = 0 : i64,
    sculptor.task_domain = "digital",
    sculptor.task_kind = "digital.producer",
    sculptor.task_name = "producer"
  } {
    return %scalar, %buffer : f16, memref<2x3xf32>
  }

  func.func private @task_consumer(%scalar: f16, %buffer: memref<2x3xf32>)
      -> (f16, memref<2x3xf32>) attributes {
    sculptor.source_layer = "test",
    sculptor.source_task_ordinal = 1 : i64,
    sculptor.task_domain = "digital",
    sculptor.task_kind = "digital.consumer",
    sculptor.task_name = "consumer"
  } {
    return %scalar, %buffer : f16, memref<2x3xf32>
  }
}

// CHECK-LABEL: func.func private @generate_task_graph()
// CHECK: %[[GRAPH:.*]] = sculptor.task_graph.create
// CHECK-NEXT: sculptor.task_graph.input %[[GRAPH]]
// CHECK-SAME: !sculptor.task_resource<f16>
// CHECK-NEXT: sculptor.task_graph.input %[[GRAPH]]
// CHECK-SAME: !sculptor.task_resource<memref<2x3xf32>>
// CHECK-NEXT: sculptor.task_graph.output %[[GRAPH]]
// CHECK-SAME: !sculptor.task_resource<f16>
// CHECK-NEXT: sculptor.task_graph.output %[[GRAPH]]
// CHECK-SAME: !sculptor.task_resource<memref<2x3xf32>>
