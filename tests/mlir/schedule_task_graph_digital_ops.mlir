// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 schedule=simple-budget" | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func private @task_matrix() -> !sculptor.logical.array

  func.func private @task_copy(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %extracted = tensor.extract_slice %arg0[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> to tensor<1x4xf32>
    return %extracted : tensor<1x4xf32>
  }

  func.func private @task_bias(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %cst = arith.constant dense<1.000000e+00> : tensor<4xf32>
    %expanded = tensor.expand_shape %cst [[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
    %empty = tensor.empty() : tensor<1x4xf32>
    %result = linalg.add ins(%arg0, %expanded : tensor<1x4xf32>, tensor<1x4xf32>) outs(%empty : tensor<1x4xf32>) -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>
  }

  func.func private @task_sigmoid(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %empty = tensor.empty() : tensor<1x4xf32>
    %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x4xf32>) outs(%empty : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %neg = arith.negf %in : f32
      %exp = math.exp %neg : f32
      %denom = arith.addf %exp, %cst : f32
      %sigmoid = arith.divf %cst, %denom : f32
      linalg.yield %sigmoid : f32
    } -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>
  }

  // CHECK: func.func private @generate_task_graph()
  // CHECK-SAME: sculptor.schedule.total_digital_ops = 20 : i64
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %array = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %copy_out = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %bias_out = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>

    // CHECK: sculptor.task.create {{.*}} @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup"{{.*}}sculptor.runtime.digital_ops = 0 : i64
    %setup = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix", source_layer = "layer0", source_task_ordinal = 0, inputs[], outputs[%array], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // CHECK: sculptor.task.create {{.*}} @task_copy, domain = "digital", task_kind = "digital.vector_tile"{{.*}}sculptor.runtime.digital_ops = 0 : i64
    %copy = sculptor.task.create %graph, @task_copy, domain = "digital", task_kind = "digital.vector_tile", task_name = "copy", source_layer = "layer0", source_task_ordinal = 1, inputs[%input], outputs[%copy_out], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>) -> !sculptor.task
    // CHECK: sculptor.task.create {{.*}} @task_bias, domain = "digital", task_kind = "digital.bias_add"{{.*}}sculptor.runtime.digital_ops = 4 : i64
    %bias = sculptor.task.create %graph, @task_bias, domain = "digital", task_kind = "digital.bias_add", task_name = "bias", source_layer = "layer0", source_task_ordinal = 2, inputs[%copy_out], outputs[%bias_out], deps[%copy] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task
    // CHECK: sculptor.task.create {{.*}} @task_sigmoid, domain = "digital", task_kind = "digital.activation"{{.*}}sculptor.runtime.digital_ops = 16 : i64
    %sigmoid = sculptor.task.create %graph, @task_sigmoid, domain = "digital", task_kind = "digital.activation", task_name = "sigmoid", source_layer = "forward", source_task_ordinal = 0, inputs[%bias_out], outputs[%output], deps[%bias] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task_resource<tensor<1x4xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}
