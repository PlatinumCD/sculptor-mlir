// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=separable-regions require-change=true" --canonicalize --cse | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=all require-change=true" -o /dev/null
// RUN: sed 's/sculptor.runtime.result_indices/sculptor.runtime.core_id = 0 : i64, sculptor.runtime.result_indices/g' %s | not sculptor-mlir-opt - --sculptor-optimize-task-graph="patterns=separable-regions" 2>&1 | FileCheck %s --check-prefix=STAGE
// RUN: sed 's/%arg0\[0, 1, 0\]/%arg0[0, 0, 0]/' %s | not sculptor-mlir-opt - --sculptor-optimize-task-graph="patterns=separable-regions require-change=true" 2>&1 | FileCheck %s --check-prefix=NOCHANGE

#identity3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func private @task_recombine(%arg0: tensor<1x4xf32>,
                               %arg1: tensor<1x4xf32>)
      -> tensor<1x2x4xf32> {
    %empty = tensor.empty() : tensor<1x2x4xf32>
    %expanded0 = tensor.expand_shape %arg0 [[0, 1], [2]]
        output_shape [1, 1, 4]
        : tensor<1x4xf32> into tensor<1x1x4xf32>
    %inserted0 = tensor.insert_slice %expanded0 into %empty[0, 0, 0]
        [1, 1, 4] [1, 1, 1]
        : tensor<1x1x4xf32> into tensor<1x2x4xf32>
    %expanded1 = tensor.expand_shape %arg1 [[0, 1], [2]]
        output_shape [1, 1, 4]
        : tensor<1x4xf32> into tensor<1x1x4xf32>
    %inserted1 = tensor.insert_slice %expanded1 into %inserted0[0, 1, 0]
        [1, 1, 4] [1, 1, 1]
        : tensor<1x1x4xf32> into tensor<1x2x4xf32>
    return %inserted1 : tensor<1x2x4xf32>
  }

  func.func private @task_pointwise(%arg0: tensor<1x2x4xf32>)
      -> tensor<1x2x4xf32> {
    %empty = tensor.empty() : tensor<1x2x4xf32>
    %result = linalg.generic {
        indexing_maps = [#identity3, #identity3],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<1x2x4xf32>)
        outs(%empty : tensor<1x2x4xf32>) {
      ^bb0(%input: f32, %output: f32):
        %squared = arith.mulf %input, %input : f32
        linalg.yield %squared : f32
      } -> tensor<1x2x4xf32>
    return %result : tensor<1x2x4xf32>
  }

  func.func private @task_slice0(%arg0: tensor<1x2x4xf32>)
      -> tensor<1x4xf32> {
    %slice = tensor.extract_slice %arg0[0, 0, 0] [1, 1, 4] [1, 1, 1]
        : tensor<1x2x4xf32> to tensor<1x1x4xf32>
    %collapsed = tensor.collapse_shape %slice [[0, 1], [2]]
        : tensor<1x1x4xf32> into tensor<1x4xf32>
    return %collapsed : tensor<1x4xf32>
  }

  func.func private @task_slice1(%arg0: tensor<1x2x4xf32>)
      -> tensor<1x4xf32> {
    %slice = tensor.extract_slice %arg0[0, 1, 0] [1, 1, 4] [1, 1, 1]
        : tensor<1x2x4xf32> to tensor<1x1x4xf32>
    %collapsed = tensor.collapse_shape %slice [[0, 1], [2]]
        : tensor<1x1x4xf32> into tensor<1x4xf32>
    return %collapsed : tensor<1x4xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input0 = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %input1 = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %combined = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2x4xf32>>
    %activated = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2x4xf32>>
    %output0 = sculptor.task_graph.output %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %output1 = sculptor.task_graph.output %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %recombine = sculptor.task.create %graph, @task_recombine,
        domain = "digital", task_kind = "digital.output_recombine",
        task_name = "recombine", source_layer = "test", source_task_ordinal = 0,
        inputs[%input0, %input1], outputs[%combined], deps[] {
          sculptor.runtime.result_indices = array<i64: 0>
        } : (!sculptor.task_graph,
             !sculptor.task_resource<tensor<1x4xf32>>,
             !sculptor.task_resource<tensor<1x4xf32>>,
             !sculptor.task_resource<tensor<1x2x4xf32>>) -> !sculptor.task
    %activation = sculptor.task.create %graph, @task_pointwise,
        domain = "digital", task_kind = "digital.activation",
        task_name = "activation", source_layer = "test", source_task_ordinal = 1,
        inputs[%combined], outputs[%activated], deps[%recombine] {
          sculptor.runtime.result_indices = array<i64: 0>
        } : (!sculptor.task_graph,
             !sculptor.task_resource<tensor<1x2x4xf32>>,
             !sculptor.task_resource<tensor<1x2x4xf32>>,
             !sculptor.task) -> !sculptor.task
    %slice0 = sculptor.task.create %graph, @task_slice0,
        domain = "digital", task_kind = "digital.token_extract",
        task_name = "slice0", source_layer = "test", source_task_ordinal = 2,
        inputs[%activated], outputs[%output0], deps[%activation] {
          sculptor.runtime.result_indices = array<i64: 0>
        } : (!sculptor.task_graph,
             !sculptor.task_resource<tensor<1x2x4xf32>>,
             !sculptor.task_resource<tensor<1x4xf32>>,
             !sculptor.task) -> !sculptor.task
    %slice1 = sculptor.task.create %graph, @task_slice1,
        domain = "digital", task_kind = "digital.token_extract",
        task_name = "slice1", source_layer = "test", source_task_ordinal = 3,
        inputs[%activated], outputs[%output1], deps[%activation] {
          sculptor.runtime.result_indices = array<i64: 0>
        } : (!sculptor.task_graph,
             !sculptor.task_resource<tensor<1x2x4xf32>>,
             !sculptor.task_resource<tensor<1x4xf32>>,
             !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @generate_task_graph
// CHECK-NOT: task_kind = "digital.output_recombine"
// CHECK: @task_slice0_separable_activation{{.*}}task_kind = "digital.activation"{{.*}}inputs[%{{.*}}]{{.*}}deps[]{{.*}}sculptor.optimization.separable_region
// CHECK: @task_slice1_separable_activation{{.*}}task_kind = "digital.activation"{{.*}}inputs[%{{.*}}]{{.*}}deps[]{{.*}}sculptor.optimization.separable_region
// CHECK-NOT: !sculptor.task_resource<tensor<1x2x4xf32>>
// CHECK: func.func private @task_slice0_separable_activation
// CHECK: arith.mulf
// CHECK: func.func private @task_slice1_separable_activation
// CHECK: arith.mulf

// STAGE: error: optimization pattern 'separable-regions' must run before task-graph scheduling
// NOCHANGE: error: no selected task-graph optimization pattern applied
