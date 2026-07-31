// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=elementwise-slices require-change=true" --canonicalize --cse | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-optimize-task-graph="patterns=elementwise-slices" | FileCheck %s --check-prefix=OVERLAP

#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func private @task_elementwise_slices(
      %arg0: tensor<1x1x6xf32>, %arg1: tensor<1x1x6xf32>,
      %arg2: tensor<1x1x6xf32>, %arg3: tensor<1x1x6xf32>)
      -> (tensor<1x1x6xf32>, tensor<1x1x6xf32>,
          tensor<1x1x6xf32>, tensor<1x1x6xf32>) {
    %combined = tensor.empty() : tensor<1x4x6xf32>
    %insert0 = tensor.insert_slice %arg0 into %combined[0, 0, 0] [1, 1, 6] [1, 1, 1]
        : tensor<1x1x6xf32> into tensor<1x4x6xf32>
    %insert1 = tensor.insert_slice %arg1 into %insert0[0, 1, 0] [1, 1, 6] [1, 1, 1]
        : tensor<1x1x6xf32> into tensor<1x4x6xf32>
    %insert2 = tensor.insert_slice %arg2 into %insert1[0, 2, 0] [1, 1, 6] [1, 1, 1]
        : tensor<1x1x6xf32> into tensor<1x4x6xf32>
    %insert3 = tensor.insert_slice %arg3 into %insert2[0, 3, 0] [1, 1, 6] [1, 1, 1]
        : tensor<1x1x6xf32> into tensor<1x4x6xf32>
    %init = tensor.empty() : tensor<1x4x6xf32>
    %activated = linalg.generic {
        indexing_maps = [#identity, #identity],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%insert3 : tensor<1x4x6xf32>)
        outs(%init : tensor<1x4x6xf32>) {
      ^bb0(%input: f32, %output: f32):
        %one = arith.constant 1.0 : f32
        %result = arith.addf %input, %one : f32
        linalg.yield %result : f32
      } -> tensor<1x4x6xf32>
    %slice0 = tensor.extract_slice %activated[0, 0, 0] [1, 1, 6] [1, 1, 1]
        : tensor<1x4x6xf32> to tensor<1x1x6xf32>
    %slice1 = tensor.extract_slice %activated[0, 1, 0] [1, 1, 6] [1, 1, 1]
        : tensor<1x4x6xf32> to tensor<1x1x6xf32>
    %slice2 = tensor.extract_slice %activated[0, 2, 0] [1, 1, 6] [1, 1, 1]
        : tensor<1x4x6xf32> to tensor<1x1x6xf32>
    %slice3 = tensor.extract_slice %activated[0, 3, 0] [1, 1, 6] [1, 1, 1]
        : tensor<1x4x6xf32> to tensor<1x1x6xf32>
    return %slice0, %slice1, %slice2, %slice3
        : tensor<1x1x6xf32>, tensor<1x1x6xf32>,
          tensor<1x1x6xf32>, tensor<1x1x6xf32>
  }

  func.func private @overlapping_slices(
      %arg0: tensor<1x1x6xf32>, %arg1: tensor<1x1x6xf32>)
      -> (tensor<1x1x6xf32>, tensor<1x1x6xf32>) {
    %combined = tensor.empty() : tensor<1x1x8xf32>
    %insert0 = tensor.insert_slice %arg0 into %combined[0, 0, 0] [1, 1, 6] [1, 1, 1]
        : tensor<1x1x6xf32> into tensor<1x1x8xf32>
    %insert1 = tensor.insert_slice %arg1 into %insert0[0, 0, 2] [1, 1, 6] [1, 1, 1]
        : tensor<1x1x6xf32> into tensor<1x1x8xf32>
    %init = tensor.empty() : tensor<1x1x8xf32>
    %activated = linalg.generic {
        indexing_maps = [#identity, #identity],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%insert1 : tensor<1x1x8xf32>)
        outs(%init : tensor<1x1x8xf32>) {
      ^bb0(%input: f32, %output: f32):
        %one = arith.constant 1.0 : f32
        %result = arith.addf %input, %one : f32
        linalg.yield %result : f32
      } -> tensor<1x1x8xf32>
    %slice0 = tensor.extract_slice %activated[0, 0, 0] [1, 1, 6] [1, 1, 1]
        : tensor<1x1x8xf32> to tensor<1x1x6xf32>
    %slice1 = tensor.extract_slice %activated[0, 0, 2] [1, 1, 6] [1, 1, 1]
        : tensor<1x1x8xf32> to tensor<1x1x6xf32>
    return %slice0, %slice1 : tensor<1x1x6xf32>, tensor<1x1x6xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input0 = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x6xf32>>
    %input1 = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x6xf32>>
    %input2 = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x6xf32>>
    %input3 = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x6xf32>>
    %output0 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x6xf32>>
    %output1 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x6xf32>>
    %output2 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x6xf32>>
    %output3 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x6xf32>>
    %task = sculptor.task.create %graph, @task_elementwise_slices,
        domain = "digital", task_kind = "digital.elementwise",
        task_name = "elementwise_slices", source_layer = "test",
        source_task_ordinal = 0,
        inputs[%input0, %input1, %input2, %input3],
        outputs[%output0, %output1, %output2, %output3], deps[] {
          sculptor.runtime.core_id = 0 : i64,
          sculptor.runtime.result_indices = array<i64: 0, 1, 2, 3>
        } : (!sculptor.task_graph,
             !sculptor.task_resource<tensor<1x1x6xf32>>,
             !sculptor.task_resource<tensor<1x1x6xf32>>,
             !sculptor.task_resource<tensor<1x1x6xf32>>,
             !sculptor.task_resource<tensor<1x1x6xf32>>,
             !sculptor.task_resource<tensor<1x1x6xf32>>,
             !sculptor.task_resource<tensor<1x1x6xf32>>,
             !sculptor.task_resource<tensor<1x1x6xf32>>,
             !sculptor.task_resource<tensor<1x1x6xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @task_elementwise_slices(
// CHECK-NOT: tensor<1x4x6xf32>
// CHECK-NOT: tensor.insert_slice
// CHECK-COUNT-4: linalg.generic
// CHECK-SAME: sculptor.optimization.elementwise_slice =
// CHECK: return

// OVERLAP-LABEL: func.func private @overlapping_slices(
// OVERLAP: tensor.insert_slice
// OVERLAP: tensor.insert_slice
// OVERLAP: linalg.generic
