// RUN: sculptor-mlir-opt %s --sculptor-lower-to-golem="array-rows=4 array-cols=2" --sculptor-assemble-task-graph | FileCheck %s --check-prefix=MARKED
// RUN: sculptor-mlir-opt %s --sculptor-lower-to-golem="array-rows=4 array-cols=2" --sculptor-assemble-task-graph --sculptor-balance-task-graph-reductions | FileCheck %s --check-prefix=TREE
// RUN: sculptor-mlir-opt %s --sculptor-lower-to-golem="array-rows=4 array-cols=2" --sculptor-lower-golem-to-task-graph="balance-task-graph-reductions=true cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" | FileCheck %s --check-prefix=PIPELINE

module {
  func.func @forward(%arg0: tensor<1x8xf32>) -> tensor<1x3xf32> {
    %result = call @wide_reduction(%arg0)
        : (tensor<1x8xf32>) -> tensor<1x3xf32>
    return %result : tensor<1x3xf32>
  }

  func.func @wide_reduction(%arg0: tensor<1x8xf32>) -> tensor<1x3xf32> {
    %weight = arith.constant dense_resource<wide_weight> : tensor<3x8xf32>
    %result = sculptor.mvm %arg0, %weight
        : (tensor<1x8xf32>, tensor<3x8xf32>) -> tensor<1x3xf32>
    return %result : tensor<1x3xf32>
  }
}

// MARKED-LABEL: func.func private @task_wide_reduction_tile_recombine
// MARKED-SAME: sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>
// MARKED-LABEL: func.func private @generate_task_graph()
// MARKED: task_kind = "digital.tile_recombine"
// MARKED-SAME: task_name = "wide_reduction_tile_recombine"
// MARKED-SAME: inputs[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
// MARKED-SAME: sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>

// TREE-NOT: func.func private @task_wide_reduction_tile_recombine
// TREE: func.func private @__sculptor_reduce_add_
// TREE-LABEL: func.func private @generate_task_graph()
// TREE: task_name = "wide_reduction_tile_recombine.level0.lane0"
// TREE: task_name = "wide_reduction_tile_recombine.level0.lane1"
// TREE: task_name = "wide_reduction_tile_recombine.level1.lane0"
// TREE-NOT: task_name = "wide_reduction_tile_recombine"

// PIPELINE-LABEL: func.func private @generate_task_graph()
// PIPELINE-SAME: sculptor.runtime.resource_count
// PIPELINE-SAME: sculptor.schedule.task_count
// PIPELINE-SAME: sculptor.timing.critical_path_ns

{-#
  dialect_resources: {
    builtin: {
      wide_weight: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F"
    }
  }
#-}
