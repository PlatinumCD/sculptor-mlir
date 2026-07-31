// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions="reduction-width=4" | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=STAR
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions="reduction-width=4" --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=TREE
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions="reduction-width=4" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" | FileCheck %s --check-prefix=PLACED
// RUN: not sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions="reduction-width=4" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=snake" 2>&1 | FileCheck %s --check-prefix=INSUFFICIENT-CORES
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions="reduction-width=4" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" --sculptor-export-task-graph-vis="output=%t.graphml format=graphml" -o /dev/null
// RUN: FileCheck %s --check-prefix=GRAPHML --input-file=%t.graphml
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions="reduction-width=4" --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" --sculptor-finalize-task-graph-resources --sculptor-export-task-graph-sim-model="output=%t.json" -o /dev/null
// RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json

module {
  func.func private @star32(
      %arg0: tensor<256xf32>,
      %arg1: tensor<256xf32>,
      %arg2: tensor<256xf32>,
      %arg3: tensor<256xf32>,
      %arg4: tensor<256xf32>,
      %arg5: tensor<256xf32>,
      %arg6: tensor<256xf32>,
      %arg7: tensor<256xf32>,
      %arg8: tensor<256xf32>,
      %arg9: tensor<256xf32>,
      %arg10: tensor<256xf32>,
      %arg11: tensor<256xf32>,
      %arg12: tensor<256xf32>,
      %arg13: tensor<256xf32>,
      %arg14: tensor<256xf32>,
      %arg15: tensor<256xf32>,
      %arg16: tensor<256xf32>,
      %arg17: tensor<256xf32>,
      %arg18: tensor<256xf32>,
      %arg19: tensor<256xf32>,
      %arg20: tensor<256xf32>,
      %arg21: tensor<256xf32>,
      %arg22: tensor<256xf32>,
      %arg23: tensor<256xf32>,
      %arg24: tensor<256xf32>,
      %arg25: tensor<256xf32>,
      %arg26: tensor<256xf32>,
      %arg27: tensor<256xf32>,
      %arg28: tensor<256xf32>,
      %arg29: tensor<256xf32>,
      %arg30: tensor<256xf32>,
      %arg31: tensor<256xf32>) -> tensor<256xf32> {
    return %arg0 : tensor<256xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<256xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<256xf32>>
    %wide = sculptor.task.create %graph, @star32, domain = "digital", task_kind = "digital.reduction", task_name = "wide", source_layer = "reduction", source_task_ordinal = 0, inputs[%input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input, %input], outputs[%output], deps[] {sculptor.runtime.digital_ops = 7936 : i64, sculptor.runtime.result_indices = array<i64: 0>, sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>, !sculptor.task_resource<tensor<256xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK-NOT: func.func private @star32
// CHECK-LABEL: func.func private @generate_task_graph()
// CHECK-COUNT-4: task_name = "wide.level0.lane{{[0-3]}}"
// CHECK: task_name = "wide.level1.root"
// CHECK-NOT: task_name = "wide.level"

// STAR-LABEL: func.func private @generate_task_graph()
// STAR-SAME: sculptor.timing.critical_path_ns = 3.984000e+03 : f64
// STAR-SAME: sculptor.timing.execution_depth = 1 : i64
// STAR-SAME: sculptor.timing.task_count = 1 : i64
// STAR: sculptor.timing.intrinsic_latency_ns = 3.984000e+03 : f64
// STAR-SAME: sculptor.workload.digital_ops = 7936 : i64

// TREE-LABEL: func.func private @generate_task_graph()
// TREE-SAME: sculptor.timing.critical_path_ns = 3.912000e+03 : f64
// TREE-SAME: sculptor.timing.execution_depth = 2 : i64
// TREE-SAME: sculptor.timing.task_count = 5 : i64
// TREE: task_name = "wide.level1.root"
// TREE-SAME: sculptor.timing.critical_path_remaining_ns = 1.439000e+03 : f64
// TREE-SAME: sculptor.timing.intrinsic_latency_ns = 1.439000e+03 : f64
// TREE-SAME: sculptor.workload.digital_ops = 768 : i64

// PLACED: task_name = "wide.level0.lane0"
// PLACED-SAME: sculptor.runtime.core_id = 0 : i64
// PLACED: task_name = "wide.level0.lane1"
// PLACED-SAME: sculptor.runtime.core_id = 1 : i64
// PLACED: task_name = "wide.level0.lane2"
// PLACED-SAME: sculptor.runtime.core_id = 2 : i64
// PLACED: task_name = "wide.level0.lane3"
// PLACED-SAME: sculptor.runtime.core_id = 3 : i64
// PLACED: task_name = "wide.level1.root"
// PLACED-SAME: sculptor.runtime.core_id = 0 : i64

// GRAPHML: <key id="reduction_tree_id" for="node" attr.name="reduction_tree_id" attr.type="long"/>
// GRAPHML: <key id="reduction_lane" for="node" attr.name="reduction_lane" attr.type="long"/>
// GRAPHML: <key id="reduction_width" for="node" attr.name="reduction_width" attr.type="long"/>
// GRAPHML: <data key="task_name">wide.level0.lane0</data>
// GRAPHML: <data key="island_id">0</data>
// GRAPHML: <data key="core_id">0</data>
// GRAPHML: <data key="reduction_tree_id">0</data>
// GRAPHML: <data key="reduction_level">0</data>
// GRAPHML: <data key="reduction_lane">0</data>
// GRAPHML: <data key="reduction_width">4</data>

// JSON: "name": "wide.level0.lane0"
// JSON: "core_id": 0
// JSON: "island_id": 0
// JSON: "reduction_tree_id": 0
// JSON: "reduction_level": 0
// JSON: "reduction_lane": 0
// JSON: "reduction_width": 4

// INSUFFICIENT-CORES: error: could not find a distinct core for a reduction lane
