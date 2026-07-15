// RUN: not sculptor-mlir-opt %s --sculptor-lower-to-golem="array-rows=2 array-cols=3" --sculptor-lower-golem-to-task-graph="cores=2 arrays-per-core=2" 2>&1 | FileCheck %s
// RUN: sculptor-mlir-opt %s --sculptor-lower-to-golem="array-rows=4 array-cols=4" --sculptor-assemble-task-graph --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=snake" --sculptor-fuse-task-graph --sculptor-finalize-task-graph-resources | FileCheck %s --check-prefix=SCHEDULED --implicit-check-not=golem_analog_mvm
// RUN: sculptor-mlir-opt %s --sculptor-lower-to-golem="array-rows=4 array-cols=4" --sculptor-lower-golem-to-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 analog-mvm-latency-ns=75 analog-io-bits-per-cycle=128 analog-io-shared=false digital-clock-ghz=1.5 digital-issue-width=4 digital-vector-bits-per-cycle=512 network-link-bits-per-cycle=64 network-hop-latency-cycles=2 network-pipelined=false schedule=snake" | FileCheck %s --check-prefix=PIPELINE

module {
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = call @tile_set_load(%arg0)
        : (tensor<1x4xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }

  func.func @tile_set_load(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %weight = arith.constant dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>
    %result = sculptor.mvm %arg0, %weight
        : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<1x3xf32>
    return %result : tensor<1x3xf32>
  }
}

// CHECK: expected task graph schedule name

// PIPELINE-LABEL: func.func private @generate_task_graph
// PIPELINE-SAME: sculptor.timing.critical_path_ns
// PIPELINE-SAME: sculptor.timing.model = #sculptor.timing_model<analogMVMLatencyNs = 75 : i64
// PIPELINE-SAME: digitalClockGHz = 1.500000e+00 : f64
// PIPELINE-SAME: networkPipelined = false>
// PIPELINE-SAME: sculptor.timing.placement_aware = true
// PIPELINE: task_name = "tile_set_load_matrix_tile_0_0"
// PIPELINE-SAME: sculptor.schedule.island_id = 0 : i64
// PIPELINE: task_kind = "mixed.fused"
// PIPELINE-SAME: sculptor.schedule.island_id = 0 : i64
// PIPELINE-SAME: sculptor.timing.intrinsic_latency_ns = 76.333333333333343 : f64

// SCHEDULED-LABEL: func.func private @generate_task_graph()
// SCHEDULED-SAME: sculptor.runtime.input_slots = [0]
// SCHEDULED-SAME: sculptor.runtime.output_slots = [1]
// SCHEDULED-SAME: sculptor.runtime.resource_count = 3 : i64
// SCHEDULED-SAME: sculptor.runtime.temp_count = 1 : i64
// SCHEDULED-SAME: sculptor.schedule.graph_score = 0 : i64
// SCHEDULED-SAME: sculptor.schedule.task_count = 2 : i64
// SCHEDULED: sculptor.task_graph.input
// SCHEDULED-SAME: sculptor.runtime.slot = 0 : i64
// SCHEDULED: sculptor.task_graph.output
// SCHEDULED-SAME: sculptor.runtime.slot = 1 : i64
// SCHEDULED: sculptor.task_graph.intermediate
// SCHEDULED-SAME: sculptor.runtime.slot = 2 : i64
// SCHEDULED: task_kind = "sculptor.matrix_setup"
// SCHEDULED: task_kind = "mixed.fused"
// SCHEDULED-LABEL: func.func private @task_tile_set_load_same_core_component
// SCHEDULED: sculptor.array.load
// SCHEDULED: sculptor.array.execute
// SCHEDULED: sculptor.array.store

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_4_torch.float32: "0x040000000AD7233C0AD7A33C8FC2F53C0AD7233DCDCC4C3D8FC2753D295C8F3D0AD7A33DEC51B83DCDCCCC3DAE47E13D8FC2F53D"
    }
  }
#-}
