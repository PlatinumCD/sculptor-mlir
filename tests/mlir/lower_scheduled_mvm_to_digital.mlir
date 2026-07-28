// RUN: sculptor-mlir-opt %s --sculptor-lower-to-golem="array-rows=4 array-cols=4" --sculptor-assemble-task-graph --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=snake" --sculptor-lower-scheduled-mvm-to-digital --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=DIGITAL --implicit-check-not="!sculptor.logical.array" --implicit-check-not="sculptor.array."
// RUN: sculptor-mlir-opt %s --sculptor-lower-to-golem="array-rows=4 array-cols=4" --sculptor-assemble-task-graph --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=snake" --sculptor-lower-scheduled-mvm-to-digital --sculptor-fuse-task-graph --sculptor-analyze-task-graph-timing --sculptor-lower-golem-to-llvm-shims --sculptor-partition-task-graph-by-core --sculptor-extract-core-module="core-id=0" --sculptor-finalize-task-graph-resources | FileCheck %s --check-prefix=CORE --implicit-check-not="!sculptor.logical.array" --implicit-check-not="sculptor.array."
// RUN: sculptor-mlir-opt %s --sculptor-lower-to-golem="array-rows=2 array-cols=3" --sculptor-assemble-task-graph --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=4 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=2 schedule=snake" --sculptor-lower-scheduled-mvm-to-digital --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=MULTICORE --implicit-check-not="!sculptor.logical.array" --implicit-check-not="sculptor.array."
// RUN: not sculptor-mlir-opt %s --sculptor-lower-to-golem="array-rows=4 array-cols=4" --sculptor-assemble-task-graph --sculptor-build-task-graph-islands --sculptor-lower-scheduled-mvm-to-digital 2>&1 | FileCheck %s --check-prefix=UNSCHEDULED

module {
  func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = call @linear(%arg0)
        : (tensor<1x4xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }

  func.func @linear(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %weight = arith.constant dense_resource<weight> : tensor<3x4xf32>
    %result = sculptor.mvm %arg0, %weight
        : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<1x3xf32>
    return %result : tensor<1x3xf32>
  }
}

// DIGITAL-LABEL: func.func private @task_linear_matrix_tile_0_0_0()
// DIGITAL-SAME: sculptor.runtime.core_id = 0 : i64
// DIGITAL-SAME: sculptor.runtime.local_array_id = 0 : i64
// DIGITAL-SAME: sculptor.runtime.physical_array_id = 0 : i64
// DIGITAL-SAME: sculptor.task_kind = "sculptor.matrix_setup"
// DIGITAL-NEXT: return

// DIGITAL-LABEL: func.func private @task_linear_mvm_0_0_2(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32>
// DIGITAL-SAME: sculptor.runtime.core_id = 0 : i64
// DIGITAL-SAME: sculptor.runtime.digital_ops = 32 : i64
// DIGITAL-SAME: sculptor.runtime.local_array_id = 0 : i64
// DIGITAL-SAME: sculptor.runtime.physical_array_id = 0 : i64
// DIGITAL-SAME: sculptor.task_domain = "digital"
// DIGITAL-SAME: sculptor.task_kind = "digital.matmul"
// DIGITAL: arith.constant {{.*}} : tensor<4x4xf32>
// DIGITAL: linalg.fill
// DIGITAL: linalg.matmul_transpose_b
// DIGITAL-SAME: tensor<1x4xf32>, tensor<4x4xf32>
// DIGITAL: tensor.extract_slice {{.*}} [1, 3] {{.*}} tensor<1x4xf32> to tensor<1x3xf32>

// DIGITAL-LABEL: func.func private @generate_task_graph()
// DIGITAL-SAME: sculptor.schedule.graph_score = 0 : i64
// DIGITAL-SAME: sculptor.schedule.task_count = 4 : i64
// DIGITAL-SAME: sculptor.schedule.total_digital_ops = 32 : i64
// DIGITAL-SAME: sculptor.timing.critical_path_ns = 4.000000e+00 : f64
// DIGITAL: sculptor.task.create {{.*}} domain = "analog", task_kind = "sculptor.matrix_setup"
// DIGITAL-SAME: inputs[], outputs[], deps[]
// DIGITAL-SAME: sculptor.runtime.core_id = 0 : i64
// DIGITAL-SAME: sculptor.schedule.island_id = 0 : i64
// DIGITAL: sculptor.task.create {{.*}} domain = "digital", task_kind = "digital.matmul"
// DIGITAL-SAME: inputs[%{{[0-9]+}}], outputs[%{{[0-9]+}}], deps[%{{[0-9]+}}, %{{[0-9]+}}]
// DIGITAL-SAME: sculptor.runtime.core_id = 0 : i64
// DIGITAL-SAME: sculptor.runtime.digital_ops = 32 : i64
// DIGITAL-SAME: sculptor.schedule.island_id = 0 : i64
// DIGITAL-SAME: sculptor.timing.analog_execute_latency_ns = 0.000000e+00 : f64
// DIGITAL-SAME: sculptor.timing.digital_ops = 32 : i64
// DIGITAL-SAME: sculptor.timing.intrinsic_latency_ns = 4.000000e+00 : f64

// CORE-LABEL: module attributes
// CORE-SAME: sculptor.runtime.core_id = 0 : i64
// CORE-LABEL: func.func private @task_linear_matrix_tile_0_0_0()
// CORE-NEXT: return
// CORE-LABEL: func.func private @task_linear_same_core_component
// CORE: linalg.matmul_transpose_b
// CORE-LABEL: func.func private @generate_task_graph()
// CORE-SAME: sculptor.runtime.input_slots = [0]
// CORE-SAME: sculptor.runtime.output_slots = [1]
// CORE-SAME: sculptor.schedule.task_count = 2 : i64
// CORE: task_kind = "sculptor.matrix_setup"
// CORE-SAME: sculptor.runtime.task_index = 0 : i64
// CORE: task_kind = "mixed.fused"
// CORE-SAME: sculptor.runtime.digital_ops = 32 : i64
// CORE-SAME: sculptor.runtime.task_index = 1 : i64

// MULTICORE-LABEL: func.func private @task_linear_mvm_0_0_6
// MULTICORE-SAME: sculptor.runtime.core_id = 0 : i64
// MULTICORE-SAME: sculptor.runtime.digital_ops = 12 : i64
// MULTICORE-SAME: sculptor.task_kind = "digital.matmul"
// MULTICORE-LABEL: func.func private @task_linear_mvm_1_0_7
// MULTICORE-SAME: sculptor.runtime.core_id = 3 : i64
// MULTICORE-SAME: sculptor.runtime.digital_ops = 12 : i64
// MULTICORE-SAME: sculptor.task_kind = "digital.matmul"
// MULTICORE-LABEL: func.func private @task_linear_mvm_0_1_8
// MULTICORE-SAME: sculptor.runtime.core_id = 1 : i64
// MULTICORE-SAME: sculptor.runtime.digital_ops = 12 : i64
// MULTICORE-SAME: sculptor.task_kind = "digital.matmul"
// MULTICORE-LABEL: func.func private @task_linear_mvm_1_1_9
// MULTICORE-SAME: sculptor.runtime.core_id = 2 : i64
// MULTICORE-SAME: sculptor.runtime.digital_ops = 12 : i64
// MULTICORE-SAME: sculptor.task_kind = "digital.matmul"
// MULTICORE-LABEL: func.func private @generate_task_graph()
// MULTICORE-SAME: sculptor.schedule.graph_score = 68 : i64
// MULTICORE-SAME: sculptor.schedule.inter_core_transfer_bytes = 40 : i64
// MULTICORE-SAME: sculptor.schedule.task_count = 11 : i64
// MULTICORE-SAME: sculptor.schedule.total_digital_ops = 51 : i64

// UNSCHEDULED: expected scheduled core placement
// UNSCHEDULED-SAME: run --sculptor-schedule-task-graph before --sculptor-lower-scheduled-mvm-to-digital

{-#
  dialect_resources: {
    builtin: {
      weight: "0x040000000000803F0000004000004040000080400000A0400000C0400000E0400000004100001041000020410000304100004041"
    }
  }
#-}
