// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital digital-clock-ghz=1 digital-issue-width=2 digital-vector-bits-per-cycle=256" | FileCheck %s --check-prefix=TIMING
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" --sculptor-lower-scheduled-mvm-to-digital | FileCheck %s --check-prefix=LOWERING

module {
  func.func private @task_setup() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<8x8xf32>
    %array = sculptor.array.set %matrix : tensor<8x8xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<1x8xf32>, %arg1: !sculptor.logical.array) -> tensor<1x8xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x8xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x8xf32>
    return %stored : tensor<1x8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %array = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "linear", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm", source_layer = "linear", source_task_ordinal = 1, inputs[%input, %array], outputs[%output], deps[%setup] {sculptor.runtime.digital_ops = 8 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// TIMING: task_name = "mvm"
// TIMING-SAME: sculptor.timing.analog_execute_latency_ns = 0.000000e+00 : f64
// TIMING-SAME: sculptor.timing.digital_ops = 8 : i64
// TIMING-SAME: sculptor.timing.digital_replacement_ops = 128 : i64
// TIMING-SAME: sculptor.timing.intrinsic_latency_ns = 1.700000e+01 : f64

// LOWERING-LABEL: func.func private @task_mvm(%arg0: tensor<1x8xf32>)
// LOWERING-SAME: sculptor.runtime.digital_ops = 136 : i64
// LOWERING-SAME: sculptor.task_kind = "digital.matmul"
// LOWERING: linalg.matmul_transpose_b
// LOWERING-LABEL: func.func private @generate_task_graph
// LOWERING: task_kind = "digital.matmul", task_name = "mvm"
// LOWERING-SAME: sculptor.runtime.digital_ops = 136 : i64
