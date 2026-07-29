// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=analog" --sculptor-schedule-task-graph="cores=2 arrays-per-core=2 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy-timing" | FileCheck %s --check-prefix=ANALOG
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=2 arrays-per-core=2 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy-timing" | FileCheck %s --check-prefix=DIGITAL
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=2 arrays-per-core=2 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy-timing" --sculptor-lower-scheduled-mvm-to-digital | FileCheck %s --check-prefix=DIGITAL-LOWERED
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=analog" --sculptor-schedule-task-graph="cores=2 arrays-per-core=2 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy-timing" -o %t.analog.0
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=analog" --sculptor-schedule-task-graph="cores=2 arrays-per-core=2 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy-timing" -o %t.analog.1
// RUN: cmp %t.analog.0 %t.analog.1
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=2 arrays-per-core=2 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy-timing" -o %t.digital.0
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=2 arrays-per-core=2 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy-timing" -o %t.digital.1
// RUN: cmp %t.digital.0 %t.digital.1

module {
  func.func private @task_setup_0() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<8x8xf32>
    %array = sculptor.array.set %matrix : tensor<8x8xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_setup_1() -> !sculptor.logical.array {
    %matrix = arith.constant dense<2.000000e+00> : tensor<8x8xf32>
    %array = sculptor.array.set %matrix : tensor<8x8xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm_0(%arg0: tensor<1x8xf32>, %arg1: !sculptor.logical.array) -> tensor<1x8xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x8xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x8xf32>
    return %stored : tensor<1x8xf32>
  }

  func.func private @task_mvm_1(%arg0: tensor<1x8xf32>, %arg1: !sculptor.logical.array) -> tensor<1x8xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x8xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x8xf32>
    return %stored : tensor<1x8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input0 = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %input1 = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %output0 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %output1 = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %array0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup0 = sculptor.task.create %graph, @task_setup_0, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup0", source_layer = "branch0", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_setup_1, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup1", source_layer = "branch1", source_task_ordinal = 0, inputs[], outputs[%array1], deps[] {sculptor.schedule.island_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm0 = sculptor.task.create %graph, @task_mvm_0, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm0", source_layer = "branch0", source_task_ordinal = 1, inputs[%input0, %array0], outputs[%output0], deps[%setup0] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task) -> !sculptor.task
    %mvm1 = sculptor.task.create %graph, @task_mvm_1, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm1", source_layer = "branch1", source_task_ordinal = 1, inputs[%input1, %array1], outputs[%output1], deps[%setup1] {sculptor.schedule.island_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// ANALOG-LABEL: func.func private @generate_task_graph
// ANALOG-SAME: sculptor.schedule.placement_cost_mode = "analog"
// ANALOG: task_name = "mvm0"
// ANALOG-SAME: sculptor.runtime.core_id = 0 : i64
// ANALOG: task_name = "mvm1"
// ANALOG-SAME: sculptor.runtime.core_id = 0 : i64

// DIGITAL-LABEL: func.func private @generate_task_graph
// DIGITAL-SAME: sculptor.schedule.placement_cost_mode = "digital"
// DIGITAL: task_name = "mvm0"
// DIGITAL-SAME: sculptor.runtime.core_id = 0 : i64
// DIGITAL: task_name = "mvm1"
// DIGITAL-SAME: sculptor.runtime.core_id = 1 : i64

// DIGITAL-LOWERED-LABEL: func.func private @task_mvm_0
// DIGITAL-LOWERED-SAME: sculptor.runtime.core_id = 0 : i64
// DIGITAL-LOWERED-SAME: sculptor.task_kind = "digital.matmul"
// DIGITAL-LOWERED-LABEL: func.func private @task_mvm_1
// DIGITAL-LOWERED-SAME: sculptor.runtime.core_id = 1 : i64
// DIGITAL-LOWERED-SAME: sculptor.task_kind = "digital.matmul"
