// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=greedy greedy-heuristic=transfer-cost,lookahead=1,beam=1,scope=cardinal" | FileCheck %s --check-prefix=GREEDY
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=greedy-timing greedy-heuristic=transfer-cost,lookahead=1,beam=1,scope=cardinal" | FileCheck %s --check-prefix=TIMING
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=greedy greedy-heuristic=transfer-cost,lookahead=1,beam=1,scope=cardinal" --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=GREEDY-EXACT
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=greedy-timing greedy-heuristic=transfer-cost,lookahead=1,beam=1,scope=cardinal" --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=TIMING-EXACT
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=greedy greedy-heuristic=transfer-cost,lookahead=3,beam=1,scope=cardinal" | FileCheck %s --check-prefix=GREEDY-LOOKAHEAD
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=greedy greedy-heuristic=transfer-cost,lookahead=1,beam=3,scope=cardinal" | FileCheck %s --check-prefix=GREEDY-BEAM
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=greedy-timing greedy-heuristic=transfer-cost,lookahead=3,beam=1,scope=cardinal" | FileCheck %s --check-prefix=TIMING-LOOKAHEAD
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=greedy-timing greedy-heuristic=transfer-cost,lookahead=1,beam=3,scope=cardinal" | FileCheck %s --check-prefix=TIMING-BEAM
// RUN: not sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=3 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=3 schedule=greedy-timing greedy-heuristic=transfer-cost" 2>&1 | FileCheck %s --check-prefix=MISSING-TIMING

module {
  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x2xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x2xf32>
    return %stored : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output_b = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output_c = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %a_to_c = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array_a = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array_b = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array_c = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup_a = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "a_matrix", source_layer = "a", source_task_ordinal = 0, inputs[], outputs[%array_a], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup_b = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "b_matrix", source_layer = "b", source_task_ordinal = 0, inputs[], outputs[%array_b], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup_c = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "c_matrix", source_layer = "c", source_task_ordinal = 0, inputs[], outputs[%array_c], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm_a = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "a_mvm", source_layer = "a", source_task_ordinal = 1, inputs[%input, %array_a], outputs[%a_to_c], deps[%setup_a] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm_b = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "b_mvm", source_layer = "b", source_task_ordinal = 1, inputs[%input, %array_b], outputs[%output_b], deps[%setup_b] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm_c = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "c_mvm", source_layer = "c", source_task_ordinal = 1, inputs[%a_to_c, %array_c], outputs[%output_c], deps[%setup_c, %mvm_a] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// GREEDY-LABEL: func.func private @generate_task_graph
// GREEDY-SAME: sculptor.schedule.graph_score = 16 : i64
// GREEDY-SAME: sculptor.schedule.total_transfer_cost = 16 : i64
// GREEDY: task_name = "a_matrix"
// GREEDY-SAME: sculptor.runtime.core_id = 0 : i64
// GREEDY: task_name = "b_matrix"
// GREEDY-SAME: sculptor.runtime.core_id = 1 : i64
// GREEDY: task_name = "c_matrix"
// GREEDY-SAME: sculptor.runtime.core_id = 2 : i64

// TIMING-LABEL: func.func private @generate_task_graph
// TIMING-SAME: sculptor.schedule.graph_score = 8 : i64
// TIMING-SAME: sculptor.schedule.total_transfer_cost = 8 : i64
// TIMING: task_name = "a_matrix"
// TIMING-SAME: sculptor.runtime.core_id = 0 : i64
// TIMING: task_name = "b_matrix"
// TIMING-SAME: sculptor.runtime.core_id = 2 : i64
// TIMING: task_name = "c_matrix"
// TIMING-SAME: sculptor.runtime.core_id = 1 : i64

// GREEDY-EXACT-LABEL: func.func private @generate_task_graph
// GREEDY-EXACT-SAME: sculptor.schedule.graph_score = 16 : i64
// GREEDY-EXACT-SAME: sculptor.timing.critical_path_ns = 2.070000e+02 : f64
// GREEDY-EXACT-SAME: sculptor.timing.total_network_latency_ns = 3.000000e+00 : f64

// TIMING-EXACT-LABEL: func.func private @generate_task_graph
// TIMING-EXACT-SAME: sculptor.schedule.graph_score = 8 : i64
// TIMING-EXACT-SAME: sculptor.timing.critical_path_ns = 2.060000e+02 : f64
// TIMING-EXACT-SAME: sculptor.timing.total_network_latency_ns = 2.000000e+00 : f64

// GREEDY-LOOKAHEAD-LABEL: func.func private @generate_task_graph
// GREEDY-LOOKAHEAD-SAME: sculptor.schedule.graph_score = 16 : i64
// GREEDY-LOOKAHEAD-SAME: sculptor.schedule.greedy_beam_width = 1 : i64
// GREEDY-LOOKAHEAD-SAME: sculptor.schedule.greedy_lookahead = 3 : i64

// GREEDY-BEAM-LABEL: func.func private @generate_task_graph
// GREEDY-BEAM-SAME: sculptor.schedule.graph_score = 16 : i64
// GREEDY-BEAM-SAME: sculptor.schedule.greedy_beam_width = 3 : i64
// GREEDY-BEAM-SAME: sculptor.schedule.greedy_lookahead = 1 : i64

// TIMING-LOOKAHEAD-LABEL: func.func private @generate_task_graph
// TIMING-LOOKAHEAD-SAME: sculptor.schedule.graph_score = 8 : i64
// TIMING-LOOKAHEAD-SAME: sculptor.schedule.greedy_beam_width = 1 : i64
// TIMING-LOOKAHEAD-SAME: sculptor.schedule.greedy_lookahead = 3 : i64

// TIMING-BEAM-LABEL: func.func private @generate_task_graph
// TIMING-BEAM-SAME: sculptor.schedule.graph_score = 8 : i64

// MISSING-TIMING: expected pre-placement timing attribute 'sculptor.timing.placement_aware'
// MISSING-TIMING: failed to load pre-placement scheduling timing profile
// TIMING-BEAM-SAME: sculptor.schedule.greedy_beam_width = 3 : i64
// TIMING-BEAM-SAME: sculptor.schedule.greedy_lookahead = 1 : i64
