// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=random" | FileCheck %s --check-prefix=RANDOM
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=snake" | FileCheck %s --check-prefix=SNAKE
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy greedy-lookahead=1 greedy-candidate-scope=cardinal" | FileCheck %s --check-prefix=GREEDY

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
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %tmp = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array0 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup0 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer0_matrix", source_layer = "layer0", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer1_matrix", source_layer = "layer1", source_task_ordinal = 0, inputs[], outputs[%array1], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm0 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "layer0_mvm", source_layer = "layer0", source_task_ordinal = 1, inputs[%input, %array0], outputs[%tmp], deps[%setup0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm1 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "layer1_mvm", source_layer = "layer1", source_task_ordinal = 1, inputs[%tmp, %array1], outputs[%output], deps[%setup1, %mvm0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// RANDOM-LABEL: func.func private @generate_task_graph
// RANDOM-SAME: sculptor.schedule.graph_score = 8 : i64
// RANDOM-SAME: sculptor.schedule.total_transfer_cost = 8 : i64

// SNAKE-LABEL: func.func private @generate_task_graph
// SNAKE-SAME: sculptor.schedule.graph_score = 8 : i64
// SNAKE-SAME: sculptor.schedule.total_transfer_cost = 8 : i64

// GREEDY-LABEL: func.func private @generate_task_graph
// GREEDY-SAME: sculptor.schedule.graph_score = 8 : i64
// GREEDY-SAME: sculptor.schedule.greedy_candidate_scope = "cardinal"
// GREEDY-SAME: sculptor.schedule.total_transfer_cost = 8 : i64
