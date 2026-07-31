// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=6 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=3 schedule=snake" | FileCheck %s --check-prefix=SCHEDULED
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=6 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=3 schedule=snake" --sculptor-fuse-task-graph --sculptor-analyze-task-graph-timing --sculptor-finalize-task-graph-resources | FileCheck %s --check-prefix=FINAL
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=6 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=3 schedule=random random-seed=0" -o /dev/null
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=6 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=3 schedule=greedy greedy-heuristic=transfer-cost,lookahead=2,beam=1,scope=cardinal" -o /dev/null
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=6 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=3 schedule=greedy-timing greedy-heuristic=transfer-cost,lookahead=2,beam=1,scope=cardinal" -o /dev/null
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=6 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=3 schedule=greedy-timing greedy-heuristic=transfer-cost,spatial-link-pressure,lookahead=1,beam=16,scope=diagonal" | FileCheck %s --check-prefix=TIMING-RERANK
// RUN: sculptor-mlir-opt %s --sculptor-balance-task-graph-reductions --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=6 arrays-per-core=1 topology=mesh mesh-rows=2 mesh-cols=3 schedule=annealing annealing-initial-schedule=snake annealing-move-set=basic annealing-steps-per-temperature=2 annealing-initial-temperature=2 annealing-final-temperature=1 annealing-cooling-rate=0.5" -o /dev/null

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

  func.func private @star_add(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>, %arg2: tensor<1x2xf32>, %arg3: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @consume(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @bridge(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %arg0 : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %leaf0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %leaf1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %leaf2 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %leaf3 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %reduced = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %bridged = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array2 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array3 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup0 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix0", source_layer = "reduction", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix1", source_layer = "reduction", source_task_ordinal = 1, inputs[], outputs[%array1], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup2 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix2", source_layer = "reduction", source_task_ordinal = 2, inputs[], outputs[%array2], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup3 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix3", source_layer = "reduction", source_task_ordinal = 3, inputs[], outputs[%array3], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm0 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm0", source_layer = "reduction", source_task_ordinal = 4, inputs[%input, %array0], outputs[%leaf0], deps[%setup0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm1 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm1", source_layer = "reduction", source_task_ordinal = 5, inputs[%input, %array1], outputs[%leaf1], deps[%setup1] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm2 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm2", source_layer = "reduction", source_task_ordinal = 6, inputs[%input, %array2], outputs[%leaf2], deps[%setup2] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm3 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm3", source_layer = "reduction", source_task_ordinal = 7, inputs[%input, %array3], outputs[%leaf3], deps[%setup3] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %star = sculptor.task.create %graph, @star_add, domain = "digital", task_kind = "digital.reduction", task_name = "star", source_layer = "reduction", source_task_ordinal = 8, inputs[%leaf0, %leaf1, %leaf2, %leaf3], outputs[%reduced], deps[%mvm0, %mvm1, %mvm2, %mvm3] {sculptor.runtime.result_indices = array<i64: 0>, sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task, !sculptor.task, !sculptor.task) -> !sculptor.task
    %bridge = sculptor.task.create %graph, @bridge, domain = "digital", task_kind = "digital.bridge", task_name = "bridge", source_layer = "bridge", source_task_ordinal = 9, inputs[%reduced], outputs[%bridged], deps[%star] {sculptor.runtime.result_indices = array<i64: 0>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %sink = sculptor.task.create %graph, @consume, domain = "digital", task_kind = "digital.consume", task_name = "sink", source_layer = "reduction", source_task_ordinal = 10, inputs[%bridged], outputs[%output], deps[%bridge] {sculptor.runtime.result_indices = array<i64: 0>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// SCHEDULED-NOT: func.func private @star_add
// SCHEDULED-LABEL: func.func private @generate_task_graph()
// SCHEDULED-SAME: sculptor.schedule.task_count = 13 : i64
// SCHEDULED-SAME: sculptor.task_graph.generation = 1 : i64
// SCHEDULED-NOT: sculptor.timing.
// SCHEDULED-DAG: task_name = "star.level0.lane0"{{.*}}sculptor.runtime.core_id = 1 : i64{{.*}}sculptor.schedule.island_id = 8 : i64
// SCHEDULED-DAG: task_name = "star.level0.lane1"{{.*}}sculptor.runtime.core_id = 2 : i64{{.*}}sculptor.schedule.island_id = 9 : i64
// SCHEDULED-DAG: task_name = "star.level1.root"{{.*}}sculptor.runtime.core_id = 2 : i64{{.*}}sculptor.schedule.island_id = 10 : i64
// SCHEDULED-NOT: task_name = "star"
// SCHEDULED: task_name = "bridge"{{.*}}sculptor.runtime.core_id = 5 : i64{{.*}}sculptor.schedule.island_id = 3 : i64
// SCHEDULED: task_name = "sink"{{.*}}sculptor.schedule.island_id = 3 : i64

// The spatial proxy's first complete candidate loses once reduction placement,
// core serialization, and routed communication are replayed exactly.
// TIMING-RERANK-LABEL: func.func private @generate_task_graph()
// TIMING-RERANK-SAME: sculptor.schedule.predicted_makespan_ns = 2.730000e+02 : f64
// TIMING-RERANK-SAME: sculptor.schedule.timing_rerank_candidate_count = 16 : i64
// TIMING-RERANK-SAME: sculptor.schedule.timing_rerank_selected_proxy_rank = 1 : i64
// TIMING-RERANK-SAME: sculptor.timing.critical_path_ns = 2.730000e+02 : f64

// FINAL-NOT: func.func private @star_add
// FINAL-LABEL: func.func private @generate_task_graph()
// FINAL-SAME: sculptor.runtime.resource_count
// FINAL-SAME: sculptor.schedule.graph_score
// FINAL: return %{{.*}} : !sculptor.task_graph
