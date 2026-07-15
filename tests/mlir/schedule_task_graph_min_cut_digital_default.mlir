// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands | FileCheck %s --check-prefix=ISLANDS
// RUN: not sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=snake" 2>&1 | FileCheck %s --check-prefix=MISSING-ISLANDS
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=snake" | FileCheck %s --check-prefix=SCHEDULED
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=snake" --sculptor-fuse-task-graph --sculptor-finalize-task-graph-resources | FileCheck %s --check-prefix=DEFAULT
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-analyze-task-graph-timing --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=random" --sculptor-fuse-task-graph --sculptor-finalize-task-graph-resources | FileCheck %s --check-prefix=DEFAULT

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

  func.func private @task_down(%arg0: tensor<1x2xf32>) -> tensor<1x1xf32> {
    %empty = tensor.empty() : tensor<1x1xf32>
    return %empty : tensor<1x1xf32>
  }

  func.func private @task_up(%arg0: tensor<1x1xf32>) -> tensor<1x2xf32> {
    %empty = tensor.empty() : tensor<1x2xf32>
    return %empty : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 100 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 100 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %tmp0 = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 100 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %tmp1 = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 4 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %tmp2 = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 100 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup0 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer_matrix0", source_layer = "layer", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer_matrix1", source_layer = "layer", source_task_ordinal = 1, inputs[], outputs[%array1], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm0 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "layer_mvm0", source_layer = "layer", source_task_ordinal = 2, inputs[%input, %array0], outputs[%tmp0], deps[%setup0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %digital0 = sculptor.task.create %graph, @task_down, domain = "digital", task_kind = "digital.compute", task_name = "layer_digital0", source_layer = "layer", source_task_ordinal = 3, inputs[%tmp0], outputs[%tmp1], deps[%mvm0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task) -> !sculptor.task
    %digital1 = sculptor.task.create %graph, @task_up, domain = "digital", task_kind = "digital.compute", task_name = "layer_digital1", source_layer = "layer", source_task_ordinal = 4, inputs[%tmp1], outputs[%tmp2], deps[%digital0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm1 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "layer_mvm1", source_layer = "layer", source_task_ordinal = 5, inputs[%tmp2, %array1], outputs[%output], deps[%setup1, %digital1] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// ISLANDS-LABEL: func.func private @generate_task_graph
// ISLANDS: task_name = "layer_matrix0"
// ISLANDS-SAME: sculptor.schedule.island_id = 0 : i64
// ISLANDS: task_name = "layer_mvm0"
// ISLANDS-SAME: sculptor.schedule.island_id = 0 : i64
// ISLANDS: task_name = "layer_digital0"
// ISLANDS-SAME: sculptor.schedule.island_id = 0 : i64
// ISLANDS: task_name = "layer_digital1"
// ISLANDS-SAME: sculptor.schedule.island_id = 1 : i64
// ISLANDS: task_name = "layer_mvm1"
// ISLANDS-SAME: sculptor.schedule.island_id = 1 : i64

// MISSING-ISLANDS: error: expected logical island ID; run --sculptor-build-task-graph-islands before --sculptor-schedule-task-graph

// SCHEDULED-LABEL: func.func private @generate_task_graph
// SCHEDULED-SAME: sculptor.schedule.task_count = 6 : i64
// SCHEDULED-NOT: sculptor.runtime.input_slots
// SCHEDULED: task_name = "layer_mvm0"
// SCHEDULED-NOT: task_kind = "mixed.fused"
// SCHEDULED: task_name = "layer_mvm1"

// DEFAULT-LABEL: func.func private @generate_task_graph
// DEFAULT-SAME: sculptor.schedule.graph_score = 4 : i64
// DEFAULT-SAME: sculptor.schedule.task_count = 4 : i64
// DEFAULT: sculptor.task.create {{.*}} task_name = "layer_matrix0"
// DEFAULT-SAME: sculptor.runtime.core_id = [[SRC_CORE:[0-9]+]] : i64
// DEFAULT-SAME: sculptor.schedule.island_id = 0 : i64
// DEFAULT: sculptor.task.create {{.*}} task_kind = "mixed.fused"
// DEFAULT-SAME: task_name = "layer_same_core_component_core_{{[0-9]+}}_2_3"
// DEFAULT-SAME: sculptor.runtime.core_id = [[SRC_CORE]] : i64
// DEFAULT-SAME: sculptor.schedule.island_id = 0 : i64
// DEFAULT: sculptor.task.create {{.*}} task_name = "layer_matrix1"
// DEFAULT-SAME: sculptor.runtime.core_id = [[DST_CORE:[0-9]+]] : i64
// DEFAULT-SAME: sculptor.schedule.island_id = 1 : i64
// DEFAULT: sculptor.task.create {{.*}} task_kind = "mixed.fused"
// DEFAULT-SAME: task_name = "layer_same_core_component_core_{{[0-9]+}}_4_5"
// DEFAULT-SAME: sculptor.runtime.core_id = [[DST_CORE]] : i64
// DEFAULT-SAME: sculptor.schedule.island_id = 1 : i64
