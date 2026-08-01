// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands | FileCheck %s --check-prefix=LEGACY
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands="digital-assignment=legacy" | FileCheck %s --check-prefix=LEGACY
// RUN: sculptor-mlir-opt %s --sculptor-build-task-graph-islands="digital-assignment=multi-terminal-balanced" | FileCheck %s --check-prefix=BALANCED
// RUN: not sculptor-mlir-opt %s --sculptor-build-task-graph-islands="digital-assignment=unknown" 2>&1 | FileCheck %s --check-prefix=INVALID

module {
  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<1x1xf32>
    %array = sculptor.array.set %matrix : tensor<1x1xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<1x1xf32>, %arg1: !sculptor.logical.array) -> tensor<1x1xf32> {
    return %arg0 : tensor<1x1xf32>
  }

  func.func private @task_digital(%arg0: tensor<1x1xf32>) -> tensor<1x1xf32> {
    return %arg0 : tensor<1x1xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 4 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 4 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %mvm0_out = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 4 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %digital0_out = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 4 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %digital1_out = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 4 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %digital2_out = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 4 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %unused = sculptor.task_graph.intermediate %graph {sculptor.runtime.byte_size = 4 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1xf32>>
    %array0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array2 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>

    %setup0 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup0", source_layer = "layer", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup1", source_layer = "layer", source_task_ordinal = 1, inputs[], outputs[%array1], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup2 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup2", source_layer = "layer", source_task_ordinal = 2, inputs[], outputs[%array2], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm0 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm0", source_layer = "layer", source_task_ordinal = 3, inputs[%input, %array0], outputs[%mvm0_out], deps[%setup0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task) -> !sculptor.task
    %digital0 = sculptor.task.create %graph, @task_digital, domain = "digital", task_kind = "digital.compute", task_name = "digital0", source_layer = "layer", source_task_ordinal = 4, inputs[%mvm0_out], outputs[%digital0_out], deps[%mvm0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task) -> !sculptor.task
    %digital1 = sculptor.task.create %graph, @task_digital, domain = "digital", task_kind = "digital.compute", task_name = "digital1", source_layer = "layer", source_task_ordinal = 5, inputs[%digital0_out], outputs[%digital1_out], deps[%digital0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task) -> !sculptor.task
    %digital2 = sculptor.task.create %graph, @task_digital, domain = "digital", task_kind = "digital.compute", task_name = "digital2", source_layer = "layer", source_task_ordinal = 6, inputs[%digital1_out], outputs[%digital2_out], deps[%digital1] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task) -> !sculptor.task
    %mvm1 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm1", source_layer = "layer", source_task_ordinal = 7, inputs[%digital2_out, %array1], outputs[%output], deps[%setup1, %digital2] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %mvm2 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm2", source_layer = "layer", source_task_ordinal = 8, inputs[%digital1_out, %array2], outputs[%unused], deps[%setup2, %digital1] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x1xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// LEGACY: sculptor.schedule.island_assignment_policy = "legacy"
// LEGACY: task_name = "digital0"
// LEGACY-SAME: sculptor.schedule.island_id = 0 : i64
// LEGACY: task_name = "digital1"
// LEGACY-SAME: sculptor.schedule.island_id = 0 : i64
// LEGACY: task_name = "digital2"
// LEGACY-SAME: sculptor.schedule.island_id = 0 : i64

// BALANCED: sculptor.schedule.island_assignment_policy = "multi-terminal-balanced"
// BALANCED: task_name = "digital0"
// BALANCED-SAME: sculptor.schedule.island_id = 0 : i64
// BALANCED: task_name = "digital1"
// BALANCED-SAME: sculptor.schedule.island_id = 2 : i64
// BALANCED: task_name = "digital2"
// BALANCED-SAME: sculptor.schedule.island_id = 1 : i64

// INVALID: error: unknown digital island assignment policy 'unknown'; expected legacy or multi-terminal-balanced
