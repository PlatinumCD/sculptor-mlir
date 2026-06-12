// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=1 arrays-per-core=2 schedule=simple-budget" 2>&1 | FileCheck %s --check-prefix=ERR

module {
  func.func private @task_matrix_0() -> !sculptor.logical.array
  func.func private @task_matrix_1() -> !sculptor.logical.array

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %array0 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>

    %setup0 = sculptor.task.create %graph, @task_matrix_0, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix_0", source_layer = "linear_0", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_matrix_1, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "matrix_1", source_layer = "linear_1", source_task_ordinal = 0, inputs[], outputs[%array1], deps[%setup0] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}

// ERR: expected sculptor.matrix_setup task to have no task dependencies before matrix setup front-loading
