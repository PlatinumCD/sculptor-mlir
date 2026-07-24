// RUN: sculptor-mlir-opt %s --sculptor-partition-task-graph-by-core --split-input-file --verify-diagnostics

module attributes {sculptor.schedule.num_cores = 1 : i64} {
  func.func private @task() -> tensor<1xf32>

  // expected-error @below {{sculptor-partition-task-graph-by-core must run before --sculptor-finalize-task-graph-resources; found stale runtime layout attribute 'sculptor.runtime.resource_count'}}
  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.runtime.resource_count = 1 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %task = sculptor.task.create %graph, @task, domain = "digital", task_kind = "test", task_name = "task", source_layer = "test", source_task_ordinal = 0, inputs[], outputs[%output], deps[] {sculptor.runtime.core_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module attributes {sculptor.schedule.num_cores = 2 : i64} {
  func.func private @producer(tensor<1xf32>)
  func.func private @consumer() -> tensor<1xf32>

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    %producer = sculptor.task.create %graph, @producer, domain = "digital", task_kind = "test", task_name = "producer", source_layer = "test", source_task_ordinal = 0, inputs[%input], outputs[], deps[] {sculptor.runtime.core_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1xf32>>) -> !sculptor.task
    // expected-error @below {{cannot partition cross-core control-only dependency from global task 0 on core 0 to global task 1 on core 1: no routed tensor edge can satisfy the dependency}}
    %consumer = sculptor.task.create %graph, @consumer, domain = "digital", task_kind = "test", task_name = "consumer", source_layer = "test", source_task_ordinal = 1, inputs[], outputs[%output], deps[%producer] {sculptor.runtime.core_id = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}
