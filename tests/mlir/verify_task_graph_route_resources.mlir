// RUN: sculptor-mlir-opt %s --split-input-file --verify-diagnostics

module {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    // expected-error @below {{requires integer attribute 'sculptor.deployment.route_id'}}
    %route = sculptor.task_graph.route_input %graph {sculptor.deployment.global_resource_id = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    // expected-error @below {{expected a route boundary to carry a ranked tensor resource}}
    %route = sculptor.task_graph.route_output %graph {sculptor.deployment.global_resource_id = 0 : i64, sculptor.deployment.route_id = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.runtime_handle>
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @consume(tensor<1xf32>)

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %route = sculptor.task_graph.route_output %graph {sculptor.deployment.global_resource_id = 0 : i64, sculptor.deployment.route_id = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    // expected-error @below {{expected route_output resources to be task outputs}}
    %task = sculptor.task.create %graph, @consume, domain = "digital", task_kind = "test", task_name = "consume", source_layer = "test", source_task_ordinal = 0, inputs[%route], outputs[], deps[] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}
