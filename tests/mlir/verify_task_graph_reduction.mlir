// RUN: sculptor-mlir-opt %s --split-input-file --verify-diagnostics

module {
  func.func private @reduce(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    // expected-error @+1 {{expected 'sculptor.task.reduction' to be a #sculptor.task_reduction attribute}}
    %task = sculptor.task.create %graph, @reduce, domain = "digital", task_kind = "digital.reduction", task_name = "wrong_attr", source_layer = "reduction", source_task_ordinal = 0, inputs[%input, %input], outputs[%output], deps[] {sculptor.task.reduction = "add"} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @reduce(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    // expected-error @+1 {{expected task reduction metadata to permit reassociation}}
    %task = sculptor.task.create %graph, @reduce, domain = "digital", task_kind = "digital.reduction", task_name = "ordered", source_layer = "reduction", source_task_ordinal = 0, inputs[%input, %input], outputs[%output], deps[] {sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = false>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @reduce(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    // expected-error @+1 {{expected a marked task reduction to use the digital domain}}
    %task = sculptor.task.create %graph, @reduce, domain = "analog", task_kind = "digital.reduction", task_name = "analog", source_layer = "reduction", source_task_ordinal = 0, inputs[%input, %input], outputs[%output], deps[] {sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @reduce(%arg0: tensor<8xf32>, %arg1: tensor<4xf32>) -> tensor<8xf32> {
    return %arg0 : tensor<8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input8 = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    %input4 = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<4xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<8xf32>>
    // expected-error @+1 {{expected every task reduction input to match the output tensor type}}
    %task = sculptor.task.create %graph, @reduce, domain = "digital", task_kind = "digital.reduction", task_name = "mismatch", source_layer = "reduction", source_task_ordinal = 0, inputs[%input8, %input4], outputs[%output], deps[] {sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<8xf32>>, !sculptor.task_resource<tensor<4xf32>>, !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @reduce(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    return %arg0 : tensor<?xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<?xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<?xf32>>
    // expected-error @+1 {{expected a marked task reduction to use a non-scalar, statically shaped floating-point tensor resource}}
    %task = sculptor.task.create %graph, @reduce, domain = "digital", task_kind = "digital.reduction", task_name = "dynamic", source_layer = "reduction", source_task_ordinal = 0, inputs[%input, %input], outputs[%output], deps[] {sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<?xf32>>, !sculptor.task_resource<tensor<?xf32>>, !sculptor.task_resource<tensor<?xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @reduce(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    return %arg0 : tensor<f32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<f32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<f32>>
    // expected-error @+1 {{expected a marked task reduction to use a non-scalar, statically shaped floating-point tensor resource}}
    %task = sculptor.task.create %graph, @reduce, domain = "digital", task_kind = "digital.reduction", task_name = "scalar", source_layer = "reduction", source_task_ordinal = 0, inputs[%input, %input], outputs[%output], deps[] {sculptor.task.reduction = #sculptor.task_reduction<kind = add, reassociate = true>} : (!sculptor.task_graph, !sculptor.task_resource<tensor<f32>>, !sculptor.task_resource<tensor<f32>>, !sculptor.task_resource<tensor<f32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}
