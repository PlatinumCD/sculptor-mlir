// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-emit-golem-tile-abi -verify-diagnostics

// Missing isolated-core identity.
// expected-error@+1 {{expected required attribute 'sculptor.runtime.core_id'}}
module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = []
} {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}

// -----

// A deployment with nested cores was not extracted.
// expected-error@+1 {{requires one extracted core and cannot package nested core modules}}
module attributes {sculptor.runtime.core_id = 0 : i64} {
  module @core_0 {
  }
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}

// -----

// A local task must retain its stable deployment identity.
module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  llvm.func @setup()
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    // expected-error@+1 {{expected required attribute 'sculptor.deployment.global_task_id'}}
    %setup = sculptor.task.create %graph, @setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "linear", source_task_ordinal = 0, inputs[], outputs[], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.input_slots = [], sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.output_slots = [], sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

// A local task index is required independently of the global task ID.
module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  llvm.func @setup()
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    // expected-error@+1 {{expected required attribute 'sculptor.runtime.task_index'}}
    %setup = sculptor.task.create %graph, @setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "linear", source_task_ordinal = 0, inputs[], outputs[], deps[] {sculptor.deployment.global_task_id = 0 : i64, sculptor.runtime.core_id = 0 : i64, sculptor.runtime.input_slots = [], sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.output_slots = [], sculptor.runtime.physical_array_id = 0 : i64} : (!sculptor.task_graph) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

// Global task IDs must remain unique in the isolated core.
module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  llvm.func @setup0()
  llvm.func @setup1()
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %setup0 = sculptor.task.create %graph, @setup0, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup0", source_layer = "linear0", source_task_ordinal = 0, inputs[], outputs[], deps[] {sculptor.deployment.global_task_id = 4 : i64, sculptor.runtime.core_id = 0 : i64, sculptor.runtime.input_slots = [], sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.output_slots = [], sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph) -> !sculptor.task
    // expected-error@+1 {{duplicate sculptor.deployment.global_task_id 4}}
    %setup1 = sculptor.task.create %graph, @setup1, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup1", source_layer = "linear1", source_task_ordinal = 0, inputs[], outputs[], deps[] {sculptor.deployment.global_task_id = 4 : i64, sculptor.runtime.core_id = 0 : i64, sculptor.runtime.input_slots = [], sculptor.runtime.local_array_id = 1 : i64, sculptor.runtime.output_slots = [], sculptor.runtime.physical_array_id = 1 : i64, sculptor.runtime.task_index = 1 : i64} : (!sculptor.task_graph) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

// Standard function lowering must have converted every task implementation.
module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  // expected-error@+1 {{expected every task implementation to be an llvm.func}}
  func.func private @not_lowered()
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}

// -----

// A task's declarative callee must resolve to an LLVM function.
module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    // expected-error@+1 {{task callee 'missing' is missing or is not an llvm.func}}
    %setup = sculptor.task.create %graph, @missing, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "linear", source_task_ordinal = 0, inputs[], outputs[], deps[] {sculptor.deployment.global_task_id = 0 : i64, sculptor.runtime.core_id = 0 : i64, sculptor.runtime.input_slots = [], sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.output_slots = [], sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

// Matrix setup is a zero-input, zero-output boot ABI.
module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [{
    global_resource_id = 0 : i64,
    input_index = 0 : i64,
    owner_core = 0 : i64
  }],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  llvm.func @bad_setup(!llvm.ptr, !llvm.ptr, i64, i64, i64)
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.deployment.global_resource_id = 0 : i64, sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    // expected-error@+1 {{matrix setup task must have zero tensor inputs and zero tensor outputs}}
    %setup = sculptor.task.create %graph, @bad_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "linear", source_task_ordinal = 0, inputs[%input], outputs[], deps[] {sculptor.deployment.global_task_id = 0 : i64, sculptor.runtime.core_id = 0 : i64, sculptor.runtime.input_slots = [0], sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.output_slots = [], sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

// Boot cannot wait for a packet-dispatched task.
module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  llvm.func @dispatch()
  llvm.func @setup()
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %dispatch = sculptor.task.create %graph, @dispatch, domain = "digital", task_kind = "digital.compute", task_name = "dispatch", source_layer = "linear", source_task_ordinal = 0, inputs[], outputs[], deps[] {sculptor.deployment.global_task_id = 0 : i64, sculptor.runtime.core_id = 0 : i64, sculptor.runtime.input_slots = [], sculptor.runtime.output_slots = [], sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph) -> !sculptor.task
    // expected-error@+1 {{boot task with global task ID 1 depends on dispatch task 0}}
    %setup = sculptor.task.create %graph, @setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "linear", source_task_ordinal = 1, inputs[], outputs[], deps[%dispatch] {sculptor.deployment.global_task_id = 1 : i64, sculptor.runtime.core_id = 0 : i64, sculptor.runtime.input_slots = [], sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.output_slots = [], sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.task_index = 1 : i64} : (!sculptor.task_graph, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

// Rank three is intentionally outside the first Golem Tensor ABI contract.
module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [{
    global_resource_id = 0 : i64,
    input_index = 0 : i64,
    owner_core = 0 : i64
  }],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    // expected-error@+1 {{supports only statically shaped f32 tensors with ranks zero through two}}
    %input = sculptor.task_graph.input %graph {sculptor.deployment.global_resource_id = 0 : i64, sculptor.runtime.byte_size = 4 : i64, sculptor.runtime.slot = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x1x1xf32>>
    return %graph : !sculptor.task_graph
  }
}

// -----

// Route records and route-boundary resources must agree exactly.
// expected-error@+1 {{route metadata does not match local route resource slot metadata for route 0}}
module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [
    #sculptor.deployment_route<
      id = 0 : i64,
      sourceCore = 0 : i64,
      sourceTask = 1 : i64,
      sourceOutput = 0 : i64,
      destinationCore = 1 : i64,
      destinationTask = 3 : i64,
      destinationInput = 0 : i64,
      resourceId = 2 : i64,
      byteSize = 4 : i64
    >
  ],
  sculptor.runtime.core_id = 0 : i64
} {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %route = sculptor.task_graph.route_output %graph {sculptor.deployment.global_resource_id = 2 : i64, sculptor.deployment.route_id = 1 : i64, sculptor.runtime.byte_size = 4 : i64, sculptor.runtime.slot = 0 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1xf32>>
    return %graph : !sculptor.task_graph
  }
}

// -----

// The packaging boundary rejects any unrelated Sculptor operation left behind.
module attributes {
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [],
  sculptor.runtime.core_id = 0 : i64
} {
  llvm.func @stray_sculptor_op() {
    // expected-error@+1 {{Sculptor operation remains after Golem tile ABI packaging}}
    %stray = sculptor.task_graph.create : !sculptor.task_graph
    llvm.return
  }
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}
