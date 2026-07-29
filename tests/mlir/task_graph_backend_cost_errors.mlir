// RUN: sculptor-mlir-opt %s --split-input-file --verify-diagnostics --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital"

module {
  func.func private @task_setup() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<8x8xf32>
    %array = sculptor.array.set %matrix : tensor<8x8xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<1x8xf32>, %arg1: !sculptor.logical.array) -> tensor<1x8xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x8xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x8xf32>
    return %stored : tensor<1x8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %array0 = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array_missing = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "missing_setup", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // expected-error @below {{expected logical-array input to be produced by a matrix-setup task}}
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm", source_layer = "missing_setup", source_task_ordinal = 1, inputs[%input, %array_missing], outputs[%output], deps[%setup] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @task_setup() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<8x8xf32>
    %array = sculptor.array.set %matrix : tensor<8x8xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_digital(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
    return %arg0 : tensor<1x8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %array = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "invalid", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // expected-error @below {{expected non-negative digital operation count}}
    %task = sculptor.task.create %graph, @task_digital, domain = "digital", task_kind = "digital.compute", task_name = "digital", source_layer = "invalid", source_task_ordinal = 1, inputs[%input], outputs[%output], deps[%setup] {sculptor.runtime.digital_ops = -1 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @task_setup() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<8x8xf32>
    %array = sculptor.array.set %matrix : tensor<8x8xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<?x8xf32>, %arg1: !sculptor.logical.array) -> tensor<?x8xf32> {
    // expected-error @below {{expected vector tensor shape to be static}}
    sculptor.array.load %arg0, %arg1 : tensor<?x8xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<?x8xf32>
    return %stored : tensor<?x8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<?x8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<?x8xf32>>
    %array = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "dynamic", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm", source_layer = "dynamic", source_task_ordinal = 1, inputs[%input, %array], outputs[%output], deps[%setup] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<?x8xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<?x8xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @task_setup() -> !sculptor.logical.array {
    %matrix = tensor.empty() : tensor<3000000000x3000000000xf32>
    %array = sculptor.array.set %matrix : tensor<3000000000x3000000000xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<1x3000000000xf32>, %arg1: !sculptor.logical.array) -> tensor<1x3000000000xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x3000000000xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x3000000000xf32>
    return %stored : tensor<1x3000000000xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3000000000xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x3000000000xf32>>
    %array = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "overflow", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // expected-error @below {{digital matmul operation count overflow}}
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm", source_layer = "overflow", source_task_ordinal = 1, inputs[%input, %array], outputs[%output], deps[%setup] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x3000000000xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x3000000000xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @task_setup() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<8x8xf32>
    %array = sculptor.array.set %matrix : tensor<8x8xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_stream(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
    return %arg0 : tensor<1x8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %array = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "conv", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    // expected-error @below {{digital MVM cost mode requires an explicit 'sculptor.runtime.digital_replacement_ops' count}}
    %stream = sculptor.task.create %graph, @task_stream, domain = "digital", task_kind = "mixed.streaming_conv_mvm", task_name = "stream", source_layer = "conv", source_task_ordinal = 1, inputs[%input], outputs[%output], deps[%setup] {sculptor.runtime.analog_execution_counts = [1], sculptor.runtime.analog_load_bytes = 32 : i64, sculptor.runtime.analog_store_bytes = 32 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}
