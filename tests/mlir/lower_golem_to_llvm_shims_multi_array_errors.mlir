// RUN: not sculptor-mlir-opt %s --sculptor-lower-golem-to-llvm-shims 2>&1 | FileCheck %s

module {
  func.func private @setup_0() -> !sculptor.logical.array attributes {sculptor.runtime.local_array_id = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64}
  func.func private @setup_1() -> !sculptor.logical.array attributes {sculptor.runtime.local_array_id = 1 : i64, sculptor.runtime.physical_array_id = 1 : i64}
  func.func private @multi_array(
      %input: tensor<1x4xf32>,
      %array0: !sculptor.logical.array,
      %array1: !sculptor.logical.array) -> tensor<1x4xf32>

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %output = sculptor.task_graph.output %graph
        : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x4xf32>>
    %array0 = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.intermediate %graph
        : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>

    %setup0 = sculptor.task.create %graph, @setup_0,
        domain = "analog", task_kind = "sculptor.matrix_setup",
        task_name = "setup_0", source_layer = "test",
        source_task_ordinal = 0,
        inputs[], outputs[%array0], deps[]
        {
          sculptor.runtime.local_array_id = 0 : i64,
          sculptor.runtime.physical_array_id = 0 : i64
        }
        : (!sculptor.task_graph,
           !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @setup_1,
        domain = "analog", task_kind = "sculptor.matrix_setup",
        task_name = "setup_1", source_layer = "test",
        source_task_ordinal = 1,
        inputs[], outputs[%array1], deps[]
        {
          sculptor.runtime.local_array_id = 1 : i64,
          sculptor.runtime.physical_array_id = 1 : i64
        }
        : (!sculptor.task_graph,
           !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task

    // CHECK: duplicate array binding for input index 1
    %compute = sculptor.task.create %graph, @multi_array,
        domain = "digital", task_kind = "mixed.streaming_conv_mvm",
        task_name = "multi_array", source_layer = "test",
        source_task_ordinal = 2,
        inputs[%input, %array0, %array1], outputs[%output],
        deps[%setup0, %setup1]
        {
          sculptor.runtime.array_bindings = [
            {
              input_index = 1 : i64,
              local_array_id = 0 : i64,
              physical_array_id = 0 : i64
            },
            {
              input_index = 1 : i64,
              local_array_id = 1 : i64,
              physical_array_id = 1 : i64
            }
          ]
        }
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x4xf32>>,
           !sculptor.task_resource<!sculptor.logical.array>,
           !sculptor.task_resource<!sculptor.logical.array>,
           !sculptor.task_resource<tensor<1x4xf32>>,
           !sculptor.task,
           !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}
