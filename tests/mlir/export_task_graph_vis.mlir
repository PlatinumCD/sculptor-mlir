// RUN: sculptor-mlir-opt --help | FileCheck %s --check-prefix=HELP
// RUN: sculptor-mlir-opt %s --sculptor-export-task-graph-vis="output=%t.dot format=dot" > /dev/null
// RUN: FileCheck %s --check-prefix=DOT --input-file=%t.dot
// RUN: sculptor-mlir-opt %s --sculptor-export-task-graph-vis="output=%t.graphml format=graphml" > /dev/null
// RUN: FileCheck %s --check-prefix=GRAPHML --input-file=%t.graphml

module {
  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }
  func.func private @task_vector(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>
  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x2xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x2xf32>
    return %stored : tensor<1x2xf32>
  }
  func.func private @task_bias(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>

  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.schedule.analog_arrays = [0 : i64, 1 : i64],
    sculptor.schedule.arrays_per_core = 1 : i64,
    sculptor.schedule.core_transfer_bytes = [0 : i64, 8 : i64, 8 : i64, 0 : i64],
    sculptor.schedule.core_transfer_cost = [0 : i64, 8 : i64, 8 : i64, 0 : i64],
    sculptor.schedule.dependency_count = 3 : i64,
    sculptor.schedule.inter_core_transfer_bytes = 16 : i64,
    sculptor.schedule.logical_array_to_analog_array = [0 : i64],
    sculptor.schedule.mesh_cols = 2 : i64,
    sculptor.schedule.mesh_rows = 1 : i64,
    sculptor.schedule.num_analog_arrays = 2 : i64,
    sculptor.schedule.num_cores = 2 : i64,
    sculptor.schedule.num_logical_arrays = 1 : i64,
    sculptor.schedule.task_count = 4 : i64,
    sculptor.schedule.topology = "mesh",
    sculptor.schedule.total_digital_ops = 5 : i64,
    sculptor.schedule.total_transfer_cost = 16 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 1 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 0 : i64, sculptor.runtime.slot = 2 : i64, sculptor.runtime.temp_index = 0 : i64, sculptor.runtime.temp_offset = 0 : i64, sculptor.schedule.logical_array_index = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %tile = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 3 : i64, sculptor.runtime.temp_index = 1 : i64, sculptor.runtime.temp_offset = 0 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %mvm_out = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 8 : i64, sculptor.runtime.slot = 4 : i64, sculptor.runtime.temp_index = 2 : i64, sculptor.runtime.temp_offset = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>

    %setup = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "linear_matrix_tile_0_0", source_layer = "linear_0", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.task_index = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %vector = sculptor.task.create %graph, @task_vector, domain = "digital", task_kind = "digital.vector_tile", task_name = "linear_vector_tile_0", source_layer = "linear_0", source_task_ordinal = 1, inputs[%input], outputs[%tile], deps[] {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.digital_ops = 3 : i64, sculptor.runtime.task_index = 1 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>) -> !sculptor.task
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "linear_mvm_0_0", source_layer = "linear_0", source_task_ordinal = 2, inputs[%tile, %array], outputs[%mvm_out], deps[%setup, %vector] {sculptor.runtime.core_id = 0 : i64, sculptor.runtime.digital_ops = 0 : i64, sculptor.runtime.physical_array_id = 0 : i64, sculptor.runtime.task_index = 2 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    %bias = sculptor.task.create %graph, @task_bias, domain = "digital", task_kind = "digital.bias_add", task_name = "linear_bias_add", source_layer = "linear_0", source_task_ordinal = 3, inputs[%mvm_out], outputs[%output], deps[%mvm] {sculptor.runtime.core_id = 1 : i64, sculptor.runtime.digital_ops = 2 : i64, sculptor.runtime.task_index = 3 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task

    return %graph : !sculptor.task_graph
  }
}

// HELP: --sculptor-export-task-graph-vis

// DOT: digraph analog_task_graph
// DOT: subgraph cluster_graph_0
// DOT: label="@generate_task_graph"
// DOT: subgraph cluster_0_linear_0
// DOT: label="linear_0"
// DOT: label="sculptor.matrix_setup\nlinear_matrix_tile_0_0"
// DOT-SAME: fillcolor="#fef3c7"
// DOT: label="digital.vector_tile\nlinear_vector_tile_0"
// DOT-SAME: fillcolor="#dbeafe"
// DOT: label="sculptor.mvm\nlinear_mvm_0_0"
// DOT: label="digital.bias_add\nlinear_bias_add"
// DOT: task_0_0 -> task_0_2 [style="dotted", color="#b45309"]
// DOT: task_0_1 -> task_0_2 [style="solid", color="#334155"]
// DOT: task_0_2 -> task_0_3 [style="solid", color="#334155"]

// GRAPHML: <?xml version="1.0" encoding="UTF-8"?>
// GRAPHML: <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
// GRAPHML: <key id="task_index" for="node" attr.name="task_index" attr.type="long"/>
// GRAPHML: <key id="edge_kind" for="edge" attr.name="edge_kind" attr.type="string"/>
// GRAPHML: <key id="transfer_cost" for="edge" attr.name="transfer_cost" attr.type="long"/>
// GRAPHML: <graph id="generate_task_graph" edgedefault="directed">
// GRAPHML: <node id="task_0_0">
// GRAPHML: <data key="task_index">0</data>
// GRAPHML: <data key="task_name">linear_matrix_tile_0_0</data>
// GRAPHML: <data key="task_kind">sculptor.matrix_setup</data>
// GRAPHML: <data key="core_id">0</data>
// GRAPHML: <data key="physical_array_id">0</data>
// GRAPHML: <data key="local_array_id">0</data>
// GRAPHML: <data key="analog_ops">1</data>
// GRAPHML: <node id="task_0_2">
// GRAPHML: <data key="task_index">2</data>
// GRAPHML: <data key="task_name">linear_mvm_0_0</data>
// GRAPHML: <data key="task_kind">sculptor.mvm</data>
// GRAPHML: <data key="analog_ops">3</data>
// GRAPHML: <edge id="edge_0_control_0" source="task_0_0" target="task_0_2">
// GRAPHML: <data key="edge_kind">control</data>
// GRAPHML: <data key="logical_array_dependency">true</data>
// GRAPHML: <edge id="edge_0_data_0" source="task_0_1" target="task_0_2">
// GRAPHML: <data key="edge_kind">data</data>
// GRAPHML: <data key="producer_task">1</data>
// GRAPHML: <data key="consumer_task">2</data>
// GRAPHML: <data key="resource_id">3</data>
// GRAPHML: <data key="byte_size">8</data>
// GRAPHML: <data key="source_core">1</data>
// GRAPHML: <data key="destination_core">0</data>
// GRAPHML: <data key="mesh_distance">1</data>
// GRAPHML: <data key="transfer_cost">8</data>
// GRAPHML: <data key="inter_core">true</data>
// GRAPHML: </graphml>
