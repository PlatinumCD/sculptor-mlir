// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing 2>&1 | FileCheck %s

module {
  func.func private @task_vector_loop(%input: tensor<8xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0.0 : f32
    %initial = vector.splat %zero : vector<4xf32>
    %unused = scf.for %index = %c0 to %c2 step %c1
        iter_args(%accumulator = %initial) -> vector<4xf32> {
      %sum = arith.addf %accumulator, %accumulator : vector<4xf32>
      scf.yield %sum : vector<4xf32>
    }
    return %input : tensor<8xf32>
  }

  func.func private @task_tensor_slice(%input: tensor<8xf32>) -> tensor<4xf32> {
    %slice = tensor.extract_slice %input[0] [4] [1]
        : tensor<8xf32> to tensor<4xf32>
    return %slice : tensor<4xf32>
  }

  func.func private @task_fallback(%input: tensor<8xf32>) -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %unused = tensor.dim %input, %c0 : tensor<8xf32>
    return %input : tensor<8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<8xf32>>
    %vector_output = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<8xf32>>
    %slice_output = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<4xf32>>
    %fallback_output = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<8xf32>>

    %vector_task = sculptor.task.create %graph, @task_vector_loop,
        domain = "digital", task_kind = "digital.vector_loop",
        task_name = "vector_loop", source_layer = "cost",
        source_task_ordinal = 0, inputs[%input], outputs[%vector_output], deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<8xf32>>,
           !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    %slice_task = sculptor.task.create %graph, @task_tensor_slice,
        domain = "digital", task_kind = "digital.tensor_slice",
        task_name = "tensor_slice", source_layer = "cost",
        source_task_ordinal = 1, inputs[%input], outputs[%slice_output], deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<8xf32>>,
           !sculptor.task_resource<tensor<4xf32>>) -> !sculptor.task
    %fallback_task = sculptor.task.create %graph, @task_fallback,
        domain = "digital", task_kind = "digital.fallback",
        task_name = "fallback", source_layer = "cost",
        source_task_ordinal = 2, inputs[%input], outputs[%fallback_output], deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<8xf32>>,
           !sculptor.task_resource<tensor<8xf32>>) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK: warning: task body requires conservative fallback cost model 'golem-qemu-v1' revision 1

// CHECK-LABEL: task_name = "vector_loop"
// CHECK-SAME: sculptor.timing.control_instruction_estimate = 14 : i64
// CHECK-SAME: sculptor.timing.cost_confidence = "high"
// CHECK-SAME: sculptor.timing.cost_source = "static-analysis"
// CHECK-SAME: sculptor.timing.load_instruction_estimate = 8 : i64
// CHECK-SAME: sculptor.timing.predicted_cpu_cycles = 3.300000e+01 : f64
// CHECK-SAME: sculptor.timing.runtime_dispatch_cycles = 8 : i64
// CHECK-SAME: sculptor.timing.store_instruction_estimate = 8 : i64
// CHECK-SAME: sculptor.timing.task_entry_cycles = 4 : i64
// CHECK-SAME: sculptor.timing.task_exit_cycles = 4 : i64
// CHECK-SAME: sculptor.timing.vector_instruction_estimate = 3 : i64
// CHECK-SAME: sculptor.workload.local_bytes_read = 32 : i64
// CHECK-SAME: sculptor.workload.local_bytes_written = 32 : i64
// CHECK-SAME: sculptor.workload.loop_iterations = 2 : i64

// CHECK-LABEL: task_name = "tensor_slice"
// CHECK-SAME: sculptor.timing.cost_confidence = "high"
// CHECK-SAME: sculptor.timing.cost_source = "static-analysis"
// CHECK-SAME: sculptor.timing.load_instruction_estimate = 12 : i64
// CHECK-SAME: sculptor.timing.predicted_cpu_cycles = 3.300000e+01 : f64
// CHECK-SAME: sculptor.timing.scalar_instruction_estimate = 0 : i64
// CHECK-SAME: sculptor.timing.store_instruction_estimate = 12 : i64
// CHECK-SAME: sculptor.workload.local_bytes_read = 48 : i64
// CHECK-SAME: sculptor.workload.local_bytes_written = 48 : i64

// CHECK-LABEL: task_name = "fallback"
// CHECK-SAME: sculptor.timing.cost_confidence = "medium"
// CHECK-SAME: sculptor.timing.cost_source = "calibrated-fallback"
// CHECK-SAME: sculptor.timing.predicted_cpu_cycles = 4.000000e+01 : f64
