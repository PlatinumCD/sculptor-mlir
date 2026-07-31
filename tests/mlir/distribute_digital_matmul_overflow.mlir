// RUN: not sculptor-mlir-opt %s --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=2 min-ops-per-shard=1" 2>&1 | FileCheck %s

module {
  func.func private @task_overflow_matmul(
      %lhs: tensor<3037000500x3037000500xf32>,
      %rhs: tensor<3037000500x2xf32>)
      -> tensor<3037000500x2xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<3037000500x2xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<3037000500x2xf32>)
        -> tensor<3037000500x2xf32>
    %result = linalg.matmul
        ins(%lhs, %rhs
            : tensor<3037000500x3037000500xf32>,
              tensor<3037000500x2xf32>)
        outs(%init : tensor<3037000500x2xf32>)
        -> tensor<3037000500x2xf32>
    return %result : tensor<3037000500x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %lhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<
            tensor<3037000500x3037000500xf32>>
    %rhs = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<3037000500x2xf32>>
    %output = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<3037000500x2xf32>>
    %matmul = sculptor.task.create %graph, @task_overflow_matmul,
        domain = "digital",
        task_kind = "digital.matmul",
        task_name = "overflow",
        source_layer = "test",
        source_task_ordinal = 0,
        inputs[%lhs, %rhs],
        outputs[%output],
        deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<
               tensor<3037000500x3037000500xf32>>,
           !sculptor.task_resource<tensor<3037000500x2xf32>>,
           !sculptor.task_resource<tensor<3037000500x2xf32>>)
        -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// CHECK: digital matmul minimum shard operation count overflows signed 64-bit arithmetic
