// RUN: not sculptor-mlir-opt %s --sculptor-plan-core-scratchpad="bytes=0" 2>&1 | FileCheck %s --check-prefix=CAPACITY
// RUN: not sculptor-mlir-opt %s --sculptor-plan-core-scratchpad="bytes=128 alignment=3" 2>&1 | FileCheck %s --check-prefix=ALIGN

// CAPACITY: error: scratchpad mode requires a positive byte capacity
// ALIGN: error: scratchpad alignment must be a positive power of two
module attributes {sculptor.runtime.core_id = 0 : i64} {
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}
