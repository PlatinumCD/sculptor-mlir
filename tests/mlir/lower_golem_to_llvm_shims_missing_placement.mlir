// RUN: not sculptor-mlir-opt %s --sculptor-lower-golem-to-llvm-shims 2>&1 | FileCheck %s

module {
  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    // CHECK: expected enclosing task function 'task_matrix' to carry runtime attr 'sculptor.runtime.local_array_id'
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }
}
