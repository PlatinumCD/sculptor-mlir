// RUN: sculptor-mlir-opt %s --sculptor-materialize-tasks | FileCheck %s --implicit-check-not=forward_compute --implicit-check-not=digital.compute --implicit-check-not="call @layer" --implicit-check-not="func.func @layer"

module {
  // CHECK-LABEL: func.func @forward
  // CHECK: %[[TASK:.*]] = call @task_layer_use_const_0(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK: return %[[TASK]] : tensor<1x2xf32>
  func.func @forward(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %0 = call @layer(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }

  func.func @layer(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %cst = arith.constant dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>
    %0 = sculptor.task_region kind = "digital.bias_add" name = "use_const"(%arg0, %cst) {
    ^bb0(%input: tensor<1x2xf32>, %bias: tensor<1x2xf32>):
      %empty = tensor.empty() : tensor<1x2xf32>
      %sum = linalg.add ins(%input, %bias : tensor<1x2xf32>, tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
      sculptor.yield %sum : tensor<1x2xf32>
    } : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }

  // CHECK-LABEL: func.func private @task_layer_use_const_0(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK-SAME: sculptor.source_layer = "layer"
  // CHECK-SAME: sculptor.task_kind = "digital.bias_add"
  // CHECK-SAME: sculptor.task_name = "use_const"
  // CHECK: arith.constant dense<{{.*}}> : tensor<1x2xf32>
  // CHECK: linalg.add
  // CHECK: return
}
