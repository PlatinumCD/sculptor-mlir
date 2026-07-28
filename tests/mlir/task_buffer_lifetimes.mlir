// RUN: sculptor-mlir-opt %s --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation-pipeline --convert-bufferization-to-memref --optimize-allocation-liveness | FileCheck %s

module {
  func.func private @task_digital_temporaries(
      %arg0: tensor<4xf32>) -> tensor<4xf32> {
    %tmp0Init = tensor.empty() : tensor<4xf32>
    %tmp0 = linalg.add
        ins(%arg0, %arg0 : tensor<4xf32>, tensor<4xf32>)
        outs(%tmp0Init : tensor<4xf32>) -> tensor<4xf32>
    %tmp1Init = tensor.empty() : tensor<4xf32>
    %tmp1 = linalg.add
        ins(%tmp0, %arg0 : tensor<4xf32>, tensor<4xf32>)
        outs(%tmp1Init : tensor<4xf32>) -> tensor<4xf32>
    %outputInit = tensor.empty() : tensor<4xf32>
    %output = linalg.add
        ins(%tmp1, %arg0 : tensor<4xf32>, tensor<4xf32>)
        outs(%outputInit : tensor<4xf32>) -> tensor<4xf32>
    return %output : tensor<4xf32>
  }

  // The declarative graph intentionally survives function bufferization.
  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}

// CHECK-LABEL: func.func private @task_digital_temporaries(
// CHECK: %[[TMP0:.*]] = memref.alloc() {{.*}} : memref<4xf32>
// CHECK: linalg.add {{.*}} outs(%[[TMP0]] : memref<4xf32>)
// CHECK: %[[TMP1:.*]] = memref.alloc() {{.*}} : memref<4xf32>
// CHECK: linalg.add {{.*}}%[[TMP0]]{{.*}} outs(%[[TMP1]] : memref<4xf32>)
// CHECK-NEXT: memref.dealloc %[[TMP0]] : memref<4xf32>
// CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}} : memref<4xf32>
// CHECK: linalg.add {{.*}}%[[TMP1]]{{.*}} outs(%[[OUTPUT]] : memref<4xf32>)
// CHECK-NEXT: memref.dealloc %[[TMP1]] : memref<4xf32>
// CHECK-NOT: memref.dealloc %[[OUTPUT]]
// CHECK: return %[[OUTPUT]] : memref<4xf32>
// CHECK-LABEL: func.func private @generate_task_graph()
// CHECK: sculptor.task_graph.create
