// RUN: sculptor-mlir-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect

module {
  func.func @valid_no_result() {
    sculptor.task_region kind = "digital.vector_tile"() {
      sculptor.yield
    } : () -> ()
    return
  }
}

// -----

module {
  func.func @valid_one_result(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %0 = sculptor.task_region kind = "digital.bias_add"(%arg0) {
    ^bb0(%input: tensor<1x4xf32>):
      sculptor.yield %input : tensor<1x4xf32>
    } : (tensor<1x4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}

// -----

module {
  func.func @valid_multi_result(
      %arg0: tensor<1x2xf32>,
      %arg1: tensor<1x2xf32>
  ) -> (tensor<1x2xf32>, tensor<1x2xf32>) {
    %0:2 = sculptor.task_region kind = "digital.tile_recombine"(%arg0, %arg1) {
    ^bb0(%lhs: tensor<1x2xf32>, %rhs: tensor<1x2xf32>):
      sculptor.yield %lhs, %rhs : tensor<1x2xf32>, tensor<1x2xf32>
    } : (tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
    return %0#0, %0#1 : tensor<1x2xf32>, tensor<1x2xf32>
  }
}

// -----

module {
  func.func @valid_custom_kind() {
    sculptor.task_region kind = "linear.input_passthrough"() {
      sculptor.yield
    } : () -> ()
    return
  }
}

// -----

module {
  func.func @invalid_empty_kind() {
    // expected-error @+1 {{expected kind to be a non-empty string}}
    sculptor.task_region kind = ""() {
      sculptor.yield
    } : () -> ()
    return
  }
}

// -----

module {
  func.func @invalid_region_arg_count(%arg0: tensor<1x4xf32>) {
    // expected-error @+1 {{expected region argument count (0) to match input count (1)}}
    sculptor.task_region kind = "digital.vector_tile"(%arg0) {
      sculptor.yield
    } : (tensor<1x4xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @invalid_region_arg_type(%arg0: tensor<1x4xf32>) {
    // expected-error @+1 {{expected region argument type at index 0 ('tensor<1x5xf32>') to match input type ('tensor<1x4xf32>')}}
    sculptor.task_region kind = "digital.vector_tile"(%arg0) {
    ^bb0(%input: tensor<1x5xf32>):
      sculptor.yield
    } : (tensor<1x4xf32>) -> ()
    return
  }
}

// -----

module {
  func.func @invalid_yield_count(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    // expected-error @+1 {{expected yield value count (0) to match result count (1)}}
    %0 = sculptor.task_region kind = "digital.bias_add"(%arg0) {
    ^bb0(%input: tensor<1x4xf32>):
      sculptor.yield
    } : (tensor<1x4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}

// -----

module {
  func.func @invalid_yield_type(%arg0: tensor<1x4xf32>) -> tensor<1x5xf32> {
    // expected-error @+1 {{expected yield type at index 0 ('tensor<1x4xf32>') to match result type ('tensor<1x5xf32>')}}
    %0 = sculptor.task_region kind = "digital.bias_add"(%arg0) {
    ^bb0(%input: tensor<1x4xf32>):
      sculptor.yield %input : tensor<1x4xf32>
    } : (tensor<1x4xf32>) -> tensor<1x5xf32>
    return %0 : tensor<1x5xf32>
  }
}

// -----

module {
  func.func @invalid_yield_parent() {
    "test.region"() ({
      // expected-error @+1 {{expected to terminate sculptor.task_region}}
      sculptor.yield
    }) : () -> ()
    return
  }
}
