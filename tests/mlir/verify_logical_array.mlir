// RUN: sculptor-mlir-opt %s --split-input-file --verify-diagnostics

module {
  func.func @valid_logical_array(%matrix: tensor<2x3xf32>,
                                 %vector: tensor<1x3xf32>) {
    %array = sculptor.array.set %matrix
        : tensor<2x3xf32> -> !sculptor.logical.array
    sculptor.array.load %vector, %array
        : tensor<1x3xf32>, !sculptor.logical.array
    %partial = sculptor.array.execute %array
        : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %partial
        : !sculptor.array.result -> tensor<1x2xf32>
    return
  }
}

// -----

module {
  func.func @invalid_set_rank(%matrix: tensor<2x3x4xf32>) {
    // expected-error @+1 {{expected matrix tensor rank to be 2}}
    %array = sculptor.array.set %matrix
        : tensor<2x3x4xf32> -> !sculptor.logical.array
    return
  }
}

// -----

module {
  func.func @invalid_set_dynamic_shape(%matrix: tensor<?x3xf32>) {
    // expected-error @+1 {{expected matrix tensor shape to be static}}
    %array = sculptor.array.set %matrix
        : tensor<?x3xf32> -> !sculptor.logical.array
    return
  }
}

// -----

module {
  func.func @invalid_set_element_type(%matrix: tensor<2x3xi32>) {
    // expected-error @+1 {{expected matrix tensor element type to be f32}}
    %array = sculptor.array.set %matrix
        : tensor<2x3xi32> -> !sculptor.logical.array
    return
  }
}

// -----

module {
  func.func @invalid_load_rank(%matrix: tensor<2x3xf32>,
                               %vector: tensor<3xf32>) {
    %array = sculptor.array.set %matrix
        : tensor<2x3xf32> -> !sculptor.logical.array
    // expected-error @+1 {{expected vector tensor rank to be 2}}
    sculptor.array.load %vector, %array
        : tensor<3xf32>, !sculptor.logical.array
    return
  }
}

// -----

module {
  func.func @invalid_load_dynamic_shape(%matrix: tensor<2x3xf32>,
                                        %vector: tensor<1x?xf32>) {
    %array = sculptor.array.set %matrix
        : tensor<2x3xf32> -> !sculptor.logical.array
    // expected-error @+1 {{expected vector tensor shape to be static}}
    sculptor.array.load %vector, %array
        : tensor<1x?xf32>, !sculptor.logical.array
    return
  }
}

// -----

module {
  func.func @invalid_load_element_type(%matrix: tensor<2x3xf32>,
                                       %vector: tensor<1x3xi32>) {
    %array = sculptor.array.set %matrix
        : tensor<2x3xf32> -> !sculptor.logical.array
    // expected-error @+1 {{expected vector tensor element type to be f32}}
    sculptor.array.load %vector, %array
        : tensor<1x3xi32>, !sculptor.logical.array
    return
  }
}

// -----

module {
  func.func @invalid_load_leading_dim(%matrix: tensor<2x3xf32>,
                                      %vector: tensor<2x3xf32>) {
    %array = sculptor.array.set %matrix
        : tensor<2x3xf32> -> !sculptor.logical.array
    // expected-error @+1 {{expected vector leading dimension (2) to be 1}}
    sculptor.array.load %vector, %array
        : tensor<2x3xf32>, !sculptor.logical.array
    return
  }
}

// -----

module {
  func.func @invalid_store_rank(%matrix: tensor<2x3xf32>) {
    %array = sculptor.array.set %matrix
        : tensor<2x3xf32> -> !sculptor.logical.array
    %partial = sculptor.array.execute %array
        : !sculptor.logical.array -> !sculptor.array.result
    // expected-error @+1 {{expected output tensor rank to be 2}}
    %stored = sculptor.array.store %partial
        : !sculptor.array.result -> tensor<2xf32>
    return
  }
}

// -----

module {
  func.func @invalid_store_dynamic_shape(%matrix: tensor<2x3xf32>) {
    %array = sculptor.array.set %matrix
        : tensor<2x3xf32> -> !sculptor.logical.array
    %partial = sculptor.array.execute %array
        : !sculptor.logical.array -> !sculptor.array.result
    // expected-error @+1 {{expected output tensor shape to be static}}
    %stored = sculptor.array.store %partial
        : !sculptor.array.result -> tensor<1x?xf32>
    return
  }
}

// -----

module {
  func.func @invalid_store_element_type(%matrix: tensor<2x3xf32>) {
    %array = sculptor.array.set %matrix
        : tensor<2x3xf32> -> !sculptor.logical.array
    %partial = sculptor.array.execute %array
        : !sculptor.logical.array -> !sculptor.array.result
    // expected-error @+1 {{expected output tensor element type to be f32}}
    %stored = sculptor.array.store %partial
        : !sculptor.array.result -> tensor<1x2xi32>
    return
  }
}

// -----

module {
  func.func @invalid_store_leading_dim(%matrix: tensor<2x3xf32>) {
    %array = sculptor.array.set %matrix
        : tensor<2x3xf32> -> !sculptor.logical.array
    %partial = sculptor.array.execute %array
        : !sculptor.logical.array -> !sculptor.array.result
    // expected-error @+1 {{expected output leading dimension (2) to be 1}}
    %stored = sculptor.array.store %partial
        : !sculptor.array.result -> tensor<2x2xf32>
    return
  }
}
