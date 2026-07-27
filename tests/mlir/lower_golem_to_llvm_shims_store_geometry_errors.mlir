// RUN: sculptor-mlir-opt %s --sculptor-lower-golem-to-llvm-shims --split-input-file --verify-diagnostics

module {
  func.func private @missing_physical_shape() -> tensor<1x1xf32>
      attributes {sculptor.runtime.local_array_id = 0 : i64} {
    %matrix = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %array = sculptor.array.set %matrix : tensor<1x1xf32> -> !sculptor.logical.array
    %execution = sculptor.array.execute %array : !sculptor.logical.array -> !sculptor.array.result
    // expected-error @below {{expected physical array shape attribute 'sculptor.tile_physical_shape'}}
    // expected-error @below {{failed to legalize operation 'sculptor.array.store' that was explicitly marked illegal}}
    %result = sculptor.array.store %execution {sculptor.tile_valid_shape = [1, 1]} : !sculptor.array.result -> tensor<1x1xf32>
    return %result : tensor<1x1xf32>
  }
}

// -----

module {
  func.func private @nonpositive_physical_rows() -> tensor<1x1xf32>
      attributes {sculptor.runtime.local_array_id = 0 : i64} {
    %matrix = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %array = sculptor.array.set %matrix : tensor<1x1xf32> -> !sculptor.logical.array
    %execution = sculptor.array.execute %array : !sculptor.logical.array -> !sculptor.array.result
    // expected-error @below {{expected positive physical array row count}}
    // expected-error @below {{failed to legalize operation 'sculptor.array.store' that was explicitly marked illegal}}
    %result = sculptor.array.store %execution {sculptor.tile_physical_shape = [0, 4], sculptor.tile_valid_shape = [1, 1]} : !sculptor.array.result -> tensor<1x1xf32>
    return %result : tensor<1x1xf32>
  }
}

// -----

module {
  func.func private @nonpositive_valid_rows() -> tensor<1x0xf32>
      attributes {sculptor.runtime.local_array_id = 0 : i64} {
    %matrix = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %array = sculptor.array.set %matrix : tensor<1x1xf32> -> !sculptor.logical.array
    %execution = sculptor.array.execute %array : !sculptor.logical.array -> !sculptor.array.result
    // expected-error @below {{logical store width must be between one and physical array rows}}
    // expected-error @below {{failed to legalize operation 'sculptor.array.store' that was explicitly marked illegal}}
    %result = sculptor.array.store %execution {sculptor.tile_physical_shape = [4, 4], sculptor.tile_valid_shape = [0, 1]} : !sculptor.array.result -> tensor<1x0xf32>
    return %result : tensor<1x0xf32>
  }
}

// -----

module {
  func.func private @valid_rows_exceed_physical_rows() -> tensor<1x5xf32>
      attributes {sculptor.runtime.local_array_id = 0 : i64} {
    %matrix = arith.constant dense<0.000000e+00> : tensor<1x1xf32>
    %array = sculptor.array.set %matrix : tensor<1x1xf32> -> !sculptor.logical.array
    %execution = sculptor.array.execute %array : !sculptor.logical.array -> !sculptor.array.result
    // expected-error @below {{logical store width must be between one and physical array rows}}
    // expected-error @below {{failed to legalize operation 'sculptor.array.store' that was explicitly marked illegal}}
    %result = sculptor.array.store %execution {sculptor.tile_physical_shape = [4, 4], sculptor.tile_valid_shape = [5, 1]} : !sculptor.array.result -> tensor<1x5xf32>
    return %result : tensor<1x5xf32>
  }
}
