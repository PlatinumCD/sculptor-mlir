// RUN: sculptor-mlir-opt %s --sculptor-extract-core-module="core-id=1" --split-input-file --verify-diagnostics

// expected-error @+1 {{requested core 1 is not an active core in the deployment}}
module attributes {sculptor.deployment.active_core_ids = [0]} {
  module @core_0 attributes {sculptor.runtime.core_id = 0 : i64} {
  }
}

// -----

// expected-error @+1 {{found multiple nested deployment modules for core 1}}
module attributes {sculptor.deployment.active_core_ids = [1]} {
  module @first attributes {sculptor.runtime.core_id = 1 : i64} {
  }
  module @second attributes {sculptor.runtime.core_id = 1 : i64} {
  }
}
