#ifndef SCULPTOR_OPS_TD
#define SCULPTOR_OPS_TD

//===- SculptorOps.td -------------------------------------------------------===//
//
// Aggregate include point for all operations in the analog dialect.
//
//===----------------------------------------------------------------------===//

include "sculptor-mlir/Dialect/Sculptor/IR/Ops/SculptorNNOps.td"
include "sculptor-mlir/Dialect/Sculptor/IR/Ops/SculptorGolemOps.td"
include "sculptor-mlir/Dialect/Sculptor/IR/Ops/SculptorTaskGraphOps.td"
include "sculptor-mlir/Dialect/Sculptor/IR/Ops/SculptorExecutionOps.td"
include "sculptor-mlir/Dialect/Sculptor/IR/Ops/SculptorTaskRegionOps.td"

#endif // SCULPTOR_OPS_TD
