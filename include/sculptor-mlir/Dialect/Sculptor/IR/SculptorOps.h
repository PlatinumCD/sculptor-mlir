#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTOROPS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTOROPS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

//===----------------------------------------------------------------------===//
// Analog Ops
//===----------------------------------------------------------------------===//

// Exposes the TableGen-generated operation classes for analog storage,
// placement, task-graph, and array-execution IR.
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorNNOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h.inc"

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTOROPS_H
