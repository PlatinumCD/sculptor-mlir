#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTORTYPES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTORTYPES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

//===- Generated includes -------------------------------------------------===//

// Exposes the TableGen-generated type classes for analog containers,
// tiled array views, and task-graph handles/resources.
#define GET_TYPEDEF_CLASSES
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h.inc"

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTORTYPES_H
