#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::sculptor;

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorNNOpsEnums.cpp.inc"

#define GET_OP_CLASSES
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.cpp.inc"

// Registers the TableGen-generated operation classes with the analog dialect.
void SculptorDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.cpp.inc"
      >();
}
