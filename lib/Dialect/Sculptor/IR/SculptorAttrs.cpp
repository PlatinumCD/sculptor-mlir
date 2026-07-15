#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.cpp.inc"

using namespace mlir;
using namespace mlir::sculptor;

void SculptorDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.cpp.inc"
      >();
}
