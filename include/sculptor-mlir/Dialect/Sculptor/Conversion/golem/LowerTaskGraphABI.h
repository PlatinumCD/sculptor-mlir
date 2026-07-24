#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_GOLEM_LOWERTASKGRAPHABI_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_GOLEM_LOWERTASKGRAPHABI_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace sculptor {
namespace golem {

// Migrates logical-array task resources into scheduled array bindings and
// explicit setup dependencies before task callable signatures are converted.
LogicalResult lowerTaskGraphABI(ModuleOp module, TypeConverter &typeConverter);

} // namespace golem
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_GOLEM_LOWERTASKGRAPHABI_H
