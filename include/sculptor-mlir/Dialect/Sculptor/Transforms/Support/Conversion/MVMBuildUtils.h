#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_MVMBUILDUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_MVMBUILDUTILS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace sculptor {
namespace mvm_build {

// Builds the analog MVM primitive without encoding layer or backend semantics.
inline Value buildMVM(Location loc, RankedTensorType resultTy, Value input,
                      Value weight, OpBuilder &builder) {
  return builder.create<sculptor::MVMOp>(loc, resultTy, input, weight).getResult();
}

} // namespace mvm_build
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_MVMBUILDUTILS_H
