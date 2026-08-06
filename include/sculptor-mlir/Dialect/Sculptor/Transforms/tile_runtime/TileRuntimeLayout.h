#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILE_RUNTIMELAYOUT_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILE_RUNTIMELAYOUT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sculptor {

LogicalResult rebuildTileRuntimeLayout(func::FuncOp taskGraphFunc);

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILE_RUNTIMELAYOUT_H
