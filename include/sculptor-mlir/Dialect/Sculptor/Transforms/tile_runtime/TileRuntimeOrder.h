#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILE_RUNTIMEORDER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILE_RUNTIMEORDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace tile_runtime {

// Assigns deterministic per-core indices in global task order.
llvm::SmallVector<unsigned>
buildLocalRuntimeOrder(llvm::ArrayRef<int64_t> coreByTask);

} // namespace tile_runtime
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILE_RUNTIMEORDER_H
