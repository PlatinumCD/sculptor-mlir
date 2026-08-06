#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_OUTLINETILEROUTINES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_OUTLINETILEROUTINES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct OutlineTileRoutinesPass
    : public PassWrapper<OutlineTileRoutinesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineTileRoutinesPass)

  StringRef getArgument() const final {
    return "sculptor-outline-tile-routines";
  }
  StringRef getDescription() const final {
    return "Outline a locked logical-tile placement into per-tile routines";
  }

  void runOnOperation() override;
};

void registerOutlineTileRoutinesPass();

} // namespace sculptor
} // namespace mlir

#endif
