#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_INSTRUMENTTILEHEAP_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_INSTRUMENTTILEHEAP_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sculptor {

struct InstrumentTileHeapPass
    : public PassWrapper<InstrumentTileHeapPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InstrumentTileHeapPass)

  StringRef getArgument() const final {
    return "sculptor-instrument-tile-heap";
  }
  StringRef getDescription() const final {
    return "Route tile malloc/free calls through optional heap profiling";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() override;
};

void registerInstrumentTileHeapPass();

} // namespace mlir::sculptor

#endif
