#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FINALIZETILERUNTIMEGRAPH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FINALIZETILERUNTIMEGRAPH_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct FinalizeTileRuntimeGraphPass
    : public PassWrapper<FinalizeTileRuntimeGraphPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeTileRuntimeGraphPass)

  StringRef getArgument() const final {
    return "sculptor-finalize-tile-runtime-graph";
  }

  StringRef getDescription() const final {
    return "Assign tile-local runtime slots and storage offsets";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerFinalizeTileRuntimeGraphPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FINALIZETILERUNTIMEGRAPH_H
