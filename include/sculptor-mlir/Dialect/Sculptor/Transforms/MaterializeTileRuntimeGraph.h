#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MATERIALIZETILERUNTIMEGRAPH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MATERIALIZETILERUNTIMEGRAPH_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct MaterializeTileRuntimeGraphPass
    : public PassWrapper<MaterializeTileRuntimeGraphPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeTileRuntimeGraphPass)

  StringRef getArgument() const final {
    return "sculptor-materialize-tile-runtime-graph";
  }
  StringRef getDescription() const final {
    return "Materialize an extracted RA tile as a runtime task graph";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerMaterializeTileRuntimeGraphPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MATERIALIZETILERUNTIMEGRAPH_H
