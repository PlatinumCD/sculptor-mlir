#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FINALIZETASKGRAPHRESOURCES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FINALIZETASKGRAPHRESOURCES_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct FinalizeTaskGraphResourcesPass
    : public PassWrapper<FinalizeTaskGraphResourcesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeTaskGraphResourcesPass)

  StringRef getArgument() const final {
    return "sculptor-finalize-task-graph-resources";
  }

  StringRef getDescription() const final {
    return "Assign runtime slots and storage offsets to surviving task graph "
           "resources";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerFinalizeTaskGraphResourcesPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_FINALIZETASKGRAPHRESOURCES_H
