#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PARTITIONTASKGRAPHBYCORE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PARTITIONTASKGRAPHBYCORE_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct PartitionTaskGraphByCorePass
    : public PassWrapper<PartitionTaskGraphByCorePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionTaskGraphByCorePass)

  StringRef getArgument() const final {
    return "sculptor-partition-task-graph-by-core";
  }

  StringRef getDescription() const final {
    return "Partition a scheduled task graph into isolated per-core modules";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerPartitionTaskGraphByCorePass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PARTITIONTASKGRAPHBYCORE_H
