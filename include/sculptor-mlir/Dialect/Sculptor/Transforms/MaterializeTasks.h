#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MATERIALIZETASKS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MATERIALIZETASKS_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

// Materializes outlined sculptor.task_region operations into directly callable
// task functions. The initial shell is behavior-preserving while the
// materialization pipeline is built out incrementally.
struct MaterializeTasksPass
    : public mlir::PassWrapper<MaterializeTasksPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeTasksPass)

  mlir::StringRef getArgument() const final {
    return "sculptor-materialize-tasks";
  }

  mlir::StringRef getDescription() const final {
    return "Materialize analog task regions into task functions";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::sculptor::SculptorDialect, mlir::func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerMaterializeTasksPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MATERIALIZETASKS_H
