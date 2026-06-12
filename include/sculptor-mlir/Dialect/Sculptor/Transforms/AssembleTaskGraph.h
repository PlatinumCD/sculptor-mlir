#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_ASSEMBLETASKGRAPH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_ASSEMBLETASKGRAPH_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

// Defines the module pass that finds forward and materializes its task-graph
// generator function.
struct AssembleTaskGraphPass
    : public mlir::PassWrapper<AssembleTaskGraphPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssembleTaskGraphPass)

  // Provides the stable command-line name used to schedule this transform.
  mlir::StringRef getArgument() const final {
    return "sculptor-assemble-task-graph";
  }

  // Summarizes the pass behavior for MLIR pass registration and help text.
  mlir::StringRef getDescription() const final {
    return "Assemble an analog task graph from forward";
  }

  // Ensures generated task-graph operations can be parsed and materialized.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::sculptor::SculptorDialect>();
  }

  // Rewrites the module around forward by outlining task boundaries and
  // creating the companion task-graph construction function.
  void runOnOperation() override;
};

// Registers the task-graph assembly pass with MLIR's global pass registry.
void registerAssembleTaskGraphPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_ASSEMBLETASKGRAPH_H
