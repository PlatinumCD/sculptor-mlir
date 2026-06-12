#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_EMITRUNTIMEGRAPH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_EMITRUNTIMEGRAPH_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct EmitRuntimeGraphPass
    : public mlir::PassWrapper<EmitRuntimeGraphPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitRuntimeGraphPass)

  mlir::StringRef getArgument() const final {
    return "sculptor-emit-runtime-graph";
  }

  mlir::StringRef getDescription() const final {
    return "Emit generic runtime graph metadata and task-entry shims";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::sculptor::SculptorDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override;
};

void registerEmitRuntimeGraphPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_EMITRUNTIMEGRAPH_H
