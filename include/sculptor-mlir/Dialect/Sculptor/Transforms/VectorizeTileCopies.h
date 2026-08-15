#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_VECTORIZETILECOPIES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_VECTORIZETILECOPIES_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::sculptor {

struct VectorizeTileCopiesPass
    : public PassWrapper<VectorizeTileCopiesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizeTileCopiesPass)

  Option<int64_t> vectorBits{
      *this, "vector-bits",
      llvm::cl::desc("Preferred vector width for tile-local pack and copy "
                     "materialization"),
      llvm::cl::init(256)};

  VectorizeTileCopiesPass() = default;
  VectorizeTileCopiesPass(const VectorizeTileCopiesPass &pass)
      : PassWrapper(pass),
        vectorBits(*this, "vector-bits",
                   llvm::cl::desc(
                       "Preferred vector width for tile-local pack and copy "
                       "materialization"),
                   llvm::cl::init(256)) {
    vectorBits = pass.vectorBits;
  }

  StringRef getArgument() const final {
    return "sculptor-vectorize-tile-copies";
  }
  StringRef getDescription() const final {
    return "Classify and vectorize unavoidable tile-local materialization";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void registerVectorizeTileCopiesPass();

} // namespace mlir::sculptor

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_VECTORIZETILECOPIES_H
