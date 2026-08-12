#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPANDDIGITALWORK_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPANDDIGITALWORK_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {

struct ExpandDigitalWorkPass
    : public PassWrapper<ExpandDigitalWorkPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandDigitalWorkPass)

  Option<int64_t> parallelWorkers{
      *this, "parallel-workers",
      llvm::cl::desc("Target number of independent digital work units"),
      llvm::cl::init(4)};

  Option<bool> requireChange{
      *this, "require-change",
      llvm::cl::desc("Fail when no digital operation can be expanded"),
      llvm::cl::init(false)};

  Option<std::string> dataflow{
      *this, "dataflow",
      llvm::cl::desc("Digital dataflow mode: bulk or sharded"),
      llvm::cl::init("bulk")};

  Option<std::string> tilingPolicy{
      *this, "tiling-policy",
      llvm::cl::desc(
          "Digital tiling policy: dimension-first or communication-aware"),
      llvm::cl::init("dimension-first")};

  Option<int64_t> shardPropagationDepth{
      *this, "shard-propagation-depth",
      llvm::cl::desc("Maximum shard propagation depth; zero is unbounded"),
      llvm::cl::init(0)};

  Option<bool> requireCompleteShardChain{
      *this, "require-complete-shard-chain",
      llvm::cl::desc("Fail when an eligible shard chain reaches a boundary"),
      llvm::cl::init(false)};

  Option<std::string> reductionTree{
      *this, "reduction-tree",
      llvm::cl::desc("Associative reduction policy: none or balanced"),
      llvm::cl::init("none")};

  Option<int64_t> reductionFanIn{
      *this, "reduction-fan-in",
      llvm::cl::desc("Maximum reduction fan-in; only two is supported"),
      llvm::cl::init(2)};

  Option<int64_t> reductionMinimumWidth{
      *this, "reduction-min-width",
      llvm::cl::desc("Minimum number of leaves for reduction balancing"),
      llvm::cl::init(3)};

  ExpandDigitalWorkPass() = default;
  ExpandDigitalWorkPass(const ExpandDigitalWorkPass &pass);

  StringRef getArgument() const final { return "sculptor-expand-digital-work"; }

  StringRef getDescription() const final {
    return "Expose balanced parallel-only digital work units through "
           "TilingInterface";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, func::FuncDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void registerExpandDigitalWorkPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_EXPANDDIGITALWORK_H
