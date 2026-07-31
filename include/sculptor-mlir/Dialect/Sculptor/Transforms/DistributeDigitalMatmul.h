#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_DISTRIBUTEDIGITALMATMUL_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_DISTRIBUTEDIGITALMATMUL_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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

struct DistributeDigitalMatmulPass
    : public PassWrapper<DistributeDigitalMatmulPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DistributeDigitalMatmulPass)

  Option<int64_t> maxShards{
      *this, "max-shards",
      llvm::cl::desc("Maximum number of digital matmul shards"),
      llvm::cl::init(8)};

  Option<int64_t> minOpsPerShard{
      *this, "min-ops-per-shard",
      llvm::cl::desc("Minimum scalar matmul operations in every shard"),
      llvm::cl::init(65536)};

  Option<std::string> strategy{
      *this, "strategy",
      llvm::cl::desc("auto, output-columns, output-rows, two-dimensional, or "
                     "attention-heads"),
      llvm::cl::init("auto")};

  Option<std::string> placementPolicy{
      *this, "placement-policy",
      llvm::cl::desc("unconstrained, prefer-distinct, or require-distinct"),
      llvm::cl::init("prefer-distinct")};

  Option<bool> requireChange{
      *this, "require-change",
      llvm::cl::desc("Fail when no eligible digital matmul is distributed"),
      llvm::cl::init(false)};

  DistributeDigitalMatmulPass() = default;
  DistributeDigitalMatmulPass(const DistributeDigitalMatmulPass &pass);

  StringRef getArgument() const final {
    return "sculptor-distribute-digital-matmul";
  }

  StringRef getDescription() const final {
    return "Distribute eligible digital matmuls into independently placeable "
           "task-graph shards";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SculptorDialect, arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void registerDistributeDigitalMatmulPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_DISTRIBUTEDIGITALMATMUL_H
