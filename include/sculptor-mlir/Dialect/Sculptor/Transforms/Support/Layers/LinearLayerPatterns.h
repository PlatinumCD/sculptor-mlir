#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_LINEARLAYERPATTERNS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_LINEARLAYERPATTERNS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/CommonLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/MatchedSubgraphUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/OperandRelationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <optional>

namespace mlir {
namespace sculptor {
namespace layer_patterns {

// Carries the ops and boundary values that make up one linear layer slice.
struct LinearMatch {
  Operation *weightConstant = nullptr;
  Operation *weightTransposeEmpty = nullptr;
  Operation *weightTranspose = nullptr;
  Operation *outputEmpty = nullptr;
  Operation *outputFill = nullptr;
  Operation *outputFillConstant = nullptr;
  Operation *matmulOp = nullptr;
  Operation *biasConstant = nullptr;
  Operation *biasAddOp = nullptr;

  Operation *root = nullptr;
  llvm::SmallVector<Operation *> ops;
  llvm::SmallVector<Value> inputs;
  llvm::SmallVector<Value> outputs;
};

// Collects the full matched linear subgraph in cloning order.
inline void collectLinearOps(LinearMatch &match) {
  match.ops.clear();

  match_utils::appendUniqueOp(match.ops, match.weightConstant);
  match_utils::appendUniqueOp(match.ops, match.weightTransposeEmpty);
  match_utils::appendUniqueOp(match.ops, match.weightTranspose);
  match_utils::appendUniqueOp(match.ops, match.outputEmpty);
  match_utils::appendUniqueOp(match.ops, match.outputFillConstant);
  match_utils::appendUniqueOp(match.ops, match.outputFill);
  match_utils::appendUniqueOp(match.ops, match.matmulOp);
  match_utils::appendUniqueOp(match.ops, match.biasConstant);
  match_utils::appendUniqueOp(match.ops, match.biasAddOp);
}

// Computes the complete match boundary before rewriting.
inline void finalizeLinearMatch(LinearMatch &match) {
  collectLinearOps(match);
  match_utils::collectInputs(match.ops, match.inputs);
  match_utils::collectOutputs(match.root, match.outputs);
}

// Validates the shared matmul core and records its weight and output-init ops.
inline LogicalResult matchLinearCore(LinearMatch &match) {
  Operation *op = match.matmulOp;
  if (!layer_utils::hasDpsInputsAndOperands(op, 2, 3))
    return failure();

  auto weightTranspose =
      layer_utils::operandProducerOfType<linalg::TransposeOp>(op, 0);
  unsigned inputIndex = 0;
  if (!weightTranspose)
    weightTranspose =
        layer_utils::operandProducerOfType<linalg::TransposeOp>(op, 1);
  else
    inputIndex = 1;
  if (!weightTranspose)
    return failure();

  if (!layer_utils::hasDpsInputsAndOperands(weightTranspose.getOperation(), 1,
                                            2))
    return failure();

  auto weightConstant = layer_utils::operandProducerOfType<arith::ConstantOp>(
      weightTranspose.getOperation(), 0);
  auto weightTransposeEmpty =
      layer_utils::operandProducerOfType<tensor::EmptyOp>(
          weightTranspose.getOperation(), 1);
  if (!weightConstant || !weightTransposeEmpty)
    return failure();

  match.inputs.clear();
  match.inputs.push_back(op->getOperand(inputIndex));

  auto outputInit = matchFillOutputInit(op, 2);
  if (!outputInit)
    return failure();

  match.weightConstant = weightConstant.getOperation();
  match.weightTransposeEmpty = weightTransposeEmpty.getOperation();
  match.weightTranspose = weightTranspose.getOperation();
  match.outputFill = outputInit->outputFill;
  match.outputFillConstant = outputInit->outputFillConstant;
  match.outputEmpty = outputInit->outputEmpty;
  return success();
}

// Recognizes linear layers where a bias constant is added to the matmul result.
inline std::optional<LinearMatch> matchLinearWithBias(Operation *op) {
  if (!layer_utils::isAddfGeneric(op) || !layer_utils::hasInputs(op, 2) ||
      !layer_utils::operandProducersAreEither<arith::ConstantOp,
                                              linalg::MatmulOp>(op))
    return std::nullopt;

  auto biasConstant =
      layer_utils::operandProducerOfType<arith::ConstantOp>(op, 0);
  auto matmulOp = layer_utils::operandProducerOfType<linalg::MatmulOp>(op, 1);
  if (!biasConstant || !matmulOp) {
    biasConstant = layer_utils::operandProducerOfType<arith::ConstantOp>(op, 1);
    matmulOp = layer_utils::operandProducerOfType<linalg::MatmulOp>(op, 0);
  }
  if (!biasConstant || !matmulOp)
    return std::nullopt;

  LinearMatch match;
  match.root = op;
  match.biasAddOp = op;
  match.biasConstant = biasConstant.getOperation();
  match.matmulOp = matmulOp.getOperation();
  if (failed(matchLinearCore(match)))
    return std::nullopt;

  finalizeLinearMatch(match);
  return match;
}

// Recognizes linear layers represented directly by the matmul result.
inline std::optional<LinearMatch> matchLinearWithoutBias(Operation *op) {
  auto matmulOp = llvm::dyn_cast_or_null<linalg::MatmulOp>(op);
  if (!matmulOp)
    return std::nullopt;

  LinearMatch match;
  match.root = op;
  match.matmulOp = matmulOp.getOperation();
  if (failed(matchLinearCore(match)))
    return std::nullopt;

  finalizeLinearMatch(match);
  return match;
}

} // namespace layer_patterns
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_LAYERS_LINEARLAYERPATTERNS_H
