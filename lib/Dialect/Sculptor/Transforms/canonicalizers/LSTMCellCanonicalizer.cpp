#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Canonicalization/CanonicalRewriteUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/CommonLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/LinalgMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/MatchedSubgraphUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/OperandRelationUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/RecurrentLayerPatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <optional>

namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace layer_utils = mlir::sculptor::layer_utils;
namespace match_utils = mlir::sculptor::match_utils;
namespace canonicalizer_utils = mlir::sculptor::canonicalizer_utils;
namespace linalg_match = mlir::sculptor::linalg_match;

namespace {

using mlir::arith::ConstantOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::tensor::CollapseShapeOp;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;

struct LSTMCellMatmulBranchMatch {
  mlir::Value activationInput;
  mlir::Operation *weightConstant = nullptr;
  mlir::Operation *transposeEmpty = nullptr;
  mlir::Operation *transpose = nullptr;
  mlir::Operation *matmul = nullptr;
  mlir::Operation *fill = nullptr;
  mlir::Operation *fillConstant = nullptr;
  mlir::Operation *biasAdd = nullptr;
  mlir::Operation *biasConstant = nullptr;
};

struct LSTMCellMatch {
  mlir::Operation *sharedMatmulOutputEmpty = nullptr;
  mlir::Operation *sharedFillOp = nullptr;
  mlir::Operation *sharedFillConstant = nullptr;
  mlir::Operation *preActivationAddOp = nullptr;
  mlir::Operation *preActivationCollapseOp = nullptr;

  LSTMCellMatmulBranchMatch inputBranch;
  LSTMCellMatmulBranchMatch hiddenBranch;

  layer_patterns::RecurrentGateIndexingScaffoldMatch indexing;
  layer_patterns::RecurrentGateSliceMatch inputGate;
  layer_patterns::RecurrentGateSliceMatch forgetGate;
  layer_patterns::RecurrentGateSliceMatch candidateGate;
  layer_patterns::RecurrentGateSliceMatch outputGate;

  mlir::Operation *sharedHiddenOutputEmpty = nullptr;
  mlir::Operation *sigmoidOneConstant = nullptr;
  mlir::Operation *forgetCellMulOp = nullptr;
  mlir::Operation *inputCandidateMulOp = nullptr;
  mlir::Operation *cellAddOp = nullptr;
  mlir::Operation *cellTanhOp = nullptr;
  mlir::Operation *hiddenMulOp = nullptr;
  mlir::Operation *root = nullptr;

  mlir::Value cellStateInput;
  bool hasBias = false;
  llvm::SmallVector<mlir::Operation *> ops;
};

static mlir::LogicalResult matchCollapsedExtractGeneric(
    mlir::Operation *op, mlir::Operation *&indexValueOp,
    mlir::Operation *&collapsedVectorOp, mlir::Operation *&zeroConst,
    mlir::Operation *&extentConst, mlir::Operation *&gatherEmpty) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic)
    return mlir::failure();

  if (!layer_utils::hasDpsInputsAndOperands(op, 1, 2))
    return mlir::failure();

  auto outputEmpty = layer_utils::operandProducerOfType<EmptyOp>(op, 1);
  if (!outputEmpty)
    return mlir::failure();

  mlir::Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return mlir::failure();

  mlir::Block &block = region.front();
  if (block.empty())
    return mlir::failure();

  auto it = block.begin();
  auto e = block.end();
  auto cmp = llvm::dyn_cast<mlir::arith::CmpIOp>(&*it++);
  auto add = (it != e) ? llvm::dyn_cast<mlir::arith::AddIOp>(&*it++)
                       : mlir::arith::AddIOp();
  auto select = (it != e) ? llvm::dyn_cast<mlir::arith::SelectOp>(&*it++)
                          : mlir::arith::SelectOp();
  auto cast = (it != e) ? llvm::dyn_cast<mlir::arith::IndexCastOp>(&*it++)
                        : mlir::arith::IndexCastOp();
  auto extract = (it != e) ? llvm::dyn_cast<mlir::tensor::ExtractOp>(&*it++)
                           : mlir::tensor::ExtractOp();
  auto yield = (it != e) ? llvm::dyn_cast<mlir::linalg::YieldOp>(&*it++)
                         : mlir::linalg::YieldOp();
  if (!cmp || !add || !select || !cast || !extract || !yield || it != e)
    return mlir::failure();

  if (cmp.getPredicate() != mlir::arith::CmpIPredicate::slt)
    return mlir::failure();

  mlir::Value wrappedIndex = cmp.getLhs();
  mlir::Value zeroValue = cmp.getRhs();
  mlir::Value extentValue;
  if (add.getLhs() == wrappedIndex)
    extentValue = add.getRhs();
  else if (add.getRhs() == wrappedIndex)
    extentValue = add.getLhs();
  else
    return mlir::failure();

  if (select.getCondition() != cmp.getResult() ||
      select.getTrueValue() != add.getResult() ||
      select.getFalseValue() != wrappedIndex ||
      cast.getIn() != select.getResult() || extract.getIndices().size() != 1 ||
      extract.getIndices().front() != cast.getResult() ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != extract.getResult())
    return mlir::failure();

  zeroConst = layer_utils::producerOf(zeroValue);
  extentConst = layer_utils::producerOf(extentValue);
  collapsedVectorOp = layer_utils::producerOf(extract.getTensor());
  indexValueOp = layer_utils::operandProducer(op, 0);
  gatherEmpty = outputEmpty.getOperation();
  if (!llvm::dyn_cast_or_null<ConstantOp>(zeroConst) ||
      !llvm::dyn_cast_or_null<ConstantOp>(extentConst) || !collapsedVectorOp ||
      !indexValueOp)
    return mlir::failure();

  return mlir::success();
}

// Reuses one gate indexing scaffold across all LSTMCell gate slices.
static mlir::LogicalResult matchSharedGateIndexScaffold(
    CollapseShapeOp collapsedIndices, mlir::Operation *expectedZeroConst,
    mlir::Operation *expectedExtentConst,
    layer_patterns::RecurrentGateIndexingScaffoldMatch &scaffold) {
  if (scaffold.collapsedIndices) {
    return scaffold.collapsedIndices == collapsedIndices.getOperation() &&
                   scaffold.zeroIndexConstant == expectedZeroConst &&
                   scaffold.indexExtentConstant == expectedExtentConst
               ? mlir::success()
               : mlir::failure();
  }

  mlir::Operation *combinedIndices =
      layer_utils::producerOf(collapsedIndices.getSrc());
  if (!combinedIndices || !layer_utils::isAddiGeneric(combinedIndices))
    return mlir::failure();

  if (!layer_utils::hasDpsInputsAndOperands(combinedIndices, 2, 3))
    return mlir::failure();

  auto combinedIndicesEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(combinedIndices, 2);
  if (!combinedIndicesEmpty)
    return mlir::failure();

  mlir::Operation *baseOffsetGeneric =
      layer_utils::operandProducer(combinedIndices, 0);
  auto rangeExpand =
      layer_utils::operandProducerOfType<ExpandShapeOp>(combinedIndices, 1);
  if (!baseOffsetGeneric || !rangeExpand)
    return mlir::failure();

  mlir::Operation *rangeGeneric = layer_utils::producerOf(rangeExpand.getSrc());
  if (!rangeGeneric ||
      mlir::failed(layer_patterns::matchIndexRangeGeneric(rangeGeneric)))
    return mlir::failure();

  auto rangeEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(rangeGeneric, 0);
  if (!rangeEmpty)
    return mlir::failure();

  if (mlir::failed(layer_patterns::matchMultiplyByConstantGeneric(
          baseOffsetGeneric, expectedExtentConst)))
    return mlir::failure();

  auto baseOffsetEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(baseOffsetGeneric, 1);
  auto zeroIndexExpand =
      layer_utils::operandProducerOfType<ExpandShapeOp>(baseOffsetGeneric, 0);
  if (!baseOffsetEmpty || !zeroIndexExpand)
    return mlir::failure();

  mlir::Operation *zeroIndexGeneric =
      layer_utils::producerOf(zeroIndexExpand.getSrc());
  if (!zeroIndexGeneric ||
      mlir::failed(layer_patterns::matchYieldingConstantGeneric(
          zeroIndexGeneric, expectedZeroConst)))
    return mlir::failure();

  auto zeroIndexEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(zeroIndexGeneric, 0);
  if (!zeroIndexEmpty)
    return mlir::failure();

  scaffold.zeroIndexConstant = expectedZeroConst;
  scaffold.indexExtentConstant = expectedExtentConst;
  scaffold.zeroIndexEmpty = zeroIndexEmpty.getOperation();
  scaffold.zeroIndexGeneric = zeroIndexGeneric;
  scaffold.zeroIndexExpand = zeroIndexExpand.getOperation();
  scaffold.baseOffsetEmpty = baseOffsetEmpty.getOperation();
  scaffold.baseOffsetGeneric = baseOffsetGeneric;
  scaffold.rangeEmpty = rangeEmpty.getOperation();
  scaffold.rangeGeneric = rangeGeneric;
  scaffold.rangeExpand = rangeExpand.getOperation();
  scaffold.combinedIndicesEmpty = combinedIndicesEmpty.getOperation();
  scaffold.combinedIndicesGeneric = combinedIndices;
  scaffold.collapsedIndices = collapsedIndices.getOperation();
  return mlir::success();
}

static mlir::LogicalResult
matchGateSlice(mlir::Operation *activationOp,
               mlir::Operation *sharedHiddenOutputEmpty, int64_t expectedOffset,
               int64_t gateWidth, bool expectSigmoid,
               layer_patterns::RecurrentGateSliceMatch &gate,
               layer_patterns::RecurrentGateIndexingScaffoldMatch &scaffold,
               mlir::Operation *&sharedCollapsedPreattivation,
               mlir::Operation *&sharedSigmoidOneConstant) {
  if (expectSigmoid) {
    if (!layer_utils::isSigmoidGeneric(activationOp))
      return mlir::failure();

    mlir::Operation *sigmoidOneConstant =
        layer_patterns::getSigmoidUnitConstant(activationOp);
    if (!sigmoidOneConstant)
      return mlir::failure();

    if (sharedSigmoidOneConstant &&
        sharedSigmoidOneConstant != sigmoidOneConstant)
      return mlir::failure();
    sharedSigmoidOneConstant = sigmoidOneConstant;
  } else if (!layer_utils::isTanhGeneric(activationOp)) {
    return mlir::failure();
  }

  if (!layer_utils::hasDpsInputsAndOperands(activationOp, 1, 2))
    return mlir::failure();

  auto activationOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(activationOp, 1);
  if (!activationOutputEmpty ||
      activationOutputEmpty.getOperation() != sharedHiddenOutputEmpty)
    return mlir::failure();

  auto expand =
      layer_utils::operandProducerOfType<ExpandShapeOp>(activationOp, 0);
  if (!expand)
    return mlir::failure();

  mlir::Operation *gather = layer_utils::producerOf(expand.getSrc());
  mlir::Operation *indexValueOp = nullptr;
  mlir::Operation *collapsedVectorOp = nullptr;
  mlir::Operation *zeroConst = nullptr;
  mlir::Operation *extentConst = nullptr;
  mlir::Operation *gatherEmpty = nullptr;
  if (!gather || mlir::failed(matchCollapsedExtractGeneric(
                     gather, indexValueOp, collapsedVectorOp, zeroConst,
                     extentConst, gatherEmpty)))
    return mlir::failure();

  if (!layer_utils::constantOpHasI64Value(zeroConst, 0) ||
      !layer_utils::constantOpHasI64Value(extentConst, gateWidth))
    return mlir::failure();

  if (sharedCollapsedPreattivation &&
      sharedCollapsedPreattivation != collapsedVectorOp)
    return mlir::failure();
  sharedCollapsedPreattivation = collapsedVectorOp;

  if (scaffold.gatherEmpty && scaffold.gatherEmpty != gatherEmpty)
    return mlir::failure();
  scaffold.gatherEmpty = gatherEmpty;

  if (expectedOffset == 0) {
    auto collapsedIndices = llvm::dyn_cast<CollapseShapeOp>(indexValueOp);
    if (!collapsedIndices ||
        mlir::failed(matchSharedGateIndexScaffold(collapsedIndices, zeroConst,
                                                  extentConst, scaffold)))
      return mlir::failure();
  } else {
    mlir::Operation *baseIndicesOp =
        layer_utils::operandProducer(indexValueOp, 0);
    auto collapsedIndices =
        llvm::dyn_cast_or_null<CollapseShapeOp>(baseIndicesOp);
    if (!collapsedIndices ||
        mlir::failed(matchSharedGateIndexScaffold(collapsedIndices, zeroConst,
                                                  extentConst, scaffold)))
      return mlir::failure();

    mlir::Operation *offsetConstant = nullptr;
    if (mlir::failed(layer_patterns::matchOffsetAddGeneric(
            indexValueOp, scaffold.collapsedIndices, scaffold.rangeEmpty,
            expectedOffset, offsetConstant)))
      return mlir::failure();

    gate.offsetAdd = indexValueOp;
    gate.offsetConstant = offsetConstant;
  }

  gate.gather = gather;
  gate.expand = expand.getOperation();
  gate.activation = activationOp;
  return mlir::success();
}

static mlir::LogicalResult
matchLSTMMatmulBranchCore(MatmulOp matmul, mlir::Operation *sharedOutputEmpty,
                          LSTMCellMatmulBranchMatch &branch) {
  if (!layer_utils::hasDpsInputsAndOperands(matmul.getOperation(), 2, 3))
    return mlir::failure();

  auto outputInit =
      layer_patterns::matchFillOutputInit(matmul.getOperation(), 2);
  if (!outputInit || outputInit->outputEmpty != sharedOutputEmpty)
    return mlir::failure();

  auto firstTranspose =
      layer_utils::operandProducerOfType<TransposeOp>(matmul.getOperation(), 0);
  auto secondTranspose =
      layer_utils::operandProducerOfType<TransposeOp>(matmul.getOperation(), 1);
  if (static_cast<bool>(firstTranspose) == static_cast<bool>(secondTranspose))
    return mlir::failure();

  auto transpose = firstTranspose ? firstTranspose : secondTranspose;
  if (!linalg_match::hasPermutation(transpose, {1, 0}) ||
      !layer_utils::hasDpsInputsAndOperands(transpose.getOperation(), 1, 2))
    return mlir::failure();

  auto transposeEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(transpose.getOperation(), 1);
  auto weightConstant = layer_utils::operandProducerOfType<ConstantOp>(
      transpose.getOperation(), 0);
  if (!transposeEmpty || !weightConstant)
    return mlir::failure();

  branch.activationInput =
      firstTranspose ? matmul.getOperand(1) : matmul.getOperand(0);
  branch.weightConstant = weightConstant.getOperation();
  branch.transposeEmpty = transposeEmpty.getOperation();
  branch.transpose = transpose.getOperation();
  branch.matmul = matmul.getOperation();
  branch.fill = outputInit->outputFill;
  branch.fillConstant = outputInit->outputFillConstant;
  return mlir::success();
}

static mlir::LogicalResult
matchLSTMMatmulBranchWithBias(mlir::Operation *branchAdd,
                              mlir::Operation *sharedOutputEmpty,
                              LSTMCellMatmulBranchMatch &branch) {
  if (!layer_utils::isAddfGeneric(branchAdd) ||
      !layer_utils::hasDpsInputsAndOperands(branchAdd, 2, 3))
    return mlir::failure();

  auto branchOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(branchAdd, 2);
  if (!branchOutputEmpty ||
      branchOutputEmpty.getOperation() != sharedOutputEmpty ||
      !layer_utils::operandProducersAreEither<MatmulOp, ConstantOp>(branchAdd))
    return mlir::failure();

  auto matmul = layer_utils::operandProducerOfType<MatmulOp>(branchAdd, 0);
  auto biasConstant =
      layer_utils::operandProducerOfType<ConstantOp>(branchAdd, 1);
  if (!matmul || !biasConstant) {
    matmul = layer_utils::operandProducerOfType<MatmulOp>(branchAdd, 1);
    biasConstant = layer_utils::operandProducerOfType<ConstantOp>(branchAdd, 0);
  }

  if (!matmul || !biasConstant ||
      mlir::failed(
          matchLSTMMatmulBranchCore(matmul, sharedOutputEmpty, branch)))
    return mlir::failure();

  branch.biasAdd = branchAdd;
  branch.biasConstant = biasConstant.getOperation();
  return mlir::success();
}

static mlir::LogicalResult
matchLSTMMatmulBranchWithoutBias(mlir::Operation *branchMatmul,
                                 mlir::Operation *sharedOutputEmpty,
                                 LSTMCellMatmulBranchMatch &branch) {
  auto matmul = llvm::dyn_cast_or_null<MatmulOp>(branchMatmul);
  if (!matmul)
    return mlir::failure();

  return matchLSTMMatmulBranchCore(matmul, sharedOutputEmpty, branch);
}

static bool isRecurrentBranch(const LSTMCellMatmulBranchMatch &branch,
                              mlir::RankedTensorType outputType) {
  auto activationType =
      layer_utils::getStaticRank2TensorType(branch.activationInput);
  auto weightType =
      layer_utils::getStaticRank2TensorType(branch.weightConstant);
  if (!activationType || !weightType)
    return false;

  int64_t batchSize = outputType.getShape()[0];
  int64_t hiddenSize = outputType.getShape()[1];
  return activationType.getShape()[0] == batchSize &&
         activationType.getShape()[1] == hiddenSize &&
         weightType.getShape()[0] == hiddenSize * 4 &&
         weightType.getShape()[1] == hiddenSize;
}

static bool isDistinctInputBranch(const LSTMCellMatmulBranchMatch &branch,
                                  mlir::RankedTensorType outputType) {
  auto activationType =
      layer_utils::getStaticRank2TensorType(branch.activationInput);
  auto weightType =
      layer_utils::getStaticRank2TensorType(branch.weightConstant);
  if (!activationType || !weightType)
    return false;

  int64_t batchSize = outputType.getShape()[0];
  int64_t hiddenSize = outputType.getShape()[1];
  int64_t inputSize = activationType.getShape()[1];
  return activationType.getShape()[0] == batchSize && inputSize != hiddenSize &&
         weightType.getShape()[0] == hiddenSize * 4 &&
         weightType.getShape()[1] == inputSize;
}

static bool
hasBlockArgumentActivation(const LSTMCellMatmulBranchMatch &branch) {
  return branch.activationInput &&
         !layer_utils::producerOf(branch.activationInput);
}

static bool hasProducedActivation(const LSTMCellMatmulBranchMatch &branch) {
  return branch.activationInput &&
         layer_utils::producerOf(branch.activationInput);
}

// Classifies the two LSTMCell matmul branches as input and hidden.
static bool assignBranchRoles(const LSTMCellMatmulBranchMatch &firstBranch,
                              const LSTMCellMatmulBranchMatch &secondBranch,
                              mlir::RankedTensorType outputType,
                              LSTMCellMatch &match) {
  bool firstIsInput = isDistinctInputBranch(firstBranch, outputType);
  bool firstIsRecurrent = isRecurrentBranch(firstBranch, outputType);
  bool secondIsInput = isDistinctInputBranch(secondBranch, outputType);
  bool secondIsRecurrent = isRecurrentBranch(secondBranch, outputType);

  if (firstIsInput && !firstIsRecurrent && secondIsRecurrent &&
      !secondIsInput) {
    match.inputBranch = firstBranch;
    match.hiddenBranch = secondBranch;
    return true;
  }

  if (secondIsInput && !secondIsRecurrent && firstIsRecurrent &&
      !firstIsInput) {
    match.inputBranch = secondBranch;
    match.hiddenBranch = firstBranch;
    return true;
  }

  if (firstIsRecurrent && secondIsRecurrent && !firstIsInput &&
      !secondIsInput) {
    if (hasProducedActivation(firstBranch) &&
        hasBlockArgumentActivation(secondBranch)) {
      match.inputBranch = firstBranch;
      match.hiddenBranch = secondBranch;
      return true;
    }

    if (hasProducedActivation(secondBranch) &&
        hasBlockArgumentActivation(firstBranch)) {
      match.inputBranch = secondBranch;
      match.hiddenBranch = firstBranch;
      return true;
    }
  }

  return false;
}

static mlir::LogicalResult
matchFinalHiddenMul(mlir::Operation *op, mlir::Operation *&outputGate,
                    mlir::Operation *&cellTanh,
                    mlir::Operation *&sharedHiddenOutputEmpty) {
  if (!layer_utils::isMulfGeneric(op) ||
      !layer_utils::hasDpsInputsAndOperands(op, 2, 3))
    return mlir::failure();

  auto outputEmpty = layer_utils::operandProducerOfType<EmptyOp>(op, 2);
  if (!outputEmpty)
    return mlir::failure();

  mlir::Operation *firstInput = layer_utils::operandProducer(op, 0);
  mlir::Operation *secondInput = layer_utils::operandProducer(op, 1);
  if (layer_utils::isSigmoidGeneric(firstInput) &&
      layer_utils::isTanhGeneric(secondInput)) {
    outputGate = firstInput;
    cellTanh = secondInput;
  } else if (layer_utils::isSigmoidGeneric(secondInput) &&
             layer_utils::isTanhGeneric(firstInput)) {
    outputGate = secondInput;
    cellTanh = firstInput;
  } else {
    return mlir::failure();
  }

  sharedHiddenOutputEmpty = outputEmpty.getOperation();
  return mlir::success();
}

static mlir::LogicalResult
matchForgetCellMul(mlir::Operation *op,
                   mlir::Operation *sharedHiddenOutputEmpty,
                   mlir::Operation *&forgetGate, mlir::Value &cellStateInput) {
  if (!layer_utils::isMulfGeneric(op) ||
      !layer_utils::hasDpsInputsAndOperands(op, 2, 3))
    return mlir::failure();

  auto outputEmpty = layer_utils::operandProducerOfType<EmptyOp>(op, 2);
  if (!outputEmpty || outputEmpty.getOperation() != sharedHiddenOutputEmpty)
    return mlir::failure();

  mlir::Operation *firstInput = layer_utils::operandProducer(op, 0);
  mlir::Operation *secondInput = layer_utils::operandProducer(op, 1);
  if (layer_utils::isSigmoidGeneric(firstInput) &&
      !layer_utils::isSigmoidGeneric(secondInput)) {
    forgetGate = firstInput;
    cellStateInput = op->getOperand(1);
    return mlir::success();
  }

  if (layer_utils::isSigmoidGeneric(secondInput) &&
      !layer_utils::isSigmoidGeneric(firstInput)) {
    forgetGate = secondInput;
    cellStateInput = op->getOperand(0);
    return mlir::success();
  }

  return mlir::failure();
}

static mlir::LogicalResult matchInputCandidateMul(
    mlir::Operation *op, mlir::Operation *sharedHiddenOutputEmpty,
    mlir::Operation *&inputGate, mlir::Operation *&candidateGate) {
  if (!layer_utils::isMulfGeneric(op) ||
      !layer_utils::hasDpsInputsAndOperands(op, 2, 3))
    return mlir::failure();

  auto outputEmpty = layer_utils::operandProducerOfType<EmptyOp>(op, 2);
  if (!outputEmpty || outputEmpty.getOperation() != sharedHiddenOutputEmpty)
    return mlir::failure();

  mlir::Operation *firstInput = layer_utils::operandProducer(op, 0);
  mlir::Operation *secondInput = layer_utils::operandProducer(op, 1);
  if (layer_utils::isSigmoidGeneric(firstInput) &&
      layer_utils::isTanhGeneric(secondInput)) {
    inputGate = firstInput;
    candidateGate = secondInput;
    return mlir::success();
  }

  if (layer_utils::isSigmoidGeneric(secondInput) &&
      layer_utils::isTanhGeneric(firstInput)) {
    inputGate = secondInput;
    candidateGate = firstInput;
    return mlir::success();
  }

  return mlir::failure();
}

static void appendBranchOps(llvm::SmallVectorImpl<mlir::Operation *> &ops,
                            const LSTMCellMatmulBranchMatch &branch) {
  match_utils::appendUniqueOp(ops, branch.weightConstant);
  match_utils::appendUniqueOp(ops, branch.transposeEmpty);
  match_utils::appendUniqueOp(ops, branch.transpose);
  match_utils::appendUniqueOp(ops, branch.fillConstant);
  match_utils::appendUniqueOp(ops, branch.fill);
  match_utils::appendUniqueOp(ops, branch.matmul);
  match_utils::appendUniqueOp(ops, branch.biasConstant);
  match_utils::appendUniqueOp(ops, branch.biasAdd);
}

static void collectMatchedOps(LSTMCellMatch &match) {
  match.ops.clear();

  match_utils::appendUniqueOp(match.ops, match.sharedMatmulOutputEmpty);
  appendBranchOps(match.ops, match.inputBranch);
  appendBranchOps(match.ops, match.hiddenBranch);
  match_utils::appendUniqueOp(match.ops, match.preActivationAddOp);
  match_utils::appendUniqueOp(match.ops, match.preActivationCollapseOp);
  layer_patterns::collectRecurrentGateIndexingOps(match.indexing, match.ops);
  match_utils::appendUniqueOp(match.ops, match.sharedHiddenOutputEmpty);
  match_utils::appendUniqueOp(match.ops, match.sigmoidOneConstant);
  layer_patterns::collectRecurrentGateSliceOps(match.inputGate, match.ops);
  layer_patterns::collectRecurrentGateSliceOps(match.forgetGate, match.ops);
  layer_patterns::collectRecurrentGateSliceOps(match.candidateGate, match.ops);
  layer_patterns::collectRecurrentGateSliceOps(match.outputGate, match.ops);
  match_utils::appendUniqueOp(match.ops, match.forgetCellMulOp);
  match_utils::appendUniqueOp(match.ops, match.inputCandidateMulOp);
  match_utils::appendUniqueOp(match.ops, match.cellAddOp);
  match_utils::appendUniqueOp(match.ops, match.cellTanhOp);
  match_utils::appendUniqueOp(match.ops, match.hiddenMulOp);
}

// Matches the complete LSTMCell gate and state-update pattern.
template <typename BranchMatcher>
static std::optional<LSTMCellMatch>
matchLSTMCell(mlir::Operation *op, bool hasBias, BranchMatcher branchMatcher) {
  if (!op || op->getNumResults() != 1)
    return std::nullopt;

  auto outputType = layer_utils::getStaticRank2TensorType(op);
  if (!outputType)
    return std::nullopt;

  int64_t hiddenSize = outputType.getShape()[1];
  int64_t gateWidth = hiddenSize * 4;

  mlir::Operation *outputGateActivation = nullptr;
  mlir::Operation *cellTanh = nullptr;
  mlir::Operation *sharedHiddenOutputEmpty = nullptr;
  if (mlir::failed(matchFinalHiddenMul(op, outputGateActivation, cellTanh,
                                       sharedHiddenOutputEmpty)))
    return std::nullopt;

  if (!layer_utils::hasDpsInputsAndOperands(cellTanh, 1, 2) ||
      !layer_utils::isTanhGeneric(cellTanh))
    return std::nullopt;

  auto cellTanhOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(cellTanh, 1);
  if (!cellTanhOutputEmpty ||
      cellTanhOutputEmpty.getOperation() != sharedHiddenOutputEmpty)
    return std::nullopt;

  mlir::Operation *cellAdd = layer_utils::operandProducer(cellTanh, 0);
  if (!layer_utils::isAddfGeneric(cellAdd) ||
      !layer_utils::hasDpsInputsAndOperands(cellAdd, 2, 3))
    return std::nullopt;

  auto cellAddOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(cellAdd, 2);
  if (!cellAddOutputEmpty ||
      cellAddOutputEmpty.getOperation() != sharedHiddenOutputEmpty)
    return std::nullopt;

  auto cellType = layer_utils::getStaticRank2TensorType(cellAdd);
  if (!cellType || cellType != outputType)
    return std::nullopt;

  mlir::Operation *forgetCellMul = nullptr;
  mlir::Operation *inputCandidateMul = nullptr;
  mlir::Operation *firstCellAddInput = layer_utils::operandProducer(cellAdd, 0);
  mlir::Operation *secondCellAddInput =
      layer_utils::operandProducer(cellAdd, 1);
  mlir::Operation *forgetGateActivation = nullptr;
  mlir::Operation *inputGateActivation = nullptr;
  mlir::Operation *candidateGateActivation = nullptr;
  mlir::Value cellStateInput;
  if (mlir::succeeded(
          matchForgetCellMul(firstCellAddInput, sharedHiddenOutputEmpty,
                             forgetGateActivation, cellStateInput)) &&
      mlir::succeeded(matchInputCandidateMul(
          secondCellAddInput, sharedHiddenOutputEmpty, inputGateActivation,
          candidateGateActivation))) {
    forgetCellMul = firstCellAddInput;
    inputCandidateMul = secondCellAddInput;
  } else if (mlir::succeeded(
                 matchForgetCellMul(secondCellAddInput, sharedHiddenOutputEmpty,
                                    forgetGateActivation, cellStateInput)) &&
             mlir::succeeded(matchInputCandidateMul(
                 firstCellAddInput, sharedHiddenOutputEmpty,
                 inputGateActivation, candidateGateActivation))) {
    forgetCellMul = secondCellAddInput;
    inputCandidateMul = firstCellAddInput;
  } else {
    return std::nullopt;
  }

  auto cellStateType = layer_utils::getStaticRank2TensorType(cellStateInput);
  if (!cellStateType || cellStateType != outputType)
    return std::nullopt;

  layer_patterns::RecurrentGateIndexingScaffoldMatch indexing;
  mlir::Operation *collapsedPreattivation = nullptr;
  mlir::Operation *sigmoidOneConstant = nullptr;
  layer_patterns::RecurrentGateSliceMatch outputGate;
  if (mlir::failed(matchGateSlice(outputGateActivation, sharedHiddenOutputEmpty,
                                  3 * hiddenSize, gateWidth, true, outputGate,
                                  indexing, collapsedPreattivation,
                                  sigmoidOneConstant)))
    return std::nullopt;

  layer_patterns::RecurrentGateSliceMatch forgetGate;
  if (mlir::failed(matchGateSlice(forgetGateActivation, sharedHiddenOutputEmpty,
                                  hiddenSize, gateWidth, true, forgetGate,
                                  indexing, collapsedPreattivation,
                                  sigmoidOneConstant)))
    return std::nullopt;

  layer_patterns::RecurrentGateSliceMatch inputGate;
  if (mlir::failed(matchGateSlice(inputGateActivation, sharedHiddenOutputEmpty,
                                  0, gateWidth, true, inputGate, indexing,
                                  collapsedPreattivation, sigmoidOneConstant)))
    return std::nullopt;

  layer_patterns::RecurrentGateSliceMatch candidateGate;
  if (mlir::failed(matchGateSlice(candidateGateActivation,
                                  sharedHiddenOutputEmpty, 2 * hiddenSize,
                                  gateWidth, false, candidateGate, indexing,
                                  collapsedPreattivation, sigmoidOneConstant)))
    return std::nullopt;

  auto collapsedPreattivationOp =
      llvm::dyn_cast_or_null<CollapseShapeOp>(collapsedPreattivation);
  auto collapsedType = collapsedPreattivationOp
                           ? llvm::dyn_cast<mlir::RankedTensorType>(
                                 collapsedPreattivationOp.getResult().getType())
                           : mlir::RankedTensorType();
  if (!collapsedPreattivationOp || !collapsedType ||
      !collapsedType.hasStaticShape() || collapsedType.getRank() != 1 ||
      collapsedType.getShape()[0] != gateWidth)
    return std::nullopt;

  mlir::Operation *preActivationAdd =
      layer_utils::producerOf(collapsedPreattivationOp.getSrc());
  if (!layer_utils::isAddfGeneric(preActivationAdd) ||
      !layer_utils::hasDpsInputsAndOperands(preActivationAdd, 2, 3))
    return std::nullopt;

  auto preActivationType =
      layer_utils::getStaticRank2TensorType(preActivationAdd);
  if (!preActivationType ||
      preActivationType.getShape()[0] != outputType.getShape()[0] ||
      preActivationType.getShape()[1] != gateWidth)
    return std::nullopt;

  auto matmulOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(preActivationAdd, 2);
  if (!matmulOutputEmpty)
    return std::nullopt;

  LSTMCellMatmulBranchMatch firstBranch;
  LSTMCellMatmulBranchMatch secondBranch;
  if (mlir::failed(
          branchMatcher(layer_utils::operandProducer(preActivationAdd, 0),
                        matmulOutputEmpty.getOperation(), firstBranch)) ||
      mlir::failed(
          branchMatcher(layer_utils::operandProducer(preActivationAdd, 1),
                        matmulOutputEmpty.getOperation(), secondBranch)))
    return std::nullopt;

  if (firstBranch.fill != secondBranch.fill)
    return std::nullopt;

  LSTMCellMatch match;
  match.sharedMatmulOutputEmpty = matmulOutputEmpty.getOperation();
  match.sharedFillOp = firstBranch.fill;
  match.sharedFillConstant = firstBranch.fillConstant;
  match.preActivationAddOp = preActivationAdd;
  match.preActivationCollapseOp = collapsedPreattivationOp.getOperation();
  match.indexing = indexing;
  match.inputGate = inputGate;
  match.forgetGate = forgetGate;
  match.candidateGate = candidateGate;
  match.outputGate = outputGate;
  match.sharedHiddenOutputEmpty = sharedHiddenOutputEmpty;
  match.sigmoidOneConstant = sigmoidOneConstant;
  match.forgetCellMulOp = forgetCellMul;
  match.inputCandidateMulOp = inputCandidateMul;
  match.cellAddOp = cellAdd;
  match.cellTanhOp = cellTanh;
  match.hiddenMulOp = op;
  match.root = op;
  match.cellStateInput = cellStateInput;
  match.hasBias = hasBias;
  if (!assignBranchRoles(firstBranch, secondBranch, outputType, match))
    return std::nullopt;

  collectMatchedOps(match);
  return match;
}

static std::optional<LSTMCellMatch> matchLSTMCellWithBias(mlir::Operation *op) {
  return matchLSTMCell(op, /*hasBias=*/true, matchLSTMMatmulBranchWithBias);
}

static std::optional<LSTMCellMatch>
matchLSTMCellWithoutBias(mlir::Operation *op) {
  return matchLSTMCell(op, /*hasBias=*/false, matchLSTMMatmulBranchWithoutBias);
}

static void replaceExternalUses(mlir::Value oldValue, mlir::Value newValue,
                                llvm::ArrayRef<mlir::Operation *> matchedOps) {
  for (mlir::OpOperand &use : llvm::make_early_inc_range(oldValue.getUses())) {
    if (!match_utils::containsOp(matchedOps, use.getOwner()))
      use.set(newValue);
  }
}

static void rewriteLSTMCellMatchToSculptorOp(const LSTMCellMatch &match,
                                           mlir::RewriterBase &rewriter) {
  mlir::Value wIh =
      canonicalizer_utils::firstResult(match.inputBranch.weightConstant);
  mlir::Value wHh =
      canonicalizer_utils::firstResult(match.hiddenBranch.weightConstant);
  if (!wIh || !wHh)
    return;

  mlir::Value bIh;
  mlir::Value bHh;
  if (match.hasBias) {
    bIh = canonicalizer_utils::firstResult(match.inputBranch.biasConstant);
    bHh = canonicalizer_utils::firstResult(match.hiddenBranch.biasConstant);
    if (!bIh || !bHh)
      return;
  }

  rewriter.setInsertionPoint(match.cellAddOp);
  auto lstmCellOp = rewriter.create<mlir::sculptor::NNLSTMCellOp>(
      match.root->getLoc(),
      mlir::TypeRange{match.root->getResult(0).getType(),
                      match.cellAddOp->getResult(0).getType()},
      match.inputBranch.activationInput, match.hiddenBranch.activationInput,
      match.cellStateInput, wIh, wHh, bIh, bHh,
      rewriter.getBoolAttr(match.hasBias));

  match.root->getResult(0).replaceAllUsesWith(lstmCellOp.getH());
  replaceExternalUses(match.cellAddOp->getResult(0), lstmCellOp.getC(),
                      match.ops);
  canonicalizer_utils::eraseDeadMatchedOps(match.ops, rewriter);
}

class LSTMCellCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  explicit LSTMCellCanonicalizer(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "lstm_cell"; }

  void canonicalize(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(func, rewriter, matchLSTMCellWithBias,
                                      rewriteLSTMCellMatchToSculptorOp);
    layer_patterns::rewriteAllMatches(func, rewriter, matchLSTMCellWithoutBias,
                                      rewriteLSTMCellMatchToSculptorOp);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerLSTMCellCanonicalizer(LayerCanonicalizers &canonicalizers,
                                   MLIRContext *context) {
  canonicalizers.push_back(std::make_unique<LSTMCellCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
