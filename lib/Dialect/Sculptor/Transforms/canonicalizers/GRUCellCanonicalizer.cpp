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

struct GRUMatmulBranchMatch {
  mlir::Value activationInput;
  mlir::Operation *weightConstant = nullptr;
  mlir::Operation *transposeEmpty = nullptr;
  mlir::Operation *transpose = nullptr;
  mlir::Operation *matmul = nullptr;
  mlir::Operation *fillEmpty = nullptr;
  mlir::Operation *fill = nullptr;
  mlir::Operation *fillConstant = nullptr;
  mlir::Operation *biasAdd = nullptr;
  mlir::Operation *biasConstant = nullptr;
  mlir::Operation *collapse = nullptr;
};

struct GRUCellMatch {
  GRUMatmulBranchMatch firstBranch;
  GRUMatmulBranchMatch secondBranch;
  GRUMatmulBranchMatch inputBranch;
  GRUMatmulBranchMatch hiddenBranch;
  layer_patterns::RecurrentGateIndexingScaffoldMatch indexing;
  layer_patterns::RecurrentGateSliceMatch firstResetGate;
  layer_patterns::RecurrentGateSliceMatch secondResetGate;
  layer_patterns::RecurrentGateSliceMatch firstUpdateGate;
  layer_patterns::RecurrentGateSliceMatch secondUpdateGate;
  layer_patterns::RecurrentGateSliceMatch directNewGate;
  layer_patterns::RecurrentGateSliceMatch resetNewGate;

  mlir::Operation *sharedOutputEmpty = nullptr;
  mlir::Operation *resetAdd = nullptr;
  mlir::Operation *resetGate = nullptr;
  mlir::Operation *updateAdd = nullptr;
  mlir::Operation *updateGate = nullptr;
  mlir::Operation *resetNewMul = nullptr;
  mlir::Operation *candidateAdd = nullptr;
  mlir::Operation *candidateTanh = nullptr;
  mlir::Operation *hiddenMinusCandidate = nullptr;
  mlir::Operation *updateBlendMul = nullptr;
  mlir::Operation *root = nullptr;
  mlir::Operation *sigmoidOneConstant = nullptr;

  bool hasBias = false;
  llvm::SmallVector<mlir::Operation *> ops;
};

static bool hasCollapsedGateVectorType(mlir::Operation *op, int64_t gateWidth) {
  auto collapse = llvm::dyn_cast_or_null<CollapseShapeOp>(op);
  auto type = collapse ? llvm::dyn_cast<mlir::RankedTensorType>(
                             collapse.getResult().getType())
                       : mlir::RankedTensorType();
  return collapse && type && type.hasStaticShape() && type.getRank() == 1 &&
         type.getShape()[0] == gateWidth;
}

static mlir::LogicalResult matchCollapsedExtractGeneric(
    mlir::Operation *op, mlir::Operation *&indexValueOp,
    mlir::Operation *&collapsedVectorOp, mlir::Operation *&zeroConst,
    mlir::Operation *&extentConst, mlir::Operation *&gatherEmpty) {
  auto generic = llvm::dyn_cast_or_null<mlir::linalg::GenericOp>(op);
  if (!generic || !layer_utils::hasDpsInputsAndOperands(op, 1, 2))
    return mlir::failure();

  auto outputEmpty = layer_utils::operandProducerOfType<EmptyOp>(op, 1);
  if (!outputEmpty || !generic.getRegion().hasOneBlock())
    return mlir::failure();

  mlir::Block &block = generic.getRegion().front();
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
  if (!cmp || !add || !select || !cast || !extract || !yield || it != e ||
      cmp.getPredicate() != mlir::arith::CmpIPredicate::slt)
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

// Reuses one gate indexing scaffold across all GRUCell gate slices.
static mlir::LogicalResult matchSharedGateIndexScaffold(
    CollapseShapeOp collapsedIndices, mlir::Operation *expectedZeroConst,
    mlir::Operation *expectedExtentConst, int64_t expectedExtent,
    layer_patterns::RecurrentGateIndexingScaffoldMatch &scaffold) {
  if (!layer_utils::constantOpHasI64Value(expectedExtentConst, expectedExtent))
    return mlir::failure();

  if (scaffold.collapsedIndices) {
    return scaffold.collapsedIndices == collapsedIndices.getOperation() &&
                   scaffold.zeroIndexConstant == expectedZeroConst &&
                   scaffold.indexExtentConstant == expectedExtentConst
               ? mlir::success()
               : mlir::failure();
  }

  mlir::Operation *combinedIndices =
      layer_utils::producerOf(collapsedIndices.getSrc());
  if (!combinedIndices || !layer_utils::isAddiGeneric(combinedIndices) ||
      !layer_utils::hasDpsInputsAndOperands(combinedIndices, 2, 3))
    return mlir::failure();

  auto combinedIndicesEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(combinedIndices, 2);
  mlir::Operation *baseOffsetGeneric =
      layer_utils::operandProducer(combinedIndices, 0);
  auto rangeExpand =
      layer_utils::operandProducerOfType<ExpandShapeOp>(combinedIndices, 1);
  if (!combinedIndicesEmpty || !baseOffsetGeneric || !rangeExpand)
    return mlir::failure();

  mlir::Operation *rangeGeneric = layer_utils::producerOf(rangeExpand.getSrc());
  auto rangeEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(rangeGeneric, 0);
  if (!rangeGeneric || !rangeEmpty ||
      mlir::failed(layer_patterns::matchIndexRangeGeneric(rangeGeneric)) ||
      mlir::failed(layer_patterns::matchMultiplyByConstantGeneric(
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
  auto zeroIndexEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(zeroIndexGeneric, 0);
  if (!zeroIndexGeneric || !zeroIndexEmpty ||
      mlir::failed(layer_patterns::matchYieldingConstantGeneric(
          zeroIndexGeneric, expectedZeroConst)))
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

static mlir::LogicalResult matchGateSliceValue(
    mlir::Value value, mlir::Operation *expectedCollapsedVector,
    int64_t expectedOffset, int64_t expectedExtent,
    layer_patterns::RecurrentGateSliceMatch &gate,
    layer_patterns::RecurrentGateIndexingScaffoldMatch &scaffold) {
  auto expand = layer_utils::producerOfType<ExpandShapeOp>(value);
  if (!expand)
    return mlir::failure();

  mlir::Operation *gather = layer_utils::producerOf(expand.getSrc());
  mlir::Operation *indexValueOp = nullptr;
  mlir::Operation *collapsedVectorOp = nullptr;
  mlir::Operation *zeroConst = nullptr;
  mlir::Operation *extentConst = nullptr;
  mlir::Operation *gatherEmpty = nullptr;
  if (!gather ||
      mlir::failed(matchCollapsedExtractGeneric(gather, indexValueOp,
                                                collapsedVectorOp, zeroConst,
                                                extentConst, gatherEmpty)) ||
      collapsedVectorOp != expectedCollapsedVector)
    return mlir::failure();

  if (scaffold.gatherEmpty && scaffold.gatherEmpty != gatherEmpty)
    return mlir::failure();
  scaffold.gatherEmpty = gatherEmpty;

  if (expectedOffset == 0) {
    auto collapsedIndices = llvm::dyn_cast<CollapseShapeOp>(indexValueOp);
    if (!collapsedIndices || mlir::failed(matchSharedGateIndexScaffold(
                                 collapsedIndices, zeroConst, extentConst,
                                 expectedExtent, scaffold)))
      return mlir::failure();
  } else {
    mlir::Operation *baseIndicesOp =
        layer_utils::operandProducer(indexValueOp, 0);
    auto collapsedIndices =
        llvm::dyn_cast_or_null<CollapseShapeOp>(baseIndicesOp);
    if (!collapsedIndices || mlir::failed(matchSharedGateIndexScaffold(
                                 collapsedIndices, zeroConst, extentConst,
                                 expectedExtent, scaffold)))
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
  return mlir::success();
}

static mlir::LogicalResult
matchGatePairAdd(mlir::Operation *op, mlir::Operation *firstCollapsedVector,
                 mlir::Operation *secondCollapsedVector, int64_t expectedOffset,
                 int64_t expectedExtent,
                 layer_patterns::RecurrentGateSliceMatch &firstGate,
                 layer_patterns::RecurrentGateSliceMatch &secondGate,
                 layer_patterns::RecurrentGateIndexingScaffoldMatch &scaffold) {
  if (!layer_utils::isAddfGeneric(op) ||
      !layer_utils::hasDpsInputsAndOperands(op, 2, 3))
    return mlir::failure();

  if (mlir::succeeded(matchGateSliceValue(
          op->getOperand(0), firstCollapsedVector, expectedOffset,
          expectedExtent, firstGate, scaffold)) &&
      mlir::succeeded(matchGateSliceValue(
          op->getOperand(1), secondCollapsedVector, expectedOffset,
          expectedExtent, secondGate, scaffold)))
    return mlir::success();

  if (mlir::succeeded(matchGateSliceValue(
          op->getOperand(1), firstCollapsedVector, expectedOffset,
          expectedExtent, firstGate, scaffold)) &&
      mlir::succeeded(matchGateSliceValue(
          op->getOperand(0), secondCollapsedVector, expectedOffset,
          expectedExtent, secondGate, scaffold)))
    return mlir::success();

  return mlir::failure();
}

static mlir::LogicalResult
matchGRUMatmulBranchCore(MatmulOp matmul, GRUMatmulBranchMatch &branch) {
  if (!layer_utils::hasDpsInputsAndOperands(matmul.getOperation(), 2, 3))
    return mlir::failure();

  auto outputInit =
      layer_patterns::matchFillOutputInit(matmul.getOperation(), 2);
  if (!outputInit)
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

  auto weightConstant = layer_utils::operandProducerOfType<ConstantOp>(
      transpose.getOperation(), 0);
  auto transposeEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(transpose.getOperation(), 1);
  if (!weightConstant || !transposeEmpty)
    return mlir::failure();

  branch.activationInput =
      firstTranspose ? matmul.getOperand(1) : matmul.getOperand(0);
  branch.weightConstant = weightConstant.getOperation();
  branch.transposeEmpty = transposeEmpty.getOperation();
  branch.transpose = transpose.getOperation();
  branch.matmul = matmul.getOperation();
  branch.fillEmpty = outputInit->outputEmpty;
  branch.fill = outputInit->outputFill;
  branch.fillConstant = outputInit->outputFillConstant;
  return mlir::success();
}

static mlir::LogicalResult
matchGRUPreactivationBranch(CollapseShapeOp collapse, bool hasBias,
                            GRUMatmulBranchMatch &branch) {
  if (hasBias) {
    mlir::Operation *biasAdd = layer_utils::producerOf(collapse.getSrc());
    if (!layer_utils::isAddfGeneric(biasAdd) ||
        !layer_utils::hasDpsInputsAndOperands(biasAdd, 2, 3) ||
        !layer_utils::operandProducersAreEither<MatmulOp, ConstantOp>(biasAdd))
      return mlir::failure();

    auto matmul = layer_utils::operandProducerOfType<MatmulOp>(biasAdd, 0);
    auto biasConstant =
        layer_utils::operandProducerOfType<ConstantOp>(biasAdd, 1);
    if (!matmul || !biasConstant) {
      matmul = layer_utils::operandProducerOfType<MatmulOp>(biasAdd, 1);
      biasConstant = layer_utils::operandProducerOfType<ConstantOp>(biasAdd, 0);
    }

    if (!matmul || !biasConstant ||
        mlir::failed(matchGRUMatmulBranchCore(matmul, branch)))
      return mlir::failure();

    branch.biasAdd = biasAdd;
    branch.biasConstant = biasConstant.getOperation();
  } else {
    auto matmul = layer_utils::producerOfType<MatmulOp>(collapse.getSrc());
    if (!matmul || mlir::failed(matchGRUMatmulBranchCore(matmul, branch)))
      return mlir::failure();
  }

  branch.collapse = collapse.getOperation();
  return mlir::success();
}

static mlir::LogicalResult
matchGateActivation(mlir::Operation *activation, mlir::Operation *expectedInput,
                    mlir::Operation *expectedOutputEmpty,
                    mlir::Operation *&unitConstant) {
  if (!layer_utils::isSigmoidGeneric(activation) ||
      !layer_utils::hasDpsInputsAndOperands(activation, 1, 2) ||
      layer_utils::operandProducer(activation, 0) != expectedInput)
    return mlir::failure();

  auto outputEmpty = layer_utils::operandProducerOfType<EmptyOp>(activation, 1);
  if (!outputEmpty || outputEmpty.getOperation() != expectedOutputEmpty)
    return mlir::failure();

  mlir::Operation *foundUnitConstant =
      layer_patterns::getSigmoidUnitConstant(activation);
  if (!foundUnitConstant)
    return mlir::failure();
  if (unitConstant && unitConstant != foundUnitConstant)
    return mlir::failure();
  unitConstant = foundUnitConstant;
  return mlir::success();
}

// Matches the reset-gated new-state dependency in a GRUCell.
static mlir::LogicalResult matchNewGatePath(
    mlir::Operation *candidateAdd, mlir::Operation *firstCollapsedVector,
    mlir::Operation *secondCollapsedVector, mlir::Operation *resetGate,
    mlir::Operation *sharedOutputEmpty, int64_t newGateOffset,
    int64_t gateWidth, layer_patterns::RecurrentGateSliceMatch &directNewGate,
    layer_patterns::RecurrentGateSliceMatch &resetNewGate,
    layer_patterns::RecurrentGateIndexingScaffoldMatch &scaffold,
    mlir::Operation *&resetNewMul) {
  if (!layer_utils::isAddfGeneric(candidateAdd) ||
      !layer_utils::hasDpsInputsAndOperands(candidateAdd, 2, 3))
    return mlir::failure();

  auto matchOrdered = [&](unsigned directIndex, unsigned resetIndex,
                          mlir::Operation *directCollapsed,
                          mlir::Operation *resetCollapsed) {
    layer_patterns::RecurrentGateSliceMatch directGate;
    layer_patterns::RecurrentGateSliceMatch resetGateSlice;
    if (mlir::failed(matchGateSliceValue(candidateAdd->getOperand(directIndex),
                                         directCollapsed, newGateOffset,
                                         gateWidth, directGate, scaffold)))
      return mlir::failure();

    mlir::Operation *mul =
        layer_utils::operandProducer(candidateAdd, resetIndex);
    if (!layer_utils::isMulfGeneric(mul) ||
        !layer_utils::hasDpsInputsAndOperands(mul, 2, 3))
      return mlir::failure();

    auto outputEmpty = layer_utils::operandProducerOfType<EmptyOp>(mul, 2);
    if (!outputEmpty || outputEmpty.getOperation() != sharedOutputEmpty)
      return mlir::failure();

    if (layer_utils::operandProducer(mul, 0) == resetGate) {
      if (mlir::failed(matchGateSliceValue(mul->getOperand(1), resetCollapsed,
                                           newGateOffset, gateWidth,
                                           resetGateSlice, scaffold)))
        return mlir::failure();
    } else if (layer_utils::operandProducer(mul, 1) == resetGate) {
      if (mlir::failed(matchGateSliceValue(mul->getOperand(0), resetCollapsed,
                                           newGateOffset, gateWidth,
                                           resetGateSlice, scaffold)))
        return mlir::failure();
    } else {
      return mlir::failure();
    }

    directNewGate = directGate;
    resetNewGate = resetGateSlice;
    resetNewMul = mul;
    return mlir::success();
  };

  if (mlir::succeeded(
          matchOrdered(0, 1, firstCollapsedVector, secondCollapsedVector)) ||
      mlir::succeeded(
          matchOrdered(1, 0, firstCollapsedVector, secondCollapsedVector)) ||
      mlir::succeeded(
          matchOrdered(0, 1, secondCollapsedVector, firstCollapsedVector)) ||
      mlir::succeeded(
          matchOrdered(1, 0, secondCollapsedVector, firstCollapsedVector)))
    return mlir::success();

  return mlir::failure();
}

static bool isRecurrentBranch(const GRUMatmulBranchMatch &branch,
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
         weightType.getShape()[0] == hiddenSize * 3 &&
         weightType.getShape()[1] == hiddenSize;
}

static bool isDistinctInputBranch(const GRUMatmulBranchMatch &branch,
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
         weightType.getShape()[0] == hiddenSize * 3 &&
         weightType.getShape()[1] == inputSize;
}

static bool hasBlockArgumentActivation(const GRUMatmulBranchMatch &branch) {
  return branch.activationInput &&
         !layer_utils::producerOf(branch.activationInput);
}

static bool hasProducedActivation(const GRUMatmulBranchMatch &branch) {
  return branch.activationInput &&
         layer_utils::producerOf(branch.activationInput);
}

// Classifies the two GRUCell matmul branches as input and hidden.
static bool assignBranchRoles(const GRUMatmulBranchMatch &firstBranch,
                              const GRUMatmulBranchMatch &secondBranch,
                              mlir::RankedTensorType outputType,
                              GRUCellMatch &match) {
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

// Checks the final GRUCell blend uses previous hidden minus candidate.
static bool hasExpectedHiddenMinusCandidate(const GRUCellMatch &match) {
  if (!match.hiddenMinusCandidate ||
      !layer_utils::hasDpsInputsAndOperands(match.hiddenMinusCandidate, 2, 3))
    return false;

  return match.hiddenMinusCandidate->getOperand(0) ==
             match.hiddenBranch.activationInput &&
         layer_utils::operandProducer(match.hiddenMinusCandidate, 1) ==
             match.candidateTanh;
}

static void collectBranchOps(const GRUMatmulBranchMatch &branch,
                             llvm::SmallVectorImpl<mlir::Operation *> &ops) {
  match_utils::appendUniqueOp(ops, branch.weightConstant);
  match_utils::appendUniqueOp(ops, branch.transposeEmpty);
  match_utils::appendUniqueOp(ops, branch.transpose);
  match_utils::appendUniqueOp(ops, branch.fillEmpty);
  match_utils::appendUniqueOp(ops, branch.fillConstant);
  match_utils::appendUniqueOp(ops, branch.fill);
  match_utils::appendUniqueOp(ops, branch.matmul);
  match_utils::appendUniqueOp(ops, branch.biasConstant);
  match_utils::appendUniqueOp(ops, branch.biasAdd);
  match_utils::appendUniqueOp(ops, branch.collapse);
}

static void collectMatchedOps(GRUCellMatch &match) {
  match.ops.clear();

  collectBranchOps(match.firstBranch, match.ops);
  collectBranchOps(match.secondBranch, match.ops);
  layer_patterns::collectRecurrentGateIndexingOps(match.indexing, match.ops);
  layer_patterns::collectRecurrentGateSliceOps(match.firstResetGate, match.ops);
  layer_patterns::collectRecurrentGateSliceOps(match.secondResetGate,
                                               match.ops);
  match_utils::appendUniqueOp(match.ops, match.sharedOutputEmpty);
  match_utils::appendUniqueOp(match.ops, match.resetAdd);
  match_utils::appendUniqueOp(match.ops, match.sigmoidOneConstant);
  match_utils::appendUniqueOp(match.ops, match.resetGate);
  layer_patterns::collectRecurrentGateSliceOps(match.firstUpdateGate,
                                               match.ops);
  layer_patterns::collectRecurrentGateSliceOps(match.secondUpdateGate,
                                               match.ops);
  match_utils::appendUniqueOp(match.ops, match.updateAdd);
  match_utils::appendUniqueOp(match.ops, match.updateGate);
  layer_patterns::collectRecurrentGateSliceOps(match.directNewGate, match.ops);
  layer_patterns::collectRecurrentGateSliceOps(match.resetNewGate, match.ops);
  match_utils::appendUniqueOp(match.ops, match.resetNewMul);
  match_utils::appendUniqueOp(match.ops, match.candidateAdd);
  match_utils::appendUniqueOp(match.ops, match.candidateTanh);
  match_utils::appendUniqueOp(match.ops, match.hiddenMinusCandidate);
  match_utils::appendUniqueOp(match.ops, match.updateBlendMul);
  match_utils::appendUniqueOp(match.ops, match.root);
}

static std::optional<GRUCellMatch> matchGRUCell(mlir::Operation *op,
                                                bool hasBias) {
  if (!layer_utils::isAddfGeneric(op) ||
      !layer_utils::hasDpsInputsAndOperands(op, 2, 3) ||
      op->getNumResults() != 1)
    return std::nullopt;

  auto outputType = layer_utils::getStaticRank2TensorType(op);
  if (!outputType)
    return std::nullopt;
  int64_t hiddenSize = outputType.getShape()[1];
  int64_t gateWidth = hiddenSize * 3;

  auto outputEmpty = layer_utils::operandProducerOfType<EmptyOp>(op, 2);
  if (!outputEmpty)
    return std::nullopt;

  mlir::Operation *candidateTanh = nullptr;
  mlir::Operation *updateBlendMul = nullptr;
  if (layer_utils::isTanhGeneric(layer_utils::operandProducer(op, 0))) {
    candidateTanh = layer_utils::operandProducer(op, 0);
    updateBlendMul = layer_utils::operandProducer(op, 1);
  } else if (layer_utils::isTanhGeneric(layer_utils::operandProducer(op, 1))) {
    candidateTanh = layer_utils::operandProducer(op, 1);
    updateBlendMul = layer_utils::operandProducer(op, 0);
  } else {
    return std::nullopt;
  }

  if (!layer_utils::isMulfGeneric(updateBlendMul) ||
      !layer_utils::hasDpsInputsAndOperands(updateBlendMul, 2, 3))
    return std::nullopt;

  auto blendOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(updateBlendMul, 2);
  if (!blendOutputEmpty ||
      blendOutputEmpty.getOperation() != outputEmpty.getOperation())
    return std::nullopt;

  mlir::Operation *hiddenMinusCandidate = nullptr;
  mlir::Operation *updateGate = nullptr;
  if (layer_utils::isSubfGeneric(
          layer_utils::operandProducer(updateBlendMul, 0)) &&
      layer_utils::isSigmoidGeneric(
          layer_utils::operandProducer(updateBlendMul, 1))) {
    hiddenMinusCandidate = layer_utils::operandProducer(updateBlendMul, 0);
    updateGate = layer_utils::operandProducer(updateBlendMul, 1);
  } else if (layer_utils::isSubfGeneric(
                 layer_utils::operandProducer(updateBlendMul, 1)) &&
             layer_utils::isSigmoidGeneric(
                 layer_utils::operandProducer(updateBlendMul, 0))) {
    hiddenMinusCandidate = layer_utils::operandProducer(updateBlendMul, 1);
    updateGate = layer_utils::operandProducer(updateBlendMul, 0);
  } else {
    return std::nullopt;
  }

  if (!layer_utils::hasDpsInputsAndOperands(hiddenMinusCandidate, 2, 3) ||
      layer_utils::operandProducer(hiddenMinusCandidate, 1) != candidateTanh)
    return std::nullopt;

  auto subOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(hiddenMinusCandidate, 2);
  if (!subOutputEmpty ||
      subOutputEmpty.getOperation() != outputEmpty.getOperation())
    return std::nullopt;

  if (!layer_utils::hasDpsInputsAndOperands(candidateTanh, 1, 2))
    return std::nullopt;

  auto tanhOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(candidateTanh, 1);
  if (!tanhOutputEmpty ||
      tanhOutputEmpty.getOperation() != outputEmpty.getOperation())
    return std::nullopt;

  mlir::Operation *candidateAdd =
      layer_utils::operandProducer(candidateTanh, 0);
  mlir::Operation *updateAdd = layer_utils::operandProducer(updateGate, 0);
  mlir::Operation *resetAdd = nullptr;
  if (!candidateAdd || !updateAdd)
    return std::nullopt;

  GRUCellMatch match;
  match.sharedOutputEmpty = outputEmpty.getOperation();
  match.root = op;
  match.candidateTanh = candidateTanh;
  match.candidateAdd = candidateAdd;
  match.updateBlendMul = updateBlendMul;
  match.hiddenMinusCandidate = hiddenMinusCandidate;
  match.updateGate = updateGate;
  match.updateAdd = updateAdd;
  match.hasBias = hasBias;

  auto matchWithCollapsedOrder =
      [&](mlir::Operation *firstCollapsedVector,
          mlir::Operation *secondCollapsedVector) -> mlir::LogicalResult {
    GRUCellMatch candidate = match;
    auto firstCollapse =
        llvm::dyn_cast_or_null<CollapseShapeOp>(firstCollapsedVector);
    auto secondCollapse =
        llvm::dyn_cast_or_null<CollapseShapeOp>(secondCollapsedVector);
    if (!firstCollapse || !secondCollapse ||
        !hasCollapsedGateVectorType(firstCollapsedVector, gateWidth) ||
        !hasCollapsedGateVectorType(secondCollapsedVector, gateWidth) ||
        mlir::failed(matchGRUPreactivationBranch(firstCollapse, hasBias,
                                                 candidate.firstBranch)) ||
        mlir::failed(matchGRUPreactivationBranch(secondCollapse, hasBias,
                                                 candidate.secondBranch)))
      return mlir::failure();

    if (candidate.firstBranch.fill != candidate.secondBranch.fill)
      return mlir::failure();

    if (mlir::failed(matchGatePairAdd(
            resetAdd, firstCollapsedVector, secondCollapsedVector, 0, gateWidth,
            candidate.firstResetGate, candidate.secondResetGate,
            candidate.indexing)))
      return mlir::failure();

    if (mlir::failed(matchGateActivation(candidate.resetGate, resetAdd,
                                         candidate.sharedOutputEmpty,
                                         candidate.sigmoidOneConstant)))
      return mlir::failure();

    if (mlir::failed(matchGatePairAdd(
            candidate.updateAdd, firstCollapsedVector, secondCollapsedVector,
            hiddenSize, gateWidth, candidate.firstUpdateGate,
            candidate.secondUpdateGate, candidate.indexing)))
      return mlir::failure();

    if (mlir::failed(matchGateActivation(
            candidate.updateGate, candidate.updateAdd,
            candidate.sharedOutputEmpty, candidate.sigmoidOneConstant)))
      return mlir::failure();

    if (mlir::failed(matchNewGatePath(
            candidate.candidateAdd, firstCollapsedVector, secondCollapsedVector,
            candidate.resetGate, candidate.sharedOutputEmpty, 2 * hiddenSize,
            gateWidth, candidate.directNewGate, candidate.resetNewGate,
            candidate.indexing, candidate.resetNewMul)))
      return mlir::failure();

    if (!assignBranchRoles(candidate.firstBranch, candidate.secondBranch,
                           outputType, candidate))
      return mlir::failure();

    if (!hasExpectedHiddenMinusCandidate(candidate))
      return mlir::failure();

    match = candidate;
    return mlir::success();
  };

  if (layer_utils::isMulfGeneric(
          layer_utils::operandProducer(candidateAdd, 0))) {
    match.resetNewMul = layer_utils::operandProducer(candidateAdd, 0);
    resetAdd = layer_utils::operandProducer(match.resetNewMul, 0);
    if (!layer_utils::isSigmoidGeneric(resetAdd))
      resetAdd = layer_utils::operandProducer(match.resetNewMul, 1);
  } else if (layer_utils::isMulfGeneric(
                 layer_utils::operandProducer(candidateAdd, 1))) {
    match.resetNewMul = layer_utils::operandProducer(candidateAdd, 1);
    resetAdd = layer_utils::operandProducer(match.resetNewMul, 0);
    if (!layer_utils::isSigmoidGeneric(resetAdd))
      resetAdd = layer_utils::operandProducer(match.resetNewMul, 1);
  } else {
    return std::nullopt;
  }

  match.resetGate = resetAdd;
  resetAdd = layer_utils::operandProducer(match.resetGate, 0);
  match.resetAdd = resetAdd;
  if (!resetAdd)
    return std::nullopt;

  auto firstResetExpand =
      layer_utils::operandProducerOfType<ExpandShapeOp>(resetAdd, 0);
  auto secondResetExpand =
      layer_utils::operandProducerOfType<ExpandShapeOp>(resetAdd, 1);
  if (!firstResetExpand || !secondResetExpand)
    return std::nullopt;

  mlir::Operation *firstGather =
      layer_utils::producerOf(firstResetExpand.getSrc());
  mlir::Operation *secondGather =
      layer_utils::producerOf(secondResetExpand.getSrc());
  mlir::Operation *wrappedIndex = nullptr;
  mlir::Operation *firstCollapsedVector = nullptr;
  mlir::Operation *secondCollapsedVector = nullptr;
  mlir::Operation *zeroConstant = nullptr;
  mlir::Operation *extentConstant = nullptr;
  mlir::Operation *gatherEmpty = nullptr;
  if (!firstGather || !secondGather ||
      mlir::failed(matchCollapsedExtractGeneric(
          firstGather, wrappedIndex, firstCollapsedVector, zeroConstant,
          extentConstant, gatherEmpty)) ||
      mlir::failed(matchCollapsedExtractGeneric(
          secondGather, wrappedIndex, secondCollapsedVector, zeroConstant,
          extentConstant, gatherEmpty)))
    return std::nullopt;

  if (mlir::failed(
          matchWithCollapsedOrder(firstCollapsedVector, secondCollapsedVector)))
    return std::nullopt;

  collectMatchedOps(match);
  return match;
}

static std::optional<GRUCellMatch> matchGRUCellWithBias(mlir::Operation *op) {
  return matchGRUCell(op, /*hasBias=*/true);
}

static std::optional<GRUCellMatch>
matchGRUCellWithoutBias(mlir::Operation *op) {
  return matchGRUCell(op, /*hasBias=*/false);
}

static void rewriteGRUCellMatchToSculptorOp(const GRUCellMatch &match,
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

  rewriter.setInsertionPoint(match.root);
  auto gruCellOp = rewriter.create<mlir::sculptor::NNGRUCellOp>(
      match.root->getLoc(), match.root->getResult(0).getType(),
      match.inputBranch.activationInput, match.hiddenBranch.activationInput,
      wIh, wHh, bIh, bHh, rewriter.getBoolAttr(match.hasBias));

  match.root->getResult(0).replaceAllUsesWith(gruCellOp.getH());
  canonicalizer_utils::eraseDeadMatchedOps(match.ops, rewriter);
}

class GRUCellCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  explicit GRUCellCanonicalizer(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "gru_cell"; }

  void canonicalize(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(func, rewriter, matchGRUCellWithBias,
                                      rewriteGRUCellMatchToSculptorOp);
    layer_patterns::rewriteAllMatches(func, rewriter, matchGRUCellWithoutBias,
                                      rewriteGRUCellMatchToSculptorOp);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerGRUCellCanonicalizer(LayerCanonicalizers &canonicalizers,
                                  MLIRContext *context) {
  canonicalizers.push_back(std::make_unique<GRUCellCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
