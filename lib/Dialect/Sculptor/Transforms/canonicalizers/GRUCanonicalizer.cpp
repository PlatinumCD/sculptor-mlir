#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/TensorTypeUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/LinalgMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/OperandRelationUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/RecurrentLayerPatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <utility>

namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace layer_utils = mlir::sculptor::layer_utils;
namespace tensor_type = mlir::sculptor::tensor_type;
namespace linalg_match = mlir::sculptor::linalg_match;

namespace {

using mlir::arith::AddFOp;
using mlir::arith::ConstantOp;
using mlir::func::ReturnOp;
using mlir::linalg::BatchMatmulOp;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::tensor::CollapseShapeOp;
using mlir::tensor::ConcatOp;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;
using mlir::tensor::ExtractSliceOp;

struct GRUTypes {
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType hiddenStateTy;
  mlir::RankedTensorType sequenceResultTy;
  mlir::RankedTensorType hiddenResultTy;
  int64_t layerCount = 0;
  int64_t sequenceLength = 0;
  int64_t batchSize = 0;
  int64_t inputSize = 0;
  int64_t hiddenSize = 0;
  bool hasBias = false;
};

struct LayerInputProjection {
  mlir::Value source;
  ConstantOp weightConstant;
  TransposeOp weightTranspose;
  BatchMatmulOp batchMatmulOp;
  MatmulOp matmulOp;
  GenericOp biasAddOp;
  ConstantOp biasConstant;
  ConcatOp previousLayerConcat;
};

struct RecurrentProjection {
  mlir::Value activation;
  ConstantOp weightConstant;
  TransposeOp weightTranspose;
  MatmulOp matmulOp;
  CollapseShapeOp collapseOp;
  GenericOp biasAddOp;
  ConstantOp biasConstant;
};

struct GRUGateSlices {
  layer_patterns::RecurrentGateIndexingScaffoldMatch inputIndexing;
  layer_patterns::RecurrentGateIndexingScaffoldMatch hiddenIndexing;
  layer_patterns::RecurrentGateSliceMatch inputResetGate;
  layer_patterns::RecurrentGateSliceMatch hiddenResetGate;
  layer_patterns::RecurrentGateSliceMatch inputUpdateGate;
  layer_patterns::RecurrentGateSliceMatch hiddenUpdateGate;
  layer_patterns::RecurrentGateSliceMatch inputNewGate;
  layer_patterns::RecurrentGateSliceMatch hiddenNewGate;
};

struct GRUStep {
  mlir::Value hiddenOutput;
  GenericOp finalHiddenAddOp;
  GenericOp candidateTanhOp;
  GenericOp updateBlendMulOp;
  GenericOp hiddenMinusCandidateOp;
  GenericOp candidateAddOp;
  GenericOp resetNewMulOp;
  GenericOp resetGateOp;
  GenericOp resetAddOp;
  GenericOp updateGateOp;
  GenericOp updateAddOp;
  CollapseShapeOp inputCollapseOp;
  ExtractSliceOp inputSliceOp;
  RecurrentProjection recurrentProjection;
  GRUGateSlices gates;
};

struct GRUHiddenUpdate {
  GenericOp finalHiddenAddOp;
  GenericOp candidateTanhOp;
  GenericOp updateBlendMulOp;
  GenericOp hiddenMinusCandidateOp;
  GenericOp candidateAddOp;
  GenericOp updateGateOp;
  GenericOp updateAddOp;
  EmptyOp sharedOutputEmpty;
};

struct GRUResetUpdateGates {
  GenericOp resetGateOp;
  GenericOp resetAddOp;
};

struct GRUCollapsedResetInputs {
  mlir::Operation *firstCollapsedVector = nullptr;
  mlir::Operation *secondCollapsedVector = nullptr;
};

struct GRULayer {
  ConcatOp outputConcat;
  LayerInputProjection inputProjection;
  llvm::SmallVector<GRUStep, 4> steps;
};

struct GRUMatch {
  ReturnOp returnOp;
  TransposeOp sequenceTranspose;
  ConcatOp sequenceConcat;
  ExpandShapeOp hiddenExpand;
  ConcatOp hiddenConcat;
  llvm::SmallVector<GRULayer, 4> layers;
  GRUTypes types;
};

static bool isAddfGeneric(GenericOp genericOp) {
  if (!linalg_match::hasExpectedBodyShape(genericOp, /*inputCount=*/2))
    return false;

  mlir::Block &body = genericOp.getRegion().front();
  auto addOp = llvm::dyn_cast<AddFOp>(body.getOperations().front());
  auto yieldOp =
      llvm::dyn_cast<mlir::linalg::YieldOp>(body.getOperations().back());
  return addOp && yieldOp && yieldOp.getValues().size() == 1 &&
         ((addOp.getOperand(0) == body.getArgument(0) &&
           addOp.getOperand(1) == body.getArgument(1)) ||
          (addOp.getOperand(0) == body.getArgument(1) &&
           addOp.getOperand(1) == body.getArgument(0))) &&
         yieldOp.getValues().front() == addOp.getResult();
}

static bool isYieldInputGeneric(GenericOp genericOp) {
  if (!genericOp || genericOp.getInputs().size() != 1 ||
      genericOp.getOutputs().size() != 1 || genericOp.getNumResults() != 1 ||
      !genericOp.getRegion().hasOneBlock())
    return false;

  mlir::Block &body = genericOp.getRegion().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 1)
    return false;

  auto yieldOp = llvm::dyn_cast<mlir::linalg::YieldOp>(body.front());
  return yieldOp && yieldOp.getValues().size() == 1 &&
         yieldOp.getValues().front() == body.getArgument(0);
}

static mlir::FailureOr<RecurrentProjection>
matchRecurrentProjection(mlir::Operation *collapsedVectorOp,
                         const GRUTypes &types) {
  int64_t gateWidth = 3 * types.hiddenSize;
  auto collapse = llvm::dyn_cast_or_null<CollapseShapeOp>(collapsedVectorOp);
  if (!collapse ||
      !tensor_type::hasStaticF32Shape(collapse.getResult(), {gateWidth}) ||
      !tensor_type::hasStaticF32Shape(collapse.getSrc(),
                                      {types.batchSize, gateWidth}))
    return mlir::failure();

  mlir::Value matmulValue = collapse.getSrc();
  GenericOp biasAddOp;
  ConstantOp biasConstant;
  if (types.hasBias) {
    auto biasMatch = layer_patterns::matchProjectionBiasAdd(
        matmulValue, {types.batchSize, gateWidth}, gateWidth);
    if (mlir::failed(biasMatch))
      return mlir::failure();
    biasAddOp = layer_utils::producerOfType<GenericOp>(matmulValue);
    biasConstant = biasMatch->second;
    matmulValue = biasMatch->first;
  } else if (isAddfGeneric(
                 layer_utils::producerOfType<GenericOp>(matmulValue))) {
    return mlir::failure();
  }

  auto projection = layer_patterns::matchMatmulProjection(
      matmulValue, types.batchSize, types.hiddenSize, gateWidth);
  if (mlir::failed(projection))
    return mlir::failure();

  return RecurrentProjection{projection->activation,
                             projection->weightConstant,
                             projection->weightTranspose,
                             projection->matmulOp,
                             collapse,
                             biasAddOp,
                             biasConstant};
}

static mlir::LogicalResult matchCollapsedExtractGeneric(
    mlir::Operation *op, mlir::Operation *&indexValueOp,
    mlir::Operation *&collapsedVectorOp, mlir::Operation *&zeroConst,
    mlir::Operation *&extentConst, mlir::Operation *&gatherEmpty) {
  auto generic = llvm::dyn_cast_or_null<GenericOp>(op);
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

// Reuses one gate indexing scaffold across all GRU gate slices.
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
  if (!combinedIndices || !layer_utils::isAddiGeneric(combinedIndices))
    return mlir::failure();

  if (!layer_utils::hasDpsInputsAndOperands(combinedIndices, 2, 3))
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
  if (!rangeGeneric ||
      mlir::failed(layer_patterns::matchIndexRangeGeneric(rangeGeneric)))
    return mlir::failure();

  auto rangeEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(rangeGeneric, 0);
  if (!rangeEmpty)
    return mlir::failure();

  mlir::Operation *scaledBaseOffsetGeneric = baseOffsetGeneric;
  if (mlir::failed(layer_patterns::matchMultiplyByConstantGeneric(
          scaledBaseOffsetGeneric, expectedExtentConst))) {
    if (!layer_utils::isAddiGeneric(baseOffsetGeneric) ||
        !layer_utils::hasDpsInputsAndOperands(baseOffsetGeneric, 2, 3))
      return mlir::failure();

    mlir::Operation *firstBaseInput =
        layer_utils::operandProducer(baseOffsetGeneric, 0);
    mlir::Operation *secondBaseInput =
        layer_utils::operandProducer(baseOffsetGeneric, 1);
    if (!firstBaseInput || firstBaseInput != secondBaseInput)
      return mlir::failure();

    scaledBaseOffsetGeneric = firstBaseInput;
    if (mlir::failed(layer_patterns::matchMultiplyByConstantGeneric(
            scaledBaseOffsetGeneric, expectedExtentConst)))
      return mlir::failure();
  }

  auto baseOffsetEmpty = layer_utils::operandProducerOfType<EmptyOp>(
      baseOffsetGeneric, baseOffsetGeneric->getNumOperands() - 1);
  auto zeroIndexExpand = layer_utils::operandProducerOfType<ExpandShapeOp>(
      scaledBaseOffsetGeneric, 0);
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

// Matches one GRU gate slice from a shared fused preactivation.
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

static mlir::LogicalResult matchGatePairAdd(
    mlir::Operation *op, mlir::Operation *inputCollapsedVector,
    mlir::Operation *hiddenCollapsedVector, int64_t inputExpectedOffset,
    int64_t hiddenExpectedOffset, int64_t expectedExtent,
    layer_patterns::RecurrentGateSliceMatch &inputGate,
    layer_patterns::RecurrentGateSliceMatch &hiddenGate,
    layer_patterns::RecurrentGateIndexingScaffoldMatch &inputScaffold,
    layer_patterns::RecurrentGateIndexingScaffoldMatch &hiddenScaffold) {
  if (!layer_utils::isAddfGeneric(op) ||
      !layer_utils::hasDpsInputsAndOperands(op, 2, 3))
    return mlir::failure();

  if (mlir::succeeded(matchGateSliceValue(
          op->getOperand(0), inputCollapsedVector, inputExpectedOffset,
          expectedExtent, inputGate, inputScaffold)) &&
      mlir::succeeded(matchGateSliceValue(
          op->getOperand(1), hiddenCollapsedVector, hiddenExpectedOffset,
          expectedExtent, hiddenGate, hiddenScaffold)))
    return mlir::success();

  if (mlir::succeeded(matchGateSliceValue(
          op->getOperand(1), inputCollapsedVector, inputExpectedOffset,
          expectedExtent, inputGate, inputScaffold)) &&
      mlir::succeeded(matchGateSliceValue(
          op->getOperand(0), hiddenCollapsedVector, hiddenExpectedOffset,
          expectedExtent, hiddenGate, hiddenScaffold)))
    return mlir::success();

  return mlir::failure();
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

// Matches the reset-gated new-state dependency in a GRU step.
static mlir::LogicalResult matchNewGatePath(
    mlir::Operation *candidateAdd, mlir::Operation *inputCollapsedVector,
    mlir::Operation *hiddenCollapsedVector, mlir::Operation *resetGate,
    mlir::Operation *sharedOutputEmpty, int64_t inputNewGateOffset,
    int64_t hiddenNewGateOffset, int64_t gateWidth,
    layer_patterns::RecurrentGateSliceMatch &inputNewGate,
    layer_patterns::RecurrentGateSliceMatch &hiddenNewGate,
    layer_patterns::RecurrentGateIndexingScaffoldMatch &inputScaffold,
    layer_patterns::RecurrentGateIndexingScaffoldMatch &hiddenScaffold,
    mlir::Operation *&resetNewMul) {
  if (!layer_utils::isAddfGeneric(candidateAdd) ||
      !layer_utils::hasDpsInputsAndOperands(candidateAdd, 2, 3))
    return mlir::failure();

  auto candidateAddEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(candidateAdd, 2);
  if (!candidateAddEmpty ||
      candidateAddEmpty.getOperation() != sharedOutputEmpty)
    return mlir::failure();

  auto matchOrdered = [&](unsigned inputIndex,
                          unsigned resetIndex) -> mlir::LogicalResult {
    layer_patterns::RecurrentGateSliceMatch candidateInputGate;
    layer_patterns::RecurrentGateSliceMatch candidateHiddenGate;
    if (mlir::failed(matchGateSliceValue(
            candidateAdd->getOperand(inputIndex), inputCollapsedVector,
            inputNewGateOffset, gateWidth, candidateInputGate, inputScaffold)))
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
      if (mlir::failed(matchGateSliceValue(
              mul->getOperand(1), hiddenCollapsedVector, hiddenNewGateOffset,
              gateWidth, candidateHiddenGate, hiddenScaffold)))
        return mlir::failure();
    } else if (layer_utils::operandProducer(mul, 1) == resetGate) {
      if (mlir::failed(matchGateSliceValue(
              mul->getOperand(0), hiddenCollapsedVector, hiddenNewGateOffset,
              gateWidth, candidateHiddenGate, hiddenScaffold)))
        return mlir::failure();
    } else {
      return mlir::failure();
    }

    inputNewGate = candidateInputGate;
    hiddenNewGate = candidateHiddenGate;
    resetNewMul = mul;
    return mlir::success();
  };

  if (mlir::succeeded(matchOrdered(0, 1)) ||
      mlir::succeeded(matchOrdered(1, 0)))
    return mlir::success();

  return mlir::failure();
}

static mlir::LogicalResult
matchGRUGateStructure(mlir::Operation *resetAdd, mlir::Operation *updateAdd,
                      mlir::Operation *candidateAdd, mlir::Operation *resetGate,
                      mlir::Operation *sharedOutputEmpty, int64_t hiddenSize,
                      mlir::Operation *inputCollapsedVector,
                      mlir::Operation *hiddenCollapsedVector, int64_t timestep,
                      GRUGateSlices &gates, mlir::Operation *&resetNewMul) {
  int64_t gateWidth = 3 * hiddenSize;
  int64_t inputTimestepOffset = timestep * gateWidth;
  if (mlir::failed(matchGatePairAdd(
          resetAdd, inputCollapsedVector, hiddenCollapsedVector,
          /*inputExpectedOffset=*/inputTimestepOffset,
          /*hiddenExpectedOffset=*/0, gateWidth, gates.inputResetGate,
          gates.hiddenResetGate, gates.inputIndexing, gates.hiddenIndexing)))
    return mlir::failure();

  if (mlir::failed(matchGatePairAdd(
          updateAdd, inputCollapsedVector, hiddenCollapsedVector,
          /*inputExpectedOffset=*/inputTimestepOffset + hiddenSize,
          /*hiddenExpectedOffset=*/hiddenSize, gateWidth, gates.inputUpdateGate,
          gates.hiddenUpdateGate, gates.inputIndexing, gates.hiddenIndexing)))
    return mlir::failure();

  if (mlir::failed(matchNewGatePath(
          candidateAdd, inputCollapsedVector, hiddenCollapsedVector, resetGate,
          sharedOutputEmpty,
          /*inputNewGateOffset=*/inputTimestepOffset + 2 * hiddenSize,
          /*hiddenNewGateOffset=*/2 * hiddenSize, gateWidth, gates.inputNewGate,
          gates.hiddenNewGate, gates.inputIndexing, gates.hiddenIndexing,
          resetNewMul)))
    return mlir::failure();

  return mlir::success();
}

static bool isSameConstantOp(ConstantOp lhs, ConstantOp rhs) {
  return lhs && rhs && lhs.getOperation() == rhs.getOperation();
}

static bool validatePreviousHiddenInput(mlir::Value previousHidden,
                                        int64_t layer, int64_t timestep,
                                        llvm::ArrayRef<GRUStep> previousSteps,
                                        const GRUTypes &types,
                                        mlir::Value hiddenStateArgument) {
  if (timestep == 0)
    return layer_patterns::matchesInitialRecurrentStateSlice(
        previousHidden, hiddenStateArgument, layer, types.batchSize,
        types.hiddenSize);

  return !previousSteps.empty() &&
         previousHidden == previousSteps.back().hiddenOutput;
}

static bool validateRecurrentActivation(mlir::Value activation, int64_t layer,
                                        int64_t timestep,
                                        llvm::ArrayRef<GRUStep> previousSteps,
                                        const GRUTypes &types,
                                        mlir::Value hiddenStateArgument) {
  if (timestep == 0)
    return layer_patterns::matchesInitialRecurrentStateCollapsed(
        activation, hiddenStateArgument, layer, types.batchSize,
        types.hiddenSize);

  return !previousSteps.empty() &&
         layer_patterns::matchesCollapsedRecurrentValue(
             activation, previousSteps.back().hiddenOutput, types.batchSize,
             types.hiddenSize);
}

static mlir::FailureOr<GRUHiddenUpdate>
matchGRUHiddenUpdate(mlir::Value hiddenOutput, int64_t layer, int64_t timestep,
                     llvm::ArrayRef<GRUStep> previousSteps,
                     const GRUTypes &types, mlir::Value hiddenStateArgument) {
  auto finalHiddenAdd = layer_utils::producerOfType<GenericOp>(hiddenOutput);
  if (!finalHiddenAdd ||
      !layer_utils::isAddfGeneric(finalHiddenAdd.getOperation()) ||
      !layer_utils::hasDpsInputsAndOperands(finalHiddenAdd.getOperation(), 2,
                                            3))
    return mlir::failure();

  auto sharedOutputEmpty = layer_utils::operandProducerOfType<EmptyOp>(
      finalHiddenAdd.getOperation(), 2);
  if (!sharedOutputEmpty)
    return mlir::failure();

  mlir::Operation *candidateTanhOp = nullptr;
  mlir::Operation *updateBlendMulOp = nullptr;
  if (layer_utils::isTanhGeneric(
          layer_utils::operandProducer(finalHiddenAdd, 0))) {
    candidateTanhOp = layer_utils::operandProducer(finalHiddenAdd, 0);
    updateBlendMulOp = layer_utils::operandProducer(finalHiddenAdd, 1);
  } else if (layer_utils::isTanhGeneric(
                 layer_utils::operandProducer(finalHiddenAdd, 1))) {
    candidateTanhOp = layer_utils::operandProducer(finalHiddenAdd, 1);
    updateBlendMulOp = layer_utils::operandProducer(finalHiddenAdd, 0);
  } else {
    return mlir::failure();
  }

  auto candidateTanh = llvm::dyn_cast_or_null<GenericOp>(candidateTanhOp);
  auto updateBlendMul = llvm::dyn_cast_or_null<GenericOp>(updateBlendMulOp);
  if (!layer_utils::isMulfGeneric(updateBlendMul) ||
      !layer_utils::hasDpsInputsAndOperands(updateBlendMul, 2, 3))
    return mlir::failure();

  auto blendOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(updateBlendMul, 2);
  if (!blendOutputEmpty ||
      blendOutputEmpty.getOperation() != sharedOutputEmpty.getOperation())
    return mlir::failure();

  mlir::Operation *hiddenMinusCandidateOp = nullptr;
  mlir::Operation *updateGateOp = nullptr;
  if (layer_utils::isSubfGeneric(
          layer_utils::operandProducer(updateBlendMul, 0)) &&
      layer_utils::isSigmoidGeneric(
          layer_utils::operandProducer(updateBlendMul, 1))) {
    hiddenMinusCandidateOp = layer_utils::operandProducer(updateBlendMul, 0);
    updateGateOp = layer_utils::operandProducer(updateBlendMul, 1);
  } else if (layer_utils::isSubfGeneric(
                 layer_utils::operandProducer(updateBlendMul, 1)) &&
             layer_utils::isSigmoidGeneric(
                 layer_utils::operandProducer(updateBlendMul, 0))) {
    hiddenMinusCandidateOp = layer_utils::operandProducer(updateBlendMul, 1);
    updateGateOp = layer_utils::operandProducer(updateBlendMul, 0);
  } else {
    return mlir::failure();
  }

  auto hiddenMinusCandidate =
      llvm::dyn_cast_or_null<GenericOp>(hiddenMinusCandidateOp);
  auto updateGate = llvm::dyn_cast_or_null<GenericOp>(updateGateOp);
  if (!hiddenMinusCandidate || !updateGate)
    return mlir::failure();

  if (!layer_utils::hasDpsInputsAndOperands(hiddenMinusCandidate, 2, 3) ||
      layer_utils::operandProducer(hiddenMinusCandidate, 1) != candidateTanh ||
      !validatePreviousHiddenInput(hiddenMinusCandidate->getOperand(0), layer,
                                   timestep, previousSteps, types,
                                   hiddenStateArgument))
    return mlir::failure();

  auto subOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(hiddenMinusCandidate, 2);
  if (!subOutputEmpty ||
      subOutputEmpty.getOperation() != sharedOutputEmpty.getOperation())
    return mlir::failure();

  if (!layer_utils::hasDpsInputsAndOperands(candidateTanh, 1, 2))
    return mlir::failure();

  auto tanhOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(candidateTanh, 1);
  if (!tanhOutputEmpty ||
      tanhOutputEmpty.getOperation() != sharedOutputEmpty.getOperation())
    return mlir::failure();

  mlir::Operation *candidateAdd =
      layer_utils::operandProducer(candidateTanh, 0);
  mlir::Operation *updateAdd = layer_utils::operandProducer(updateGate, 0);
  if (!candidateAdd || !updateAdd)
    return mlir::failure();

  auto candidateAddGeneric = llvm::dyn_cast_or_null<GenericOp>(candidateAdd);
  auto updateAddGeneric = llvm::dyn_cast_or_null<GenericOp>(updateAdd);
  if (!candidateAddGeneric || !updateAddGeneric)
    return mlir::failure();

  return GRUHiddenUpdate{finalHiddenAdd,      candidateTanh,
                         updateBlendMul,      hiddenMinusCandidate,
                         candidateAddGeneric, updateGate,
                         updateAddGeneric,    sharedOutputEmpty};
}

static mlir::FailureOr<GRUResetUpdateGates>
matchGRUResetUpdateGates(GRUHiddenUpdate hiddenUpdate) {
  mlir::Operation *resetNewMul = nullptr;
  mlir::Operation *resetGate = nullptr;
  if (layer_utils::isMulfGeneric(
          layer_utils::operandProducer(hiddenUpdate.candidateAddOp, 0))) {
    resetNewMul = layer_utils::operandProducer(hiddenUpdate.candidateAddOp, 0);
    resetGate = layer_utils::operandProducer(resetNewMul, 0);
    if (!layer_utils::isSigmoidGeneric(resetGate))
      resetGate = layer_utils::operandProducer(resetNewMul, 1);
  } else if (layer_utils::isMulfGeneric(layer_utils::operandProducer(
                 hiddenUpdate.candidateAddOp, 1))) {
    resetNewMul = layer_utils::operandProducer(hiddenUpdate.candidateAddOp, 1);
    resetGate = layer_utils::operandProducer(resetNewMul, 0);
    if (!layer_utils::isSigmoidGeneric(resetGate))
      resetGate = layer_utils::operandProducer(resetNewMul, 1);
  } else {
    return mlir::failure();
  }

  if (!resetGate)
    return mlir::failure();
  auto resetGateGeneric = llvm::dyn_cast_or_null<GenericOp>(resetGate);
  if (!resetGateGeneric)
    return mlir::failure();

  mlir::Operation *resetAdd = layer_utils::operandProducer(resetGate, 0);
  if (!resetAdd)
    return mlir::failure();
  auto resetAddGeneric = llvm::dyn_cast_or_null<GenericOp>(resetAdd);
  if (!resetAddGeneric)
    return mlir::failure();

  mlir::Operation *sigmoidOneConstant = nullptr;
  if (mlir::failed(matchGateActivation(
          resetGateGeneric.getOperation(), resetAddGeneric.getOperation(),
          hiddenUpdate.sharedOutputEmpty.getOperation(), sigmoidOneConstant)) ||
      mlir::failed(matchGateActivation(
          hiddenUpdate.updateGateOp.getOperation(),
          hiddenUpdate.updateAddOp.getOperation(),
          hiddenUpdate.sharedOutputEmpty.getOperation(), sigmoidOneConstant)))
    return mlir::failure();

  return GRUResetUpdateGates{resetGateGeneric, resetAddGeneric};
}

static mlir::FailureOr<GRUCollapsedResetInputs>
matchCollapsedResetInputs(GenericOp resetAdd) {
  auto firstResetExpand =
      layer_utils::operandProducerOfType<ExpandShapeOp>(resetAdd, 0);
  auto secondResetExpand =
      layer_utils::operandProducerOfType<ExpandShapeOp>(resetAdd, 1);
  if (!firstResetExpand || !secondResetExpand)
    return mlir::failure();

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
    return mlir::failure();

  return GRUCollapsedResetInputs{firstCollapsedVector, secondCollapsedVector};
}

static mlir::FailureOr<GRUStep> matchGRUStepWithCollapsedOrder(
    mlir::Value hiddenOutput, GRUHiddenUpdate hiddenUpdate,
    GRUResetUpdateGates resetUpdateGates, mlir::Operation *inputCollapsedVector,
    mlir::Operation *hiddenCollapsedVector, int64_t layer, int64_t timestep,
    llvm::ArrayRef<GRUStep> previousSteps, const GRUTypes &types,
    mlir::Value hiddenStateArgument) {
  int64_t gateWidth = 3 * types.hiddenSize;
  auto inputSlice = layer_patterns::matchTimestepProjectionSlice(
      inputCollapsedVector->getResult(0), timestep,
      llvm::ArrayRef<int64_t>{gateWidth}, types.sequenceLength, types.batchSize,
      gateWidth);
  if (mlir::failed(inputSlice))
    return mlir::failure();

  auto recurrentProjection =
      matchRecurrentProjection(hiddenCollapsedVector, types);
  if (mlir::failed(recurrentProjection))
    return mlir::failure();

  if (!validateRecurrentActivation(recurrentProjection->activation, layer,
                                   timestep, previousSteps, types,
                                   hiddenStateArgument))
    return mlir::failure();

  GRUGateSlices gates;
  mlir::Operation *matchedResetNewMul = nullptr;
  if (mlir::failed(matchGRUGateStructure(
          resetUpdateGates.resetAddOp.getOperation(),
          hiddenUpdate.updateAddOp.getOperation(),
          hiddenUpdate.candidateAddOp.getOperation(),
          resetUpdateGates.resetGateOp.getOperation(),
          hiddenUpdate.sharedOutputEmpty.getOperation(), types.hiddenSize,
          inputCollapsedVector, hiddenCollapsedVector, timestep, gates,
          matchedResetNewMul)))
    return mlir::failure();

  return GRUStep{hiddenOutput,
                 hiddenUpdate.finalHiddenAddOp,
                 hiddenUpdate.candidateTanhOp,
                 hiddenUpdate.updateBlendMulOp,
                 hiddenUpdate.hiddenMinusCandidateOp,
                 hiddenUpdate.candidateAddOp,
                 llvm::cast<GenericOp>(matchedResetNewMul),
                 resetUpdateGates.resetGateOp,
                 resetUpdateGates.resetAddOp,
                 hiddenUpdate.updateGateOp,
                 hiddenUpdate.updateAddOp,
                 inputSlice->collapse,
                 inputSlice->slice,
                 *recurrentProjection,
                 gates};
}

// Matches one GRU timestep through candidate and update blending.
static mlir::FailureOr<GRUStep>
matchGRUStep(mlir::Value hiddenOutput, int64_t layer, int64_t timestep,
             llvm::ArrayRef<GRUStep> previousSteps, const GRUTypes &types,
             mlir::Value hiddenStateArgument) {
  if (!tensor_type::hasStaticF32Shape(hiddenOutput,
                                      {1, types.batchSize, types.hiddenSize}))
    return mlir::failure();

  auto hiddenUpdate = matchGRUHiddenUpdate(
      hiddenOutput, layer, timestep, previousSteps, types, hiddenStateArgument);
  if (mlir::failed(hiddenUpdate))
    return mlir::failure();

  auto resetUpdateGates = matchGRUResetUpdateGates(*hiddenUpdate);
  if (mlir::failed(resetUpdateGates))
    return mlir::failure();

  auto resetInputs = matchCollapsedResetInputs(resetUpdateGates->resetAddOp);
  if (mlir::failed(resetInputs))
    return mlir::failure();

  auto firstOrder = matchGRUStepWithCollapsedOrder(
      hiddenOutput, *hiddenUpdate, *resetUpdateGates,
      resetInputs->firstCollapsedVector, resetInputs->secondCollapsedVector,
      layer, timestep, previousSteps, types, hiddenStateArgument);
  if (mlir::succeeded(firstOrder))
    return firstOrder;

  return matchGRUStepWithCollapsedOrder(
      hiddenOutput, *hiddenUpdate, *resetUpdateGates,
      resetInputs->secondCollapsedVector, resetInputs->firstCollapsedVector,
      layer, timestep, previousSteps, types, hiddenStateArgument);
}

static bool hasSharedLayerRecurrentParameters(const GRULayer &layer,
                                              bool hasBias) {
  if (layer.steps.empty())
    return false;

  const RecurrentProjection &firstProjection =
      layer.steps.front().recurrentProjection;
  if (!firstProjection.weightConstant)
    return false;
  if (hasBias && !firstProjection.biasConstant)
    return false;

  for (const GRUStep &step : layer.steps) {
    const RecurrentProjection &projection = step.recurrentProjection;
    if (!isSameConstantOp(projection.weightConstant,
                          firstProjection.weightConstant))
      return false;
    if (hasBias && !isSameConstantOp(projection.biasConstant,
                                     firstProjection.biasConstant))
      return false;
  }

  return true;
}

static mlir::FailureOr<LayerInputProjection>
matchFirstLayerInputProjection(mlir::Value source, const GRUTypes &types,
                               mlir::Value inputArgument) {
  int64_t gateWidth = 3 * types.hiddenSize;
  if (!tensor_type::hasStaticF32Shape(
          source, {types.sequenceLength, types.batchSize, gateWidth}))
    return mlir::failure();

  LayerInputProjection projection;
  projection.source = source;

  if (auto expand = layer_utils::producerOfType<ExpandShapeOp>(source)) {
    if (!tensor_type::hasStaticF32Shape(
            expand.getSrc(),
            {types.sequenceLength * types.batchSize, gateWidth}))
      return mlir::failure();

    mlir::Value matmulValue = expand.getSrc();
    if (types.hasBias) {
      auto biasMatch = layer_patterns::matchProjectionBiasAdd(
          matmulValue, {types.sequenceLength * types.batchSize, gateWidth},
          gateWidth);
      if (mlir::failed(biasMatch))
        return mlir::failure();
      projection.biasAddOp =
          layer_utils::producerOfType<GenericOp>(matmulValue);
      projection.biasConstant = biasMatch->second;
      matmulValue = biasMatch->first;
    } else if (isAddfGeneric(
                   layer_utils::producerOfType<GenericOp>(matmulValue))) {
      return mlir::failure();
    }

    auto weightProjection = layer_patterns::matchMatmulProjection(
        matmulValue, types.sequenceLength * types.batchSize, types.inputSize,
        gateWidth);
    if (mlir::failed(weightProjection))
      return mlir::failure();

    auto inputCollapse = layer_utils::producerOfType<CollapseShapeOp>(
        weightProjection->activation);
    if (!inputCollapse ||
        !tensor_type::hasStaticF32Shape(
            inputCollapse.getResult(),
            {types.sequenceLength * types.batchSize, types.inputSize}))
      return mlir::failure();

    auto inputTranspose =
        layer_utils::producerOfType<TransposeOp>(inputCollapse.getSrc());
    if (!inputTranspose || inputTranspose.getInput() != inputArgument ||
        !linalg_match::hasPermutation(inputTranspose, {1, 0, 2}) ||
        !tensor_type::hasStaticF32Shape(
            inputTranspose.getResult().front(),
            {types.sequenceLength, types.batchSize, types.inputSize}) ||
        !tensor_type::hasStaticF32Shape(
            inputTranspose.getInit(),
            {types.sequenceLength, types.batchSize, types.inputSize}))
      return mlir::failure();

    auto inputTransposeInit =
        layer_utils::producerOfType<EmptyOp>(inputTranspose.getInit());
    if (!inputTransposeInit)
      return mlir::failure();

    projection.weightConstant = weightProjection->weightConstant;
    projection.weightTranspose = weightProjection->weightTranspose;
    projection.matmulOp = weightProjection->matmulOp;
    return projection;
  }

  mlir::Value batchMatmulValue = source;
  if (types.hasBias) {
    auto biasMatch = layer_patterns::matchProjectionBiasAdd(
        batchMatmulValue, {types.sequenceLength, types.batchSize, gateWidth},
        gateWidth);
    if (mlir::failed(biasMatch))
      return mlir::failure();
    projection.biasAddOp =
        layer_utils::producerOfType<GenericOp>(batchMatmulValue);
    projection.biasConstant = biasMatch->second;
    batchMatmulValue = biasMatch->first;
  } else if (isAddfGeneric(
                 layer_utils::producerOfType<GenericOp>(batchMatmulValue))) {
    return mlir::failure();
  }

  auto batchMatmul =
      layer_utils::producerOfType<BatchMatmulOp>(batchMatmulValue);
  if (!batchMatmul || batchMatmul.getInputs().size() != 2 ||
      batchMatmul.getOutputs().size() != 1 ||
      !tensor_type::hasStaticF32Shape(
          batchMatmul.getResult(0),
          {types.sequenceLength, types.batchSize, gateWidth}))
    return mlir::failure();

  auto batchMatmulTy =
      llvm::cast<mlir::RankedTensorType>(batchMatmul.getResult(0).getType());
  if (!layer_patterns::isZeroInitializedOutput(batchMatmul.getOutputs()[0],
                                               batchMatmulTy))
    return mlir::failure();

  auto inputTranspose =
      layer_utils::producerOfType<TransposeOp>(batchMatmul.getInputs()[0]);
  if (!inputTranspose || inputTranspose.getInput() != inputArgument ||
      !linalg_match::hasPermutation(inputTranspose, {1, 0, 2}) ||
      !tensor_type::hasStaticF32Shape(
          inputTranspose.getResult().front(),
          {types.sequenceLength, types.batchSize, types.inputSize}) ||
      !tensor_type::hasStaticF32Shape(
          inputTranspose.getInit(),
          {types.sequenceLength, types.batchSize, types.inputSize}))
    return mlir::failure();

  auto inputTransposeInit =
      layer_utils::producerOfType<EmptyOp>(inputTranspose.getInit());
  if (!inputTransposeInit)
    return mlir::failure();

  auto broadcast =
      layer_utils::producerOfType<GenericOp>(batchMatmul.getInputs()[1]);
  if (!isYieldInputGeneric(broadcast) ||
      !linalg_match::hasWeightBroadcastIndexingMaps(broadcast) ||
      !tensor_type::hasStaticF32Shape(
          broadcast.getResult(0),
          {types.sequenceLength, types.inputSize, gateWidth}) ||
      !tensor_type::hasStaticF32Shape(
          broadcast.getOutputs()[0],
          {types.sequenceLength, types.inputSize, gateWidth}))
    return mlir::failure();

  TransposeOp weightTranspose;
  auto weightConstant = layer_patterns::matchProjectionWeightTranspose(
      broadcast.getInputs()[0], types.inputSize, gateWidth, weightTranspose);
  if (mlir::failed(weightConstant))
    return mlir::failure();

  projection.batchMatmulOp = batchMatmul;
  projection.weightConstant = *weightConstant;
  projection.weightTranspose = weightTranspose;
  return projection;
}

static mlir::FailureOr<LayerInputProjection>
matchStackedLayerInputProjection(mlir::Value source, const GRUTypes &types) {
  int64_t gateWidth = 3 * types.hiddenSize;
  if (!tensor_type::hasStaticF32Shape(
          source, {types.sequenceLength, types.batchSize, gateWidth}))
    return mlir::failure();

  auto expand = layer_utils::producerOfType<ExpandShapeOp>(source);
  if (!expand ||
      !tensor_type::hasStaticF32Shape(
          expand.getSrc(), {types.sequenceLength * types.batchSize, gateWidth}))
    return mlir::failure();

  LayerInputProjection projection;
  projection.source = source;

  mlir::Value matmulValue = expand.getSrc();
  if (types.hasBias) {
    auto biasMatch = layer_patterns::matchProjectionBiasAdd(
        matmulValue, {types.sequenceLength * types.batchSize, gateWidth},
        gateWidth);
    if (mlir::failed(biasMatch))
      return mlir::failure();
    projection.biasAddOp = layer_utils::producerOfType<GenericOp>(matmulValue);
    projection.biasConstant = biasMatch->second;
    matmulValue = biasMatch->first;
  } else if (isAddfGeneric(
                 layer_utils::producerOfType<GenericOp>(matmulValue))) {
    return mlir::failure();
  }

  auto weightProjection = layer_patterns::matchMatmulProjection(
      matmulValue, types.sequenceLength * types.batchSize, types.hiddenSize,
      gateWidth);
  if (mlir::failed(weightProjection))
    return mlir::failure();

  auto collapse = layer_utils::producerOfType<CollapseShapeOp>(
      weightProjection->activation);
  if (!collapse ||
      !tensor_type::hasStaticF32Shape(
          collapse.getResult(),
          {types.sequenceLength * types.batchSize, types.hiddenSize}))
    return mlir::failure();

  auto previousConcat =
      layer_utils::producerOfType<ConcatOp>(collapse.getSrc());
  if (!layer_patterns::matchesLayerOutputConcat(
          previousConcat, types.sequenceLength, types.batchSize,
          types.hiddenSize))
    return mlir::failure();

  projection.previousLayerConcat = previousConcat;
  projection.weightConstant = weightProjection->weightConstant;
  projection.weightTranspose = weightProjection->weightTranspose;
  projection.matmulOp = weightProjection->matmulOp;
  return projection;
}

static mlir::FailureOr<GRULayer>
matchGRULayerBody(llvm::ArrayRef<mlir::Value> layerOutputs, int64_t layer,
                  const GRUTypes &types, mlir::Value inputArgument,
                  mlir::Value hiddenStateArgument) {
  if (layerOutputs.size() != static_cast<size_t>(types.sequenceLength))
    return mlir::failure();

  GRULayer layerMatch;
  layerMatch.steps.reserve(layerOutputs.size());

  mlir::Value commonInputSource;
  for (auto [timestep, output] : llvm::enumerate(layerOutputs)) {
    auto step = matchGRUStep(output, layer, timestep, layerMatch.steps, types,
                             hiddenStateArgument);
    if (mlir::failed(step))
      return mlir::failure();

    mlir::Value inputSource = step->inputSliceOp.getSource();
    if (!commonInputSource)
      commonInputSource = inputSource;
    else if (commonInputSource != inputSource)
      return mlir::failure();

    layerMatch.steps.push_back(*step);
  }

  if (!commonInputSource ||
      !hasSharedLayerRecurrentParameters(layerMatch, types.hasBias))
    return mlir::failure();

  if (layer == 0) {
    auto inputProjection =
        matchFirstLayerInputProjection(commonInputSource, types, inputArgument);
    if (mlir::failed(inputProjection))
      return mlir::failure();
    layerMatch.inputProjection = *inputProjection;
  } else {
    auto inputProjection =
        matchStackedLayerInputProjection(commonInputSource, types);
    if (mlir::failed(inputProjection))
      return mlir::failure();
    layerMatch.inputProjection = *inputProjection;
  }

  return layerMatch;
}

static mlir::FailureOr<GRUTypes> getSupportedGRUTypes(mlir::func::FuncOp func,
                                                      bool hasBias) {
  if (func.getNumArguments() != 2 || func.getNumResults() != 2 ||
      !func.getBody().hasOneBlock())
    return mlir::failure();

  auto inputTy =
      tensor_type::getStaticF32Tensor(func.getArgument(0).getType(), 3);
  auto hiddenStateTy =
      tensor_type::getStaticF32Tensor(func.getArgument(1).getType(), 3);
  auto sequenceResultTy =
      tensor_type::getStaticF32Tensor(func.getResultTypes()[0], 3);
  auto hiddenResultTy =
      tensor_type::getStaticF32Tensor(func.getResultTypes()[1], 3);
  if (mlir::failed(inputTy) || mlir::failed(hiddenStateTy) ||
      mlir::failed(sequenceResultTy) || mlir::failed(hiddenResultTy))
    return mlir::failure();

  int64_t batchSize = inputTy->getDimSize(0);
  int64_t sequenceLength = inputTy->getDimSize(1);
  int64_t inputSize = inputTy->getDimSize(2);
  int64_t layerCount = hiddenStateTy->getDimSize(0);
  int64_t hiddenBatchSize = hiddenStateTy->getDimSize(1);
  int64_t hiddenSize = hiddenStateTy->getDimSize(2);
  if (layerCount <= 0 || sequenceLength <= 0 || batchSize <= 0 ||
      inputSize <= 0 || hiddenSize <= 0 || batchSize != hiddenBatchSize)
    return mlir::failure();

  if (sequenceResultTy->getShape() !=
          llvm::ArrayRef<int64_t>({batchSize, sequenceLength, hiddenSize}) ||
      hiddenResultTy->getShape() != hiddenStateTy->getShape())
    return mlir::failure();

  return GRUTypes{*inputTy,        *hiddenStateTy, *sequenceResultTy,
                  *hiddenResultTy, layerCount,     sequenceLength,
                  batchSize,       inputSize,      hiddenSize,
                  hasBias};
}

static mlir::FailureOr<GRUMatch> matchSupportedGRU(mlir::func::FuncOp func,
                                                   bool hasBias) {
  auto types = getSupportedGRUTypes(func, hasBias);
  if (mlir::failed(types))
    return mlir::failure();

  auto returnOp =
      llvm::dyn_cast<ReturnOp>(func.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 2)
    return mlir::failure();

  auto sequenceAssembly = layer_patterns::matchFinalSequenceAssembly(
      returnOp.getOperand(0), types->sequenceLength, types->batchSize,
      types->hiddenSize);
  if (mlir::failed(sequenceAssembly))
    return mlir::failure();

  int64_t tanhCount = 0;
  func.walk([&](mlir::Operation *op) {
    if (layer_utils::isTanhGeneric(op))
      ++tanhCount;
  });
  if (tanhCount != types->layerCount * types->sequenceLength)
    return mlir::failure();

  llvm::SmallVector<llvm::SmallVector<mlir::Value>> layerOutputs(
      types->layerCount);
  layerOutputs.back().assign(sequenceAssembly->second.getInputs().begin(),
                             sequenceAssembly->second.getInputs().end());

  llvm::SmallVector<GRULayer, 4> layers;
  layers.resize(types->layerCount);
  for (int64_t layer = types->layerCount - 1; layer >= 0; --layer) {
    auto layerMatch =
        matchGRULayerBody(layerOutputs[layer], layer, *types,
                          func.getArgument(0), func.getArgument(1));
    if (mlir::failed(layerMatch))
      return mlir::failure();

    if (layer > 0) {
      ConcatOp previousLayerConcat =
          layerMatch->inputProjection.previousLayerConcat;
      layerOutputs[layer - 1].assign(previousLayerConcat.getInputs().begin(),
                                     previousLayerConcat.getInputs().end());
    }

    layerMatch->outputConcat =
        layer == types->layerCount - 1
            ? sequenceAssembly->second
            : layers[layer + 1].inputProjection.previousLayerConcat;
    layers[layer] = *layerMatch;
  }

  auto hiddenAssembly = layer_patterns::matchFinalRecurrentStateAssembly(
      returnOp.getOperand(1), types->layerCount, types->batchSize,
      types->hiddenSize, [&](auto layer) -> mlir::Value {
        auto layerIndex = static_cast<size_t>(layer);
        if (layerIndex >= layers.size() || layers[layerIndex].steps.empty())
          return {};
        return layers[layerIndex].steps.back().hiddenOutput;
      });
  if (mlir::failed(hiddenAssembly))
    return mlir::failure();

  return GRUMatch{returnOp,
                  sequenceAssembly->first,
                  sequenceAssembly->second,
                  hiddenAssembly->first,
                  hiddenAssembly->second,
                  layers,
                  *types};
}

static mlir::FailureOr<GRUMatch> matchSupportedGRU(mlir::func::FuncOp func) {
  auto noBiasMatch = matchSupportedGRU(func, /*hasBias=*/false);
  if (mlir::succeeded(noBiasMatch))
    return noBiasMatch;

  return matchSupportedGRU(func, /*hasBias=*/true);
}

static void appendRecurrentOperands(GRUMatch &match,
                                    llvm::SmallVectorImpl<mlir::Value> &ops) {
  for (GRULayer &layer : match.layers) {
    ops.push_back(layer.inputProjection.weightConstant.getResult());
    ops.push_back(
        layer.steps.front().recurrentProjection.weightConstant.getResult());
    if (!match.types.hasBias)
      continue;

    ops.push_back(layer.inputProjection.biasConstant.getResult());
    ops.push_back(
        layer.steps.front().recurrentProjection.biasConstant.getResult());
  }
}

static void rewriteGRUMatchToSculptorOp(mlir::func::FuncOp func, GRUMatch &match,
                                      mlir::RewriterBase &rewriter) {
  if (!func.getBody().hasOneBlock())
    return;

  llvm::SmallVector<mlir::Operation *> oldBodyOps;
  for (mlir::Operation &op : func.getBody().front().without_terminator())
    oldBodyOps.push_back(&op);

  llvm::SmallVector<mlir::Value> recurrentOperands;
  appendRecurrentOperands(match, recurrentOperands);

  rewriter.setInsertionPoint(match.returnOp);
  auto gruOp = rewriter.create<mlir::sculptor::NNGRUOp>(
      match.returnOp.getLoc(), func.getResultTypes(), func.getArgument(0),
      func.getArgument(1), recurrentOperands, /*batch_first=*/true,
      match.types.hasBias, static_cast<uint64_t>(match.types.hiddenSize),
      static_cast<uint64_t>(match.types.layerCount));
  rewriter.replaceOpWithNewOp<ReturnOp>(match.returnOp, gruOp.getResults());

  for (mlir::Operation *op : llvm::reverse(oldBodyOps)) {
    if (op->use_empty())
      rewriter.eraseOp(op);
  }
}

class GRUCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  explicit GRUCanonicalizer(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "gru"; }

  void canonicalize(mlir::func::FuncOp func) const override {
    auto match = matchSupportedGRU(func);
    if (mlir::failed(match))
      return;

    mlir::IRRewriter rewriter(func.getContext());
    rewriteGRUMatchToSculptorOp(func, *match, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerGRUCanonicalizer(LayerCanonicalizers &canonicalizers,
                              MLIRContext *context) {
  canonicalizers.push_back(std::make_unique<GRUCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
