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

struct LSTMTypes {
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType hiddenStateTy;
  mlir::RankedTensorType cellStateTy;
  mlir::RankedTensorType sequenceResultTy;
  mlir::RankedTensorType hiddenResultTy;
  mlir::RankedTensorType cellResultTy;
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
  ExpandShapeOp expandOp;
  GenericOp biasAddOp;
  ConstantOp biasConstant;
};

struct LSTMStep {
  mlir::Value hiddenOutput;
  mlir::Value cellOutput;
  GenericOp hiddenMulOp;
  GenericOp cellAddOp;
  GenericOp preActivationAddOp;
  CollapseShapeOp preActivationCollapseOp;
  CollapseShapeOp inputCollapseOp;
  ExtractSliceOp inputSliceOp;
  RecurrentProjection recurrentProjection;
};

struct LSTMHiddenOutput {
  GenericOp hiddenMulOp;
  GenericOp cellTanhOp;
  mlir::Operation *outputGateActivation = nullptr;
  mlir::Operation *sharedHiddenOutputEmpty = nullptr;
};

struct LSTMCellUpdate {
  GenericOp cellAddOp;
  mlir::Operation *forgetGateActivation = nullptr;
  mlir::Operation *inputGateActivation = nullptr;
  mlir::Operation *candidateGateActivation = nullptr;
  mlir::Value cellStateInput;
};

struct LSTMGatePreactivation {
  GenericOp preActivationAddOp;
  CollapseShapeOp preActivationCollapseOp;
  CollapseShapeOp inputCollapseOp;
  ExtractSliceOp inputSliceOp;
  RecurrentProjection recurrentProjection;
};

struct LSTMLayer {
  ConcatOp outputConcat;
  LayerInputProjection inputProjection;
  llvm::SmallVector<LSTMStep> steps;
};

struct LSTMMatch {
  ReturnOp returnOp;
  TransposeOp sequenceTranspose;
  ConcatOp sequenceConcat;
  ExpandShapeOp hiddenExpand;
  ConcatOp hiddenConcat;
  ExpandShapeOp cellExpand;
  ConcatOp cellConcat;
  llvm::SmallVector<LSTMLayer> layers;
  LSTMTypes types;
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
matchRecurrentProjection(mlir::Value value, const LSTMTypes &types) {
  int64_t gateWidth = 4 * types.hiddenSize;
  auto expand = layer_utils::producerOfType<ExpandShapeOp>(value);
  if (!expand ||
      !tensor_type::hasStaticF32Shape(expand.getResult(),
                                      {1, types.batchSize, gateWidth}) ||
      !tensor_type::hasStaticF32Shape(expand.getSrc(),
                                      {types.batchSize, gateWidth}))
    return mlir::failure();

  mlir::Value matmulValue = expand.getSrc();
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
                             expand,
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

// Reuses one gate indexing scaffold across all LSTM gate slices.
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

static bool validatePreviousCellInput(mlir::Value cellStateInput, int64_t layer,
                                      int64_t timestep,
                                      llvm::ArrayRef<LSTMStep> previousSteps,
                                      const LSTMTypes &types,
                                      mlir::Value cellStateArgument) {
  if (timestep == 0)
    return layer_patterns::matchesInitialRecurrentStateSlice(
        cellStateInput, cellStateArgument, layer, types.batchSize,
        types.hiddenSize);

  return !previousSteps.empty() &&
         cellStateInput == previousSteps.back().cellOutput;
}

static bool validateRecurrentActivation(mlir::Value activation, int64_t layer,
                                        int64_t timestep,
                                        llvm::ArrayRef<LSTMStep> previousSteps,
                                        const LSTMTypes &types,
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

static mlir::FailureOr<LSTMHiddenOutput>
matchLSTMHiddenOutput(mlir::Value hiddenOutput) {
  auto hiddenMulOp = layer_utils::producerOfType<GenericOp>(hiddenOutput);
  mlir::Operation *outputGateActivation = nullptr;
  mlir::Operation *cellTanh = nullptr;
  mlir::Operation *sharedHiddenOutputEmpty = nullptr;
  if (!hiddenMulOp || mlir::failed(matchFinalHiddenMul(
                          hiddenMulOp.getOperation(), outputGateActivation,
                          cellTanh, sharedHiddenOutputEmpty)))
    return mlir::failure();

  auto cellTanhOp = llvm::dyn_cast_or_null<GenericOp>(cellTanh);
  if (!layer_utils::hasDpsInputsAndOperands(cellTanh, 1, 2) ||
      !layer_utils::isTanhGeneric(cellTanh) || !cellTanhOp)
    return mlir::failure();

  auto cellTanhOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(cellTanh, 1);
  if (!cellTanhOutputEmpty ||
      cellTanhOutputEmpty.getOperation() != sharedHiddenOutputEmpty)
    return mlir::failure();

  return LSTMHiddenOutput{hiddenMulOp, cellTanhOp, outputGateActivation,
                          sharedHiddenOutputEmpty};
}

static mlir::FailureOr<LSTMCellUpdate>
matchLSTMCellUpdate(LSTMHiddenOutput hiddenOutput, int64_t layer,
                    int64_t timestep, llvm::ArrayRef<LSTMStep> previousSteps,
                    const LSTMTypes &types, mlir::Value cellStateArgument) {
  auto cellAddOp = llvm::dyn_cast_or_null<GenericOp>(
      layer_utils::operandProducer(hiddenOutput.cellTanhOp, 0));
  if (!cellAddOp || !layer_utils::isAddfGeneric(cellAddOp.getOperation()) ||
      !layer_utils::hasDpsInputsAndOperands(cellAddOp.getOperation(), 2, 3) ||
      !tensor_type::hasStaticF32Shape(cellAddOp.getResult(0),
                                      {1, types.batchSize, types.hiddenSize}))
    return mlir::failure();

  auto cellAddOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(cellAddOp.getOperation(), 2);
  if (!cellAddOutputEmpty ||
      cellAddOutputEmpty.getOperation() != hiddenOutput.sharedHiddenOutputEmpty)
    return mlir::failure();

  mlir::Operation *forgetGateActivation = nullptr;
  mlir::Operation *inputGateActivation = nullptr;
  mlir::Operation *candidateGateActivation = nullptr;
  mlir::Value cellStateInput;
  mlir::Operation *firstCellAddInput =
      layer_utils::operandProducer(cellAddOp.getOperation(), 0);
  mlir::Operation *secondCellAddInput =
      layer_utils::operandProducer(cellAddOp.getOperation(), 1);
  if (mlir::succeeded(matchForgetCellMul(
          firstCellAddInput, hiddenOutput.sharedHiddenOutputEmpty,
          forgetGateActivation, cellStateInput)) &&
      mlir::succeeded(matchInputCandidateMul(
          secondCellAddInput, hiddenOutput.sharedHiddenOutputEmpty,
          inputGateActivation, candidateGateActivation))) {
  } else if (mlir::succeeded(matchForgetCellMul(
                 secondCellAddInput, hiddenOutput.sharedHiddenOutputEmpty,
                 forgetGateActivation, cellStateInput)) &&
             mlir::succeeded(matchInputCandidateMul(
                 firstCellAddInput, hiddenOutput.sharedHiddenOutputEmpty,
                 inputGateActivation, candidateGateActivation))) {
  } else {
    return mlir::failure();
  }

  if (!validatePreviousCellInput(cellStateInput, layer, timestep, previousSteps,
                                 types, cellStateArgument))
    return mlir::failure();

  return LSTMCellUpdate{cellAddOp, forgetGateActivation, inputGateActivation,
                        candidateGateActivation, cellStateInput};
}

static mlir::FailureOr<LSTMGatePreactivation> matchLSTMGatePreactivation(
    LSTMHiddenOutput hiddenOutput, LSTMCellUpdate cellUpdate, int64_t layer,
    int64_t timestep, llvm::ArrayRef<LSTMStep> previousSteps,
    const LSTMTypes &types, mlir::Value hiddenStateArgument) {
  int64_t gateWidth = 4 * types.hiddenSize;
  layer_patterns::RecurrentGateIndexingScaffoldMatch indexing;
  mlir::Operation *collapsedPreattivation = nullptr;
  mlir::Operation *sigmoidOneConstant = nullptr;
  layer_patterns::RecurrentGateSliceMatch outputGate;
  if (mlir::failed(matchGateSlice(hiddenOutput.outputGateActivation,
                                  hiddenOutput.sharedHiddenOutputEmpty,
                                  3 * types.hiddenSize, gateWidth, true,
                                  outputGate, indexing, collapsedPreattivation,
                                  sigmoidOneConstant)))
    return mlir::failure();

  layer_patterns::RecurrentGateSliceMatch forgetGate;
  if (mlir::failed(matchGateSlice(
          cellUpdate.forgetGateActivation, hiddenOutput.sharedHiddenOutputEmpty,
          types.hiddenSize, gateWidth, true, forgetGate, indexing,
          collapsedPreattivation, sigmoidOneConstant)))
    return mlir::failure();

  layer_patterns::RecurrentGateSliceMatch inputGate;
  if (mlir::failed(matchGateSlice(cellUpdate.inputGateActivation,
                                  hiddenOutput.sharedHiddenOutputEmpty, 0,
                                  gateWidth, true, inputGate, indexing,
                                  collapsedPreattivation, sigmoidOneConstant)))
    return mlir::failure();

  layer_patterns::RecurrentGateSliceMatch candidateGate;
  if (mlir::failed(matchGateSlice(cellUpdate.candidateGateActivation,
                                  hiddenOutput.sharedHiddenOutputEmpty,
                                  2 * types.hiddenSize, gateWidth, false,
                                  candidateGate, indexing,
                                  collapsedPreattivation, sigmoidOneConstant)))
    return mlir::failure();

  auto preActivationCollapseOp =
      llvm::dyn_cast_or_null<CollapseShapeOp>(collapsedPreattivation);
  if (!preActivationCollapseOp ||
      !tensor_type::hasStaticF32Shape(preActivationCollapseOp.getResult(),
                                      {gateWidth}) ||
      !tensor_type::hasStaticF32Shape(preActivationCollapseOp.getSrc(),
                                      {1, types.batchSize, gateWidth}))
    return mlir::failure();

  auto preActivationAddOp =
      layer_utils::producerOfType<GenericOp>(preActivationCollapseOp.getSrc());
  if (!isAddfGeneric(preActivationAddOp) ||
      !tensor_type::hasStaticF32Shape(preActivationAddOp.getResult(0),
                                      {1, types.batchSize, gateWidth}) ||
      !tensor_type::hasStaticF32Shape(preActivationAddOp.getOutputs()[0],
                                      {1, types.batchSize, gateWidth}))
    return mlir::failure();

  auto firstSlice = layer_patterns::matchTimestepProjectionSlice(
      preActivationAddOp.getInputs()[0], timestep,
      llvm::ArrayRef<int64_t>{types.batchSize, gateWidth}, types.sequenceLength,
      types.batchSize, gateWidth);
  auto secondSlice = layer_patterns::matchTimestepProjectionSlice(
      preActivationAddOp.getInputs()[1], timestep,
      llvm::ArrayRef<int64_t>{types.batchSize, gateWidth}, types.sequenceLength,
      types.batchSize, gateWidth);
  if (mlir::succeeded(firstSlice) == mlir::succeeded(secondSlice))
    return mlir::failure();

  unsigned inputSliceIndex = mlir::succeeded(firstSlice) ? 0 : 1;
  if (!linalg_match::hasPreActivationAddIndexingMaps(preActivationAddOp,
                                                     inputSliceIndex))
    return mlir::failure();

  unsigned recurrentIndex = mlir::succeeded(firstSlice) ? 1 : 0;
  auto recurrentProjection = matchRecurrentProjection(
      preActivationAddOp.getInputs()[recurrentIndex], types);
  if (mlir::failed(recurrentProjection))
    return mlir::failure();

  if (!validateRecurrentActivation(recurrentProjection->activation, layer,
                                   timestep, previousSteps, types,
                                   hiddenStateArgument))
    return mlir::failure();

  auto sliceMatch = mlir::succeeded(firstSlice) ? *firstSlice : *secondSlice;
  return LSTMGatePreactivation{preActivationAddOp, preActivationCollapseOp,
                               sliceMatch.collapse, sliceMatch.slice,
                               *recurrentProjection};
}

// Matches one LSTM timestep from gate math through hidden output.
static mlir::FailureOr<LSTMStep>
matchLSTMStep(mlir::Value hiddenOutput, int64_t layer, int64_t timestep,
              llvm::ArrayRef<LSTMStep> previousSteps, const LSTMTypes &types,
              mlir::Value hiddenStateArgument, mlir::Value cellStateArgument) {
  if (!tensor_type::hasStaticF32Shape(hiddenOutput,
                                      {1, types.batchSize, types.hiddenSize}))
    return mlir::failure();

  auto hiddenOutputMatch = matchLSTMHiddenOutput(hiddenOutput);
  if (mlir::failed(hiddenOutputMatch))
    return mlir::failure();

  auto cellUpdate =
      matchLSTMCellUpdate(*hiddenOutputMatch, layer, timestep, previousSteps,
                          types, cellStateArgument);
  if (mlir::failed(cellUpdate))
    return mlir::failure();

  auto preactivation = matchLSTMGatePreactivation(
      *hiddenOutputMatch, *cellUpdate, layer, timestep, previousSteps, types,
      hiddenStateArgument);
  if (mlir::failed(preactivation))
    return mlir::failure();

  return LSTMStep{hiddenOutput,
                  cellUpdate->cellAddOp.getResult(0),
                  hiddenOutputMatch->hiddenMulOp,
                  cellUpdate->cellAddOp,
                  preactivation->preActivationAddOp,
                  preactivation->preActivationCollapseOp,
                  preactivation->inputCollapseOp,
                  preactivation->inputSliceOp,
                  preactivation->recurrentProjection};
}

static bool isSameConstantOp(ConstantOp lhs, ConstantOp rhs) {
  return lhs && rhs && lhs.getOperation() == rhs.getOperation();
}

static bool hasSharedLayerRecurrentParameters(const LSTMLayer &layer,
                                              bool hasBias) {
  if (layer.steps.empty())
    return false;

  const RecurrentProjection &firstProjection =
      layer.steps.front().recurrentProjection;
  if (!firstProjection.weightConstant)
    return false;
  if (hasBias && !firstProjection.biasConstant)
    return false;

  for (const LSTMStep &step : layer.steps) {
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
matchFirstLayerInputProjection(mlir::Value source, const LSTMTypes &types,
                               mlir::Value inputArgument) {
  int64_t gateWidth = 4 * types.hiddenSize;
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
matchStackedLayerInputProjection(mlir::Value source, const LSTMTypes &types) {
  int64_t gateWidth = 4 * types.hiddenSize;
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

static mlir::FailureOr<LSTMLayer>
matchLSTMLayerBody(llvm::ArrayRef<mlir::Value> layerOutputs, int64_t layer,
                   const LSTMTypes &types, mlir::Value inputArgument,
                   mlir::Value hiddenStateArgument,
                   mlir::Value cellStateArgument) {
  if (layerOutputs.size() != static_cast<size_t>(types.sequenceLength))
    return mlir::failure();

  LSTMLayer layerMatch;
  layerMatch.steps.reserve(layerOutputs.size());

  mlir::Value commonInputSource;
  for (auto [timestep, output] : llvm::enumerate(layerOutputs)) {
    auto step = matchLSTMStep(output, layer, timestep, layerMatch.steps, types,
                              hiddenStateArgument, cellStateArgument);
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

static mlir::FailureOr<LSTMTypes> getSupportedLSTMTypes(mlir::func::FuncOp func,
                                                        bool hasBias) {
  if (func.getNumArguments() != 3 || func.getNumResults() != 3 ||
      !func.getBody().hasOneBlock())
    return mlir::failure();

  auto inputTy =
      tensor_type::getStaticF32Tensor(func.getArgument(0).getType(), 3);
  auto hiddenStateTy =
      tensor_type::getStaticF32Tensor(func.getArgument(1).getType(), 3);
  auto cellStateTy =
      tensor_type::getStaticF32Tensor(func.getArgument(2).getType(), 3);
  auto sequenceResultTy =
      tensor_type::getStaticF32Tensor(func.getResultTypes()[0], 3);
  auto hiddenResultTy =
      tensor_type::getStaticF32Tensor(func.getResultTypes()[1], 3);
  auto cellResultTy =
      tensor_type::getStaticF32Tensor(func.getResultTypes()[2], 3);
  if (mlir::failed(inputTy) || mlir::failed(hiddenStateTy) ||
      mlir::failed(cellStateTy) || mlir::failed(sequenceResultTy) ||
      mlir::failed(hiddenResultTy) || mlir::failed(cellResultTy))
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

  if (cellStateTy->getShape() != hiddenStateTy->getShape() ||
      sequenceResultTy->getShape() !=
          llvm::ArrayRef<int64_t>({batchSize, sequenceLength, hiddenSize}) ||
      hiddenResultTy->getShape() != hiddenStateTy->getShape() ||
      cellResultTy->getShape() != hiddenStateTy->getShape())
    return mlir::failure();

  return LSTMTypes{*inputTy,          *hiddenStateTy,  *cellStateTy,
                   *sequenceResultTy, *hiddenResultTy, *cellResultTy,
                   layerCount,        sequenceLength,  batchSize,
                   inputSize,         hiddenSize,      hasBias};
}

// Verifies the full supported LSTM shape, assembly, and layer chain.
static mlir::FailureOr<LSTMMatch> matchSupportedLSTM(mlir::func::FuncOp func,
                                                     bool hasBias) {
  auto types = getSupportedLSTMTypes(func, hasBias);
  if (mlir::failed(types))
    return mlir::failure();

  auto returnOp =
      llvm::dyn_cast<ReturnOp>(func.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 3)
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
  if (tanhCount != 2 * types->layerCount * types->sequenceLength)
    return mlir::failure();

  llvm::SmallVector<llvm::SmallVector<mlir::Value>> layerOutputs(
      types->layerCount);
  layerOutputs.back().assign(sequenceAssembly->second.getInputs().begin(),
                             sequenceAssembly->second.getInputs().end());

  llvm::SmallVector<LSTMLayer> layers;
  layers.resize(types->layerCount);
  for (int64_t layer = types->layerCount - 1; layer >= 0; --layer) {
    auto layerMatch = matchLSTMLayerBody(
        layerOutputs[layer], layer, *types, func.getArgument(0),
        func.getArgument(1), func.getArgument(2));
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

  auto cellAssembly = layer_patterns::matchFinalRecurrentStateAssembly(
      returnOp.getOperand(2), types->layerCount, types->batchSize,
      types->hiddenSize, [&](auto layer) -> mlir::Value {
        auto layerIndex = static_cast<size_t>(layer);
        if (layerIndex >= layers.size() || layers[layerIndex].steps.empty())
          return {};
        return layers[layerIndex].steps.back().cellOutput;
      });
  if (mlir::failed(cellAssembly))
    return mlir::failure();

  return LSTMMatch{returnOp,
                   sequenceAssembly->first,
                   sequenceAssembly->second,
                   hiddenAssembly->first,
                   hiddenAssembly->second,
                   cellAssembly->first,
                   cellAssembly->second,
                   layers,
                   *types};
}

static mlir::FailureOr<LSTMMatch> matchSupportedLSTM(mlir::func::FuncOp func) {
  auto noBiasMatch = matchSupportedLSTM(func, /*hasBias=*/false);
  if (mlir::succeeded(noBiasMatch))
    return noBiasMatch;

  return matchSupportedLSTM(func, /*hasBias=*/true);
}

static void appendRecurrentOperands(LSTMMatch &match,
                                    llvm::SmallVectorImpl<mlir::Value> &ops) {
  for (LSTMLayer &layer : match.layers) {
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

static void rewriteLSTMMatchToSculptorOp(mlir::func::FuncOp func,
                                       LSTMMatch &match,
                                       mlir::RewriterBase &rewriter) {
  if (!func.getBody().hasOneBlock())
    return;

  llvm::SmallVector<mlir::Operation *> oldBodyOps;
  for (mlir::Operation &op : func.getBody().front().without_terminator())
    oldBodyOps.push_back(&op);

  llvm::SmallVector<mlir::Value> recurrentOperands;
  appendRecurrentOperands(match, recurrentOperands);

  rewriter.setInsertionPoint(match.returnOp);
  auto lstmOp = rewriter.create<mlir::sculptor::NNLSTMOp>(
      match.returnOp.getLoc(), func.getResultTypes(), func.getArgument(0),
      func.getArgument(1), func.getArgument(2), recurrentOperands,
      /*batch_first=*/true, match.types.hasBias,
      static_cast<uint64_t>(match.types.hiddenSize),
      static_cast<uint64_t>(match.types.layerCount));
  rewriter.replaceOpWithNewOp<ReturnOp>(match.returnOp, lstmOp.getResults());

  for (mlir::Operation *op : llvm::reverse(oldBodyOps)) {
    if (op->use_empty())
      rewriter.eraseOp(op);
  }
}

class LSTMCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  explicit LSTMCanonicalizer(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "lstm"; }

  void canonicalize(mlir::func::FuncOp func) const override {
    auto match = matchSupportedLSTM(func);
    if (mlir::failed(match))
      return;

    mlir::IRRewriter rewriter(func.getContext());
    rewriteLSTMMatchToSculptorOp(func, *match, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerLSTMCanonicalizer(LayerCanonicalizers &canonicalizers,
                               MLIRContext *context) {
  canonicalizers.push_back(std::make_unique<LSTMCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
