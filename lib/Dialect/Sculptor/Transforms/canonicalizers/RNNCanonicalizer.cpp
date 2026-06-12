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

namespace tensor_type = mlir::sculptor::tensor_type;
namespace linalg_match = mlir::sculptor::linalg_match;
namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace layer_utils = mlir::sculptor::layer_utils;

namespace {

using mlir::arith::AddFOp;
using mlir::arith::ConstantOp;
using mlir::func::ReturnOp;
using mlir::linalg::BatchMatmulOp;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::math::TanhOp;
using mlir::tensor::CollapseShapeOp;
using mlir::tensor::ConcatOp;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;
using mlir::tensor::ExtractSliceOp;

struct RNNTypes {
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
  ExpandShapeOp expandOp;
  GenericOp biasAddOp;
  ConstantOp biasConstant;
};

struct RNNStep {
  mlir::Value output;
  GenericOp tanhOp;
  GenericOp preActivationAddOp;
  CollapseShapeOp inputCollapseOp;
  ExtractSliceOp inputSliceOp;
  RecurrentProjection recurrentProjection;
};

struct RNNLayer {
  ConcatOp outputConcat;
  LayerInputProjection inputProjection;
  llvm::SmallVector<RNNStep> steps;
};

struct RNNMatch {
  ReturnOp returnOp;
  TransposeOp sequenceTranspose;
  ConcatOp sequenceConcat;
  ExpandShapeOp hiddenExpand;
  ConcatOp hiddenConcat;
  llvm::SmallVector<RNNLayer> layers;
  RNNTypes types;
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

static bool isTanhGeneric(GenericOp genericOp) {
  if (!linalg_match::hasExpectedBodyShape(genericOp, /*inputCount=*/1) ||
      !linalg_match::hasElementwiseIndexingMaps(genericOp, /*rank=*/3))
    return false;

  mlir::Block &body = genericOp.getRegion().front();
  auto tanhOp = llvm::dyn_cast<TanhOp>(body.getOperations().front());
  auto yieldOp =
      llvm::dyn_cast<mlir::linalg::YieldOp>(body.getOperations().back());
  return tanhOp && yieldOp && yieldOp.getValues().size() == 1 &&
         tanhOp.getOperand() == body.getArgument(0) &&
         yieldOp.getValues().front() == tanhOp.getResult();
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
matchRecurrentProjection(mlir::Value value, const RNNTypes &types) {
  auto expand = layer_utils::producerOfType<ExpandShapeOp>(value);
  if (!expand ||
      !tensor_type::hasStaticF32Shape(expand.getResult(),
                                      {1, types.batchSize, types.hiddenSize}) ||
      !tensor_type::hasStaticF32Shape(expand.getSrc(),
                                      {types.batchSize, types.hiddenSize}))
    return mlir::failure();

  mlir::Value matmulValue = expand.getSrc();
  GenericOp biasAddOp;
  ConstantOp biasConstant;
  if (types.hasBias) {
    auto biasMatch = layer_patterns::matchProjectionBiasAdd(
        matmulValue, {types.batchSize, types.hiddenSize}, types.hiddenSize);
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
      matmulValue, types.batchSize, types.hiddenSize, types.hiddenSize);
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

// Matches one unrolled RNN timestep against input and recurrent projections.
static mlir::FailureOr<RNNStep>
matchRNNStep(mlir::Value output, int64_t layer, int64_t timestep,
             llvm::ArrayRef<mlir::Value> layerOutputs, const RNNTypes &types,
             mlir::Value hiddenStateArgument) {
  if (!tensor_type::hasStaticF32Shape(output,
                                      {1, types.batchSize, types.hiddenSize}))
    return mlir::failure();

  auto tanhOp = layer_utils::producerOfType<GenericOp>(output);
  if (!isTanhGeneric(tanhOp) ||
      !tensor_type::hasStaticF32Shape(tanhOp.getInputs()[0],
                                      {1, types.batchSize, types.hiddenSize}) ||
      !tensor_type::hasStaticF32Shape(tanhOp.getOutputs()[0],
                                      {1, types.batchSize, types.hiddenSize}))
    return mlir::failure();

  auto preActivationAdd =
      layer_utils::producerOfType<GenericOp>(tanhOp.getInputs()[0]);
  if (!isAddfGeneric(preActivationAdd) ||
      !tensor_type::hasStaticF32Shape(preActivationAdd.getResult(0),
                                      {1, types.batchSize, types.hiddenSize}) ||
      !tensor_type::hasStaticF32Shape(preActivationAdd.getOutputs()[0],
                                      {1, types.batchSize, types.hiddenSize}))
    return mlir::failure();

  auto firstSlice = layer_patterns::matchTimestepProjectionSlice(
      preActivationAdd.getInputs()[0], timestep,
      llvm::ArrayRef<int64_t>{types.batchSize, types.hiddenSize},
      types.sequenceLength, types.batchSize, types.hiddenSize);
  auto secondSlice = layer_patterns::matchTimestepProjectionSlice(
      preActivationAdd.getInputs()[1], timestep,
      llvm::ArrayRef<int64_t>{types.batchSize, types.hiddenSize},
      types.sequenceLength, types.batchSize, types.hiddenSize);
  if (mlir::succeeded(firstSlice) == mlir::succeeded(secondSlice))
    return mlir::failure();

  unsigned inputSliceIndex = mlir::succeeded(firstSlice) ? 0 : 1;
  if (!linalg_match::hasPreActivationAddIndexingMaps(preActivationAdd,
                                                     inputSliceIndex))
    return mlir::failure();

  unsigned recurrentIndex = mlir::succeeded(firstSlice) ? 1 : 0;
  auto recurrentProjection = matchRecurrentProjection(
      preActivationAdd.getInputs()[recurrentIndex], types);
  if (mlir::failed(recurrentProjection))
    return mlir::failure();

  if (timestep == 0) {
    if (!layer_patterns::matchesInitialRecurrentStateCollapsed(
            recurrentProjection->activation, hiddenStateArgument, layer,
            types.batchSize, types.hiddenSize))
      return mlir::failure();
  } else if (!layer_patterns::matchesCollapsedRecurrentValue(
                 recurrentProjection->activation, layerOutputs[timestep - 1],
                 types.batchSize, types.hiddenSize)) {
    return mlir::failure();
  }

  auto sliceMatch = mlir::succeeded(firstSlice) ? *firstSlice : *secondSlice;
  return RNNStep{output,           tanhOp,
                 preActivationAdd, sliceMatch.collapse,
                 sliceMatch.slice, *recurrentProjection};
}

static bool isSameConstantOp(ConstantOp lhs, ConstantOp rhs) {
  return lhs && rhs && lhs.getOperation() == rhs.getOperation();
}

static bool hasSharedLayerRecurrentParameters(const RNNLayer &layer,
                                              bool hasBias) {
  if (layer.steps.empty())
    return false;

  const RecurrentProjection &firstProjection =
      layer.steps.front().recurrentProjection;
  if (!firstProjection.weightConstant)
    return false;
  if (hasBias && !firstProjection.biasConstant)
    return false;

  for (const RNNStep &step : layer.steps) {
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
matchFirstLayerInputProjection(mlir::Value source, const RNNTypes &types,
                               mlir::Value inputArgument) {
  if (!tensor_type::hasStaticF32Shape(
          source, {types.sequenceLength, types.batchSize, types.hiddenSize}))
    return mlir::failure();

  LayerInputProjection projection;
  projection.source = source;

  if (auto expand = layer_utils::producerOfType<ExpandShapeOp>(source)) {
    if (!tensor_type::hasStaticF32Shape(
            expand.getSrc(),
            {types.sequenceLength * types.batchSize, types.hiddenSize}))
      return mlir::failure();

    mlir::Value matmulValue = expand.getSrc();
    if (types.hasBias) {
      auto biasMatch = layer_patterns::matchProjectionBiasAdd(
          matmulValue,
          {types.sequenceLength * types.batchSize, types.hiddenSize},
          types.hiddenSize);
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
        types.hiddenSize);
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
        batchMatmulValue,
        {types.sequenceLength, types.batchSize, types.hiddenSize},
        types.hiddenSize);
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
          {types.sequenceLength, types.batchSize, types.hiddenSize}))
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
          {types.sequenceLength, types.inputSize, types.hiddenSize}) ||
      !tensor_type::hasStaticF32Shape(
          broadcast.getOutputs()[0],
          {types.sequenceLength, types.inputSize, types.hiddenSize}))
    return mlir::failure();

  TransposeOp weightTranspose;
  auto weightConstant = layer_patterns::matchProjectionWeightTranspose(
      broadcast.getInputs()[0], types.inputSize, types.hiddenSize,
      weightTranspose);
  if (mlir::failed(weightConstant))
    return mlir::failure();

  projection.batchMatmulOp = batchMatmul;
  projection.weightConstant = *weightConstant;
  projection.weightTranspose = weightTranspose;
  return projection;
}

static mlir::FailureOr<LayerInputProjection>
matchStackedLayerInputProjection(mlir::Value source, const RNNTypes &types) {
  if (!tensor_type::hasStaticF32Shape(
          source, {types.sequenceLength, types.batchSize, types.hiddenSize}))
    return mlir::failure();

  auto expand = layer_utils::producerOfType<ExpandShapeOp>(source);
  if (!expand || !tensor_type::hasStaticF32Shape(
                     expand.getSrc(), {types.sequenceLength * types.batchSize,
                                       types.hiddenSize}))
    return mlir::failure();

  LayerInputProjection projection;
  projection.source = source;

  mlir::Value matmulValue = expand.getSrc();
  if (types.hasBias) {
    auto biasMatch = layer_patterns::matchProjectionBiasAdd(
        matmulValue, {types.sequenceLength * types.batchSize, types.hiddenSize},
        types.hiddenSize);
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
      types.hiddenSize);
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

// Collects one RNN layer from timestep outputs and shared parameters.
static mlir::FailureOr<RNNLayer>
matchRNNLayerBody(llvm::ArrayRef<mlir::Value> layerOutputs, int64_t layer,
                  const RNNTypes &types, mlir::Value inputArgument,
                  mlir::Value hiddenStateArgument) {
  if (layerOutputs.size() != static_cast<size_t>(types.sequenceLength))
    return mlir::failure();

  RNNLayer layerMatch;
  layerMatch.steps.reserve(layerOutputs.size());

  mlir::Value commonInputSource;
  for (auto [timestep, output] : llvm::enumerate(layerOutputs)) {
    auto step = matchRNNStep(output, layer, timestep, layerOutputs, types,
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

static mlir::FailureOr<RNNTypes> getSupportedRNNTypes(mlir::func::FuncOp func,
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

  return RNNTypes{*inputTy,        *hiddenStateTy, *sequenceResultTy,
                  *hiddenResultTy, layerCount,     sequenceLength,
                  batchSize,       inputSize,      hiddenSize,
                  hasBias};
}

// Verifies the full supported RNN shape, assembly, and layer chain.
static mlir::FailureOr<RNNMatch> matchSupportedRNN(mlir::func::FuncOp func,
                                                   bool hasBias) {
  auto types = getSupportedRNNTypes(func, hasBias);
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
    if (isTanhGeneric(llvm::dyn_cast<GenericOp>(op)))
      ++tanhCount;
  });
  if (tanhCount != types->layerCount * types->sequenceLength)
    return mlir::failure();

  llvm::SmallVector<llvm::SmallVector<mlir::Value>> layerOutputs(
      types->layerCount);
  layerOutputs.back().assign(sequenceAssembly->second.getInputs().begin(),
                             sequenceAssembly->second.getInputs().end());

  llvm::SmallVector<RNNLayer> layers;
  layers.resize(types->layerCount);
  for (int64_t layer = types->layerCount - 1; layer >= 0; --layer) {
    auto layerMatch =
        matchRNNLayerBody(layerOutputs[layer], layer, *types,
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
        return layers[layerIndex].steps.back().output;
      });
  if (mlir::failed(hiddenAssembly))
    return mlir::failure();

  return RNNMatch{returnOp,
                  sequenceAssembly->first,
                  sequenceAssembly->second,
                  hiddenAssembly->first,
                  hiddenAssembly->second,
                  layers,
                  *types};
}

static mlir::FailureOr<RNNMatch> matchSupportedRNN(mlir::func::FuncOp func) {
  auto noBiasMatch = matchSupportedRNN(func, /*hasBias=*/false);
  if (mlir::succeeded(noBiasMatch))
    return noBiasMatch;

  return matchSupportedRNN(func, /*hasBias=*/true);
}

static void appendRecurrentOperands(RNNMatch &match,
                                    llvm::SmallVectorImpl<mlir::Value> &ops) {
  for (RNNLayer &layer : match.layers) {
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

// Replaces the matched RNN dataflow with a single sculptor.nn.rnn op.
static void rewriteRNNMatchToSculptorOp(mlir::func::FuncOp func, RNNMatch &match,
                                      mlir::RewriterBase &rewriter) {
  if (!func.getBody().hasOneBlock())
    return;

  llvm::SmallVector<mlir::Operation *> oldBodyOps;
  for (mlir::Operation &op : func.getBody().front().without_terminator())
    oldBodyOps.push_back(&op);

  llvm::SmallVector<mlir::Value> recurrentOperands;
  appendRecurrentOperands(match, recurrentOperands);

  rewriter.setInsertionPoint(match.returnOp);
  auto rnnOp = rewriter.create<mlir::sculptor::NNRNNOp>(
      match.returnOp.getLoc(), func.getResultTypes(), func.getArgument(0),
      func.getArgument(1), recurrentOperands, /*batch_first=*/true,
      match.types.hasBias, static_cast<uint64_t>(match.types.hiddenSize),
      static_cast<uint64_t>(match.types.layerCount));
  rewriter.replaceOpWithNewOp<ReturnOp>(match.returnOp, rnnOp.getResults());

  for (mlir::Operation *op : llvm::reverse(oldBodyOps)) {
    if (op->use_empty())
      rewriter.eraseOp(op);
  }
}

class RNNCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  explicit RNNCanonicalizer(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "rnn"; }

  void canonicalize(mlir::func::FuncOp func) const override {
    auto match = matchSupportedRNN(func);
    if (mlir::failed(match))
      return;

    mlir::IRRewriter rewriter(func.getContext());
    rewriteRNNMatchToSculptorOp(func, *match, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerRNNCanonicalizer(LayerCanonicalizers &canonicalizers,
                              MLIRContext *context) {
  canonicalizers.push_back(std::make_unique<RNNCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
