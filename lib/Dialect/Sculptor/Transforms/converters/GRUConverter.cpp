#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConstantUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/MVMBuildUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/NNLayerMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/RecurrentElementwiseUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/RecurrentGateUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/RecurrentLayerConversionUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/TensorTypeUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/SemanticOperationScope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <memory>

namespace converter_constant = mlir::sculptor::converter_constant;
namespace converter_recurrent_elementwise =
    mlir::sculptor::converter_recurrent_elementwise;
namespace converter_recurrent_layer = mlir::sculptor::converter_recurrent_layer;
namespace mvm_build = mlir::sculptor::mvm_build;
namespace tensor_type = mlir::sculptor::tensor_type;
namespace nn_layer_match = mlir::sculptor::nn_layer_match;
namespace recurrent_gate = mlir::sculptor::recurrent_gate;

namespace {

using mlir::sculptor::NNGRULayerOp;
using mlir::sculptor::NNGRUOp;
using mlir::arith::ConstantOp;
using mlir::tensor::ConcatOp;
using mlir::tensor::EmptyOp;

struct GRULayerLowering {
  NNGRULayerOp gruLayerOp;
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType hiddenStateTy;
  mlir::RankedTensorType outputTy;
  mlir::RankedTensorType hiddenResultTy;
  mlir::RankedTensorType weightIHTy;
  mlir::RankedTensorType weightHHTy;
  mlir::RankedTensorType hidden2DTy;
  mlir::RankedTensorType hiddenSliceTy;
  mlir::RankedTensorType inputSliceTy;
  mlir::RankedTensorType input2DTy;
  mlir::RankedTensorType rowInputTy;
  mlir::RankedTensorType rowHiddenTy;
  mlir::RankedTensorType rowFusedInputTy;
  mlir::RankedTensorType rowPreActivationTy;
  mlir::RankedTensorType preActivationTy;
  mlir::RankedTensorType timestepResultTy;
  mlir::RankedTensorType fusedWeightTy;
  mlir::RankedTensorType fusedBiasTy;
  ConstantOp weightIHConstant;
  ConstantOp weightHHConstant;
  ConstantOp biasIHConstant;
  ConstantOp biasHHConstant;
  int64_t layerIndex = 0;
  int64_t numLayers = 0;
  int64_t batchSize = 0;
  int64_t sequenceLength = 0;
  int64_t inputSize = 0;
  int64_t hiddenSize = 0;
  bool hasBias = false;
};

struct GRUTimestepResult {
  mlir::Value hidden;
  mlir::Value output;
};

static mlir::FailureOr<GRULayerLowering>
matchGRULayer(NNGRULayerOp op) {
  mlir::FailureOr<NNGRULayerOp> gruLayerOp = op;
  bool hasBias = (*gruLayerOp).getHasBias();
  if (!(*gruLayerOp).getBatchFirst())
    return mlir::failure();

  auto inputTy = tensor_type::getStaticF32Tensor(
      (*gruLayerOp).getInput().getType(), /*expectedRank=*/3);
  auto hiddenStateTy = tensor_type::getStaticF32Tensor(
      (*gruLayerOp).getH0().getType(), /*expectedRank=*/3);
  auto outputTy = tensor_type::getStaticF32Tensor(
      (*gruLayerOp).getOutput().getType(), /*expectedRank=*/3);
  auto hiddenResultTy = tensor_type::getStaticF32Tensor(
      (*gruLayerOp).getHn().getType(), /*expectedRank=*/3);
  auto weightIHTy = tensor_type::getStaticF32Tensor(
      (*gruLayerOp).getWIh().getType(), /*expectedRank=*/2);
  auto weightHHTy = tensor_type::getStaticF32Tensor(
      (*gruLayerOp).getWHh().getType(), /*expectedRank=*/2);
  if (mlir::failed(inputTy) || mlir::failed(hiddenStateTy) ||
      mlir::failed(outputTy) || mlir::failed(hiddenResultTy) ||
      mlir::failed(weightIHTy) || mlir::failed(weightHHTy))
    return mlir::failure();

  int64_t layerIndex = (*gruLayerOp).getLayerIndex();
  int64_t numLayers = (*gruLayerOp).getNumLayers();
  int64_t hiddenSize = (*gruLayerOp).getHiddenSize();
  int64_t batchSize = inputTy->getShape()[0];
  int64_t sequenceLength = inputTy->getShape()[1];
  int64_t inputSize = inputTy->getShape()[2];
  if (layerIndex < 0 || numLayers < 1 || layerIndex >= numLayers ||
      batchSize < 1 || sequenceLength < 1 || inputSize < 1 || hiddenSize < 1)
    return mlir::failure();

  if (hiddenStateTy->getShape() !=
          llvm::ArrayRef<int64_t>({numLayers, batchSize, hiddenSize}) ||
      outputTy->getShape() !=
          llvm::ArrayRef<int64_t>({batchSize, sequenceLength, hiddenSize}) ||
      hiddenResultTy->getShape() !=
          llvm::ArrayRef<int64_t>({1, batchSize, hiddenSize}) ||
      weightIHTy->getShape() !=
          llvm::ArrayRef<int64_t>({hiddenSize * 3, inputSize}) ||
      weightHHTy->getShape() !=
          llvm::ArrayRef<int64_t>({hiddenSize * 3, hiddenSize}))
    return mlir::failure();

  auto weightIHConstant = (*gruLayerOp).getWIh().getDefiningOp<ConstantOp>();
  auto weightHHConstant = (*gruLayerOp).getWHh().getDefiningOp<ConstantOp>();
  if (!weightIHConstant || !weightHHConstant)
    return mlir::failure();

  ConstantOp biasIHConstant;
  ConstantOp biasHHConstant;
  mlir::RankedTensorType fusedBiasTy;
  if (hasBias) {
    mlir::Value biasIH = (*gruLayerOp).getBIh();
    mlir::Value biasHH = (*gruLayerOp).getBHh();
    if (!biasIH || !biasHH)
      return mlir::failure();

    auto biasIHTy =
        tensor_type::getStaticF32Tensor(biasIH.getType(), /*expectedRank=*/1);
    auto biasHHTy =
        tensor_type::getStaticF32Tensor(biasHH.getType(), /*expectedRank=*/1);
    if (mlir::failed(biasIHTy) || mlir::failed(biasHHTy) ||
        biasIHTy->getShape() != llvm::ArrayRef<int64_t>({hiddenSize * 3}) ||
        biasHHTy->getShape() != llvm::ArrayRef<int64_t>({hiddenSize * 3}))
      return mlir::failure();

    biasIHConstant = biasIH.getDefiningOp<ConstantOp>();
    biasHHConstant = biasHH.getDefiningOp<ConstantOp>();
    if (!biasIHConstant || !biasHHConstant)
      return mlir::failure();

    fusedBiasTy = mlir::RankedTensorType::get({hiddenSize * 4},
                                              inputTy->getElementType());
  } else if ((*gruLayerOp).getBIh() || (*gruLayerOp).getBHh()) {
    return mlir::failure();
  }

  mlir::Type elementType = inputTy->getElementType();
  GRULayerLowering lowering;
  lowering.gruLayerOp = *gruLayerOp;
  lowering.inputTy = *inputTy;
  lowering.hiddenStateTy = *hiddenStateTy;
  lowering.outputTy = *outputTy;
  lowering.hiddenResultTy = *hiddenResultTy;
  lowering.weightIHTy = *weightIHTy;
  lowering.weightHHTy = *weightHHTy;
  lowering.hidden2DTy =
      mlir::RankedTensorType::get({batchSize, hiddenSize}, elementType);
  lowering.hiddenSliceTy =
      mlir::RankedTensorType::get({1, batchSize, hiddenSize}, elementType);
  lowering.inputSliceTy =
      mlir::RankedTensorType::get({batchSize, 1, inputSize}, elementType);
  lowering.input2DTy =
      mlir::RankedTensorType::get({batchSize, inputSize}, elementType);
  lowering.rowInputTy =
      mlir::RankedTensorType::get({1, inputSize}, elementType);
  lowering.rowHiddenTy =
      mlir::RankedTensorType::get({1, hiddenSize}, elementType);
  lowering.rowFusedInputTy = mlir::RankedTensorType::get(
      {1, inputSize + hiddenSize}, elementType);
  lowering.rowPreActivationTy =
      mlir::RankedTensorType::get({1, hiddenSize * 4}, elementType);
  lowering.preActivationTy =
      mlir::RankedTensorType::get({batchSize, hiddenSize * 4}, elementType);
  lowering.timestepResultTy =
      mlir::RankedTensorType::get({batchSize, 1, hiddenSize}, elementType);
  lowering.fusedWeightTy = mlir::RankedTensorType::get(
      {hiddenSize * 4, inputSize + hiddenSize}, elementType);
  lowering.fusedBiasTy = fusedBiasTy;
  lowering.weightIHConstant = weightIHConstant;
  lowering.weightHHConstant = weightHHConstant;
  lowering.biasIHConstant = biasIHConstant;
  lowering.biasHHConstant = biasHHConstant;
  lowering.layerIndex = layerIndex;
  lowering.numLayers = numLayers;
  lowering.batchSize = batchSize;
  lowering.sequenceLength = sequenceLength;
  lowering.inputSize = inputSize;
  lowering.hiddenSize = hiddenSize;
  lowering.hasBias = hasBias;
  return lowering;
}

static mlir::FailureOr<GRULayerLowering>
matchExtractedGRULayer(mlir::func::FuncOp func) {
  auto gruLayerOp = nn_layer_match::matchSingleNNLayerOp<NNGRULayerOp>(func);
  if (mlir::failed(gruLayerOp))
    return mlir::failure();

  bool hasBias = (*gruLayerOp).getHasBias();
  if (!nn_layer_match::hasLayerTypeMatchingBias(func, "gru", "gru_w_bias",
                                                hasBias) ||
      func.getNumArguments() != 2 || func.getNumResults() != 2 ||
      (*gruLayerOp).getInput() != func.getArgument(0) ||
      (*gruLayerOp).getH0() != func.getArgument(1))
    return mlir::failure();
  return matchGRULayer(*gruLayerOp);
}

static mlir::LogicalResult splitGRUStack(NNGRUOp op,
                                         mlir::RewriterBase &rewriter) {
  auto inputTy = llvm::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
  auto outputTy =
      llvm::dyn_cast<mlir::RankedTensorType>(op.getOutput().getType());
  auto hiddenTy = llvm::dyn_cast<mlir::RankedTensorType>(op.getHn().getType());
  if (!op.getBatchFirst() || !inputTy || !outputTy || !hiddenTy ||
      !inputTy.hasStaticShape() || !outputTy.hasStaticShape() ||
      !hiddenTy.hasStaticShape() || inputTy.getRank() != 3 ||
      outputTy.getRank() != 3 || hiddenTy.getRank() != 3)
    return mlir::failure();

  int64_t layerCount = op.getNumLayers();
  int64_t operandsPerLayer = op.getHasBias() ? 4 : 2;
  if (layerCount < 1 || static_cast<int64_t>(op.getRecurrentOperands().size()) !=
                            layerCount * operandsPerLayer)
    return mlir::failure();

  auto hiddenSliceTy = mlir::RankedTensorType::get(
      {1, hiddenTy.getDimSize(1), hiddenTy.getDimSize(2)},
      hiddenTy.getElementType());
  mlir::Value currentSequence = op.getInput();
  llvm::SmallVector<mlir::Value> finalHiddenStates;
  rewriter.setInsertionPoint(op);

  for (int64_t layer = 0; layer < layerCount; ++layer) {
    int64_t base = layer * operandsPerLayer;
    mlir::Value bIh;
    mlir::Value bHh;
    if (op.getHasBias()) {
      bIh = op.getRecurrentOperands()[base + 2];
      bHh = op.getRecurrentOperands()[base + 3];
    }
    auto layerOp = rewriter.create<NNGRULayerOp>(
        op.getLoc(), mlir::TypeRange{outputTy, hiddenSliceTy}, currentSequence,
        op.getH0(), op.getRecurrentOperands()[base],
        op.getRecurrentOperands()[base + 1], bIh, bHh,
        op.getBatchFirstAttr(), op.getHasBiasAttr(), op.getHiddenSizeAttr(),
        rewriter.getI64IntegerAttr(layer), op.getNumLayersAttr());
    currentSequence = layerOp.getOutput();
    finalHiddenStates.push_back(layerOp.getHn());
  }

  mlir::Value finalHidden = finalHiddenStates.front();
  if (finalHiddenStates.size() > 1)
    finalHidden =
        rewriter
            .create<ConcatOp>(op.getLoc(), hiddenTy, /*dim=*/0,
                              mlir::ValueRange(finalHiddenStates))
            .getResult();
  op.getOutput().replaceAllUsesWith(currentSequence);
  op.getHn().replaceAllUsesWith(finalHidden);
  rewriter.eraseOp(op);
  return mlir::success();
}

// Packs reset, update, input-new, and hidden-new projections.
static mlir::TypedAttr buildGRUFusedWeightAttr(GRULayerLowering &match) {
  auto maybeInputWeights =
      converter_constant::getF32ConstantValues(match.weightIHConstant);
  auto maybeHiddenWeights =
      converter_constant::getF32ConstantValues(match.weightHHConstant);
  if (mlir::failed(maybeInputWeights) || mlir::failed(maybeHiddenWeights))
    return {};

  llvm::SmallVector<float> inputWeights = *maybeInputWeights;
  llvm::SmallVector<float> hiddenWeights = *maybeHiddenWeights;
  if (static_cast<int64_t>(inputWeights.size()) !=
          match.weightIHTy.getNumElements() ||
      static_cast<int64_t>(hiddenWeights.size()) !=
          match.weightHHTy.getNumElements())
    return {};

  int64_t fusedWidth = match.inputSize + match.hiddenSize;
  llvm::SmallVector<float> fusedWeights(match.fusedWeightTy.getNumElements(),
                                        0.0f);

  for (int64_t row = 0; row < match.hiddenSize; ++row) {
    int64_t resetInputOffset = row * match.inputSize;
    int64_t resetHiddenOffset = row * match.hiddenSize;
    int64_t resetFusedOffset = row * fusedWidth;
    for (int64_t col = 0; col < match.inputSize; ++col)
      fusedWeights[resetFusedOffset + col] =
          inputWeights[resetInputOffset + col];
    for (int64_t col = 0; col < match.hiddenSize; ++col)
      fusedWeights[resetFusedOffset + match.inputSize + col] =
          hiddenWeights[resetHiddenOffset + col];

    int64_t updateInputOffset = (match.hiddenSize + row) * match.inputSize;
    int64_t updateHiddenOffset = (match.hiddenSize + row) * match.hiddenSize;
    int64_t updateFusedOffset = (match.hiddenSize + row) * fusedWidth;
    for (int64_t col = 0; col < match.inputSize; ++col)
      fusedWeights[updateFusedOffset + col] =
          inputWeights[updateInputOffset + col];
    for (int64_t col = 0; col < match.hiddenSize; ++col)
      fusedWeights[updateFusedOffset + match.inputSize + col] =
          hiddenWeights[updateHiddenOffset + col];

    int64_t inputNewOffset = (match.hiddenSize * 2 + row) * match.inputSize;
    int64_t inputNewFusedOffset = (match.hiddenSize * 2 + row) * fusedWidth;
    for (int64_t col = 0; col < match.inputSize; ++col)
      fusedWeights[inputNewFusedOffset + col] =
          inputWeights[inputNewOffset + col];

    int64_t hiddenNewOffset = (match.hiddenSize * 2 + row) * match.hiddenSize;
    int64_t hiddenNewFusedOffset =
        (match.hiddenSize * 3 + row) * fusedWidth + match.inputSize;
    for (int64_t col = 0; col < match.hiddenSize; ++col)
      fusedWeights[hiddenNewFusedOffset + col] =
          hiddenWeights[hiddenNewOffset + col];
  }

  bool useResource =
      converter_constant::isResourceBackedF32Constant(match.weightIHConstant) ||
      converter_constant::isResourceBackedF32Constant(match.weightHHConstant);
  return converter_constant::buildF32ElementsAttr(
      match.fusedWeightTy, fusedWeights, "analog_gru_layer_fused_weight_",
      useResource);
}

// Keeps GRU input-new and hidden-new bias terms separate.
static mlir::TypedAttr buildGRUFusedBiasAttr(GRULayerLowering &match) {
  auto maybeInputBias =
      converter_constant::getF32ConstantValues(match.biasIHConstant);
  auto maybeHiddenBias =
      converter_constant::getF32ConstantValues(match.biasHHConstant);
  if (mlir::failed(maybeInputBias) || mlir::failed(maybeHiddenBias))
    return {};

  llvm::SmallVector<float> inputBias = *maybeInputBias;
  llvm::SmallVector<float> hiddenBias = *maybeHiddenBias;
  int64_t gateRows = match.hiddenSize * 3;
  if (static_cast<int64_t>(inputBias.size()) != gateRows ||
      static_cast<int64_t>(hiddenBias.size()) != gateRows)
    return {};

  llvm::SmallVector<float> fusedBias(match.fusedBiasTy.getNumElements(), 0.0f);
  for (int64_t index = 0; index < match.hiddenSize; ++index) {
    fusedBias[index] = inputBias[index] + hiddenBias[index];
    fusedBias[match.hiddenSize + index] = inputBias[match.hiddenSize + index] +
                                          hiddenBias[match.hiddenSize + index];
    fusedBias[match.hiddenSize * 2 + index] =
        inputBias[match.hiddenSize * 2 + index];
    fusedBias[match.hiddenSize * 3 + index] =
        hiddenBias[match.hiddenSize * 2 + index];
  }

  bool useResource =
      converter_constant::isResourceBackedF32Constant(match.biasIHConstant) ||
      converter_constant::isResourceBackedF32Constant(match.biasHHConstant);
  return converter_constant::buildF32ElementsAttr(
      match.fusedBiasTy, fusedBias, "analog_gru_layer_fused_bias_",
      useResource);
}

// Blends the candidate and prior hidden state with the update gate.
static mlir::Value buildGRUNextHidden(GRULayerLowering &match,
                                      mlir::Value candidate,
                                      mlir::Value updateGate,
                                      mlir::Value previousHidden,
                                      mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::AffineMap hiddenMap =
      builder.getMultiDimIdentityMap(match.hidden2DTy.getRank());
  llvm::SmallVector<mlir::AffineMap, 4> indexingMaps = {hiddenMap, hiddenMap,
                                                        hiddenMap, hiddenMap};
  llvm::SmallVector<mlir::utils::IteratorType, 2> iteratorTypes(
      match.hidden2DTy.getRank(), mlir::utils::IteratorType::parallel);

  return builder
      .create<mlir::linalg::GenericOp>(
          loc, match.hidden2DTy,
          mlir::ValueRange{candidate, updateGate, previousHidden},
          mlir::ValueRange{previousHidden}, indexingMaps, iteratorTypes,
          [](mlir::OpBuilder &builder, mlir::Location nestedLoc,
             mlir::ValueRange args) {
            mlir::Value hiddenMinusCandidate =
                builder.create<mlir::arith::SubFOp>(nestedLoc, args[2],
                                                    args[0]);
            mlir::Value updateDelta = builder.create<mlir::arith::MulFOp>(
                nestedLoc, args[1], hiddenMinusCandidate);
            mlir::Value nextHidden = builder.create<mlir::arith::AddFOp>(
                nestedLoc, args[0], updateDelta);
            builder.create<mlir::linalg::YieldOp>(nestedLoc, nextHidden);
          })
      .getResult(0);
}

// Applies GRU gate order reset, update, input-new, hidden-new.
static mlir::Value buildGRUGateMath(GRULayerLowering &match,
                                    mlir::Value preActivation,
                                    mlir::Value previousHidden,
                                    mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::Value resetPre = recurrent_gate::extractBatchGate(
      loc, preActivation, /*gateOffset=*/0, match.batchSize, match.hiddenSize,
      match.hidden2DTy, builder);
  mlir::Value updatePre = recurrent_gate::extractBatchGate(
      loc, preActivation, match.hiddenSize, match.batchSize, match.hiddenSize,
      match.hidden2DTy, builder);
  mlir::Value inputNew = recurrent_gate::extractBatchGate(
      loc, preActivation, match.hiddenSize * 2, match.batchSize,
      match.hiddenSize, match.hidden2DTy, builder);
  mlir::Value hiddenNew = recurrent_gate::extractBatchGate(
      loc, preActivation, match.hiddenSize * 3, match.batchSize,
      match.hiddenSize, match.hidden2DTy, builder);

  mlir::Value resetGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.hidden2DTy, resetPre, builder);
  mlir::Value updateGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.hidden2DTy, updatePre, builder);
  mlir::Value resetHiddenNew = converter_recurrent_elementwise::buildMul(
      loc, match.hidden2DTy, resetGate, hiddenNew, builder);
  mlir::Value candidateInput = converter_recurrent_elementwise::buildAdd(
      loc, match.hidden2DTy, inputNew, resetHiddenNew, builder);
  mlir::Value candidate = converter_recurrent_elementwise::buildTanh(
      loc, match.hidden2DTy, candidateInput, builder);
  return buildGRUNextHidden(match, candidate, updateGate, previousHidden,
                            builder);
}

static mlir::Value buildInitialHiddenStage(GRULayerLowering &match,
                                            mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::Value h0 = match.gruLayerOp.getH0();
  mlir::sculptor::SemanticOperationScope scope(
      builder, "digital.hidden_extract", "gru_layer_initial_hidden_extract");
  mlir::Value initialHidden = converter_recurrent_layer::extractLayerState(
      loc, h0, match.layerIndex, match.batchSize, match.hiddenSize,
      match.hiddenSliceTy, match.hidden2DTy, builder);
  scope.annotate();
  return initialHidden;
}

static mlir::Value buildTimestepExtractStage(GRULayerLowering &match,
                                              int64_t timestep,
                                              mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::Value input = match.gruLayerOp.getInput();
  mlir::sculptor::SemanticOperationScope scope(
      builder, "digital.timestep_extract", "gru_layer_timestep_extract");
  mlir::Value timestepIndex =
      builder.create<mlir::arith::ConstantIndexOp>(loc, timestep);
  mlir::Value timestepInput =
      converter_recurrent_layer::extractBatchFirstTimestep(
          loc, input, timestepIndex, match.batchSize, match.inputSize,
          match.inputSliceTy, match.input2DTy, builder);
  scope.annotate();
  return timestepInput;
}

static mlir::Value buildInputRecombineStage(GRULayerLowering &match,
                                             mlir::Value timestepInput,
                                             mlir::Value recurrentHidden,
                                             mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::sculptor::SemanticOperationScope scope(
      builder, "digital.input_recombine", "gru_layer_input_recombine");
  mlir::Value fusedInput =
      builder
          .create<ConcatOp>(
              loc, match.rowFusedInputTy, /*dim=*/1,
              mlir::ValueRange{timestepInput, recurrentHidden})
          .getResult();
  scope.annotate();
  return fusedInput;
}

static mlir::Value buildBiasAddStage(GRULayerLowering &match,
                                      mlir::TypedAttr fusedBiasAttr,
                                      mlir::Value preActivation,
                                      mlir::OpBuilder &builder) {
  assert((!match.hasBias || fusedBiasAttr) &&
         "expected fused bias attr for biased GRU layer");
  mlir::Location loc = match.gruLayerOp.getLoc();

  mlir::Value biasResult = preActivation;
  if (match.hasBias) {
    mlir::sculptor::SemanticOperationScope scope(
        builder, "digital.bias_add", "gru_layer_bias_add");
    auto fusedBias =
        builder.create<ConstantOp>(loc, match.fusedBiasTy, fusedBiasAttr);
    mlir::Value expandedBias = converter_recurrent_layer::expandRowBias(
        loc, fusedBias.getResult(), match.rowPreActivationTy, builder);
    biasResult = converter_recurrent_layer::addBroadcastRowBias(
        loc, preActivation, expandedBias, match.rowPreActivationTy, builder);
    scope.annotate();
  }
  return biasResult;
}

struct GRUGateSlices {
  mlir::Value reset;
  mlir::Value update;
  mlir::Value inputNew;
  mlir::Value hiddenNew;
};

static GRUGateSlices buildGateSplitStage(GRULayerLowering &match,
                                          mlir::Value preActivation,
                                          mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::sculptor::SemanticOperationScope scope(
      builder, "digital.gate_split", "gru_layer_gate_split");
  mlir::Value resetPre = recurrent_gate::extractBatchGate(
      loc, preActivation, /*gateOffset=*/0, match.batchSize,
      match.hiddenSize, match.hidden2DTy, builder);
  mlir::Value updatePre = recurrent_gate::extractBatchGate(
      loc, preActivation, match.hiddenSize, match.batchSize,
      match.hiddenSize, match.hidden2DTy, builder);
  mlir::Value inputNew = recurrent_gate::extractBatchGate(
      loc, preActivation, match.hiddenSize * 2, match.batchSize,
      match.hiddenSize, match.hidden2DTy, builder);
  mlir::Value hiddenNew = recurrent_gate::extractBatchGate(
      loc, preActivation, match.hiddenSize * 3, match.batchSize,
      match.hiddenSize, match.hidden2DTy, builder);
  scope.annotate();
  return GRUGateSlices{resetPre, updatePre, inputNew, hiddenNew};
}

struct GRUGateActivations {
  mlir::Value reset;
  mlir::Value update;
};

static GRUGateActivations buildGateActivationStage(GRULayerLowering &match,
                                                    GRUGateSlices gates,
                                                    mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::sculptor::SemanticOperationScope scope(
      builder, "digital.activation", "gru_layer_gate_activation");
  mlir::Value resetGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.hidden2DTy, gates.reset, builder);
  mlir::Value updateGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.hidden2DTy, gates.update, builder);
  scope.annotate();
  return GRUGateActivations{resetGate, updateGate};
}

static mlir::Value buildCandidateUpdateStage(GRULayerLowering &match,
                                              mlir::Value resetGate,
                                              mlir::Value inputNew,
                                              mlir::Value hiddenNew,
                                              mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::sculptor::SemanticOperationScope scope(
      builder, "digital.candidate_update", "gru_layer_candidate_update");
  mlir::Value resetHiddenNew = converter_recurrent_elementwise::buildMul(
      loc, match.hidden2DTy, resetGate, hiddenNew, builder);
  mlir::Value candidateInput = converter_recurrent_elementwise::buildAdd(
      loc, match.hidden2DTy, inputNew, resetHiddenNew, builder);
  mlir::Value candidate = converter_recurrent_elementwise::buildTanh(
      loc, match.hidden2DTy, candidateInput, builder);
  scope.annotate();
  return candidate;
}

static mlir::Value buildHiddenUpdateStage(GRULayerLowering &match,
                                           mlir::Value candidate,
                                           mlir::Value updateGate,
                                           mlir::Value previousHidden,
                                           mlir::OpBuilder &builder) {
  mlir::sculptor::SemanticOperationScope scope(
      builder, "digital.hidden_update", "gru_layer_hidden_update");
  mlir::Value nextHidden = buildGRUNextHidden(
      match, candidate, updateGate, previousHidden, builder);
  scope.annotate();
  return nextHidden;
}

static mlir::Value buildOutputUpdateStage(GRULayerLowering &match,
                                           mlir::Value timestepHidden,
                                           mlir::Value sequenceOutput,
                                           int64_t timestep,
                                           mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::sculptor::SemanticOperationScope scope(
      builder, "digital.output_update", "gru_layer_output_update");
  mlir::Value timestepIndex =
      builder.create<mlir::arith::ConstantIndexOp>(loc, timestep);
  mlir::Value outputBase = timestepHidden;
  if (sequenceOutput) {
    outputBase = sequenceOutput;
  } else {
    outputBase = builder.create<EmptyOp>(loc, match.outputTy.getShape(),
                                         match.outputTy.getElementType());
  }
  mlir::Value nextOutput = converter_recurrent_layer::insertBatchFirstTimestep(
      loc, timestepHidden, outputBase, timestepIndex, match.batchSize,
      match.hiddenSize, match.timestepResultTy, builder);
  scope.annotate();
  return nextOutput;
}

static mlir::Value buildFinalHiddenStage(GRULayerLowering &match,
                                          mlir::Value finalHidden,
                                          mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::sculptor::SemanticOperationScope scope(
      builder, "digital.hidden_output", "gru_layer_hidden_output");
  mlir::Value hiddenOutput = converter_recurrent_layer::expandFinalLayerState(
      loc, finalHidden, match.hiddenResultTy, builder);
  scope.annotate();
  return hiddenOutput;
}

static GRUTimestepResult buildSectionedGRUTimestep(
    GRULayerLowering &match, int64_t timestep, mlir::Value recurrentHidden,
    mlir::Value sequenceOutput, mlir::Value fusedWeight,
    mlir::TypedAttr fusedBiasAttr, mlir::OpBuilder &builder) {
  assert(match.batchSize == 1 && "sectioned GRU lowering expects batch size 1");
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::Value timestepInput =
      buildTimestepExtractStage(match, timestep, builder);
  mlir::Value fusedInput =
      buildInputRecombineStage(match, timestepInput, recurrentHidden, builder);
  mlir::Value preActivation = mvm_build::buildMVM(
      loc, match.rowPreActivationTy, fusedInput, fusedWeight, builder);
  mlir::Value biasedPreActivation =
      buildBiasAddStage(match, fusedBiasAttr, preActivation, builder);
  GRUGateSlices slices =
      buildGateSplitStage(match, biasedPreActivation, builder);
  GRUGateActivations gates = buildGateActivationStage(match, slices, builder);
  mlir::Value candidate = buildCandidateUpdateStage(
      match, gates.reset, slices.inputNew, slices.hiddenNew, builder);
  mlir::Value nextHidden = buildHiddenUpdateStage(
      match, candidate, gates.update, recurrentHidden, builder);
  mlir::Value nextOutput = buildOutputUpdateStage(
      match, nextHidden, sequenceOutput, timestep, builder);
  return GRUTimestepResult{nextHidden, nextOutput};
}

static mlir::Value
buildGRUBatchPreActivation(GRULayerLowering &match, mlir::Value timestepInput,
                           mlir::Value recurrentHidden, mlir::Value fusedWeight,
                           mlir::Value expandedBias, mlir::OpBuilder &builder) {
  assert(match.batchSize > 0 && "expected positive batch size");
  llvm::SmallVector<mlir::Value> rowPreActivations;
  rowPreActivations.reserve(match.batchSize);
  mlir::Location loc = match.gruLayerOp.getLoc();

  for (int64_t row = 0; row < match.batchSize; ++row) {
    mlir::Value inputRow = converter_recurrent_layer::extractBatchRow(
        loc, timestepInput, match.rowInputTy, row, match.inputSize, builder);
    mlir::Value hiddenRow = converter_recurrent_layer::extractBatchRow(
        loc, recurrentHidden, match.rowHiddenTy, row, match.hiddenSize,
        builder);
    mlir::Value fusedInput =
        builder
            .create<ConcatOp>(loc, match.rowFusedInputTy, /*dim=*/1,
                              mlir::ValueRange{inputRow, hiddenRow})
            .getResult();
    mlir::Value mvmResult = mvm_build::buildMVM(
        loc, match.rowPreActivationTy, fusedInput, fusedWeight, builder);
    rowPreActivations.push_back(mvmResult);
  }

  mlir::Value preActivation;
  if (rowPreActivations.size() == 1) {
    preActivation = rowPreActivations.front();
  } else {
    preActivation = builder
                        .create<ConcatOp>(loc, match.preActivationTy, /*dim=*/0,
                                          mlir::ValueRange(rowPreActivations))
                        .getResult();
  }

  if (!match.hasBias)
    return preActivation;

  return converter_recurrent_layer::addBroadcastRowBias(
      loc, preActivation, expandedBias, match.preActivationTy, builder);
}

static mlir::LogicalResult lowerGRULayerOp(NNGRULayerOp op,
                                           mlir::RewriterBase &rewriter) {
  auto match = matchGRULayer(op);
  if (mlir::failed(match))
    return mlir::failure();

  mlir::TypedAttr fusedWeightAttr = buildGRUFusedWeightAttr(*match);
  if (!fusedWeightAttr)
    return mlir::failure();

  mlir::TypedAttr fusedBiasAttr;
  if (match->hasBias) {
    fusedBiasAttr = buildGRUFusedBiasAttr(*match);
    if (!fusedBiasAttr)
      return mlir::failure();
  }

  mlir::Location loc = match->gruLayerOp.getLoc();
  rewriter.setInsertionPoint(match->gruLayerOp);
  auto fusedWeightConstant =
      rewriter.create<ConstantOp>(loc, match->fusedWeightTy, fusedWeightAttr);

  ConstantOp fusedBiasConstant;
  mlir::Value expandedBias;
  if (match->hasBias && match->batchSize != 1) {
    fusedBiasConstant =
        rewriter.create<ConstantOp>(loc, match->fusedBiasTy, fusedBiasAttr);
    expandedBias = converter_recurrent_layer::expandRowBias(
        loc, fusedBiasConstant.getResult(), match->rowPreActivationTy,
        rewriter);
  }

  mlir::Value currentHidden;
  mlir::Value sequenceOutputInit;
  if (match->batchSize == 1) {
    currentHidden = buildInitialHiddenStage(*match, rewriter);
  } else {
    currentHidden = converter_recurrent_layer::extractLayerState(
        loc, match->gruLayerOp.getH0(), match->layerIndex, match->batchSize,
        match->hiddenSize, match->hiddenSliceTy, match->hidden2DTy, rewriter);

    sequenceOutputInit = rewriter.create<EmptyOp>(
        loc, match->outputTy.getShape(), match->outputTy.getElementType());
  }

  mlir::Value sequenceOutput = sequenceOutputInit;
  mlir::Value finalHidden = currentHidden;
  if (match->batchSize == 1) {
    for (int64_t step = 0; step < match->sequenceLength; ++step) {
      GRUTimestepResult timestepResult = buildSectionedGRUTimestep(
          *match, step, finalHidden, sequenceOutput,
          fusedWeightConstant.getResult(), fusedBiasAttr, rewriter);
      finalHidden = timestepResult.hidden;
      sequenceOutput = timestepResult.output;
    }
  } else {
    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value cSequenceLength = rewriter.create<mlir::arith::ConstantIndexOp>(
        loc, match->sequenceLength);

    llvm::SmallVector<mlir::Value, 2> initArgs = {currentHidden,
                                                  sequenceOutputInit};
    auto timestepLoop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, cSequenceLength, c1, initArgs,
        [&](mlir::OpBuilder &builder, mlir::Location loopLoc,
            mlir::Value timestep, mlir::ValueRange iterArgs) {
          mlir::Value loopHidden = iterArgs[0];
          mlir::Value loopOutput = iterArgs[1];
          mlir::Value timestepInput =
              converter_recurrent_layer::extractBatchFirstTimestep(
                  match->gruLayerOp.getLoc(), match->gruLayerOp.getInput(),
                  timestep, match->batchSize, match->inputSize,
                  match->inputSliceTy, match->input2DTy, builder);
          mlir::Value preActivation = buildGRUBatchPreActivation(
              *match, timestepInput, loopHidden,
              fusedWeightConstant.getResult(), expandedBias, builder);
          mlir::Value timestepHidden =
              buildGRUGateMath(*match, preActivation, loopHidden, builder);
          mlir::Value nextOutput =
              converter_recurrent_layer::insertBatchFirstTimestep(
                  match->gruLayerOp.getLoc(), timestepHidden, loopOutput,
                  timestep, match->batchSize, match->hiddenSize,
                  match->timestepResultTy, builder);
          builder.create<mlir::scf::YieldOp>(
              loopLoc, mlir::ValueRange{timestepHidden, nextOutput});
        });

    rewriter.setInsertionPointAfter(timestepLoop);
    finalHidden = timestepLoop.getResult(0);
    sequenceOutput = timestepLoop.getResult(1);
  }

  mlir::Value hiddenOutput =
      match->batchSize == 1
          ? buildFinalHiddenStage(*match, finalHidden, rewriter)
          : converter_recurrent_layer::expandFinalLayerState(
                match->gruLayerOp.getLoc(), finalHidden, match->hiddenResultTy,
                rewriter);

  match->gruLayerOp.getOutput().replaceAllUsesWith(sequenceOutput);
  match->gruLayerOp.getHn().replaceAllUsesWith(hiddenOutput);
  rewriter.eraseOp(match->gruLayerOp);
  converter_recurrent_layer::eraseUnusedConstants(
      {match->weightIHConstant, match->weightHHConstant, match->biasIHConstant,
       match->biasHHConstant},
      rewriter);
  return mlir::success();
}

static mlir::LogicalResult lowerGRULayerToMVM(mlir::func::FuncOp func,
                                              mlir::RewriterBase &rewriter) {
  auto match = matchExtractedGRULayer(func);
  if (mlir::failed(match))
    return mlir::failure();
  return lowerGRULayerOp(match->gruLayerOp, rewriter);
}

// Converts extracted sculptor.nn.gru_layer bodies to sculptor.mvm timestep loops.
class GRUConverter : public mlir::sculptor::LayerToMVMConverter {
public:
  mlir::StringRef getName() const override { return "gru"; }

  void lowerToMVM(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());
    (void)lowerGRULayerToMVM(func, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

LogicalResult decomposeInlineGRULayers(func::FuncOp func) {
  SmallVector<NNGRUOp> stacks;
  func.walk([&](NNGRUOp op) { stacks.push_back(op); });

  SemanticLayerRewriteListener layerListener;
  IRRewriter rewriter(func.getContext(), &layerListener);
  for (NNGRUOp op : stacks) {
    SemanticLayerRewriteScope layerScope(layerListener, op);
    if (failed(splitGRUStack(op, rewriter))) {
      op.emitOpError("cannot split supported inline GRU stack");
      return failure();
    }
  }

  SmallVector<NNGRULayerOp> layers;
  func.walk([&](NNGRULayerOp op) { layers.push_back(op); });
  for (NNGRULayerOp op : layers) {
    SemanticLayerRewriteScope layerScope(layerListener, op);
    if (failed(lowerGRULayerOp(op, rewriter))) {
      op.emitOpError("cannot decompose supported inline GRU layer");
      return failure();
    }
  }
  return success();
}

void registerGRUConverter(LayerToMVMConverters &converters,
                          LayerToMVMConverterMap &converterMap,
                          MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<GRUConverter>();
  const LayerToMVMConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["gru"] = converterPtr;
  converterMap["gru_w_bias"] = converterPtr;
}

} // namespace sculptor
} // namespace mlir
