#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConstantUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/MVMBuildUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/NNLayerMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/RecurrentElementwiseUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/RecurrentGateUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/RecurrentLayerConversionUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/TensorTypeUtils.h"

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

static mlir::Block *addTaskRegionBody(mlir::sculptor::TaskRegionOp region,
                                      mlir::ValueRange inputs) {
  mlir::Block *body = new mlir::Block();
  region.getBody().push_back(body);
  for (mlir::Value input : inputs)
    body->addArgument(input.getType(), input.getLoc());
  return body;
}

static mlir::FailureOr<GRULayerLowering>
matchExtractedGRULayer(mlir::func::FuncOp func) {
  auto gruLayerOp = nn_layer_match::matchSingleNNLayerOp<NNGRULayerOp>(func);
  if (mlir::failed(gruLayerOp))
    return mlir::failure();

  bool hasBias = (*gruLayerOp).getHasBias();
  if (!nn_layer_match::hasLayerTypeMatchingBias(func, "gru", "gru_w_bias",
                                                hasBias))
    return mlir::failure();

  if (!(*gruLayerOp).getBatchFirst())
    return mlir::failure();

  if (func.getNumArguments() != 2 || func.getNumResults() != 2 ||
      (*gruLayerOp).getInput() != func.getArgument(0) ||
      (*gruLayerOp).getH0() != func.getArgument(1))
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

static mlir::Value buildInitialHiddenRegion(GRULayerLowering &match,
                                            mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::Value h0 = match.gruLayerOp.getH0();
  auto hiddenRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.hidden2DTy}, mlir::ValueRange{h0},
      "digital.hidden_extract",
      builder.getStringAttr("gru_layer_initial_hidden_extract"));

  mlir::Block *body = addTaskRegionBody(hiddenRegion, mlir::ValueRange{h0});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value initialHidden = converter_recurrent_layer::extractLayerState(
      loc, body->getArgument(0), match.layerIndex, match.batchSize,
      match.hiddenSize, match.hiddenSliceTy, match.hidden2DTy, builder);
  builder.create<mlir::sculptor::YieldOp>(loc, initialHidden);
  return hiddenRegion.getResult(0);
}

static mlir::Value buildTimestepExtractRegion(GRULayerLowering &match,
                                              int64_t timestep,
                                              mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::Value input = match.gruLayerOp.getInput();
  auto extractRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.input2DTy}, mlir::ValueRange{input},
      "digital.timestep_extract",
      builder.getStringAttr("gru_layer_timestep_extract"));

  mlir::Block *body = addTaskRegionBody(extractRegion, mlir::ValueRange{input});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value timestepIndex =
      builder.create<mlir::arith::ConstantIndexOp>(loc, timestep);
  mlir::Value timestepInput =
      converter_recurrent_layer::extractBatchFirstTimestep(
          loc, body->getArgument(0), timestepIndex, match.batchSize,
          match.inputSize, match.inputSliceTy, match.input2DTy, builder);
  builder.create<mlir::sculptor::YieldOp>(loc, timestepInput);
  return extractRegion.getResult(0);
}

static mlir::Value buildInputRecombineRegion(GRULayerLowering &match,
                                             mlir::Value timestepInput,
                                             mlir::Value recurrentHidden,
                                             mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {timestepInput, recurrentHidden};
  auto recombineRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.rowFusedInputTy}, mlir::ValueRange(inputs),
      "digital.input_recombine",
      builder.getStringAttr("gru_layer_input_recombine"));

  mlir::Block *body = addTaskRegionBody(recombineRegion, inputs);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value fusedInput =
      builder
          .create<ConcatOp>(
              loc, match.rowFusedInputTy, /*dim=*/1,
              mlir::ValueRange{body->getArgument(0), body->getArgument(1)})
          .getResult();
  builder.create<mlir::sculptor::YieldOp>(loc, fusedInput);
  return recombineRegion.getResult(0);
}

static mlir::Value buildBiasAddRegion(GRULayerLowering &match,
                                      mlir::TypedAttr fusedBiasAttr,
                                      mlir::Value preActivation,
                                      mlir::OpBuilder &builder) {
  assert((!match.hasBias || fusedBiasAttr) &&
         "expected fused bias attr for biased GRU layer");
  mlir::Location loc = match.gruLayerOp.getLoc();

  auto biasRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.rowPreActivationTy},
      mlir::ValueRange{preActivation}, "digital.bias_add",
      builder.getStringAttr("gru_layer_bias_add"));

  mlir::Block *body =
      addTaskRegionBody(biasRegion, mlir::ValueRange{preActivation});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value biasResult = body->getArgument(0);
  if (match.hasBias) {
    auto fusedBias =
        builder.create<ConstantOp>(loc, match.fusedBiasTy, fusedBiasAttr);
    mlir::Value expandedBias = converter_recurrent_layer::expandRowBias(
        loc, fusedBias.getResult(), match.rowPreActivationTy, builder);
    biasResult = converter_recurrent_layer::addBroadcastRowBias(
        loc, body->getArgument(0), expandedBias, match.rowPreActivationTy,
        builder);
  }
  builder.create<mlir::sculptor::YieldOp>(loc, biasResult);
  return biasRegion.getResult(0);
}

struct GRUGateSlices {
  mlir::Value reset;
  mlir::Value update;
  mlir::Value inputNew;
  mlir::Value hiddenNew;
};

static GRUGateSlices buildGateSplitRegion(GRULayerLowering &match,
                                          mlir::Value preActivation,
                                          mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  auto gateSplitRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc,
      mlir::TypeRange{match.hidden2DTy, match.hidden2DTy, match.hidden2DTy,
                      match.hidden2DTy},
      mlir::ValueRange{preActivation}, "digital.gate_split",
      builder.getStringAttr("gru_layer_gate_split"));

  mlir::Block *body =
      addTaskRegionBody(gateSplitRegion, mlir::ValueRange{preActivation});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value regionPreActivation = body->getArgument(0);
  mlir::Value resetPre = recurrent_gate::extractBatchGate(
      loc, regionPreActivation, /*gateOffset=*/0, match.batchSize,
      match.hiddenSize, match.hidden2DTy, builder);
  mlir::Value updatePre = recurrent_gate::extractBatchGate(
      loc, regionPreActivation, match.hiddenSize, match.batchSize,
      match.hiddenSize, match.hidden2DTy, builder);
  mlir::Value inputNew = recurrent_gate::extractBatchGate(
      loc, regionPreActivation, match.hiddenSize * 2, match.batchSize,
      match.hiddenSize, match.hidden2DTy, builder);
  mlir::Value hiddenNew = recurrent_gate::extractBatchGate(
      loc, regionPreActivation, match.hiddenSize * 3, match.batchSize,
      match.hiddenSize, match.hidden2DTy, builder);

  builder.create<mlir::sculptor::YieldOp>(
      loc, mlir::ValueRange{resetPre, updatePre, inputNew, hiddenNew});
  return GRUGateSlices{
      gateSplitRegion.getResult(0), gateSplitRegion.getResult(1),
      gateSplitRegion.getResult(2), gateSplitRegion.getResult(3)};
}

struct GRUGateActivations {
  mlir::Value reset;
  mlir::Value update;
};

static GRUGateActivations buildGateActivationRegion(GRULayerLowering &match,
                                                    GRUGateSlices gates,
                                                    mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {gates.reset, gates.update};
  auto activationRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.hidden2DTy, match.hidden2DTy},
      mlir::ValueRange(inputs), "digital.activation",
      builder.getStringAttr("gru_layer_gate_activation"));

  mlir::Block *body = addTaskRegionBody(activationRegion, inputs);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value resetGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.hidden2DTy, body->getArgument(0), builder);
  mlir::Value updateGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.hidden2DTy, body->getArgument(1), builder);

  builder.create<mlir::sculptor::YieldOp>(
      loc, mlir::ValueRange{resetGate, updateGate});
  return GRUGateActivations{activationRegion.getResult(0),
                            activationRegion.getResult(1)};
}

static mlir::Value buildCandidateUpdateRegion(GRULayerLowering &match,
                                              mlir::Value resetGate,
                                              mlir::Value inputNew,
                                              mlir::Value hiddenNew,
                                              mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {resetGate, inputNew, hiddenNew};
  auto candidateRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.hidden2DTy}, mlir::ValueRange(inputs),
      "digital.candidate_update",
      builder.getStringAttr("gru_layer_candidate_update"));

  mlir::Block *body = addTaskRegionBody(candidateRegion, inputs);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value resetHiddenNew = converter_recurrent_elementwise::buildMul(
      loc, match.hidden2DTy, body->getArgument(0), body->getArgument(2),
      builder);
  mlir::Value candidateInput = converter_recurrent_elementwise::buildAdd(
      loc, match.hidden2DTy, body->getArgument(1), resetHiddenNew, builder);
  mlir::Value candidate = converter_recurrent_elementwise::buildTanh(
      loc, match.hidden2DTy, candidateInput, builder);
  builder.create<mlir::sculptor::YieldOp>(loc, candidate);
  return candidateRegion.getResult(0);
}

static mlir::Value buildHiddenUpdateRegion(GRULayerLowering &match,
                                           mlir::Value candidate,
                                           mlir::Value updateGate,
                                           mlir::Value previousHidden,
                                           mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {candidate, updateGate,
                                           previousHidden};
  auto hiddenUpdateRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.hidden2DTy}, mlir::ValueRange(inputs),
      "digital.hidden_update",
      builder.getStringAttr("gru_layer_hidden_update"));

  mlir::Block *body = addTaskRegionBody(hiddenUpdateRegion, inputs);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value nextHidden =
      buildGRUNextHidden(match, body->getArgument(0), body->getArgument(1),
                         body->getArgument(2), builder);
  builder.create<mlir::sculptor::YieldOp>(loc, nextHidden);
  return hiddenUpdateRegion.getResult(0);
}

static mlir::Value buildOutputUpdateRegion(GRULayerLowering &match,
                                           mlir::Value timestepHidden,
                                           mlir::Value sequenceOutput,
                                           int64_t timestep,
                                           mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {timestepHidden};
  if (sequenceOutput)
    inputs.push_back(sequenceOutput);
  auto outputRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.outputTy}, mlir::ValueRange(inputs),
      "digital.output_update",
      builder.getStringAttr("gru_layer_output_update"));

  mlir::Block *body = addTaskRegionBody(outputRegion, inputs);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value timestepIndex =
      builder.create<mlir::arith::ConstantIndexOp>(loc, timestep);
  mlir::Value outputBase = body->getArgument(0);
  if (sequenceOutput) {
    outputBase = body->getArgument(1);
  } else {
    outputBase = builder.create<EmptyOp>(loc, match.outputTy.getShape(),
                                         match.outputTy.getElementType());
  }
  mlir::Value nextOutput = converter_recurrent_layer::insertBatchFirstTimestep(
      loc, body->getArgument(0), outputBase, timestepIndex, match.batchSize,
      match.hiddenSize, match.timestepResultTy, builder);
  builder.create<mlir::sculptor::YieldOp>(loc, nextOutput);
  return outputRegion.getResult(0);
}

static mlir::Value buildFinalHiddenRegion(GRULayerLowering &match,
                                          mlir::Value finalHidden,
                                          mlir::OpBuilder &builder) {
  mlir::Location loc = match.gruLayerOp.getLoc();
  auto hiddenRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.hiddenResultTy}, mlir::ValueRange{finalHidden},
      "digital.hidden_output",
      builder.getStringAttr("gru_layer_hidden_output"));

  mlir::Block *body =
      addTaskRegionBody(hiddenRegion, mlir::ValueRange{finalHidden});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value hiddenOutput = converter_recurrent_layer::expandFinalLayerState(
      loc, body->getArgument(0), match.hiddenResultTy, builder);
  builder.create<mlir::sculptor::YieldOp>(loc, hiddenOutput);
  return hiddenRegion.getResult(0);
}

static GRUTimestepResult buildSectionedGRUTimestep(
    GRULayerLowering &match, int64_t timestep, mlir::Value recurrentHidden,
    mlir::Value sequenceOutput, mlir::Value fusedWeight,
    mlir::TypedAttr fusedBiasAttr, mlir::OpBuilder &builder) {
  assert(match.batchSize == 1 && "sectioned GRU lowering expects batch size 1");
  mlir::Location loc = match.gruLayerOp.getLoc();
  mlir::Value timestepInput =
      buildTimestepExtractRegion(match, timestep, builder);
  mlir::Value fusedInput =
      buildInputRecombineRegion(match, timestepInput, recurrentHidden, builder);
  mlir::Value preActivation = mvm_build::buildMVM(
      loc, match.rowPreActivationTy, fusedInput, fusedWeight, builder);
  mlir::Value biasedPreActivation =
      buildBiasAddRegion(match, fusedBiasAttr, preActivation, builder);
  GRUGateSlices slices =
      buildGateSplitRegion(match, biasedPreActivation, builder);
  GRUGateActivations gates = buildGateActivationRegion(match, slices, builder);
  mlir::Value candidate = buildCandidateUpdateRegion(
      match, gates.reset, slices.inputNew, slices.hiddenNew, builder);
  mlir::Value nextHidden = buildHiddenUpdateRegion(
      match, candidate, gates.update, recurrentHidden, builder);
  mlir::Value nextOutput = buildOutputUpdateRegion(
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

static mlir::LogicalResult lowerGRULayerToMVM(mlir::func::FuncOp func,
                                              mlir::RewriterBase &rewriter) {
  auto match = matchExtractedGRULayer(func);
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
    currentHidden = buildInitialHiddenRegion(*match, rewriter);
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
          ? buildFinalHiddenRegion(*match, finalHidden, rewriter)
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
