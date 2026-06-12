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

using mlir::sculptor::NNLSTMLayerOp;
using mlir::arith::ConstantOp;
using mlir::tensor::ConcatOp;
using mlir::tensor::EmptyOp;

struct LSTMLayerLowering {
  NNLSTMLayerOp lstmLayerOp;
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType hiddenStateTy;
  mlir::RankedTensorType cellStateTy;
  mlir::RankedTensorType outputTy;
  mlir::RankedTensorType hiddenResultTy;
  mlir::RankedTensorType cellResultTy;
  mlir::RankedTensorType weightIHTy;
  mlir::RankedTensorType weightHHTy;
  mlir::RankedTensorType state2DTy;
  mlir::RankedTensorType stateSliceTy;
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

struct LSTMLayerStepResults {
  mlir::Value hidden;
  mlir::Value cell;
};

struct LSTMLayerTimestepResults {
  mlir::Value hidden;
  mlir::Value cell;
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

static mlir::FailureOr<LSTMLayerLowering>
matchExtractedLSTMLayer(mlir::func::FuncOp func) {
  auto lstmLayerOp = nn_layer_match::matchSingleNNLayerOp<NNLSTMLayerOp>(func);
  if (mlir::failed(lstmLayerOp))
    return mlir::failure();

  bool hasBias = (*lstmLayerOp).getHasBias();
  if (!nn_layer_match::hasLayerTypeMatchingBias(func, "lstm", "lstm_w_bias",
                                                hasBias))
    return mlir::failure();

  if (!(*lstmLayerOp).getBatchFirst())
    return mlir::failure();

  if (func.getNumArguments() != 3 || func.getNumResults() != 3 ||
      (*lstmLayerOp).getInput() != func.getArgument(0) ||
      (*lstmLayerOp).getH0() != func.getArgument(1) ||
      (*lstmLayerOp).getC0() != func.getArgument(2))
    return mlir::failure();

  auto inputTy = tensor_type::getStaticF32Tensor(
      (*lstmLayerOp).getInput().getType(), /*expectedRank=*/3);
  auto hiddenStateTy = tensor_type::getStaticF32Tensor(
      (*lstmLayerOp).getH0().getType(), /*expectedRank=*/3);
  auto cellStateTy = tensor_type::getStaticF32Tensor(
      (*lstmLayerOp).getC0().getType(), /*expectedRank=*/3);
  auto outputTy = tensor_type::getStaticF32Tensor(
      (*lstmLayerOp).getOutput().getType(), /*expectedRank=*/3);
  auto hiddenResultTy = tensor_type::getStaticF32Tensor(
      (*lstmLayerOp).getHn().getType(), /*expectedRank=*/3);
  auto cellResultTy = tensor_type::getStaticF32Tensor(
      (*lstmLayerOp).getCn().getType(), /*expectedRank=*/3);
  auto weightIHTy = tensor_type::getStaticF32Tensor(
      (*lstmLayerOp).getWIh().getType(), /*expectedRank=*/2);
  auto weightHHTy = tensor_type::getStaticF32Tensor(
      (*lstmLayerOp).getWHh().getType(), /*expectedRank=*/2);
  if (mlir::failed(inputTy) || mlir::failed(hiddenStateTy) ||
      mlir::failed(cellStateTy) || mlir::failed(outputTy) ||
      mlir::failed(hiddenResultTy) || mlir::failed(cellResultTy) ||
      mlir::failed(weightIHTy) || mlir::failed(weightHHTy))
    return mlir::failure();

  int64_t layerIndex = (*lstmLayerOp).getLayerIndex();
  int64_t numLayers = (*lstmLayerOp).getNumLayers();
  int64_t hiddenSize = (*lstmLayerOp).getHiddenSize();
  int64_t batchSize = inputTy->getShape()[0];
  int64_t sequenceLength = inputTy->getShape()[1];
  int64_t inputSize = inputTy->getShape()[2];
  if (layerIndex < 0 || numLayers < 1 || layerIndex >= numLayers ||
      batchSize < 1 || sequenceLength < 1 || inputSize < 1 || hiddenSize < 1)
    return mlir::failure();

  if (hiddenStateTy->getShape() !=
          llvm::ArrayRef<int64_t>({numLayers, batchSize, hiddenSize}) ||
      cellStateTy->getShape() != hiddenStateTy->getShape() ||
      outputTy->getShape() !=
          llvm::ArrayRef<int64_t>({batchSize, sequenceLength, hiddenSize}) ||
      hiddenResultTy->getShape() !=
          llvm::ArrayRef<int64_t>({1, batchSize, hiddenSize}) ||
      cellResultTy->getShape() != hiddenResultTy->getShape() ||
      weightIHTy->getShape() !=
          llvm::ArrayRef<int64_t>({hiddenSize * 4, inputSize}) ||
      weightHHTy->getShape() !=
          llvm::ArrayRef<int64_t>({hiddenSize * 4, hiddenSize}))
    return mlir::failure();

  auto weightIHConstant = (*lstmLayerOp).getWIh().getDefiningOp<ConstantOp>();
  auto weightHHConstant = (*lstmLayerOp).getWHh().getDefiningOp<ConstantOp>();
  if (!weightIHConstant || !weightHHConstant)
    return mlir::failure();

  ConstantOp biasIHConstant;
  ConstantOp biasHHConstant;
  mlir::RankedTensorType fusedBiasTy;
  if (hasBias) {
    mlir::Value biasIH = (*lstmLayerOp).getBIh();
    mlir::Value biasHH = (*lstmLayerOp).getBHh();
    if (!biasIH || !biasHH)
      return mlir::failure();

    auto biasIHTy = tensor_type::getStaticF32Tensor(biasIH.getType(),
                                                    /*expectedRank=*/1);
    auto biasHHTy = tensor_type::getStaticF32Tensor(biasHH.getType(),
                                                    /*expectedRank=*/1);
    if (mlir::failed(biasIHTy) || mlir::failed(biasHHTy) ||
        biasIHTy->getShape() != llvm::ArrayRef<int64_t>({hiddenSize * 4}) ||
        biasHHTy->getShape() != llvm::ArrayRef<int64_t>({hiddenSize * 4}))
      return mlir::failure();

    biasIHConstant = biasIH.getDefiningOp<ConstantOp>();
    biasHHConstant = biasHH.getDefiningOp<ConstantOp>();
    if (!biasIHConstant || !biasHHConstant)
      return mlir::failure();

    fusedBiasTy = mlir::RankedTensorType::get({hiddenSize * 4},
                                              inputTy->getElementType());
  } else if ((*lstmLayerOp).getBIh() || (*lstmLayerOp).getBHh()) {
    return mlir::failure();
  }

  mlir::Type elementType = inputTy->getElementType();
  LSTMLayerLowering lowering;
  lowering.lstmLayerOp = *lstmLayerOp;
  lowering.inputTy = *inputTy;
  lowering.hiddenStateTy = *hiddenStateTy;
  lowering.cellStateTy = *cellStateTy;
  lowering.outputTy = *outputTy;
  lowering.hiddenResultTy = *hiddenResultTy;
  lowering.cellResultTy = *cellResultTy;
  lowering.weightIHTy = *weightIHTy;
  lowering.weightHHTy = *weightHHTy;
  lowering.state2DTy =
      mlir::RankedTensorType::get({batchSize, hiddenSize}, elementType);
  lowering.stateSliceTy =
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

static mlir::TypedAttr buildLSTMFusedWeightAttr(LSTMLayerLowering &match) {
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

  int64_t outputSize = match.hiddenSize * 4;
  int64_t fusedWidth = match.inputSize + match.hiddenSize;
  llvm::SmallVector<float> fusedWeights(match.fusedWeightTy.getNumElements(),
                                        0.0f);
  for (int64_t row = 0; row < outputSize; ++row) {
    int64_t inputOffset = row * match.inputSize;
    int64_t hiddenOffset = row * match.hiddenSize;
    int64_t fusedOffset = row * fusedWidth;
    for (int64_t col = 0; col < match.inputSize; ++col)
      fusedWeights[fusedOffset + col] = inputWeights[inputOffset + col];
    for (int64_t col = 0; col < match.hiddenSize; ++col)
      fusedWeights[fusedOffset + match.inputSize + col] =
          hiddenWeights[hiddenOffset + col];
  }

  bool useResource =
      converter_constant::isResourceBackedF32Constant(match.weightIHConstant) ||
      converter_constant::isResourceBackedF32Constant(match.weightHHConstant);
  return converter_constant::buildF32ElementsAttr(
      match.fusedWeightTy, fusedWeights, "analog_lstm_layer_fused_weight_",
      useResource);
}

static mlir::TypedAttr buildLSTMFusedBiasAttr(LSTMLayerLowering &match) {
  auto maybeInputBias =
      converter_constant::getF32ConstantValues(match.biasIHConstant);
  auto maybeHiddenBias =
      converter_constant::getF32ConstantValues(match.biasHHConstant);
  if (mlir::failed(maybeInputBias) || mlir::failed(maybeHiddenBias))
    return {};

  llvm::SmallVector<float> inputBias = *maybeInputBias;
  llvm::SmallVector<float> hiddenBias = *maybeHiddenBias;
  if (static_cast<int64_t>(inputBias.size()) !=
          match.fusedBiasTy.getNumElements() ||
      static_cast<int64_t>(hiddenBias.size()) !=
          match.fusedBiasTy.getNumElements())
    return {};

  llvm::SmallVector<float> fusedBias(match.fusedBiasTy.getNumElements(), 0.0f);
  for (int64_t index = 0, end = match.fusedBiasTy.getNumElements(); index < end;
       ++index)
    fusedBias[index] = inputBias[index] + hiddenBias[index];

  bool useResource =
      converter_constant::isResourceBackedF32Constant(match.biasIHConstant) ||
      converter_constant::isResourceBackedF32Constant(match.biasHHConstant);
  return converter_constant::buildF32ElementsAttr(
      match.fusedBiasTy, fusedBias, "analog_lstm_layer_fused_bias_",
      useResource);
}

// Computes the next LSTM cell state from forget and input gates.
static mlir::Value
buildLSTMCellState(LSTMLayerLowering &match, mlir::Value inputGate,
                   mlir::Value forgetGate, mlir::Value candidateGate,
                   mlir::Value previousCell, mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  mlir::AffineMap stateMap =
      builder.getMultiDimIdentityMap(match.state2DTy.getRank());
  llvm::SmallVector<mlir::AffineMap, 5> indexingMaps = {
      stateMap, stateMap, stateMap, stateMap, stateMap};
  llvm::SmallVector<mlir::utils::IteratorType, 2> iteratorTypes(
      match.state2DTy.getRank(), mlir::utils::IteratorType::parallel);

  return builder
      .create<mlir::linalg::GenericOp>(
          loc, match.state2DTy,
          mlir::ValueRange{forgetGate, previousCell, inputGate, candidateGate},
          mlir::ValueRange{previousCell}, indexingMaps, iteratorTypes,
          [](mlir::OpBuilder &builder, mlir::Location nestedLoc,
             mlir::ValueRange args) {
            mlir::Value forgetCell = builder.create<mlir::arith::MulFOp>(
                nestedLoc, args[0], args[1]);
            mlir::Value inputCandidate = builder.create<mlir::arith::MulFOp>(
                nestedLoc, args[2], args[3]);
            mlir::Value nextCell = builder.create<mlir::arith::AddFOp>(
                nestedLoc, forgetCell, inputCandidate);
            builder.create<mlir::linalg::YieldOp>(nestedLoc, nextCell);
          })
      .getResult(0);
}

// Computes the next LSTM hidden state from the output gate.
static mlir::Value buildLSTMHiddenState(LSTMLayerLowering &match,
                                        mlir::Value outputGate,
                                        mlir::Value cellState,
                                        mlir::Value previousHidden,
                                        mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  mlir::AffineMap stateMap =
      builder.getMultiDimIdentityMap(match.state2DTy.getRank());
  llvm::SmallVector<mlir::AffineMap, 3> indexingMaps = {stateMap, stateMap,
                                                        stateMap};
  llvm::SmallVector<mlir::utils::IteratorType, 2> iteratorTypes(
      match.state2DTy.getRank(), mlir::utils::IteratorType::parallel);

  return builder
      .create<mlir::linalg::GenericOp>(
          loc, match.state2DTy, mlir::ValueRange{outputGate, cellState},
          mlir::ValueRange{previousHidden}, indexingMaps, iteratorTypes,
          [](mlir::OpBuilder &builder, mlir::Location nestedLoc,
             mlir::ValueRange args) {
            mlir::Value tanhCell =
                builder.create<mlir::math::TanhOp>(nestedLoc, args[1]);
            mlir::Value nextHidden = builder.create<mlir::arith::MulFOp>(
                nestedLoc, args[0], tanhCell);
            builder.create<mlir::linalg::YieldOp>(nestedLoc, nextHidden);
          })
      .getResult(0);
}

// Applies LSTM gate order i, f, g, o after the fused MVM.
static LSTMLayerStepResults buildLSTMGateMath(LSTMLayerLowering &match,
                                              mlir::Value preActivation,
                                              mlir::Value previousHidden,
                                              mlir::Value previousCell,
                                              mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  mlir::Value iSlice = recurrent_gate::extractBatchGate(
      loc, preActivation, /*gateOffset=*/0, match.batchSize, match.hiddenSize,
      match.state2DTy, builder);
  mlir::Value fSlice = recurrent_gate::extractBatchGate(
      loc, preActivation, match.hiddenSize, match.batchSize, match.hiddenSize,
      match.state2DTy, builder);
  mlir::Value gSlice = recurrent_gate::extractBatchGate(
      loc, preActivation, match.hiddenSize * 2, match.batchSize,
      match.hiddenSize, match.state2DTy, builder);
  mlir::Value oSlice = recurrent_gate::extractBatchGate(
      loc, preActivation, match.hiddenSize * 3, match.batchSize,
      match.hiddenSize, match.state2DTy, builder);

  mlir::Value inputGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.state2DTy, iSlice, builder);
  mlir::Value forgetGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.state2DTy, fSlice, builder);
  mlir::Value candidateGate = converter_recurrent_elementwise::buildTanh(
      loc, match.state2DTy, gSlice, builder);
  mlir::Value outputGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.state2DTy, oSlice, builder);

  mlir::Value nextCell = buildLSTMCellState(
      match, inputGate, forgetGate, candidateGate, previousCell, builder);
  mlir::Value nextHidden = buildLSTMHiddenState(match, outputGate, nextCell,
                                                previousHidden, builder);
  return LSTMLayerStepResults{nextHidden, nextCell};
}

static mlir::Value buildInitialHiddenRegion(LSTMLayerLowering &match,
                                            mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  mlir::Value h0 = match.lstmLayerOp.getH0();
  auto hiddenRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.state2DTy}, mlir::ValueRange{h0},
      "digital.hidden_extract",
      builder.getStringAttr("lstm_layer_initial_hidden_extract"));

  mlir::Block *body = addTaskRegionBody(hiddenRegion, mlir::ValueRange{h0});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value initialHidden = converter_recurrent_layer::extractLayerState(
      loc, body->getArgument(0), match.layerIndex, match.batchSize,
      match.hiddenSize, match.stateSliceTy, match.state2DTy, builder);
  builder.create<mlir::sculptor::YieldOp>(loc, initialHidden);
  return hiddenRegion.getResult(0);
}

static mlir::Value buildInitialCellRegion(LSTMLayerLowering &match,
                                          mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  mlir::Value c0 = match.lstmLayerOp.getC0();
  auto cellRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.state2DTy}, mlir::ValueRange{c0},
      "digital.cell_extract",
      builder.getStringAttr("lstm_layer_initial_cell_extract"));

  mlir::Block *body = addTaskRegionBody(cellRegion, mlir::ValueRange{c0});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value initialCell = converter_recurrent_layer::extractLayerState(
      loc, body->getArgument(0), match.layerIndex, match.batchSize,
      match.hiddenSize, match.stateSliceTy, match.state2DTy, builder);
  builder.create<mlir::sculptor::YieldOp>(loc, initialCell);
  return cellRegion.getResult(0);
}

static mlir::Value buildTimestepExtractRegion(LSTMLayerLowering &match,
                                              int64_t timestep,
                                              mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  mlir::Value input = match.lstmLayerOp.getInput();
  auto extractRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.input2DTy}, mlir::ValueRange{input},
      "digital.timestep_extract",
      builder.getStringAttr("lstm_layer_timestep_extract"));

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

static mlir::Value buildInputRecombineRegion(LSTMLayerLowering &match,
                                             mlir::Value timestepInput,
                                             mlir::Value recurrentHidden,
                                             mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {timestepInput, recurrentHidden};
  auto recombineRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.rowFusedInputTy}, mlir::ValueRange(inputs),
      "digital.input_recombine",
      builder.getStringAttr("lstm_layer_input_recombine"));

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

static mlir::Value buildBiasAddRegion(LSTMLayerLowering &match,
                                      mlir::TypedAttr fusedBiasAttr,
                                      mlir::Value preActivation,
                                      mlir::OpBuilder &builder) {
  assert((!match.hasBias || fusedBiasAttr) &&
         "expected fused bias attr for biased LSTM layer");
  mlir::Location loc = match.lstmLayerOp.getLoc();

  auto biasRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.rowPreActivationTy},
      mlir::ValueRange{preActivation}, "digital.bias_add",
      builder.getStringAttr("lstm_layer_bias_add"));

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

struct LSTMGateSlices {
  mlir::Value i;
  mlir::Value f;
  mlir::Value g;
  mlir::Value o;
};

static LSTMGateSlices buildGateSplitRegion(LSTMLayerLowering &match,
                                           mlir::Value preActivation,
                                           mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  auto gateSplitRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc,
      mlir::TypeRange{match.state2DTy, match.state2DTy, match.state2DTy,
                      match.state2DTy},
      mlir::ValueRange{preActivation}, "digital.gate_split",
      builder.getStringAttr("lstm_layer_gate_split"));

  mlir::Block *body =
      addTaskRegionBody(gateSplitRegion, mlir::ValueRange{preActivation});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value regionPreActivation = body->getArgument(0);
  mlir::Value iSlice = recurrent_gate::extractBatchGate(
      loc, regionPreActivation, /*gateOffset=*/0, match.batchSize,
      match.hiddenSize, match.state2DTy, builder);
  mlir::Value fSlice = recurrent_gate::extractBatchGate(
      loc, regionPreActivation, match.hiddenSize, match.batchSize,
      match.hiddenSize, match.state2DTy, builder);
  mlir::Value gSlice = recurrent_gate::extractBatchGate(
      loc, regionPreActivation, match.hiddenSize * 2, match.batchSize,
      match.hiddenSize, match.state2DTy, builder);
  mlir::Value oSlice = recurrent_gate::extractBatchGate(
      loc, regionPreActivation, match.hiddenSize * 3, match.batchSize,
      match.hiddenSize, match.state2DTy, builder);

  builder.create<mlir::sculptor::YieldOp>(
      loc, mlir::ValueRange{iSlice, fSlice, gSlice, oSlice});
  return LSTMGateSlices{
      gateSplitRegion.getResult(0), gateSplitRegion.getResult(1),
      gateSplitRegion.getResult(2), gateSplitRegion.getResult(3)};
}

struct LSTMGateActivations {
  mlir::Value input;
  mlir::Value forget;
  mlir::Value candidate;
  mlir::Value output;
};

static LSTMGateActivations buildGateActivationRegion(LSTMLayerLowering &match,
                                                     LSTMGateSlices gates,
                                                     mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {gates.i, gates.f, gates.g, gates.o};
  auto activationRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc,
      mlir::TypeRange{match.state2DTy, match.state2DTy, match.state2DTy,
                      match.state2DTy},
      mlir::ValueRange(inputs), "digital.activation",
      builder.getStringAttr("lstm_layer_gate_activation"));

  mlir::Block *body = addTaskRegionBody(activationRegion, inputs);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value inputGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.state2DTy, body->getArgument(0), builder);
  mlir::Value forgetGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.state2DTy, body->getArgument(1), builder);
  mlir::Value candidateGate = converter_recurrent_elementwise::buildTanh(
      loc, match.state2DTy, body->getArgument(2), builder);
  mlir::Value outputGate = converter_recurrent_elementwise::buildSigmoid(
      loc, match.state2DTy, body->getArgument(3), builder);

  builder.create<mlir::sculptor::YieldOp>(
      loc, mlir::ValueRange{inputGate, forgetGate, candidateGate, outputGate});
  return LSTMGateActivations{
      activationRegion.getResult(0), activationRegion.getResult(1),
      activationRegion.getResult(2), activationRegion.getResult(3)};
}

static mlir::Value buildCellUpdateRegion(LSTMLayerLowering &match,
                                         LSTMGateActivations gates,
                                         mlir::Value previousCell,
                                         mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {gates.forget, previousCell,
                                           gates.input, gates.candidate};
  auto cellUpdateRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.state2DTy}, mlir::ValueRange(inputs),
      "digital.cell_update", builder.getStringAttr("lstm_layer_cell_update"));

  mlir::Block *body = addTaskRegionBody(cellUpdateRegion, inputs);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value nextCell =
      buildLSTMCellState(match, body->getArgument(2), body->getArgument(0),
                         body->getArgument(3), body->getArgument(1), builder);
  builder.create<mlir::sculptor::YieldOp>(loc, nextCell);
  return cellUpdateRegion.getResult(0);
}

static mlir::Value buildHiddenUpdateRegion(LSTMLayerLowering &match,
                                           mlir::Value outputGate,
                                           mlir::Value nextCell,
                                           mlir::Value previousHidden,
                                           mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {outputGate, nextCell,
                                           previousHidden};
  auto hiddenUpdateRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.state2DTy}, mlir::ValueRange(inputs),
      "digital.hidden_update",
      builder.getStringAttr("lstm_layer_hidden_update"));

  mlir::Block *body = addTaskRegionBody(hiddenUpdateRegion, inputs);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value nextHidden =
      buildLSTMHiddenState(match, body->getArgument(0), body->getArgument(1),
                           body->getArgument(2), builder);
  builder.create<mlir::sculptor::YieldOp>(loc, nextHidden);
  return hiddenUpdateRegion.getResult(0);
}

static mlir::Value buildOutputUpdateRegion(LSTMLayerLowering &match,
                                           mlir::Value timestepHidden,
                                           mlir::Value sequenceOutput,
                                           int64_t timestep,
                                           mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {timestepHidden};
  if (sequenceOutput)
    inputs.push_back(sequenceOutput);
  auto outputRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.outputTy}, mlir::ValueRange(inputs),
      "digital.output_update",
      builder.getStringAttr("lstm_layer_output_update"));

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

static mlir::Value buildFinalHiddenRegion(LSTMLayerLowering &match,
                                          mlir::Value finalHidden,
                                          mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  auto hiddenRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.hiddenResultTy}, mlir::ValueRange{finalHidden},
      "digital.hidden_output",
      builder.getStringAttr("lstm_layer_hidden_output"));

  mlir::Block *body =
      addTaskRegionBody(hiddenRegion, mlir::ValueRange{finalHidden});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value hiddenOutput = converter_recurrent_layer::expandFinalLayerState(
      loc, body->getArgument(0), match.hiddenResultTy, builder);
  builder.create<mlir::sculptor::YieldOp>(loc, hiddenOutput);
  return hiddenRegion.getResult(0);
}

static mlir::Value buildFinalCellRegion(LSTMLayerLowering &match,
                                        mlir::Value finalCell,
                                        mlir::OpBuilder &builder) {
  mlir::Location loc = match.lstmLayerOp.getLoc();
  auto cellRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.cellResultTy}, mlir::ValueRange{finalCell},
      "digital.cell_output", builder.getStringAttr("lstm_layer_cell_output"));

  mlir::Block *body =
      addTaskRegionBody(cellRegion, mlir::ValueRange{finalCell});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value cellOutput = converter_recurrent_layer::expandFinalLayerState(
      loc, body->getArgument(0), match.cellResultTy, builder);
  builder.create<mlir::sculptor::YieldOp>(loc, cellOutput);
  return cellRegion.getResult(0);
}

static LSTMLayerTimestepResults buildSectionedLSTMTimestep(
    LSTMLayerLowering &match, int64_t timestep, mlir::Value recurrentHidden,
    mlir::Value recurrentCell, mlir::Value sequenceOutput,
    mlir::Value fusedWeight, mlir::TypedAttr fusedBiasAttr,
    mlir::OpBuilder &builder) {
  assert(match.batchSize == 1 &&
         "sectioned LSTM lowering expects batch size 1");
  mlir::Location loc = match.lstmLayerOp.getLoc();
  mlir::Value timestepInput =
      buildTimestepExtractRegion(match, timestep, builder);
  mlir::Value fusedInput =
      buildInputRecombineRegion(match, timestepInput, recurrentHidden, builder);
  mlir::Value preActivation = mvm_build::buildMVM(
      loc, match.rowPreActivationTy, fusedInput, fusedWeight, builder);
  mlir::Value biasedPreActivation =
      buildBiasAddRegion(match, fusedBiasAttr, preActivation, builder);
  LSTMGateSlices slices =
      buildGateSplitRegion(match, biasedPreActivation, builder);
  LSTMGateActivations gates = buildGateActivationRegion(match, slices, builder);
  mlir::Value nextCell =
      buildCellUpdateRegion(match, gates, recurrentCell, builder);
  mlir::Value nextHidden = buildHiddenUpdateRegion(
      match, gates.output, nextCell, recurrentHidden, builder);
  mlir::Value nextOutput = buildOutputUpdateRegion(
      match, nextHidden, sequenceOutput, timestep, builder);
  return LSTMLayerTimestepResults{nextHidden, nextCell, nextOutput};
}

// Builds batched fused preactivation before LSTM gate math.
static mlir::Value buildLSTMBatchPreActivation(LSTMLayerLowering &match,
                                               mlir::Value timestepInput,
                                               mlir::Value recurrentHidden,
                                               mlir::Value fusedWeight,
                                               mlir::Value expandedBias,
                                               mlir::OpBuilder &builder) {
  assert(match.batchSize > 0 && "expected positive batch size");
  llvm::SmallVector<mlir::Value> rowPreActivations;
  rowPreActivations.reserve(match.batchSize);
  mlir::Location loc = match.lstmLayerOp.getLoc();

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

static mlir::LogicalResult lowerLSTMLayerToMVM(mlir::func::FuncOp func,
                                               mlir::RewriterBase &rewriter) {
  auto match = matchExtractedLSTMLayer(func);
  if (mlir::failed(match))
    return mlir::failure();

  mlir::TypedAttr fusedWeightAttr = buildLSTMFusedWeightAttr(*match);
  if (!fusedWeightAttr)
    return mlir::failure();

  mlir::TypedAttr fusedBiasAttr;
  if (match->hasBias) {
    fusedBiasAttr = buildLSTMFusedBiasAttr(*match);
    if (!fusedBiasAttr)
      return mlir::failure();
  }

  mlir::Location loc = match->lstmLayerOp.getLoc();
  rewriter.setInsertionPoint(match->lstmLayerOp);
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
  mlir::Value currentCell;
  mlir::Value sequenceOutputInit;
  if (match->batchSize == 1) {
    currentHidden = buildInitialHiddenRegion(*match, rewriter);
    currentCell = buildInitialCellRegion(*match, rewriter);
  } else {
    currentHidden = converter_recurrent_layer::extractLayerState(
        loc, match->lstmLayerOp.getH0(), match->layerIndex, match->batchSize,
        match->hiddenSize, match->stateSliceTy, match->state2DTy, rewriter);
    currentCell = converter_recurrent_layer::extractLayerState(
        loc, match->lstmLayerOp.getC0(), match->layerIndex, match->batchSize,
        match->hiddenSize, match->stateSliceTy, match->state2DTy, rewriter);
    sequenceOutputInit = rewriter.create<EmptyOp>(
        loc, match->outputTy.getShape(), match->outputTy.getElementType());
  }

  mlir::Value sequenceOutput = sequenceOutputInit;
  mlir::Value finalHidden = currentHidden;
  mlir::Value finalCell = currentCell;
  if (match->batchSize == 1) {
    for (int64_t step = 0; step < match->sequenceLength; ++step) {
      LSTMLayerTimestepResults timestepResult = buildSectionedLSTMTimestep(
          *match, step, finalHidden, finalCell, sequenceOutput,
          fusedWeightConstant.getResult(), fusedBiasAttr, rewriter);
      finalHidden = timestepResult.hidden;
      finalCell = timestepResult.cell;
      sequenceOutput = timestepResult.output;
    }
  } else {
    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value cSequenceLength = rewriter.create<mlir::arith::ConstantIndexOp>(
        loc, match->sequenceLength);

    llvm::SmallVector<mlir::Value, 3> initArgs = {currentHidden, currentCell,
                                                  sequenceOutputInit};
    auto timestepLoop = rewriter.create<mlir::scf::ForOp>(
        loc, c0, cSequenceLength, c1, initArgs,
        [&](mlir::OpBuilder &builder, mlir::Location loopLoc,
            mlir::Value timestep, mlir::ValueRange iterArgs) {
          mlir::Value loopHidden = iterArgs[0];
          mlir::Value loopCell = iterArgs[1];
          mlir::Value loopOutput = iterArgs[2];
          mlir::Value timestepInput =
              converter_recurrent_layer::extractBatchFirstTimestep(
                  match->lstmLayerOp.getLoc(), match->lstmLayerOp.getInput(),
                  timestep, match->batchSize, match->inputSize,
                  match->inputSliceTy, match->input2DTy, builder);
          mlir::Value preActivation = buildLSTMBatchPreActivation(
              *match, timestepInput, loopHidden,
              fusedWeightConstant.getResult(), expandedBias, builder);
          LSTMLayerStepResults stepResults = buildLSTMGateMath(
              *match, preActivation, loopHidden, loopCell, builder);
          mlir::Value nextOutput =
              converter_recurrent_layer::insertBatchFirstTimestep(
                  match->lstmLayerOp.getLoc(), stepResults.hidden, loopOutput,
                  timestep, match->batchSize, match->hiddenSize,
                  match->timestepResultTy, builder);
          builder.create<mlir::scf::YieldOp>(
              loopLoc, mlir::ValueRange{stepResults.hidden, stepResults.cell,
                                        nextOutput});
        });

    rewriter.setInsertionPointAfter(timestepLoop);
    finalHidden = timestepLoop.getResult(0);
    finalCell = timestepLoop.getResult(1);
    sequenceOutput = timestepLoop.getResult(2);
  }

  mlir::Value hiddenOutput =
      match->batchSize == 1
          ? buildFinalHiddenRegion(*match, finalHidden, rewriter)
          : converter_recurrent_layer::expandFinalLayerState(
                match->lstmLayerOp.getLoc(), finalHidden, match->stateSliceTy,
                rewriter);
  mlir::Value cellOutput =
      match->batchSize == 1 ? buildFinalCellRegion(*match, finalCell, rewriter)
                            : converter_recurrent_layer::expandFinalLayerState(
                                  match->lstmLayerOp.getLoc(), finalCell,
                                  match->stateSliceTy, rewriter);

  match->lstmLayerOp.getOutput().replaceAllUsesWith(sequenceOutput);
  match->lstmLayerOp.getHn().replaceAllUsesWith(hiddenOutput);
  match->lstmLayerOp.getCn().replaceAllUsesWith(cellOutput);
  rewriter.eraseOp(match->lstmLayerOp);
  converter_recurrent_layer::eraseUnusedConstants(
      {match->weightIHConstant, match->weightHHConstant, match->biasIHConstant,
       match->biasHHConstant},
      rewriter);
  return mlir::success();
}

// Converts extracted sculptor.nn.lstm_layer bodies to sculptor.mvm timestep loops.
class LSTMConverter : public mlir::sculptor::LayerToMVMConverter {
public:
  mlir::StringRef getName() const override { return "lstm"; }

  void lowerToMVM(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());
    (void)lowerLSTMLayerToMVM(func, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerLSTMConverter(LayerToMVMConverters &converters,
                           LayerToMVMConverterMap &converterMap,
                           MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<LSTMConverter>();
  const LayerToMVMConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["lstm"] = converterPtr;
  converterMap["lstm_w_bias"] = converterPtr;
}

} // namespace sculptor
} // namespace mlir
