#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConstantUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/MVMBuildUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/NNLayerMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/RecurrentElementwiseUtils.h"
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

#include <cassert>
#include <memory>
#include <utility>

namespace converter_constant = mlir::sculptor::converter_constant;
namespace converter_recurrent_elementwise =
    mlir::sculptor::converter_recurrent_elementwise;
namespace converter_recurrent_layer = mlir::sculptor::converter_recurrent_layer;
namespace mvm_build = mlir::sculptor::mvm_build;
namespace nn_layer_match = mlir::sculptor::nn_layer_match;
namespace tensor_type = mlir::sculptor::tensor_type;

namespace {

using mlir::sculptor::NNRNNLayerOp;
using mlir::arith::ConstantOp;
using mlir::tensor::ConcatOp;
using mlir::tensor::EmptyOp;

struct RNNLayerLowering {
  NNRNNLayerOp rnnLayerOp;
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
  mlir::RankedTensorType rowResultTy;
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

struct RNNTimestepResult {
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

static mlir::FailureOr<RNNLayerLowering>
matchExtractedRNNLayer(mlir::func::FuncOp func) {
  auto rnnLayerOp = nn_layer_match::matchSingleNNLayerOp<NNRNNLayerOp>(func);
  if (failed(rnnLayerOp))
    return mlir::failure();

  bool hasBias = (*rnnLayerOp).getHasBias();
  if (!nn_layer_match::hasLayerTypeMatchingBias(func, "rnn", "rnn_w_bias",
                                                hasBias))
    return mlir::failure();

  if (!(*rnnLayerOp).getBatchFirst())
    return mlir::failure();

  if (func.getNumArguments() != 2 || func.getNumResults() != 2 ||
      (*rnnLayerOp).getInput() != func.getArgument(0) ||
      (*rnnLayerOp).getH0() != func.getArgument(1))
    return mlir::failure();

  auto inputTy = tensor_type::getStaticF32Tensor(
      (*rnnLayerOp).getInput().getType(), /*expectedRank=*/3);
  auto hiddenStateTy = tensor_type::getStaticF32Tensor(
      (*rnnLayerOp).getH0().getType(), /*expectedRank=*/3);
  auto outputTy = tensor_type::getStaticF32Tensor(
      (*rnnLayerOp).getOutput().getType(), /*expectedRank=*/3);
  auto hiddenResultTy = tensor_type::getStaticF32Tensor(
      (*rnnLayerOp).getHn().getType(), /*expectedRank=*/3);
  auto weightIHTy = tensor_type::getStaticF32Tensor(
      (*rnnLayerOp).getWIh().getType(), /*expectedRank=*/2);
  auto weightHHTy = tensor_type::getStaticF32Tensor(
      (*rnnLayerOp).getWHh().getType(), /*expectedRank=*/2);
  if (failed(inputTy) || failed(hiddenStateTy) || failed(outputTy) ||
      failed(hiddenResultTy) || failed(weightIHTy) || failed(weightHHTy))
    return mlir::failure();

  int64_t layerIndex = (*rnnLayerOp).getLayerIndex();
  int64_t numLayers = (*rnnLayerOp).getNumLayers();
  int64_t hiddenSize = (*rnnLayerOp).getHiddenSize();
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
          llvm::ArrayRef<int64_t>({hiddenSize, inputSize}) ||
      weightHHTy->getShape() !=
          llvm::ArrayRef<int64_t>({hiddenSize, hiddenSize}))
    return mlir::failure();

  auto weightIHConstant = (*rnnLayerOp).getWIh().getDefiningOp<ConstantOp>();
  auto weightHHConstant = (*rnnLayerOp).getWHh().getDefiningOp<ConstantOp>();
  if (!weightIHConstant || !weightHHConstant)
    return mlir::failure();

  ConstantOp biasIHConstant;
  ConstantOp biasHHConstant;
  mlir::RankedTensorType fusedBiasTy;
  if (hasBias) {
    mlir::Value biasIH = (*rnnLayerOp).getBIh();
    mlir::Value biasHH = (*rnnLayerOp).getBHh();
    if (!biasIH || !biasHH)
      return mlir::failure();

    auto biasIHTy =
        tensor_type::getStaticF32Tensor(biasIH.getType(), /*expectedRank=*/1);
    auto biasHHTy =
        tensor_type::getStaticF32Tensor(biasHH.getType(), /*expectedRank=*/1);
    if (failed(biasIHTy) || failed(biasHHTy) ||
        biasIHTy->getShape() != llvm::ArrayRef<int64_t>({hiddenSize}) ||
        biasHHTy->getShape() != llvm::ArrayRef<int64_t>({hiddenSize}))
      return mlir::failure();

    biasIHConstant = biasIH.getDefiningOp<ConstantOp>();
    biasHHConstant = biasHH.getDefiningOp<ConstantOp>();
    if (!biasIHConstant || !biasHHConstant)
      return mlir::failure();

    fusedBiasTy =
        mlir::RankedTensorType::get({hiddenSize}, inputTy->getElementType());
  } else if ((*rnnLayerOp).getBIh() || (*rnnLayerOp).getBHh()) {
    return mlir::failure();
  }

  mlir::Type elementType = inputTy->getElementType();
  RNNLayerLowering lowering;
  lowering.rnnLayerOp = *rnnLayerOp;
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
  lowering.rowResultTy =
      mlir::RankedTensorType::get({1, hiddenSize}, elementType);
  lowering.timestepResultTy =
      mlir::RankedTensorType::get({batchSize, 1, hiddenSize}, elementType);
  lowering.fusedWeightTy = mlir::RankedTensorType::get(
      {hiddenSize, inputSize + hiddenSize}, elementType);
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

static mlir::TypedAttr buildRNNFusedWeightAttr(RNNLayerLowering &match) {
  auto maybeInputWeights =
      converter_constant::getF32ConstantValues(match.weightIHConstant);
  auto maybeHiddenWeights =
      converter_constant::getF32ConstantValues(match.weightHHConstant);
  if (failed(maybeInputWeights) || failed(maybeHiddenWeights))
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
      match.fusedWeightTy, fusedWeights, "analog_rnn_layer_fused_weight_",
      useResource);
}

static mlir::TypedAttr buildRNNFusedBiasAttr(RNNLayerLowering &match) {
  auto maybeInputBias =
      converter_constant::getF32ConstantValues(match.biasIHConstant);
  auto maybeHiddenBias =
      converter_constant::getF32ConstantValues(match.biasHHConstant);
  if (failed(maybeInputBias) || failed(maybeHiddenBias))
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
      match.fusedBiasTy, fusedBias, "analog_rnn_layer_fused_bias_",
      useResource);
}

// Applies optional bias and tanh into the loop-carried hidden state.
static mlir::Value buildRNNHiddenActivation(RNNLayerLowering &match,
                                            mlir::Value preActivation,
                                            mlir::Value recurrentHidden,
                                            mlir::Value expandedBias,
                                            mlir::OpBuilder &builder) {
  mlir::Location loc = match.rnnLayerOp.getLoc();
  mlir::Value activationInput = preActivation;
  if (match.hasBias)
    activationInput = converter_recurrent_layer::addBroadcastRowBias(
        loc, preActivation, expandedBias, match.hidden2DTy, builder);

  return converter_recurrent_elementwise::buildTanh(
      loc, activationInput, recurrentHidden, match.hidden2DTy, builder);
}

static mlir::Value buildInitialHiddenRegion(RNNLayerLowering &match,
                                            mlir::OpBuilder &builder) {
  mlir::Location loc = match.rnnLayerOp.getLoc();
  mlir::Value h0 = match.rnnLayerOp.getH0();
  auto hiddenRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.hidden2DTy}, mlir::ValueRange{h0},
      "digital.hidden_extract",
      builder.getStringAttr("rnn_layer_initial_hidden_extract"));

  mlir::Block *body = addTaskRegionBody(hiddenRegion, mlir::ValueRange{h0});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value initialHidden = converter_recurrent_layer::extractLayerState(
      loc, body->getArgument(0), match.layerIndex, match.batchSize,
      match.hiddenSize, match.hiddenSliceTy, match.hidden2DTy, builder);
  builder.create<mlir::sculptor::YieldOp>(loc, initialHidden);
  return hiddenRegion.getResult(0);
}

static mlir::Value buildTimestepExtractRegion(RNNLayerLowering &match,
                                              int64_t timestep,
                                              mlir::OpBuilder &builder) {
  mlir::Location loc = match.rnnLayerOp.getLoc();
  mlir::Value input = match.rnnLayerOp.getInput();
  auto extractRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.input2DTy}, mlir::ValueRange{input},
      "digital.timestep_extract",
      builder.getStringAttr("rnn_layer_timestep_extract"));

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

static mlir::Value buildInputRecombineRegion(RNNLayerLowering &match,
                                             mlir::Value timestepInput,
                                             mlir::Value recurrentHidden,
                                             mlir::OpBuilder &builder) {
  mlir::Location loc = match.rnnLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {timestepInput, recurrentHidden};
  auto recombineRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.rowFusedInputTy}, mlir::ValueRange(inputs),
      "digital.input_recombine",
      builder.getStringAttr("rnn_layer_input_recombine"));

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

static mlir::Value buildBiasAddRegion(RNNLayerLowering &match,
                                      mlir::TypedAttr fusedBiasAttr,
                                      mlir::Value preActivation,
                                      mlir::OpBuilder &builder) {
  assert((!match.hasBias || fusedBiasAttr) &&
         "expected fused bias attr for biased RNN layer");
  mlir::Location loc = match.rnnLayerOp.getLoc();

  auto biasRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.hidden2DTy}, mlir::ValueRange{preActivation},
      "digital.bias_add", builder.getStringAttr("rnn_layer_bias_add"));

  mlir::Block *body =
      addTaskRegionBody(biasRegion, mlir::ValueRange{preActivation});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value biasResult = body->getArgument(0);
  if (match.hasBias) {
    auto fusedBias =
        builder.create<ConstantOp>(loc, match.fusedBiasTy, fusedBiasAttr);
    mlir::Value expandedBias = converter_recurrent_layer::expandRowBias(
        loc, fusedBias.getResult(), match.rowResultTy, builder);
    biasResult = converter_recurrent_layer::addBroadcastRowBias(
        loc, body->getArgument(0), expandedBias, match.hidden2DTy, builder);
  }
  builder.create<mlir::sculptor::YieldOp>(loc, biasResult);
  return biasRegion.getResult(0);
}

static mlir::Value buildActivationRegion(RNNLayerLowering &match,
                                         mlir::Value activationInput,
                                         mlir::Value recurrentHidden,
                                         mlir::OpBuilder &builder) {
  mlir::Location loc = match.rnnLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {activationInput, recurrentHidden};
  auto activationRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.hidden2DTy}, mlir::ValueRange(inputs),
      "digital.activation", builder.getStringAttr("rnn_layer_tanh"));

  mlir::Block *body = addTaskRegionBody(activationRegion, inputs);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value result = converter_recurrent_elementwise::buildTanh(
      loc, body->getArgument(0), body->getArgument(1), match.hidden2DTy,
      builder);
  builder.create<mlir::sculptor::YieldOp>(loc, result);
  return activationRegion.getResult(0);
}

static mlir::Value buildOutputUpdateRegion(RNNLayerLowering &match,
                                           mlir::Value timestepHidden,
                                           mlir::Value sequenceOutput,
                                           int64_t timestep,
                                           mlir::OpBuilder &builder) {
  mlir::Location loc = match.rnnLayerOp.getLoc();
  llvm::SmallVector<mlir::Value> inputs = {timestepHidden};
  if (sequenceOutput)
    inputs.push_back(sequenceOutput);
  auto outputRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.outputTy}, mlir::ValueRange(inputs),
      "digital.output_update",
      builder.getStringAttr("rnn_layer_output_update"));

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

static mlir::Value buildFinalHiddenRegion(RNNLayerLowering &match,
                                          mlir::Value finalHidden,
                                          mlir::OpBuilder &builder) {
  mlir::Location loc = match.rnnLayerOp.getLoc();
  auto hiddenRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.hiddenResultTy}, mlir::ValueRange{finalHidden},
      "digital.hidden_output",
      builder.getStringAttr("rnn_layer_hidden_output"));

  mlir::Block *body =
      addTaskRegionBody(hiddenRegion, mlir::ValueRange{finalHidden});

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value hiddenOutput = converter_recurrent_layer::expandFinalLayerState(
      loc, body->getArgument(0), match.hiddenResultTy, builder);
  builder.create<mlir::sculptor::YieldOp>(loc, hiddenOutput);
  return hiddenRegion.getResult(0);
}

static RNNTimestepResult buildSectionedRNNTimestep(
    RNNLayerLowering &match, int64_t timestep, mlir::Value recurrentHidden,
    mlir::Value sequenceOutput, mlir::Value fusedWeight,
    mlir::TypedAttr fusedBiasAttr, mlir::OpBuilder &builder) {
  assert(match.batchSize == 1 && "sectioned RNN lowering expects batch size 1");
  mlir::Location loc = match.rnnLayerOp.getLoc();
  mlir::Value timestepInput =
      buildTimestepExtractRegion(match, timestep, builder);
  mlir::Value fusedInput =
      buildInputRecombineRegion(match, timestepInput, recurrentHidden, builder);
  mlir::Value preActivation = mvm_build::buildMVM(
      loc, match.rowResultTy, fusedInput, fusedWeight, builder);
  mlir::Value activationInput =
      buildBiasAddRegion(match, fusedBiasAttr, preActivation, builder);
  mlir::Value timestepHidden =
      buildActivationRegion(match, activationInput, recurrentHidden, builder);
  mlir::Value nextOutput = buildOutputUpdateRegion(
      match, timestepHidden, sequenceOutput, timestep, builder);
  return RNNTimestepResult{timestepHidden, nextOutput};
}

// Runs one recurrent timestep across all batch rows.
static mlir::Value
buildRNNBatchStep(RNNLayerLowering &match, mlir::Value timestepInput,
                  mlir::Value recurrentHidden, mlir::Value fusedWeight,
                  mlir::Value expandedBias, mlir::OpBuilder &builder) {
  assert(match.batchSize > 0 && "expected positive batch size");
  llvm::SmallVector<mlir::Value> rowPreActivations;
  rowPreActivations.reserve(match.batchSize);
  mlir::Location loc = match.rnnLayerOp.getLoc();

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
        loc, match.rowResultTy, fusedInput, fusedWeight, builder);
    rowPreActivations.push_back(mvmResult);
  }

  mlir::Value preActivation;
  if (rowPreActivations.size() == 1) {
    preActivation = rowPreActivations.front();
  } else {
    preActivation = builder
                        .create<ConcatOp>(loc, match.hidden2DTy, /*dim=*/0,
                                          mlir::ValueRange(rowPreActivations))
                        .getResult();
  }

  return buildRNNHiddenActivation(match, preActivation, recurrentHidden,
                                  expandedBias, builder);
}

// Lowers one extracted RNN layer into an sculptor.mvm timestep loop.
static mlir::LogicalResult lowerRNNLayerToMVM(mlir::func::FuncOp func,
                                              mlir::RewriterBase &rewriter) {
  auto match = matchExtractedRNNLayer(func);
  if (failed(match))
    return mlir::failure();

  mlir::TypedAttr fusedWeightAttr = buildRNNFusedWeightAttr(*match);
  if (!fusedWeightAttr)
    return mlir::failure();

  mlir::TypedAttr fusedBiasAttr;
  if (match->hasBias) {
    fusedBiasAttr = buildRNNFusedBiasAttr(*match);
    if (!fusedBiasAttr)
      return mlir::failure();
  }

  mlir::Location loc = match->rnnLayerOp.getLoc();
  rewriter.setInsertionPoint(match->rnnLayerOp);
  auto fusedWeightConstant =
      rewriter.create<ConstantOp>(loc, match->fusedWeightTy, fusedWeightAttr);

  ConstantOp fusedBiasConstant;
  mlir::Value expandedBias;
  if (match->hasBias && match->batchSize != 1) {
    fusedBiasConstant =
        rewriter.create<ConstantOp>(loc, match->fusedBiasTy, fusedBiasAttr);
    expandedBias = converter_recurrent_layer::expandRowBias(
        loc, fusedBiasConstant.getResult(), match->rowResultTy, rewriter);
  }

  mlir::Value currentHidden;
  mlir::Value sequenceOutputInit;
  if (match->batchSize == 1) {
    currentHidden = buildInitialHiddenRegion(*match, rewriter);
  } else {
    currentHidden = converter_recurrent_layer::extractLayerState(
        loc, match->rnnLayerOp.getH0(), match->layerIndex, match->batchSize,
        match->hiddenSize, match->hiddenSliceTy, match->hidden2DTy, rewriter);

    sequenceOutputInit = rewriter.create<EmptyOp>(
        loc, match->outputTy.getShape(), match->outputTy.getElementType());
  }

  mlir::Value sequenceOutput = sequenceOutputInit;
  mlir::Value finalHidden = currentHidden;
  if (match->batchSize == 1) {
    for (int64_t step = 0; step < match->sequenceLength; ++step) {
      RNNTimestepResult timestepResult = buildSectionedRNNTimestep(
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
                  match->rnnLayerOp.getLoc(), match->rnnLayerOp.getInput(),
                  timestep, match->batchSize, match->inputSize,
                  match->inputSliceTy, match->input2DTy, builder);
          mlir::Value timestepHidden = buildRNNBatchStep(
              *match, timestepInput, loopHidden,
              fusedWeightConstant.getResult(), expandedBias, builder);
          mlir::Value nextOutput =
              converter_recurrent_layer::insertBatchFirstTimestep(
                  match->rnnLayerOp.getLoc(), timestepHidden, loopOutput,
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
                match->rnnLayerOp.getLoc(), finalHidden, match->hiddenResultTy,
                rewriter);

  match->rnnLayerOp.getOutput().replaceAllUsesWith(sequenceOutput);
  match->rnnLayerOp.getHn().replaceAllUsesWith(hiddenOutput);
  rewriter.eraseOp(match->rnnLayerOp);
  converter_recurrent_layer::eraseUnusedConstants(
      {match->weightIHConstant, match->weightHHConstant, match->biasIHConstant,
       match->biasHHConstant},
      rewriter);
  return mlir::success();
}

// Converts extracted sculptor.nn.rnn_layer bodies to sculptor.mvm timestep loops.
class RNNConverter : public mlir::sculptor::LayerToMVMConverter {
public:
  mlir::StringRef getName() const override { return "rnn"; }

  void lowerToMVM(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());
    (void)lowerRNNLayerToMVM(func, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerRNNConverter(LayerToMVMConverters &converters,
                          LayerToMVMConverterMap &converterMap,
                          MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<RNNConverter>();
  const LayerToMVMConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["rnn"] = converterPtr;
  converterMap["rnn_w_bias"] = converterPtr;
}

} // namespace sculptor
} // namespace mlir
