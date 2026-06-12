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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <memory>

namespace converter_constant = mlir::sculptor::converter_constant;
namespace converter_recurrent_elementwise =
    mlir::sculptor::converter_recurrent_elementwise;
namespace converter_recurrent_layer = mlir::sculptor::converter_recurrent_layer;
namespace mvm_build = mlir::sculptor::mvm_build;
namespace nn_layer_match = mlir::sculptor::nn_layer_match;
namespace recurrent_gate = mlir::sculptor::recurrent_gate;
namespace tensor_type = mlir::sculptor::tensor_type;

namespace {

using mlir::sculptor::NNLSTMCellOp;
using mlir::arith::ConstantOp;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;

struct LSTMCellLowering {
  NNLSTMCellOp lstmCellOp;
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType hiddenTy;
  mlir::RankedTensorType cellTy;
  mlir::RankedTensorType weightIHTy;
  mlir::RankedTensorType weightHHTy;
  mlir::RankedTensorType resultHTy;
  mlir::RankedTensorType resultCTy;
  mlir::RankedTensorType fusedInputTy;
  mlir::RankedTensorType fusedWeightTy;
  mlir::RankedTensorType fusedBiasTy;
  mlir::RankedTensorType preActivationTy;
  ConstantOp weightIHConstant;
  ConstantOp weightHHConstant;
  ConstantOp biasIHConstant;
  ConstantOp biasHHConstant;
  bool hasBias = false;
};

static mlir::RankedTensorType
buildFusedWeightType(mlir::RankedTensorType weightIHTy,
                     mlir::RankedTensorType weightHHTy) {
  return mlir::RankedTensorType::get(
      {weightIHTy.getShape()[0],
       weightIHTy.getShape()[1] + weightHHTy.getShape()[1]},
      weightIHTy.getElementType());
}

static mlir::RankedTensorType
buildPreActivationType(mlir::RankedTensorType hiddenTy) {
  return mlir::RankedTensorType::get({1, hiddenTy.getShape()[1] * 4},
                                     hiddenTy.getElementType());
}

static mlir::RankedTensorType
buildFusedBiasType(mlir::RankedTensorType hiddenTy) {
  return mlir::RankedTensorType::get({hiddenTy.getShape()[1] * 4},
                                     hiddenTy.getElementType());
}

static mlir::FailureOr<LSTMCellLowering>
matchExtractedLSTMCellLayer(mlir::func::FuncOp func) {
  auto lstmCellOp = nn_layer_match::matchSingleNNLayerOp<NNLSTMCellOp>(func);
  if (mlir::failed(lstmCellOp))
    return mlir::failure();

  bool hasBias = (*lstmCellOp).getHasBias();
  if (!nn_layer_match::hasLayerTypeMatchingBias(func, "lstm_cell",
                                                "lstm_cell_w_bias", hasBias))
    return mlir::failure();

  if (func.getNumArguments() != 3 ||
      (*lstmCellOp).getInput() != func.getArgument(0) ||
      (*lstmCellOp).getHPrev() != func.getArgument(1) ||
      (*lstmCellOp).getCPrev() != func.getArgument(2))
    return mlir::failure();

  auto inputTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*lstmCellOp).getInput().getType());
  auto hiddenTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*lstmCellOp).getHPrev().getType());
  auto cellTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*lstmCellOp).getCPrev().getType());
  auto weightIHTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*lstmCellOp).getWIh().getType());
  auto weightHHTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*lstmCellOp).getWHh().getType());
  auto resultHTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*lstmCellOp).getH().getType());
  auto resultCTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*lstmCellOp).getC().getType());
  if (mlir::failed(inputTy) || mlir::failed(hiddenTy) || mlir::failed(cellTy) ||
      mlir::failed(weightIHTy) || mlir::failed(weightHHTy) ||
      mlir::failed(resultHTy) || mlir::failed(resultCTy))
    return mlir::failure();

  int64_t inputSize = (*inputTy).getShape()[1];
  int64_t hiddenSize = (*hiddenTy).getShape()[1];
  if ((*inputTy).getShape()[0] != 1 || (*hiddenTy).getShape()[0] != 1 ||
      (*cellTy).getShape()[0] != 1 || (*resultHTy).getShape()[0] != 1 ||
      (*resultCTy).getShape()[0] != 1 ||
      (*cellTy).getShape()[1] != hiddenSize ||
      (*resultHTy).getShape()[1] != hiddenSize ||
      (*resultCTy).getShape()[1] != hiddenSize)
    return mlir::failure();

  if ((*weightIHTy).getShape()[0] != hiddenSize * 4 ||
      (*weightIHTy).getShape()[1] != inputSize ||
      (*weightHHTy).getShape()[0] != hiddenSize * 4 ||
      (*weightHHTy).getShape()[1] != hiddenSize)
    return mlir::failure();

  auto weightIHConstant = (*lstmCellOp).getWIh().getDefiningOp<ConstantOp>();
  auto weightHHConstant = (*lstmCellOp).getWHh().getDefiningOp<ConstantOp>();
  if (!weightIHConstant || !weightHHConstant)
    return mlir::failure();

  ConstantOp biasIHConstant;
  ConstantOp biasHHConstant;
  if (hasBias) {
    mlir::Value biasIH = (*lstmCellOp).getBIh();
    mlir::Value biasHH = (*lstmCellOp).getBHh();
    if (!biasIH || !biasHH)
      return mlir::failure();

    auto biasIHTy =
        tensor_type::getPositiveStaticRank1F32Tensor(biasIH.getType());
    auto biasHHTy =
        tensor_type::getPositiveStaticRank1F32Tensor(biasHH.getType());
    if (mlir::failed(biasIHTy) || mlir::failed(biasHHTy) ||
        (*biasIHTy).getShape()[0] != hiddenSize * 4 ||
        (*biasHHTy).getShape()[0] != hiddenSize * 4)
      return mlir::failure();

    biasIHConstant = biasIH.getDefiningOp<ConstantOp>();
    biasHHConstant = biasHH.getDefiningOp<ConstantOp>();
    if (!biasIHConstant || !biasHHConstant)
      return mlir::failure();
  } else if ((*lstmCellOp).getBIh() || (*lstmCellOp).getBHh()) {
    return mlir::failure();
  }

  LSTMCellLowering lowering;
  lowering.lstmCellOp = *lstmCellOp;
  lowering.inputTy = *inputTy;
  lowering.hiddenTy = *hiddenTy;
  lowering.cellTy = *cellTy;
  lowering.weightIHTy = *weightIHTy;
  lowering.weightHHTy = *weightHHTy;
  lowering.resultHTy = *resultHTy;
  lowering.resultCTy = *resultCTy;
  lowering.fusedInputTy =
      recurrent_gate::buildFusedInputType(*inputTy, *hiddenTy);
  lowering.fusedWeightTy = buildFusedWeightType(*weightIHTy, *weightHHTy);
  lowering.fusedBiasTy = buildFusedBiasType(*hiddenTy);
  lowering.preActivationTy = buildPreActivationType(*hiddenTy);
  lowering.weightIHConstant = weightIHConstant;
  lowering.weightHHConstant = weightHHConstant;
  lowering.biasIHConstant = biasIHConstant;
  lowering.biasHHConstant = biasHHConstant;
  lowering.hasBias = hasBias;
  return lowering;
}

// Fuses LSTMCell i, f, g, o weights for one sculptor.mvm.
static mlir::TypedAttr buildFusedWeightAttr(LSTMCellLowering &match) {
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

  int64_t outputSize = match.fusedWeightTy.getShape()[0];
  int64_t inputSize = match.inputTy.getShape()[1];
  int64_t hiddenSize = match.hiddenTy.getShape()[1];
  int64_t fusedWidth = match.fusedWeightTy.getShape()[1];
  llvm::SmallVector<float> fusedWeights(match.fusedWeightTy.getNumElements(),
                                        0.0f);

  for (int64_t row = 0; row < outputSize; ++row) {
    int64_t inputOffset = row * inputSize;
    int64_t hiddenOffset = row * hiddenSize;
    int64_t fusedOffset = row * fusedWidth;
    for (int64_t col = 0; col < inputSize; ++col)
      fusedWeights[fusedOffset + col] = inputWeights[inputOffset + col];
    for (int64_t col = 0; col < hiddenSize; ++col)
      fusedWeights[fusedOffset + inputSize + col] =
          hiddenWeights[hiddenOffset + col];
  }

  bool useResource =
      converter_constant::isResourceBackedF32Constant(match.weightIHConstant) ||
      converter_constant::isResourceBackedF32Constant(match.weightHHConstant);
  return converter_constant::buildF32ElementsAttr(
      match.fusedWeightTy, fusedWeights, "analog_lstm_cell_fused_weight_",
      useResource);
}

// Fuses LSTMCell i, f, g, o bias terms after the MVM.
static mlir::TypedAttr buildFusedBiasAttr(LSTMCellLowering &match) {
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
      match.fusedBiasTy, fusedBias, "analog_lstm_cell_fused_bias_",
      useResource);
}

static mlir::Value buildInputRecombineRegion(LSTMCellLowering &match,
                                             mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.lstmCellOp.getLoc();
  rewriter.setInsertionPoint(match.lstmCellOp);
  auto recombineRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.fusedInputTy},
      mlir::ValueRange{match.lstmCellOp.getInput(),
                       match.lstmCellOp.getHPrev()},
      "digital.input_recombine",
      rewriter.getStringAttr("lstm_cell_input_recombine"));

  mlir::Block *body = new mlir::Block();
  recombineRegion.getBody().push_back(body);
  llvm::SmallVector<mlir::Type> inputTypes = {
      match.lstmCellOp.getInput().getType(),
      match.lstmCellOp.getHPrev().getType()};
  llvm::SmallVector<mlir::Location> inputLocs = {
      match.lstmCellOp.getInput().getLoc(),
      match.lstmCellOp.getHPrev().getLoc()};
  body->addArguments(inputTypes, inputLocs);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);
  mlir::Value fusedInput = recurrent_gate::buildFusedInput(
      loc, match.fusedInputTy, body->getArgument(0), body->getArgument(1),
      rewriter);
  rewriter.create<mlir::sculptor::YieldOp>(loc, fusedInput);
  return recombineRegion.getResult(0);
}

static mlir::FailureOr<mlir::Value>
buildBiasAddRegion(LSTMCellLowering &match, mlir::TypedAttr fusedBiasAttr,
                   mlir::Value mvmResult, mlir::RewriterBase &rewriter) {
  if (match.hasBias && !fusedBiasAttr)
    return mlir::failure();

  mlir::Location loc = match.lstmCellOp.getLoc();
  rewriter.setInsertionPoint(match.lstmCellOp);
  auto biasRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.preActivationTy}, mlir::ValueRange{mvmResult},
      "digital.bias_add", rewriter.getStringAttr("lstm_cell_bias_add"));

  mlir::Block *body = new mlir::Block();
  biasRegion.getBody().push_back(body);
  body->addArgument(mvmResult.getType(), mvmResult.getLoc());

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);
  mlir::Value biasResult = body->getArgument(0);
  if (!match.hasBias) {
    rewriter.create<mlir::sculptor::YieldOp>(loc, biasResult);
    return biasRegion.getResult(0);
  }

  auto fusedBias =
      rewriter.create<ConstantOp>(loc, match.fusedBiasTy, fusedBiasAttr);

  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation = {{0, 1}};
  mlir::Value expandedBias = rewriter.create<ExpandShapeOp>(
      loc, match.preActivationTy, fusedBias.getResult(), reassociation);
  mlir::Value biasedInit =
      rewriter.create<EmptyOp>(loc, match.preActivationTy.getShape(),
                               match.preActivationTy.getElementType());
  biasResult =
      rewriter
          .create<mlir::linalg::AddOp>(
              loc, mlir::ValueRange{body->getArgument(0), expandedBias},
              mlir::ValueRange{biasedInit})
          .getResult(0);
  rewriter.create<mlir::sculptor::YieldOp>(loc, biasResult);
  return biasRegion.getResult(0);
}

struct LSTMGateSlices {
  mlir::Value i;
  mlir::Value f;
  mlir::Value g;
  mlir::Value o;
};

static LSTMGateSlices buildGateSplitRegion(LSTMCellLowering &match,
                                           mlir::Value preActivation,
                                           mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.lstmCellOp.getLoc();
  rewriter.setInsertionPoint(match.lstmCellOp);
  auto gateSplitRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc,
      mlir::TypeRange{match.resultHTy, match.resultHTy, match.resultHTy,
                      match.resultHTy},
      mlir::ValueRange{preActivation}, "digital.gate_split",
      rewriter.getStringAttr("lstm_cell_gate_split"));

  mlir::Block *body = new mlir::Block();
  gateSplitRegion.getBody().push_back(body);
  body->addArgument(preActivation.getType(), preActivation.getLoc());

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);
  int64_t hiddenSize = match.hiddenTy.getShape()[1];
  mlir::RankedTensorType resultTy = match.resultHTy;
  mlir::Value regionPreActivation = body->getArgument(0);
  mlir::Value iSlice =
      recurrent_gate::buildGateSlice(loc, resultTy, regionPreActivation,
                                     /*gateOffset=*/0, hiddenSize, rewriter);
  mlir::Value fSlice = recurrent_gate::buildGateSlice(
      loc, resultTy, regionPreActivation, hiddenSize, hiddenSize, rewriter);
  mlir::Value gSlice = recurrent_gate::buildGateSlice(
      loc, resultTy, regionPreActivation, hiddenSize * 2, hiddenSize, rewriter);
  mlir::Value oSlice = recurrent_gate::buildGateSlice(
      loc, resultTy, regionPreActivation, hiddenSize * 3, hiddenSize, rewriter);

  rewriter.create<mlir::sculptor::YieldOp>(
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

static LSTMGateActivations
buildGateActivationRegion(LSTMCellLowering &match, LSTMGateSlices gates,
                          mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.lstmCellOp.getLoc();
  rewriter.setInsertionPoint(match.lstmCellOp);
  auto activationRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc,
      mlir::TypeRange{match.resultHTy, match.resultHTy, match.resultHTy,
                      match.resultHTy},
      mlir::ValueRange{gates.i, gates.f, gates.g, gates.o},
      "digital.activation",
      rewriter.getStringAttr("lstm_cell_gate_activation"));

  mlir::Block *body = new mlir::Block();
  activationRegion.getBody().push_back(body);
  llvm::SmallVector<mlir::Type> inputTypes = {
      gates.i.getType(), gates.f.getType(), gates.g.getType(),
      gates.o.getType()};
  llvm::SmallVector<mlir::Location> inputLocs = {
      gates.i.getLoc(), gates.f.getLoc(), gates.g.getLoc(), gates.o.getLoc()};
  body->addArguments(inputTypes, inputLocs);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);
  mlir::RankedTensorType resultTy = match.resultHTy;
  mlir::Value inputGate = converter_recurrent_elementwise::buildSigmoid(
      loc, resultTy, body->getArgument(0), rewriter);
  mlir::Value forgetGate = converter_recurrent_elementwise::buildSigmoid(
      loc, resultTy, body->getArgument(1), rewriter);
  mlir::Value candidateGate = converter_recurrent_elementwise::buildTanh(
      loc, resultTy, body->getArgument(2), rewriter);
  mlir::Value outputGate = converter_recurrent_elementwise::buildSigmoid(
      loc, resultTy, body->getArgument(3), rewriter);

  rewriter.create<mlir::sculptor::YieldOp>(
      loc, mlir::ValueRange{inputGate, forgetGate, candidateGate, outputGate});
  return LSTMGateActivations{
      activationRegion.getResult(0), activationRegion.getResult(1),
      activationRegion.getResult(2), activationRegion.getResult(3)};
}

static mlir::Value buildCellUpdateRegion(LSTMCellLowering &match,
                                         LSTMGateActivations gates,
                                         mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.lstmCellOp.getLoc();
  rewriter.setInsertionPoint(match.lstmCellOp);
  auto cellUpdateRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.resultCTy},
      mlir::ValueRange{gates.forget, match.lstmCellOp.getCPrev(), gates.input,
                       gates.candidate},
      "digital.cell_update", rewriter.getStringAttr("lstm_cell_cell_update"));

  mlir::Block *body = new mlir::Block();
  cellUpdateRegion.getBody().push_back(body);
  llvm::SmallVector<mlir::Type> inputTypes = {
      gates.forget.getType(), match.lstmCellOp.getCPrev().getType(),
      gates.input.getType(), gates.candidate.getType()};
  llvm::SmallVector<mlir::Location> inputLocs = {
      gates.forget.getLoc(), match.lstmCellOp.getCPrev().getLoc(),
      gates.input.getLoc(), gates.candidate.getLoc()};
  body->addArguments(inputTypes, inputLocs);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);
  mlir::RankedTensorType resultTy = match.resultCTy;
  mlir::Value forgetCell = converter_recurrent_elementwise::buildMul(
      loc, resultTy, body->getArgument(0), body->getArgument(1), rewriter);
  mlir::Value inputCandidate = converter_recurrent_elementwise::buildMul(
      loc, resultTy, body->getArgument(2), body->getArgument(3), rewriter);
  mlir::Value nextCell = converter_recurrent_elementwise::buildAdd(
      loc, resultTy, forgetCell, inputCandidate, rewriter);

  rewriter.create<mlir::sculptor::YieldOp>(loc, nextCell);
  return cellUpdateRegion.getResult(0);
}

static mlir::Value buildHiddenUpdateRegion(LSTMCellLowering &match,
                                           mlir::Value outputGate,
                                           mlir::Value nextCell,
                                           mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.lstmCellOp.getLoc();
  rewriter.setInsertionPoint(match.lstmCellOp);
  auto hiddenUpdateRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.resultHTy},
      mlir::ValueRange{outputGate, nextCell}, "digital.hidden_update",
      rewriter.getStringAttr("lstm_cell_hidden_update"));

  mlir::Block *body = new mlir::Block();
  hiddenUpdateRegion.getBody().push_back(body);
  llvm::SmallVector<mlir::Type> inputTypes = {outputGate.getType(),
                                              nextCell.getType()};
  llvm::SmallVector<mlir::Location> inputLocs = {outputGate.getLoc(),
                                                 nextCell.getLoc()};
  body->addArguments(inputTypes, inputLocs);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);
  mlir::RankedTensorType resultTy = match.resultHTy;
  mlir::Value tanhCell = converter_recurrent_elementwise::buildTanh(
      loc, resultTy, body->getArgument(1), rewriter);
  mlir::Value nextHidden = converter_recurrent_elementwise::buildMul(
      loc, resultTy, body->getArgument(0), tanhCell, rewriter);

  rewriter.create<mlir::sculptor::YieldOp>(loc, nextHidden);
  return hiddenUpdateRegion.getResult(0);
}

struct LSTMCellResults {
  mlir::Value h;
  mlir::Value c;
};

// Applies LSTMCell gate order i, f, g, o and state updates.
static LSTMCellResults buildGateMath(LSTMCellLowering &match,
                                     mlir::Value preActivation,
                                     mlir::RewriterBase &rewriter) {
  LSTMGateSlices slices = buildGateSplitRegion(match, preActivation, rewriter);
  LSTMGateActivations gates =
      buildGateActivationRegion(match, slices, rewriter);
  mlir::Value nextCell = buildCellUpdateRegion(match, gates, rewriter);
  mlir::Value nextHidden =
      buildHiddenUpdateRegion(match, gates.output, nextCell, rewriter);
  return LSTMCellResults{nextHidden, nextCell};
}

static mlir::LogicalResult
lowerLSTMCellLayerToMVM(mlir::func::FuncOp func, mlir::RewriterBase &rewriter) {
  auto match = matchExtractedLSTMCellLayer(func);
  if (mlir::failed(match))
    return mlir::failure();

  mlir::TypedAttr fusedWeightAttr = buildFusedWeightAttr(*match);
  if (!fusedWeightAttr)
    return mlir::failure();

  mlir::TypedAttr fusedBiasAttr;
  if ((*match).hasBias) {
    fusedBiasAttr = buildFusedBiasAttr(*match);
    if (!fusedBiasAttr)
      return mlir::failure();
  }

  mlir::Location loc = (*match).lstmCellOp.getLoc();
  rewriter.setInsertionPoint((*match).lstmCellOp);
  auto fusedWeight =
      rewriter.create<ConstantOp>(loc, (*match).fusedWeightTy, fusedWeightAttr);
  mlir::Value fusedInput = buildInputRecombineRegion(*match, rewriter);
  mlir::Value mvmResult =
      mvm_build::buildMVM(loc, (*match).preActivationTy, fusedInput,
                          fusedWeight.getResult(), rewriter);

  auto preActivation =
      buildBiasAddRegion(*match, fusedBiasAttr, mvmResult, rewriter);
  if (mlir::failed(preActivation))
    return mlir::failure();

  LSTMCellResults results = buildGateMath(*match, *preActivation, rewriter);
  (*match).lstmCellOp.getH().replaceAllUsesWith(results.h);
  (*match).lstmCellOp.getC().replaceAllUsesWith(results.c);
  rewriter.eraseOp((*match).lstmCellOp);
  converter_recurrent_layer::eraseUnusedConstants(
      {match->weightIHConstant, match->weightHHConstant, match->biasIHConstant,
       match->biasHHConstant},
      rewriter);
  return mlir::success();
}

// Converts extracted sculptor.nn.lstm_cell layer bodies to fused sculptor.mvm plus
// gate math.
class LSTMCellConverter : public mlir::sculptor::LayerToMVMConverter {
public:
  mlir::StringRef getName() const override { return "lstm_cell"; }

  void lowerToMVM(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());
    (void)lowerLSTMCellLayerToMVM(func, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerLSTMCellConverter(LayerToMVMConverters &converters,
                               LayerToMVMConverterMap &converterMap,
                               MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<LSTMCellConverter>();
  const LayerToMVMConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["lstm_cell"] = converterPtr;
  converterMap["lstm_cell_w_bias"] = converterPtr;
}

} // namespace sculptor
} // namespace mlir
