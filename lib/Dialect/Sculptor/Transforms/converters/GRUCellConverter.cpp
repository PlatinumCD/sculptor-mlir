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

using mlir::sculptor::NNGRUCellOp;
using mlir::arith::ConstantOp;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;

struct GRUCellLowering {
  NNGRUCellOp gruCellOp;
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType hiddenTy;
  mlir::RankedTensorType weightIHTy;
  mlir::RankedTensorType weightHHTy;
  mlir::RankedTensorType resultTy;
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
buildFusedWeightType(mlir::RankedTensorType inputTy,
                     mlir::RankedTensorType hiddenTy) {
  int64_t hiddenSize = hiddenTy.getShape()[1];
  return mlir::RankedTensorType::get(
      {hiddenSize * 4, inputTy.getShape()[1] + hiddenSize},
      inputTy.getElementType());
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

static mlir::FailureOr<GRUCellLowering>
matchExtractedGRUCellLayer(mlir::func::FuncOp func) {
  auto gruCellOp = nn_layer_match::matchSingleNNLayerOp<NNGRUCellOp>(func);
  if (mlir::failed(gruCellOp))
    return mlir::failure();

  bool hasBias = (*gruCellOp).getHasBias();
  if (!nn_layer_match::hasLayerTypeMatchingBias(func, "gru_cell",
                                                "gru_cell_w_bias", hasBias))
    return mlir::failure();

  if (func.getNumArguments() != 2 ||
      (*gruCellOp).getInput() != func.getArgument(0) ||
      (*gruCellOp).getHPrev() != func.getArgument(1))
    return mlir::failure();

  auto inputTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*gruCellOp).getInput().getType());
  auto hiddenTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*gruCellOp).getHPrev().getType());
  auto weightIHTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*gruCellOp).getWIh().getType());
  auto weightHHTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*gruCellOp).getWHh().getType());
  auto resultTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*gruCellOp).getH().getType());
  if (mlir::failed(inputTy) || mlir::failed(hiddenTy) ||
      mlir::failed(weightIHTy) || mlir::failed(weightHHTy) ||
      mlir::failed(resultTy))
    return mlir::failure();

  int64_t inputSize = (*inputTy).getShape()[1];
  int64_t hiddenSize = (*hiddenTy).getShape()[1];
  if ((*inputTy).getShape()[0] != 1 || (*hiddenTy).getShape()[0] != 1 ||
      (*resultTy).getShape()[0] != 1 || (*resultTy).getShape()[1] != hiddenSize)
    return mlir::failure();

  if ((*weightIHTy).getShape()[0] != hiddenSize * 3 ||
      (*weightIHTy).getShape()[1] != inputSize ||
      (*weightHHTy).getShape()[0] != hiddenSize * 3 ||
      (*weightHHTy).getShape()[1] != hiddenSize)
    return mlir::failure();

  auto weightIHConstant = (*gruCellOp).getWIh().getDefiningOp<ConstantOp>();
  auto weightHHConstant = (*gruCellOp).getWHh().getDefiningOp<ConstantOp>();
  if (!weightIHConstant || !weightHHConstant)
    return mlir::failure();

  ConstantOp biasIHConstant;
  ConstantOp biasHHConstant;
  if (hasBias) {
    mlir::Value biasIH = (*gruCellOp).getBIh();
    mlir::Value biasHH = (*gruCellOp).getBHh();
    if (!biasIH || !biasHH)
      return mlir::failure();

    auto biasIHTy =
        tensor_type::getPositiveStaticRank1F32Tensor(biasIH.getType());
    auto biasHHTy =
        tensor_type::getPositiveStaticRank1F32Tensor(biasHH.getType());
    if (mlir::failed(biasIHTy) || mlir::failed(biasHHTy) ||
        (*biasIHTy).getShape()[0] != hiddenSize * 3 ||
        (*biasHHTy).getShape()[0] != hiddenSize * 3)
      return mlir::failure();

    biasIHConstant = biasIH.getDefiningOp<ConstantOp>();
    biasHHConstant = biasHH.getDefiningOp<ConstantOp>();
    if (!biasIHConstant || !biasHHConstant)
      return mlir::failure();
  } else if ((*gruCellOp).getBIh() || (*gruCellOp).getBHh()) {
    return mlir::failure();
  }

  GRUCellLowering lowering;
  lowering.gruCellOp = *gruCellOp;
  lowering.inputTy = *inputTy;
  lowering.hiddenTy = *hiddenTy;
  lowering.weightIHTy = *weightIHTy;
  lowering.weightHHTy = *weightHHTy;
  lowering.resultTy = *resultTy;
  lowering.fusedInputTy =
      recurrent_gate::buildFusedInputType(*inputTy, *hiddenTy);
  lowering.fusedWeightTy = buildFusedWeightType(*inputTy, *hiddenTy);
  lowering.fusedBiasTy = buildFusedBiasType(*hiddenTy);
  lowering.preActivationTy = buildPreActivationType(*hiddenTy);
  lowering.weightIHConstant = weightIHConstant;
  lowering.weightHHConstant = weightHHConstant;
  lowering.biasIHConstant = biasIHConstant;
  lowering.biasHHConstant = biasHHConstant;
  lowering.hasBias = hasBias;
  return lowering;
}

// Packs GRUCell reset, update, input-new, and hidden-new projections.
static mlir::TypedAttr buildFusedWeightAttr(GRUCellLowering &match) {
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

  int64_t inputSize = match.inputTy.getShape()[1];
  int64_t hiddenSize = match.hiddenTy.getShape()[1];
  int64_t fusedWidth = match.fusedWeightTy.getShape()[1];
  llvm::SmallVector<float> fusedWeights(match.fusedWeightTy.getNumElements(),
                                        0.0f);

  for (int64_t row = 0; row < hiddenSize; ++row) {
    int64_t resetInputOffset = row * inputSize;
    int64_t resetHiddenOffset = row * hiddenSize;
    int64_t resetFusedOffset = row * fusedWidth;
    for (int64_t col = 0; col < inputSize; ++col)
      fusedWeights[resetFusedOffset + col] =
          inputWeights[resetInputOffset + col];
    for (int64_t col = 0; col < hiddenSize; ++col)
      fusedWeights[resetFusedOffset + inputSize + col] =
          hiddenWeights[resetHiddenOffset + col];

    int64_t updateInputOffset = (hiddenSize + row) * inputSize;
    int64_t updateHiddenOffset = (hiddenSize + row) * hiddenSize;
    int64_t updateFusedOffset = (hiddenSize + row) * fusedWidth;
    for (int64_t col = 0; col < inputSize; ++col)
      fusedWeights[updateFusedOffset + col] =
          inputWeights[updateInputOffset + col];
    for (int64_t col = 0; col < hiddenSize; ++col)
      fusedWeights[updateFusedOffset + inputSize + col] =
          hiddenWeights[updateHiddenOffset + col];

    int64_t inputNewOffset = (hiddenSize * 2 + row) * inputSize;
    int64_t inputNewFusedOffset = (hiddenSize * 2 + row) * fusedWidth;
    for (int64_t col = 0; col < inputSize; ++col)
      fusedWeights[inputNewFusedOffset + col] =
          inputWeights[inputNewOffset + col];

    int64_t hiddenNewOffset = (hiddenSize * 2 + row) * hiddenSize;
    int64_t hiddenNewFusedOffset =
        (hiddenSize * 3 + row) * fusedWidth + inputSize;
    for (int64_t col = 0; col < hiddenSize; ++col)
      fusedWeights[hiddenNewFusedOffset + col] =
          hiddenWeights[hiddenNewOffset + col];
  }

  bool useResource =
      converter_constant::isResourceBackedF32Constant(match.weightIHConstant) ||
      converter_constant::isResourceBackedF32Constant(match.weightHHConstant);
  return converter_constant::buildF32ElementsAttr(
      match.fusedWeightTy, fusedWeights, "analog_gru_cell_fused_weight_",
      useResource);
}

// Keeps GRUCell input-new and hidden-new bias terms separate.
static mlir::TypedAttr buildFusedBiasAttr(GRUCellLowering &match) {
  auto maybeInputBias =
      converter_constant::getF32ConstantValues(match.biasIHConstant);
  auto maybeHiddenBias =
      converter_constant::getF32ConstantValues(match.biasHHConstant);
  if (mlir::failed(maybeInputBias) || mlir::failed(maybeHiddenBias))
    return {};

  llvm::SmallVector<float> inputBias = *maybeInputBias;
  llvm::SmallVector<float> hiddenBias = *maybeHiddenBias;
  int64_t gateRows = match.hiddenTy.getShape()[1] * 3;
  if (static_cast<int64_t>(inputBias.size()) != gateRows ||
      static_cast<int64_t>(hiddenBias.size()) != gateRows)
    return {};

  llvm::SmallVector<float> fusedBias(match.fusedBiasTy.getNumElements(), 0.0f);
  int64_t hiddenSize = match.hiddenTy.getShape()[1];
  for (int64_t index = 0; index < hiddenSize; ++index) {
    fusedBias[index] = inputBias[index] + hiddenBias[index];
    fusedBias[hiddenSize + index] =
        inputBias[hiddenSize + index] + hiddenBias[hiddenSize + index];
    fusedBias[hiddenSize * 2 + index] = inputBias[hiddenSize * 2 + index];
    fusedBias[hiddenSize * 3 + index] = hiddenBias[hiddenSize * 2 + index];
  }

  bool useResource =
      converter_constant::isResourceBackedF32Constant(match.biasIHConstant) ||
      converter_constant::isResourceBackedF32Constant(match.biasHHConstant);
  return converter_constant::buildF32ElementsAttr(
      match.fusedBiasTy, fusedBias, "analog_gru_cell_fused_bias_", useResource);
}

static mlir::Value buildInputRecombineRegion(GRUCellLowering &match,
                                             mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  auto recombineRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.fusedInputTy},
      mlir::ValueRange{match.gruCellOp.getInput(), match.gruCellOp.getHPrev()},
      "digital.input_recombine",
      rewriter.getStringAttr("gru_cell_input_recombine"));

  mlir::Block *body = new mlir::Block();
  recombineRegion.getBody().push_back(body);
  llvm::SmallVector<mlir::Type> inputTypes = {
      match.gruCellOp.getInput().getType(),
      match.gruCellOp.getHPrev().getType()};
  llvm::SmallVector<mlir::Location> inputLocs = {
      match.gruCellOp.getInput().getLoc(), match.gruCellOp.getHPrev().getLoc()};
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
buildBiasAddRegion(GRUCellLowering &match, mlir::TypedAttr fusedBiasAttr,
                   mlir::Value mvmResult, mlir::RewriterBase &rewriter) {
  if (match.hasBias && !fusedBiasAttr)
    return mlir::failure();

  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  auto biasRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.preActivationTy}, mlir::ValueRange{mvmResult},
      "digital.bias_add", rewriter.getStringAttr("gru_cell_bias_add"));

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

struct GRUGateSlices {
  mlir::Value reset;
  mlir::Value update;
  mlir::Value inputNew;
  mlir::Value hiddenNew;
};

static GRUGateSlices buildGateSplitRegion(GRUCellLowering &match,
                                          mlir::Value preActivation,
                                          mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  auto gateSplitRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc,
      mlir::TypeRange{match.resultTy, match.resultTy, match.resultTy,
                      match.resultTy},
      mlir::ValueRange{preActivation}, "digital.gate_split",
      rewriter.getStringAttr("gru_cell_gate_split"));

  mlir::Block *body = new mlir::Block();
  gateSplitRegion.getBody().push_back(body);
  body->addArgument(preActivation.getType(), preActivation.getLoc());

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);
  int64_t hiddenSize = match.hiddenTy.getShape()[1];
  mlir::RankedTensorType resultTy = match.resultTy;
  mlir::Value regionPreActivation = body->getArgument(0);
  mlir::Value resetPre =
      recurrent_gate::buildGateSlice(loc, resultTy, regionPreActivation,
                                     /*gateOffset=*/0, hiddenSize, rewriter);
  mlir::Value updatePre = recurrent_gate::buildGateSlice(
      loc, resultTy, regionPreActivation, hiddenSize, hiddenSize, rewriter);
  mlir::Value inputNew = recurrent_gate::buildGateSlice(
      loc, resultTy, regionPreActivation, hiddenSize * 2, hiddenSize, rewriter);
  mlir::Value hiddenNew = recurrent_gate::buildGateSlice(
      loc, resultTy, regionPreActivation, hiddenSize * 3, hiddenSize, rewriter);

  rewriter.create<mlir::sculptor::YieldOp>(
      loc, mlir::ValueRange{resetPre, updatePre, inputNew, hiddenNew});
  return GRUGateSlices{
      gateSplitRegion.getResult(0), gateSplitRegion.getResult(1),
      gateSplitRegion.getResult(2), gateSplitRegion.getResult(3)};
}

struct GRUGateActivations {
  mlir::Value reset;
  mlir::Value update;
};

static GRUGateActivations
buildGateActivationRegion(GRUCellLowering &match, GRUGateSlices gates,
                          mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  auto activationRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.resultTy, match.resultTy},
      mlir::ValueRange{gates.reset, gates.update}, "digital.activation",
      rewriter.getStringAttr("gru_cell_gate_activation"));

  mlir::Block *body = new mlir::Block();
  activationRegion.getBody().push_back(body);
  llvm::SmallVector<mlir::Type> inputTypes = {gates.reset.getType(),
                                              gates.update.getType()};
  llvm::SmallVector<mlir::Location> inputLocs = {gates.reset.getLoc(),
                                                 gates.update.getLoc()};
  body->addArguments(inputTypes, inputLocs);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);
  mlir::RankedTensorType resultTy = match.resultTy;
  mlir::Value resetGate = converter_recurrent_elementwise::buildSigmoid(
      loc, resultTy, body->getArgument(0), rewriter);
  mlir::Value updateGate = converter_recurrent_elementwise::buildSigmoid(
      loc, resultTy, body->getArgument(1), rewriter);

  rewriter.create<mlir::sculptor::YieldOp>(
      loc, mlir::ValueRange{resetGate, updateGate});
  return GRUGateActivations{activationRegion.getResult(0),
                            activationRegion.getResult(1)};
}

static mlir::Value buildCandidateUpdateRegion(GRUCellLowering &match,
                                              mlir::Value resetGate,
                                              mlir::Value inputNew,
                                              mlir::Value hiddenNew,
                                              mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  auto candidateRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.resultTy},
      mlir::ValueRange{resetGate, inputNew, hiddenNew},
      "digital.candidate_update",
      rewriter.getStringAttr("gru_cell_candidate_update"));

  mlir::Block *body = new mlir::Block();
  candidateRegion.getBody().push_back(body);
  llvm::SmallVector<mlir::Type> inputTypes = {
      resetGate.getType(), inputNew.getType(), hiddenNew.getType()};
  llvm::SmallVector<mlir::Location> inputLocs = {
      resetGate.getLoc(), inputNew.getLoc(), hiddenNew.getLoc()};
  body->addArguments(inputTypes, inputLocs);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);
  mlir::RankedTensorType resultTy = match.resultTy;
  mlir::Value resetHiddenNew = converter_recurrent_elementwise::buildMul(
      loc, resultTy, body->getArgument(0), body->getArgument(2), rewriter);
  mlir::Value candidateInput = converter_recurrent_elementwise::buildAdd(
      loc, resultTy, body->getArgument(1), resetHiddenNew, rewriter);
  mlir::Value candidate = converter_recurrent_elementwise::buildTanh(
      loc, resultTy, candidateInput, rewriter);

  rewriter.create<mlir::sculptor::YieldOp>(loc, candidate);
  return candidateRegion.getResult(0);
}

static mlir::Value buildHiddenUpdateRegion(GRUCellLowering &match,
                                           mlir::Value candidate,
                                           mlir::Value updateGate,
                                           mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  auto hiddenUpdateRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.resultTy},
      mlir::ValueRange{candidate, updateGate, match.gruCellOp.getHPrev()},
      "digital.hidden_update",
      rewriter.getStringAttr("gru_cell_hidden_update"));

  mlir::Block *body = new mlir::Block();
  hiddenUpdateRegion.getBody().push_back(body);
  llvm::SmallVector<mlir::Type> inputTypes = {
      candidate.getType(), updateGate.getType(),
      match.gruCellOp.getHPrev().getType()};
  llvm::SmallVector<mlir::Location> inputLocs = {
      candidate.getLoc(), updateGate.getLoc(),
      match.gruCellOp.getHPrev().getLoc()};
  body->addArguments(inputTypes, inputLocs);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);
  mlir::RankedTensorType resultTy = match.resultTy;
  mlir::Value hiddenMinusCandidate = converter_recurrent_elementwise::buildSub(
      loc, resultTy, body->getArgument(2), body->getArgument(0), rewriter);
  mlir::Value updateDelta = converter_recurrent_elementwise::buildMul(
      loc, resultTy, body->getArgument(1), hiddenMinusCandidate, rewriter);
  mlir::Value nextHidden = converter_recurrent_elementwise::buildAdd(
      loc, resultTy, body->getArgument(0), updateDelta, rewriter);

  rewriter.create<mlir::sculptor::YieldOp>(loc, nextHidden);
  return hiddenUpdateRegion.getResult(0);
}

// Applies GRUCell gates while preserving the reset-new dependency.
static mlir::Value buildGateMath(GRUCellLowering &match,
                                 mlir::Value preActivation,
                                 mlir::RewriterBase &rewriter) {
  GRUGateSlices slices = buildGateSplitRegion(match, preActivation, rewriter);
  GRUGateActivations gates = buildGateActivationRegion(match, slices, rewriter);
  mlir::Value candidate = buildCandidateUpdateRegion(
      match, gates.reset, slices.inputNew, slices.hiddenNew, rewriter);
  return buildHiddenUpdateRegion(match, candidate, gates.update, rewriter);
}

static mlir::LogicalResult
lowerGRUCellLayerToMVM(mlir::func::FuncOp func, mlir::RewriterBase &rewriter) {
  auto match = matchExtractedGRUCellLayer(func);
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

  mlir::Location loc = (*match).gruCellOp.getLoc();
  rewriter.setInsertionPoint((*match).gruCellOp);
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

  mlir::Value result = buildGateMath(*match, *preActivation, rewriter);
  (*match).gruCellOp.getH().replaceAllUsesWith(result);
  rewriter.eraseOp((*match).gruCellOp);
  converter_recurrent_layer::eraseUnusedConstants(
      {match->weightIHConstant, match->weightHHConstant, match->biasIHConstant,
       match->biasHHConstant},
      rewriter);
  return mlir::success();
}

// Converts extracted sculptor.nn.gru_cell layer bodies to fused sculptor.mvm plus
// gate math.
class GRUCellConverter : public mlir::sculptor::LayerToMVMConverter {
public:
  mlir::StringRef getName() const override { return "gru_cell"; }

  void lowerToMVM(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());
    (void)lowerGRUCellLayerToMVM(func, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerGRUCellConverter(LayerToMVMConverters &converters,
                              LayerToMVMConverterMap &converterMap,
                              MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<GRUCellConverter>();
  const LayerToMVMConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["gru_cell"] = converterPtr;
  converterMap["gru_cell_w_bias"] = converterPtr;
}

} // namespace sculptor
} // namespace mlir
