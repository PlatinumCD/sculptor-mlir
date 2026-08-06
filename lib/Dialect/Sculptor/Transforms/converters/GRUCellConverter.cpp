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
matchGRUCellLayer(NNGRUCellOp op) {
  mlir::FailureOr<NNGRUCellOp> gruCellOp = op;
  bool hasBias = (*gruCellOp).getHasBias();

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

static mlir::FailureOr<GRUCellLowering>
matchExtractedGRUCellLayer(mlir::func::FuncOp func) {
  auto gruCellOp = nn_layer_match::matchSingleNNLayerOp<NNGRUCellOp>(func);
  if (mlir::failed(gruCellOp))
    return mlir::failure();

  bool hasBias = (*gruCellOp).getHasBias();
  if (!nn_layer_match::hasLayerTypeMatchingBias(func, "gru_cell",
                                                "gru_cell_w_bias", hasBias) ||
      func.getNumArguments() != 2 ||
      (*gruCellOp).getInput() != func.getArgument(0) ||
      (*gruCellOp).getHPrev() != func.getArgument(1))
    return mlir::failure();
  return matchGRUCellLayer(*gruCellOp);
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

static mlir::Value buildInputRecombineStage(GRUCellLowering &match,
                                             mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  mlir::sculptor::SemanticOperationScope scope(
      rewriter, "digital.input_recombine", "gru_cell_input_recombine");
  mlir::Value fusedInput = recurrent_gate::buildFusedInput(
      loc, match.fusedInputTy, match.gruCellOp.getInput(),
      match.gruCellOp.getHPrev(), rewriter);
  scope.annotate();
  return fusedInput;
}

static mlir::FailureOr<mlir::Value>
buildBiasAddStage(GRUCellLowering &match, mlir::TypedAttr fusedBiasAttr,
                   mlir::Value mvmResult, mlir::RewriterBase &rewriter) {
  if (match.hasBias && !fusedBiasAttr)
    return mlir::failure();

  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  mlir::Value biasResult = mvmResult;
  if (!match.hasBias)
    return biasResult;
  mlir::sculptor::SemanticOperationScope scope(
      rewriter, "digital.bias_add", "gru_cell_bias_add");

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
              loc, mlir::ValueRange{mvmResult, expandedBias},
              mlir::ValueRange{biasedInit})
          .getResult(0);
  scope.annotate();
  return biasResult;
}

struct GRUGateSlices {
  mlir::Value reset;
  mlir::Value update;
  mlir::Value inputNew;
  mlir::Value hiddenNew;
};

static GRUGateSlices buildGateSplitStage(GRUCellLowering &match,
                                          mlir::Value preActivation,
                                          mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  mlir::sculptor::SemanticOperationScope scope(
      rewriter, "digital.gate_split", "gru_cell_gate_split");
  int64_t hiddenSize = match.hiddenTy.getShape()[1];
  mlir::RankedTensorType resultTy = match.resultTy;
  mlir::Value resetPre =
      recurrent_gate::buildGateSlice(loc, resultTy, preActivation,
                                     /*gateOffset=*/0, hiddenSize, rewriter);
  mlir::Value updatePre = recurrent_gate::buildGateSlice(
      loc, resultTy, preActivation, hiddenSize, hiddenSize, rewriter);
  mlir::Value inputNew = recurrent_gate::buildGateSlice(
      loc, resultTy, preActivation, hiddenSize * 2, hiddenSize, rewriter);
  mlir::Value hiddenNew = recurrent_gate::buildGateSlice(
      loc, resultTy, preActivation, hiddenSize * 3, hiddenSize, rewriter);
  scope.annotate();
  return GRUGateSlices{resetPre, updatePre, inputNew, hiddenNew};
}

struct GRUGateActivations {
  mlir::Value reset;
  mlir::Value update;
};

static GRUGateActivations
buildGateActivationStage(GRUCellLowering &match, GRUGateSlices gates,
                          mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  mlir::sculptor::SemanticOperationScope scope(
      rewriter, "digital.activation", "gru_cell_gate_activation");
  mlir::RankedTensorType resultTy = match.resultTy;
  mlir::Value resetGate = converter_recurrent_elementwise::buildSigmoid(
      loc, resultTy, gates.reset, rewriter);
  mlir::Value updateGate = converter_recurrent_elementwise::buildSigmoid(
      loc, resultTy, gates.update, rewriter);
  scope.annotate();
  return GRUGateActivations{resetGate, updateGate};
}

static mlir::Value buildCandidateUpdateStage(GRUCellLowering &match,
                                              mlir::Value resetGate,
                                              mlir::Value inputNew,
                                              mlir::Value hiddenNew,
                                              mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  mlir::sculptor::SemanticOperationScope scope(
      rewriter, "digital.candidate_update", "gru_cell_candidate_update");
  mlir::RankedTensorType resultTy = match.resultTy;
  mlir::Value resetHiddenNew = converter_recurrent_elementwise::buildMul(
      loc, resultTy, resetGate, hiddenNew, rewriter);
  mlir::Value candidateInput = converter_recurrent_elementwise::buildAdd(
      loc, resultTy, inputNew, resetHiddenNew, rewriter);
  mlir::Value candidate = converter_recurrent_elementwise::buildTanh(
      loc, resultTy, candidateInput, rewriter);
  scope.annotate();
  return candidate;
}

static mlir::Value buildHiddenUpdateStage(GRUCellLowering &match,
                                           mlir::Value candidate,
                                           mlir::Value updateGate,
                                           mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.gruCellOp.getLoc();
  rewriter.setInsertionPoint(match.gruCellOp);
  mlir::sculptor::SemanticOperationScope scope(
      rewriter, "digital.hidden_update", "gru_cell_hidden_update");
  mlir::RankedTensorType resultTy = match.resultTy;
  mlir::Value hiddenMinusCandidate = converter_recurrent_elementwise::buildSub(
      loc, resultTy, match.gruCellOp.getHPrev(), candidate, rewriter);
  mlir::Value updateDelta = converter_recurrent_elementwise::buildMul(
      loc, resultTy, updateGate, hiddenMinusCandidate, rewriter);
  mlir::Value nextHidden = converter_recurrent_elementwise::buildAdd(
      loc, resultTy, candidate, updateDelta, rewriter);
  scope.annotate();
  return nextHidden;
}

// Applies GRUCell gates while preserving the reset-new dependency.
static mlir::Value buildGateMath(GRUCellLowering &match,
                                 mlir::Value preActivation,
                                 mlir::RewriterBase &rewriter) {
  GRUGateSlices slices = buildGateSplitStage(match, preActivation, rewriter);
  GRUGateActivations gates = buildGateActivationStage(match, slices, rewriter);
  mlir::Value candidate = buildCandidateUpdateStage(
      match, gates.reset, slices.inputNew, slices.hiddenNew, rewriter);
  return buildHiddenUpdateStage(match, candidate, gates.update, rewriter);
}

static mlir::LogicalResult
lowerGRUCellOp(NNGRUCellOp op, mlir::RewriterBase &rewriter) {
  auto match = matchGRUCellLayer(op);
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
  mlir::Value fusedInput = buildInputRecombineStage(*match, rewriter);
  mlir::Value mvmResult =
      mvm_build::buildMVM(loc, (*match).preActivationTy, fusedInput,
                          fusedWeight.getResult(), rewriter);

  auto preActivation =
      buildBiasAddStage(*match, fusedBiasAttr, mvmResult, rewriter);
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

static mlir::LogicalResult
lowerGRUCellLayerToMVM(mlir::func::FuncOp func,
                       mlir::RewriterBase &rewriter) {
  auto match = matchExtractedGRUCellLayer(func);
  if (mlir::failed(match))
    return mlir::failure();
  return lowerGRUCellOp(match->gruCellOp, rewriter);
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

LogicalResult decomposeInlineGRUCellLayers(func::FuncOp func) {
  SmallVector<NNGRUCellOp> ops;
  func.walk([&](NNGRUCellOp op) { ops.push_back(op); });

  IRRewriter rewriter(func.getContext());
  for (NNGRUCellOp op : ops) {
    if (failed(lowerGRUCellOp(op, rewriter))) {
      op.emitOpError("cannot decompose supported inline GRUCell");
      return failure();
    }
  }
  return success();
}

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
