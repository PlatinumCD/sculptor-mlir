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

using mlir::sculptor::NNRNNCellOp;
using mlir::arith::ConstantOp;
using mlir::tensor::EmptyOp;
using mlir::tensor::ExpandShapeOp;

struct RNNCellLowering {
  NNRNNCellOp rnnCellOp;
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType hiddenTy;
  mlir::RankedTensorType weightIHTy;
  mlir::RankedTensorType weightHHTy;
  mlir::RankedTensorType resultTy;
  mlir::RankedTensorType fusedInputTy;
  mlir::RankedTensorType fusedWeightTy;
  mlir::RankedTensorType fusedBiasTy;
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
buildFusedBiasType(mlir::RankedTensorType resultTy) {
  return mlir::RankedTensorType::get({resultTy.getShape()[1]},
                                     resultTy.getElementType());
}

static mlir::FailureOr<RNNCellLowering>
matchExtractedRNNCellLayer(mlir::func::FuncOp func) {
  auto rnnCellOp = nn_layer_match::matchSingleNNLayerOp<NNRNNCellOp>(func);
  if (mlir::failed(rnnCellOp))
    return mlir::failure();

  bool hasBias = (*rnnCellOp).getHasBias();
  if (!nn_layer_match::hasLayerTypeMatchingBias(func, "rnn_cell",
                                                "rnn_cell_w_bias", hasBias))
    return mlir::failure();

  if ((*rnnCellOp).getActivationAttr().getValue() != "tanh")
    return mlir::failure();

  if (func.getNumArguments() != 2 ||
      (*rnnCellOp).getInput() != func.getArgument(0) ||
      (*rnnCellOp).getHPrev() != func.getArgument(1))
    return mlir::failure();

  auto inputTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*rnnCellOp).getInput().getType());
  auto hiddenTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*rnnCellOp).getHPrev().getType());
  auto weightIHTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*rnnCellOp).getWIh().getType());
  auto weightHHTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*rnnCellOp).getWHh().getType());
  auto resultTy = tensor_type::getPositiveStaticRank2F32Tensor(
      (*rnnCellOp).getH().getType());
  if (mlir::failed(inputTy) || mlir::failed(hiddenTy) ||
      mlir::failed(weightIHTy) || mlir::failed(weightHHTy) ||
      mlir::failed(resultTy))
    return mlir::failure();

  int64_t inputSize = (*inputTy).getShape()[1];
  int64_t hiddenSize = (*hiddenTy).getShape()[1];
  if ((*inputTy).getShape()[0] != 1 || (*hiddenTy).getShape()[0] != 1 ||
      (*resultTy).getShape()[0] != 1 || (*resultTy).getShape()[1] != hiddenSize)
    return mlir::failure();

  if ((*weightIHTy).getShape()[0] != hiddenSize ||
      (*weightIHTy).getShape()[1] != inputSize ||
      (*weightHHTy).getShape()[0] != hiddenSize ||
      (*weightHHTy).getShape()[1] != hiddenSize)
    return mlir::failure();

  auto weightIHConstant = (*rnnCellOp).getWIh().getDefiningOp<ConstantOp>();
  auto weightHHConstant = (*rnnCellOp).getWHh().getDefiningOp<ConstantOp>();
  if (!weightIHConstant || !weightHHConstant)
    return mlir::failure();

  ConstantOp biasIHConstant;
  ConstantOp biasHHConstant;
  if (hasBias) {
    mlir::Value biasIH = (*rnnCellOp).getBIh();
    mlir::Value biasHH = (*rnnCellOp).getBHh();
    if (!biasIH || !biasHH)
      return mlir::failure();

    auto biasIHTy =
        tensor_type::getPositiveStaticRank1F32Tensor(biasIH.getType());
    auto biasHHTy =
        tensor_type::getPositiveStaticRank1F32Tensor(biasHH.getType());
    if (mlir::failed(biasIHTy) || mlir::failed(biasHHTy) ||
        (*biasIHTy).getShape()[0] != hiddenSize ||
        (*biasHHTy).getShape()[0] != hiddenSize)
      return mlir::failure();

    biasIHConstant = biasIH.getDefiningOp<ConstantOp>();
    biasHHConstant = biasHH.getDefiningOp<ConstantOp>();
    if (!biasIHConstant || !biasHHConstant)
      return mlir::failure();
  } else if ((*rnnCellOp).getBIh() || (*rnnCellOp).getBHh()) {
    return mlir::failure();
  }

  RNNCellLowering lowering;
  lowering.rnnCellOp = *rnnCellOp;
  lowering.inputTy = *inputTy;
  lowering.hiddenTy = *hiddenTy;
  lowering.weightIHTy = *weightIHTy;
  lowering.weightHHTy = *weightHHTy;
  lowering.resultTy = *resultTy;
  lowering.fusedInputTy =
      recurrent_gate::buildFusedInputType(*inputTy, *hiddenTy);
  lowering.fusedWeightTy = buildFusedWeightType(*weightIHTy, *weightHHTy);
  lowering.fusedBiasTy = buildFusedBiasType(*resultTy);
  lowering.weightIHConstant = weightIHConstant;
  lowering.weightHHConstant = weightHHConstant;
  lowering.biasIHConstant = biasIHConstant;
  lowering.biasHHConstant = biasHHConstant;
  lowering.hasBias = hasBias;
  return lowering;
}

// Fuses RNNCell input and hidden weights for one sculptor.mvm.
static mlir::TypedAttr buildFusedWeightAttr(RNNCellLowering &match) {
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

  int64_t hiddenSize = match.resultTy.getShape()[1];
  int64_t inputSize = match.inputTy.getShape()[1];
  int64_t fusedWidth = match.fusedWeightTy.getShape()[1];
  llvm::SmallVector<float> fusedWeights(match.fusedWeightTy.getNumElements(),
                                        0.0f);

  for (int64_t row = 0; row < hiddenSize; ++row) {
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
      match.fusedWeightTy, fusedWeights, "analog_rnn_cell_fused_weight_",
      useResource);
}

// Fuses RNNCell input and hidden biases after the MVM.
static mlir::TypedAttr buildFusedBiasAttr(RNNCellLowering &match) {
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
      match.fusedBiasTy, fusedBias, "analog_rnn_cell_fused_bias_", useResource);
}

static mlir::Value buildInputRecombineRegion(RNNCellLowering &match,
                                             mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.rnnCellOp.getLoc();
  rewriter.setInsertionPoint(match.rnnCellOp);
  auto recombineRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.fusedInputTy},
      mlir::ValueRange{match.rnnCellOp.getInput(), match.rnnCellOp.getHPrev()},
      "digital.input_recombine",
      rewriter.getStringAttr("rnn_cell_input_recombine"));

  mlir::Block *body = new mlir::Block();
  recombineRegion.getBody().push_back(body);
  llvm::SmallVector<mlir::Type> inputTypes = {
      match.rnnCellOp.getInput().getType(),
      match.rnnCellOp.getHPrev().getType()};
  llvm::SmallVector<mlir::Location> inputLocs = {
      match.rnnCellOp.getInput().getLoc(), match.rnnCellOp.getHPrev().getLoc()};
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
buildBiasAddRegion(RNNCellLowering &match, mlir::TypedAttr fusedBiasAttr,
                   mlir::Value mvmResult, mlir::RewriterBase &rewriter) {
  if (match.hasBias && !fusedBiasAttr)
    return mlir::failure();

  mlir::Location loc = match.rnnCellOp.getLoc();
  rewriter.setInsertionPoint(match.rnnCellOp);
  auto biasRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.resultTy}, mlir::ValueRange{mvmResult},
      "digital.bias_add", rewriter.getStringAttr("rnn_cell_bias_add"));

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
      loc, match.resultTy, fusedBias.getResult(), reassociation);
  mlir::Value biasedInit = rewriter.create<EmptyOp>(
      loc, match.resultTy.getShape(), match.resultTy.getElementType());
  biasResult =
      rewriter
          .create<mlir::linalg::AddOp>(
              loc, mlir::ValueRange{body->getArgument(0), expandedBias},
              mlir::ValueRange{biasedInit})
          .getResult(0);
  rewriter.create<mlir::sculptor::YieldOp>(loc, biasResult);
  return biasRegion.getResult(0);
}

static mlir::Value buildActivationRegion(RNNCellLowering &match,
                                         mlir::Value activationInput,
                                         mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.rnnCellOp.getLoc();
  rewriter.setInsertionPoint(match.rnnCellOp);
  auto activationRegion = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.resultTy}, mlir::ValueRange{activationInput},
      "digital.activation", rewriter.getStringAttr("rnn_cell_tanh"));

  mlir::Block *body = new mlir::Block();
  activationRegion.getBody().push_back(body);
  body->addArgument(activationInput.getType(), activationInput.getLoc());

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);
  mlir::Value result = converter_recurrent_elementwise::buildTanh(
      loc, match.resultTy, body->getArgument(0), rewriter);
  rewriter.create<mlir::sculptor::YieldOp>(loc, result);
  return activationRegion.getResult(0);
}

static mlir::LogicalResult
lowerRNNCellLayerToMVM(mlir::func::FuncOp func, mlir::RewriterBase &rewriter) {
  auto match = matchExtractedRNNCellLayer(func);
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

  mlir::Location loc = (*match).rnnCellOp.getLoc();
  rewriter.setInsertionPoint((*match).rnnCellOp);
  auto fusedWeight =
      rewriter.create<ConstantOp>(loc, (*match).fusedWeightTy, fusedWeightAttr);
  mlir::Value fusedInput = buildInputRecombineRegion(*match, rewriter);
  mlir::Value mvmResult = mvm_build::buildMVM(
      loc, (*match).resultTy, fusedInput, fusedWeight.getResult(), rewriter);

  auto tanhInput =
      buildBiasAddRegion(*match, fusedBiasAttr, mvmResult, rewriter);
  if (mlir::failed(tanhInput))
    return mlir::failure();

  mlir::Value result = buildActivationRegion(*match, *tanhInput, rewriter);
  (*match).rnnCellOp.getH().replaceAllUsesWith(result);
  rewriter.eraseOp((*match).rnnCellOp);
  converter_recurrent_layer::eraseUnusedConstants(
      {match->weightIHConstant, match->weightHHConstant, match->biasIHConstant,
       match->biasHHConstant},
      rewriter);
  return mlir::success();
}

// Converts extracted sculptor.nn.rnn_cell layer bodies to fused sculptor.mvm plus
// activation math.
class RNNCellConverter : public mlir::sculptor::LayerToMVMConverter {
public:
  mlir::StringRef getName() const override { return "rnn_cell"; }

  void lowerToMVM(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());
    (void)lowerRNNCellLayerToMVM(func, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerRNNCellConverter(LayerToMVMConverters &converters,
                              LayerToMVMConverterMap &converterMap,
                              MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<RNNCellConverter>();
  const LayerToMVMConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["rnn_cell"] = converterPtr;
  converterMap["rnn_cell_w_bias"] = converterPtr;
}

} // namespace sculptor
} // namespace mlir
