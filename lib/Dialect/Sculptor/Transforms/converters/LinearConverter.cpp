#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/MVMBuildUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/NNLayerMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/TensorTypeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <utility>

namespace nn_layer_match = mlir::sculptor::nn_layer_match;
namespace mvm_build = mlir::sculptor::mvm_build;
namespace tensor_type = mlir::sculptor::tensor_type;

namespace {

using mlir::sculptor::NNLinearOp;
using mlir::arith::ConstantOp;
using mlir::tensor::EmptyOp;

// Carries a validated extracted linear layer body.
struct LinearLowering {
  NNLinearOp linearOp;
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType weightTy;
  mlir::RankedTensorType resultTy;
  ConstantOp biasConstant;
};

// Validates the extracted sculptor.nn.linear body without mutating the function.
static mlir::FailureOr<LinearLowering>
matchExtractedLinearLayer(mlir::func::FuncOp func) {
  auto linearOp = nn_layer_match::matchSingleNNLayerOp<NNLinearOp>(func);
  if (mlir::failed(linearOp))
    return mlir::failure();

  bool isBiasFreeLayer = nn_layer_match::hasLayerType(func, "linear");
  bool isBiasLayer = nn_layer_match::hasLayerType(func, "linear_w_bias");
  if (!isBiasFreeLayer && !isBiasLayer)
    return mlir::failure();

  if ((*linearOp).getHasBias() != isBiasLayer)
    return mlir::failure();

  if (func.getNumArguments() != 1 ||
      (*linearOp).getInput() != func.getArgument(0))
    return mlir::failure();

  auto inputTy = tensor_type::getStaticRank2F32Tensor(
      (*linearOp).getInput().getType());
  auto weightTy = tensor_type::getStaticRank2F32Tensor(
      (*linearOp).getWeight().getType());
  auto resultTy = tensor_type::getStaticRank2F32Tensor(
      (*linearOp).getResult().getType());
  if (mlir::failed(inputTy) || mlir::failed(weightTy) || mlir::failed(resultTy))
    return mlir::failure();

  auto inputShape = (*inputTy).getShape();
  auto weightShape = (*weightTy).getShape();
  auto resultShape = (*resultTy).getShape();
  if (inputShape[0] != 1 || resultShape[0] != 1)
    return mlir::failure();

  if (inputShape[1] != weightShape[1] || resultShape[1] != weightShape[0])
    return mlir::failure();

  mlir::Value bias = (*linearOp).getBias();
  if (isBiasLayer) {
    if (!bias)
      return mlir::failure();

    auto biasTy = llvm::dyn_cast<mlir::RankedTensorType>(bias.getType());
    if (!biasTy || !biasTy.hasStaticShape() || biasTy.getRank() != 1 ||
        !biasTy.getElementType().isF32() ||
        biasTy.getShape()[0] != weightShape[0])
      return mlir::failure();

    if (!bias.getDefiningOp<ConstantOp>())
      return mlir::failure();
  } else if (bias) {
    return mlir::failure();
  }

  auto weightConstant = (*linearOp).getWeight().getDefiningOp<ConstantOp>();
  if (!weightConstant)
    return mlir::failure();

  LinearLowering lowering;
  lowering.linearOp = *linearOp;
  lowering.inputTy = *inputTy;
  lowering.weightTy = *weightTy;
  lowering.resultTy = *resultTy;
  lowering.biasConstant =
      bias ? bias.getDefiningOp<ConstantOp>() : ConstantOp{};
  return lowering;
}

// Validates one inline canonical linear operation without imposing an
// outlining or task boundary.
static mlir::FailureOr<LinearLowering>
matchInlineLinearLayer(NNLinearOp linearOp) {
  auto inputTy =
      tensor_type::getStaticRank2F32Tensor(linearOp.getInput().getType());
  auto weightTy =
      tensor_type::getStaticRank2F32Tensor(linearOp.getWeight().getType());
  auto resultTy =
      tensor_type::getStaticRank2F32Tensor(linearOp.getResult().getType());
  if (mlir::failed(inputTy) || mlir::failed(weightTy) ||
      mlir::failed(resultTy))
    return mlir::failure();

  auto inputShape = (*inputTy).getShape();
  auto weightShape = (*weightTy).getShape();
  auto resultShape = (*resultTy).getShape();
  if (inputShape[0] <= 0 || inputShape[1] <= 0 || weightShape[0] <= 0 ||
      weightShape[1] <= 0 || inputShape[1] != weightShape[1] ||
      resultShape[0] != inputShape[0] ||
      resultShape[1] != weightShape[0] ||
      !linearOp.getWeight().getDefiningOp<ConstantOp>())
    return mlir::failure();

  ConstantOp biasConstant;
  mlir::Value bias = linearOp.getBias();
  if (linearOp.getHasBias()) {
    if (!bias)
      return mlir::failure();
    auto biasTy = llvm::dyn_cast<mlir::RankedTensorType>(bias.getType());
    biasConstant = bias.getDefiningOp<ConstantOp>();
    if (!biasTy || !biasTy.hasStaticShape() || biasTy.getRank() != 1 ||
        !biasTy.getElementType().isF32() ||
        biasTy.getShape()[0] != weightShape[0] || !biasConstant)
      return mlir::failure();
  } else if (bias) {
    return mlir::failure();
  }

  LinearLowering lowering;
  lowering.linearOp = linearOp;
  lowering.inputTy = *inputTy;
  lowering.weightTy = *weightTy;
  lowering.resultTy = *resultTy;
  lowering.biasConstant = biasConstant;
  return lowering;
}

static mlir::Value buildMVM(LinearLowering &match,
                            mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.linearOp.getLoc();
  rewriter.setInsertionPoint(match.linearOp);
  return mvm_build::buildMVM(loc, match.resultTy, match.linearOp.getInput(),
                             match.linearOp.getWeight(), rewriter);
}

static mlir::FailureOr<mlir::Value>
buildPostProcess(LinearLowering &match, mlir::Value mvmResult,
                 mlir::RewriterBase &rewriter) {
  mlir::Value bias = match.linearOp.getBias();
  mlir::Location loc = match.linearOp.getLoc();
  rewriter.setInsertionPoint(match.linearOp);
  if (!bias)
    return mvmResult;

  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation = {{0, 1}};
  mlir::Value expandedBias = rewriter.create<mlir::tensor::ExpandShapeOp>(
      loc, match.resultTy, bias, reassociation);
  mlir::Value biasedInit = rewriter.create<EmptyOp>(
      loc, match.resultTy.getShape(), match.resultTy.getElementType());
  auto add = rewriter.create<mlir::linalg::AddOp>(
      loc, mlir::ValueRange{mvmResult, expandedBias},
      mlir::ValueRange{biasedInit});
  add->setAttr("sculptor.semantic.section",
               rewriter.getStringAttr("digital.bias_add"));
  add->setAttr("sculptor.semantic.name",
               rewriter.getStringAttr("linear_bias_add"));
  return add.getResult(0);
}

static mlir::LogicalResult
lowerInlineLinearLayerToMVM(NNLinearOp linearOp,
                            mlir::RewriterBase &rewriter) {
  auto match = matchInlineLinearLayer(linearOp);
  if (mlir::failed(match))
    return mlir::failure();

  mlir::Value mvmResult = buildMVM(*match, rewriter);
  mvmResult.getDefiningOp()->setAttr(
      "sculptor.semantic.kind", rewriter.getStringAttr("projection"));
  auto replacement = buildPostProcess(*match, mvmResult, rewriter);
  if (mlir::failed(replacement))
    return mlir::failure();

  linearOp.getResult().replaceAllUsesWith(*replacement);
  rewriter.eraseOp(linearOp);
  return mlir::success();
}

static mlir::LogicalResult lowerLinearLayerToMVM(mlir::func::FuncOp func,
                                                  mlir::RewriterBase &rewriter) {
  auto match = matchExtractedLinearLayer(func);
  if (mlir::failed(match))
    return mlir::failure();

  mlir::Value mvmResult = buildMVM(*match, rewriter);
  auto replacement = buildPostProcess(*match, mvmResult, rewriter);
  if (failed(replacement))
    return mlir::failure();

  (*match).linearOp.getResult().replaceAllUsesWith(*replacement);
  rewriter.eraseOp((*match).linearOp);
  if ((*match).biasConstant && (*match).biasConstant->use_empty())
    rewriter.eraseOp((*match).biasConstant);
  return mlir::success();
}

// Converts extracted sculptor.nn.linear layer bodies to one sculptor.mvm.
class LinearConverter : public mlir::sculptor::LayerToMVMConverter {
public:
  // Reports the layer_type key used by the converter dispatch table.
  mlir::StringRef getName() const override { return "linear"; }

  // Replaces a recognized linear body with the execution-level MVM op.
  void lowerToMVM(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());
    (void)lowerLinearLayerToMVM(func, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

LogicalResult decomposeInlineLinearLayers(func::FuncOp func) {
  SmallVector<NNLinearOp> linearOps;
  func.walk([&](NNLinearOp linearOp) { linearOps.push_back(linearOp); });

  IRRewriter rewriter(func.getContext());
  for (NNLinearOp linearOp : linearOps) {
    if (!linearOp || !linearOp->getBlock())
      continue;
    if (failed(lowerInlineLinearLayerToMVM(linearOp, rewriter)))
      return linearOp.emitOpError("failed to decompose inline linear layer");
  }
  return success();
}

// Registers the linear converter for both biased and bias-free extracted
// layers.
void registerLinearConverter(LayerToMVMConverters &converters,
                             LayerToMVMConverterMap &converterMap,
                             MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<LinearConverter>();
  const LayerToMVMConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["linear"] = converterPtr;
  converterMap["linear_w_bias"] = converterPtr;
}

} // namespace sculptor
} // namespace mlir
