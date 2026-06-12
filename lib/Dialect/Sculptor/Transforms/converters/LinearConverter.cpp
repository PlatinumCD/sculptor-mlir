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

static mlir::Value buildMVM(LinearLowering &match,
                            mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.linearOp.getLoc();
  rewriter.setInsertionPoint(match.linearOp);
  return mvm_build::buildMVM(loc, match.resultTy, match.linearOp.getInput(),
                             match.linearOp.getWeight(), rewriter);
}

static mlir::FailureOr<mlir::Value>
buildPostProcessRegion(LinearLowering &match,
                       mlir::Value mvmResult,
                       mlir::RewriterBase &rewriter) {
  mlir::Value bias = match.linearOp.getBias();
  mlir::Location loc = match.linearOp.getLoc();
  rewriter.setInsertionPoint(match.linearOp);

  llvm::SmallVector<mlir::Value> inputs = {mvmResult};
  llvm::SmallVector<mlir::Type> inputTypes = {mvmResult.getType()};

  auto postProcess = rewriter.create<mlir::sculptor::TaskRegionOp>(
      loc, mlir::TypeRange{match.resultTy}, mlir::ValueRange(inputs),
      "digital.bias_add", rewriter.getStringAttr("linear_bias_add"));

  mlir::Block *body = new mlir::Block();
  postProcess.getBody().push_back(body);
  llvm::SmallVector<mlir::Location> argLocs(inputTypes.size(), loc);
  body->addArguments(inputTypes, argLocs);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);

  mlir::Value postProcessResult = body->getArgument(0);
  if (!bias) {
    rewriter.create<mlir::sculptor::YieldOp>(loc, postProcessResult);
    return postProcess.getResult(0);
  }

  mlir::IRMapping mapping;
  auto clonedBiasConstant =
      llvm::cast<ConstantOp>(rewriter.clone(*match.biasConstant, mapping));
  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation = {{0, 1}};
  mlir::Value expandedBias = rewriter.create<mlir::tensor::ExpandShapeOp>(
      loc, match.resultTy, clonedBiasConstant.getResult(), reassociation);
  mlir::Value biasedInit = rewriter.create<EmptyOp>(
      loc, match.resultTy.getShape(), match.resultTy.getElementType());
  postProcessResult = rewriter
      .create<mlir::linalg::AddOp>(
          loc, mlir::ValueRange{body->getArgument(0), expandedBias},
          mlir::ValueRange{biasedInit})
      .getResult(0);
  rewriter.create<mlir::sculptor::YieldOp>(loc, postProcessResult);
  return postProcess.getResult(0);
}

static mlir::LogicalResult lowerLinearLayerToMVM(mlir::func::FuncOp func,
                                                  mlir::RewriterBase &rewriter) {
  auto match = matchExtractedLinearLayer(func);
  if (mlir::failed(match))
    return mlir::failure();

  mlir::Value mvmResult = buildMVM(*match, rewriter);
  auto replacement = buildPostProcessRegion(*match, mvmResult, rewriter);
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
