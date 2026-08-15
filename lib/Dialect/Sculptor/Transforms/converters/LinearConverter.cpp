#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/MVMBuildUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/NNLayerMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/TensorTypeUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/SemanticOperationScope.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/SemanticOperationNames.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <utility>

namespace nn_layer_match = mlir::sculptor::nn_layer_match;
namespace mvm_build = mlir::sculptor::mvm_build;
namespace tensor_type = mlir::sculptor::tensor_type;
namespace semantic_operation_names = mlir::sculptor::semantic_operation_names;

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

static mlir::FailureOr<mlir::RankedTensorType>
getStaticF32Tensor(mlir::Type type) {
  auto tensor = llvm::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape() || tensor.getRank() < 2 ||
      !tensor.getElementType().isF32() ||
      llvm::any_of(tensor.getShape(), [](int64_t dim) { return dim <= 0; }))
    return mlir::failure();
  return tensor;
}

static bool hasValidLinearShapes(mlir::RankedTensorType inputTy,
                                 mlir::RankedTensorType weightTy,
                                 mlir::RankedTensorType resultTy) {
  if (inputTy.getRank() != resultTy.getRank() || weightTy.getRank() != 2 ||
      inputTy.getDimSize(inputTy.getRank() - 1) != weightTy.getDimSize(1) ||
      resultTy.getDimSize(resultTy.getRank() - 1) != weightTy.getDimSize(0))
    return false;
  for (int64_t dim = 0; dim < inputTy.getRank() - 1; ++dim)
    if (inputTy.getDimSize(dim) != resultTy.getDimSize(dim))
      return false;
  return true;
}

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

  auto inputTy = getStaticF32Tensor((*linearOp).getInput().getType());
  auto weightTy = tensor_type::getStaticRank2F32Tensor(
      (*linearOp).getWeight().getType());
  auto resultTy = getStaticF32Tensor((*linearOp).getResult().getType());
  if (mlir::failed(inputTy) || mlir::failed(weightTy) || mlir::failed(resultTy))
    return mlir::failure();

  auto weightShape = (*weightTy).getShape();
  if (!hasValidLinearShapes(*inputTy, *weightTy, *resultTy))
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
  auto inputTy = getStaticF32Tensor(linearOp.getInput().getType());
  auto weightTy =
      tensor_type::getStaticRank2F32Tensor(linearOp.getWeight().getType());
  auto resultTy = getStaticF32Tensor(linearOp.getResult().getType());
  if (mlir::failed(inputTy) || mlir::failed(weightTy) ||
      mlir::failed(resultTy))
    return mlir::failure();

  auto weightShape = (*weightTy).getShape();
  if (!hasValidLinearShapes(*inputTy, *weightTy, *resultTy) ||
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

static mlir::Value buildMVM(LinearLowering &match, int64_t sequenceId,
                            mlir::RewriterBase &rewriter) {
  mlir::Location loc = match.linearOp.getLoc();
  rewriter.setInsertionPoint(match.linearOp);
  int64_t rank = match.inputTy.getRank();
  int64_t inputWidth = match.inputTy.getDimSize(rank - 1);
  int64_t outputWidth = match.resultTy.getDimSize(rank - 1);
  int64_t rowCount = match.inputTy.getNumElements() / inputWidth;
  if (rank == 2 && rowCount == 1)
    return mvm_build::buildMVM(loc, match.resultTy, match.linearOp.getInput(),
                               match.linearOp.getWeight(), rewriter);

  mlir::RankedTensorType sequenceInputTy = mlir::RankedTensorType::get(
      {rowCount, inputWidth}, match.inputTy.getElementType());
  mlir::RankedTensorType sequenceResultTy = mlir::RankedTensorType::get(
      {rowCount, outputWidth}, match.resultTy.getElementType());
  mlir::Value sequenceInput = match.linearOp.getInput();
  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation(2);
  if (rank > 2) {
    for (int64_t dim = 0; dim < rank - 1; ++dim)
      reassociation[0].push_back(dim);
    reassociation[1].push_back(rank - 1);
    sequenceInput = rewriter.create<mlir::tensor::CollapseShapeOp>(
        loc, sequenceInputTy, sequenceInput, reassociation);
  }

  std::string sequenceName =
      "linear_mvm_sequence_" + std::to_string(sequenceId);
  mlir::sculptor::SemanticOperationScope scope(
      rewriter, semantic_operation_names::kMVMSequenceTaskKind, sequenceName);
  mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value upper =
      rewriter.create<mlir::arith::ConstantIndexOp>(loc, rowCount);
  mlir::Value init = rewriter.create<EmptyOp>(
      loc, sequenceResultTy.getShape(), sequenceResultTy.getElementType());
  auto loop = rewriter.create<mlir::scf::ForOp>(
      loc, zero, upper, one, mlir::ValueRange{init},
      [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value row,
          mlir::ValueRange iterArgs) {
        llvm::SmallVector<mlir::OpFoldResult> inputOffsets = {
            row, builder.getIndexAttr(0)};
        llvm::SmallVector<mlir::OpFoldResult> inputSizes = {
            builder.getIndexAttr(1), builder.getIndexAttr(inputWidth)};
        llvm::SmallVector<mlir::OpFoldResult> strides = {
            builder.getIndexAttr(1), builder.getIndexAttr(1)};
        mlir::RankedTensorType rowInputTy = mlir::RankedTensorType::get(
            {1, inputWidth}, match.inputTy.getElementType());
        mlir::Value rowInput =
            builder.create<mlir::tensor::ExtractSliceOp>(
                nestedLoc, rowInputTy, sequenceInput, inputOffsets, inputSizes,
                strides);
        mlir::RankedTensorType rowResultTy = mlir::RankedTensorType::get(
            {1, outputWidth}, match.resultTy.getElementType());
        mlir::Value rowResult = mvm_build::buildMVM(
            nestedLoc, rowResultTy, rowInput, match.linearOp.getWeight(),
            builder);
        llvm::SmallVector<mlir::OpFoldResult> resultSizes = {
            builder.getIndexAttr(1), builder.getIndexAttr(outputWidth)};
        mlir::Value updated = builder.create<mlir::tensor::InsertSliceOp>(
            nestedLoc, rowResult, iterArgs[0], inputOffsets, resultSizes,
            strides);
        builder.create<mlir::scf::YieldOp>(nestedLoc, updated);
      });
  rewriter.setInsertionPointAfter(loop);
  scope.annotate();

  if (rank == 2)
    return loop.getResult(0);
  return rewriter.create<mlir::tensor::ExpandShapeOp>(
      loc, match.resultTy, loop.getResult(0), reassociation);
}

static mlir::FailureOr<mlir::Value>
buildPostProcess(LinearLowering &match, mlir::Value mvmResult,
                 mlir::RewriterBase &rewriter) {
  mlir::Value bias = match.linearOp.getBias();
  mlir::Location loc = match.linearOp.getLoc();
  rewriter.setInsertionPoint(match.linearOp);
  if (!bias)
    return mvmResult;

  mlir::Value biasedInit = rewriter.create<EmptyOp>(
      loc, match.resultTy.getShape(), match.resultTy.getElementType());
  int64_t rank = match.resultTy.getRank();
  mlir::AffineMap identity = rewriter.getMultiDimIdentityMap(rank);
  mlir::AffineExpr last = rewriter.getAffineDimExpr(rank - 1);
  mlir::AffineMap biasMap = mlir::AffineMap::get(rank, 0, {last},
                                                 rewriter.getContext());
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      rank, mlir::utils::IteratorType::parallel);
  auto add = rewriter.create<mlir::linalg::GenericOp>(
      loc, match.resultTy, mlir::ValueRange{mvmResult, bias},
      mlir::ValueRange{biasedInit},
      llvm::SmallVector<mlir::AffineMap>{identity, biasMap, identity},
      iterators,
      [](mlir::OpBuilder &builder, mlir::Location nestedLoc,
         mlir::ValueRange args) {
        mlir::Value sum = builder.create<mlir::arith::AddFOp>(
            nestedLoc, args[0], args[1]);
        builder.create<mlir::linalg::YieldOp>(nestedLoc, sum);
      });
  add->setAttr("sculptor.semantic.section",
               rewriter.getStringAttr("digital.bias_add"));
  add->setAttr("sculptor.semantic.name",
               rewriter.getStringAttr("linear_bias_add"));
  return add.getResult(0);
}

static mlir::LogicalResult
lowerInlineLinearLayerToMVM(NNLinearOp linearOp,
                            int64_t sequenceId,
                            mlir::RewriterBase &rewriter) {
  auto match = matchInlineLinearLayer(linearOp);
  if (mlir::failed(match))
    return mlir::failure();

  mlir::Value mvmResult = buildMVM(*match, sequenceId, rewriter);
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

  mlir::Value mvmResult = buildMVM(*match, /*sequenceId=*/0, rewriter);
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

  SemanticLayerRewriteListener layerListener;
  IRRewriter rewriter(func.getContext(), &layerListener);
  for (auto [sequenceId, linearOp] : llvm::enumerate(linearOps)) {
    if (!linearOp || !linearOp->getBlock())
      continue;
    SemanticLayerRewriteScope layerScope(layerListener, linearOp);
    if (failed(lowerInlineLinearLayerToMVM(
            linearOp, static_cast<int64_t>(sequenceId), rewriter)))
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
