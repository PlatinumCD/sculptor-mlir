#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConstantUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConvConversionUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/I64ArrayAttrUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/NNLayerMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/RewriteUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>

namespace nn_layer_match = mlir::sculptor::nn_layer_match;
namespace converter_constant = mlir::sculptor::converter_constant;
namespace converter_conv = mlir::sculptor::converter_conv;
namespace i64_array_attr = mlir::sculptor::i64_array_attr;

namespace {

using mlir::sculptor::NNConv1DOp;
using mlir::arith::ConstantOp;
using mlir::tensor::EmptyOp;

struct Conv1DShapeInfo {
  int64_t n;
  int64_t c;
  int64_t w;
  int64_t f;
  int64_t kw;
  int64_t ow;
};

struct Conv1DConvolutionAttrs {
  int64_t stride;
  int64_t padding;
  int64_t dilation;
};

struct Conv1DMatch {
  mlir::Operation *rootOp = nullptr;
  mlir::Value result;
  mlir::Value activation;
  ConstantOp filterRank3Const;
  ConstantOp filterRank2Const;
  mlir::Value bias;
  ConstantOp biasConstant;
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType filterRank3Ty;
  mlir::RankedTensorType filterRank2Ty;
  mlir::RankedTensorType outputTy;
  Conv1DConvolutionAttrs attrs;
  Conv1DShapeInfo shape;
  bool hasBias = false;
};

struct PreparedFilter {
  mlir::Value filterMatrix;
};

struct PreparedBias {
  mlir::Value bias;
  ConstantOp biasConstant;
};

struct Conv1DLoweringState {
  mlir::Location loc;
  mlir::Type elementType;
  mlir::RankedTensorType patchTy;
  mlir::RankedTensorType matmulResultTy;
  mlir::RankedTensorType outputTy;
  Conv1DShapeInfo shape;
  Conv1DConvolutionAttrs attrs;
  bool hasBias = false;
};

static mlir::RankedTensorType
buildFlattenedTensorType(mlir::RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();
  int64_t flattenedCols = shape[1] * shape[2];
  return mlir::RankedTensorType::get({shape[0], flattenedCols},
                                     tensorTy.getElementType());
}

static mlir::FailureOr<Conv1DConvolutionAttrs>
getSupportedConvolutionAttrs(NNConv1DOp convOp) {
  llvm::SmallVector<int64_t> padding;
  if (!i64_array_attr::extract(convOp.getPadding(), /*expectedSize=*/1,
                               padding) ||
      !i64_array_attr::allEqual(padding, 0))
    return mlir::failure();

  llvm::SmallVector<int64_t> dilations;
  if (!i64_array_attr::extract(convOp.getDilation(), /*expectedSize=*/1,
                               dilations) ||
      !i64_array_attr::allEqual(dilations, 1))
    return mlir::failure();

  llvm::SmallVector<int64_t> strides;
  if (!i64_array_attr::extract(convOp.getStride(), /*expectedSize=*/1,
                               strides) ||
      !i64_array_attr::allPositive(strides))
    return mlir::failure();

  return Conv1DConvolutionAttrs{strides[0], padding[0], dilations[0]};
}

static mlir::FailureOr<Conv1DShapeInfo>
getValidatedShapeInfo(mlir::RankedTensorType inputTy,
                      std::optional<mlir::RankedTensorType> biasTy,
                      mlir::RankedTensorType filterRank3Ty,
                      mlir::RankedTensorType filterRank2Ty,
                      mlir::RankedTensorType outputTy, int64_t stride) {
  auto inputShape = inputTy.getShape();
  auto filterShape = filterRank3Ty.getShape();
  auto filterFlatShape = filterRank2Ty.getShape();
  auto outputShape = outputTy.getShape();

  Conv1DShapeInfo shapeInfo{
      inputShape[0],  inputShape[1],  inputShape[2],
      filterShape[0], filterShape[2], outputShape[2],
  };

  if (shapeInfo.n != 1)
    return mlir::failure();

  if (filterShape[1] != shapeInfo.c)
    return mlir::failure();

  if (outputShape[0] != shapeInfo.n || outputShape[1] != shapeInfo.f)
    return mlir::failure();

  if (filterFlatShape[0] != shapeInfo.f ||
      filterFlatShape[1] != shapeInfo.c * shapeInfo.kw)
    return mlir::failure();

  if (shapeInfo.kw > shapeInfo.w)
    return mlir::failure();

  int64_t expectedOw = ((shapeInfo.w - shapeInfo.kw) / stride) + 1;
  if (shapeInfo.ow != expectedOw)
    return mlir::failure();

  if (biasTy && biasTy->getShape()[0] != shapeInfo.f)
    return mlir::failure();

  return shapeInfo;
}

static mlir::FailureOr<ConstantOp>
createFlattenedFilter(ConstantOp filterConst,
                      mlir::RankedTensorType filterRank3Ty,
                      mlir::RewriterBase &rewriter) {
  // Conv1D exposes weights as the sculptor.mvm matrix; setup is handled later by
  // ExpandMVMToGolem.
  mlir::RankedTensorType flattenedTy = buildFlattenedTensorType(filterRank3Ty);
  mlir::TypedAttr flattenedAttr =
      converter_constant::reshapeDenseOrResourceAttr(filterConst, flattenedTy);
  if (!flattenedAttr)
    return mlir::failure();

  rewriter.setInsertionPointAfter(filterConst);
  return rewriter.create<ConstantOp>(filterConst.getLoc(), flattenedTy,
                                     flattenedAttr);
}

static mlir::FailureOr<Conv1DMatch>
matchSupportedConv1D(NNConv1DOp convOp, mlir::RewriterBase &rewriter) {
  auto inputOutputTypes = converter_conv::getStaticF32InputOutputTypes(
      convOp.getInput(), convOp.getResult(), /*expectedRank=*/3);
  if (failed(inputOutputTypes))
    return mlir::failure();
  auto [inputTy, outputTy] = *inputOutputTypes;

  auto filterConstant =
      converter_conv::getStaticF32FilterConstant(convOp.getWeight(),
                                                 /*expectedRank=*/3);
  if (failed(filterConstant))
    return mlir::failure();
  auto [filterRank3Const, filterRank3Ty] = *filterConstant;

  auto convAttrs = getSupportedConvolutionAttrs(convOp);
  if (failed(convAttrs))
    return mlir::failure();

  auto biasMatch =
      converter_conv::matchOptionalBias(convOp.getHasBias(), convOp.getBias());
  if (failed(biasMatch))
    return mlir::failure();

  mlir::RankedTensorType filterRank2Ty =
      buildFlattenedTensorType(filterRank3Ty);
  auto shapeInfo =
      getValidatedShapeInfo(inputTy, biasMatch->biasTy, filterRank3Ty,
                            filterRank2Ty, outputTy, convAttrs->stride);
  if (failed(shapeInfo))
    return mlir::failure();

  auto filterRank2Const =
      createFlattenedFilter(filterRank3Const, filterRank3Ty, rewriter);
  if (failed(filterRank2Const))
    return mlir::failure();

  Conv1DMatch match;
  match.rootOp = convOp.getOperation();
  match.result = convOp.getResult();
  match.activation = convOp.getInput();
  match.filterRank3Const = filterRank3Const;
  match.filterRank2Const = *filterRank2Const;
  match.bias = biasMatch->bias;
  match.biasConstant = biasMatch->biasConstant;
  match.inputTy = inputTy;
  match.filterRank3Ty = filterRank3Ty;
  match.filterRank2Ty = filterRank2Ty;
  match.outputTy = outputTy;
  match.attrs = *convAttrs;
  match.shape = *shapeInfo;
  match.hasBias = convOp.getHasBias();
  return match;
}

static Conv1DLoweringState buildLoweringState(const Conv1DMatch &match) {
  mlir::Location loc = match.rootOp->getLoc();
  mlir::Type elementType = match.inputTy.getElementType();
  int64_t flattenedWidth = match.shape.c * match.shape.kw;
  return Conv1DLoweringState{
      .loc = loc,
      .elementType = elementType,
      .patchTy = mlir::RankedTensorType::get({1, flattenedWidth}, elementType),
      .matmulResultTy =
          mlir::RankedTensorType::get({1, match.shape.f}, elementType),
      .outputTy = match.outputTy,
      .shape = match.shape,
      .attrs = match.attrs,
      .hasBias = match.hasBias,
  };
}

static PreparedFilter prepareFilter(Conv1DMatch &match) {
  return PreparedFilter{match.filterRank2Const.getResult()};
}

static PreparedBias prepareBias(Conv1DMatch &match,
                                const Conv1DLoweringState &state,
                                mlir::OpBuilder &builder) {
  (void)state;
  (void)builder;

  PreparedBias preparedBias;
  if (!match.hasBias)
    return preparedBias;

  preparedBias.bias = match.bias;
  preparedBias.biasConstant = match.biasConstant;
  return preparedBias;
}

static mlir::Value buildIndexConstant(mlir::OpBuilder &builder,
                                      mlir::Location loc, int64_t value) {
  return builder.create<mlir::arith::ConstantIndexOp>(loc, value);
}

static mlir::Value buildFlattenedPatch(mlir::OpBuilder &builder,
                                       const Conv1DLoweringState &state,
                                       mlir::Value activation,
                                       int64_t outputIndex) {
  mlir::Value patchInit = builder.create<EmptyOp>(
      state.loc, state.patchTy.getShape(), state.elementType);

  mlir::Value zero = buildIndexConstant(builder, state.loc, 0);
  mlir::Value patch = patchInit;
  for (int64_t channel = 0; channel < state.shape.c; ++channel) {
    mlir::Value channelIndex = buildIndexConstant(builder, state.loc, channel);
    for (int64_t kernel = 0; kernel < state.shape.kw; ++kernel) {
      int64_t inputIndex = outputIndex * state.attrs.stride +
                           kernel * state.attrs.dilation - state.attrs.padding;
      mlir::Value inputIndexValue =
          buildIndexConstant(builder, state.loc, inputIndex);
      mlir::Value inputValue = builder.create<mlir::tensor::ExtractOp>(
          state.loc, activation,
          mlir::ValueRange{zero, channelIndex, inputIndexValue});

      int64_t flattenedIndex = channel * state.shape.kw + kernel;
      mlir::Value flatIndex =
          buildIndexConstant(builder, state.loc, flattenedIndex);
      patch = builder.create<mlir::tensor::InsertOp>(
          state.loc, inputValue, patch, mlir::ValueRange{zero, flatIndex});
    }
  }

  return patch;
}

static mlir::Value buildPatchPreparationRegion(mlir::OpBuilder &builder,
                                               const Conv1DMatch &match,
                                               const Conv1DLoweringState &state,
                                               int64_t outputIndex) {
  auto prepRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      state.loc, mlir::TypeRange{state.patchTy},
      mlir::ValueRange{match.activation}, "digital.conv_patch",
      builder.getStringAttr("conv1d_ow_" + std::to_string(outputIndex)));

  mlir::Block *body = new mlir::Block();
  prepRegion.getBody().push_back(body);
  body->addArgument(match.activation.getType(), match.activation.getLoc());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value patch =
      buildFlattenedPatch(builder, state, body->getArgument(0), outputIndex);
  builder.create<mlir::sculptor::YieldOp>(state.loc, patch);
  return prepRegion.getResult(0);
}

static mlir::Value buildPostProcessRegion(mlir::OpBuilder &builder,
                                          PreparedBias &preparedBias,
                                          const Conv1DLoweringState &state,
                                          mlir::Value channelResult,
                                          int64_t outputIndex) {
  auto postProcess = builder.create<mlir::sculptor::TaskRegionOp>(
      state.loc, mlir::TypeRange{state.matmulResultTy},
      mlir::ValueRange{channelResult}, "digital.bias_add",
      builder.getStringAttr("conv1d_ow_" + std::to_string(outputIndex) +
                            "_bias_add"));

  mlir::Block *body = new mlir::Block();
  postProcess.getBody().push_back(body);
  body->addArgument(channelResult.getType(), channelResult.getLoc());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value postProcessResult = body->getArgument(0);
  if (preparedBias.bias) {
    auto biasConstant = builder.create<ConstantOp>(
        preparedBias.biasConstant.getLoc(), preparedBias.biasConstant.getType(),
        preparedBias.biasConstant.getValue());
    postProcessResult = converter_conv::applyOptionalBias(
        state.loc, state.matmulResultTy, state.elementType, postProcessResult,
        biasConstant.getResult(), builder);
  }

  builder.create<mlir::sculptor::YieldOp>(state.loc, postProcessResult);
  return postProcess.getResult(0);
}

static mlir::Value buildOutputPosition(mlir::OpBuilder &builder,
                                       const Conv1DMatch &match,
                                       const PreparedFilter &preparedFilter,
                                       PreparedBias &preparedBias,
                                       const Conv1DLoweringState &state,
                                       int64_t outputIndex) {
  mlir::Value patch =
      buildPatchPreparationRegion(builder, match, state, outputIndex);
  mlir::Value channelResult =
      converter_conv::buildPatchMVM(state.loc, state.matmulResultTy, patch,
                                    preparedFilter.filterMatrix, builder);
  return buildPostProcessRegion(builder, preparedBias, state, channelResult,
                                outputIndex);
}

static mlir::FailureOr<mlir::Value>
assembleOutputTensor(mlir::OpBuilder &builder, const Conv1DLoweringState &state,
                     llvm::ArrayRef<mlir::Value> positionResults) {
  if (positionResults.empty())
    return mlir::failure();

  auto outputAssembly = builder.create<mlir::sculptor::TaskRegionOp>(
      state.loc, mlir::TypeRange{state.outputTy},
      mlir::ValueRange(positionResults), "digital.output_recombine",
      builder.getStringAttr("conv1d_output_recombine"));

  mlir::Block *body = new mlir::Block();
  outputAssembly.getBody().push_back(body);
  for (mlir::Value positionResult : positionResults)
    body->addArgument(positionResult.getType(), positionResult.getLoc());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  mlir::RankedTensorType positionExpandedTy = mlir::RankedTensorType::get(
      {state.shape.n, state.shape.f, 1}, state.elementType);
  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation = {{0},
                                                                    {1, 2}};

  llvm::SmallVector<mlir::Value> expandedPositions;
  expandedPositions.reserve(positionResults.size());
  for (mlir::BlockArgument positionResult : body->getArguments()) {
    expandedPositions.push_back(
        builder
            .create<mlir::tensor::ExpandShapeOp>(state.loc, positionExpandedTy,
                                                 positionResult, reassociation)
            .getResult());
  }

  mlir::Value output = expandedPositions.front();
  if (expandedPositions.size() > 1) {
    output = builder
                 .create<mlir::tensor::ConcatOp>(
                     state.loc, state.outputTy, /*dim=*/2,
                     mlir::ValueRange(expandedPositions))
                 .getResult();
  }

  builder.create<mlir::sculptor::YieldOp>(state.loc, output);
  return outputAssembly.getResult(0);
}

static mlir::FailureOr<mlir::Value> emitUnrolledOutputPositions(
    mlir::RewriterBase &rewriter, const Conv1DMatch &match,
    const PreparedFilter &preparedFilter, PreparedBias &preparedBias,
    const Conv1DLoweringState &state) {
  rewriter.setInsertionPointAfter(match.rootOp);
  llvm::SmallVector<mlir::Value> positionResults;
  positionResults.reserve(state.shape.ow);
  for (int64_t outputIndex = 0; outputIndex < state.shape.ow; ++outputIndex) {
    positionResults.push_back(buildOutputPosition(
        rewriter, match, preparedFilter, preparedBias, state, outputIndex));
  }

  return assembleOutputTensor(rewriter, state, positionResults);
}

static void eraseUnusedConv1DOps(Conv1DMatch &match,
                                 mlir::RewriterBase &rewriter) {
  mlir::sculptor::converter_rewrite::eraseIfUnused(match.rootOp, rewriter);
  mlir::sculptor::converter_rewrite::eraseIfUnused(
      match.filterRank3Const.getOperation(), rewriter);

  if (match.hasBias)
    mlir::sculptor::converter_rewrite::eraseIfUnused(
        match.biasConstant.getOperation(), rewriter);
}

// Converts extracted sculptor.nn.conv1d layer bodies into per-position sculptor.mvm
// execution.
class Conv1DConverter : public mlir::sculptor::LayerToMVMConverter {
public:
  mlir::StringRef getName() const override { return "conv1d"; }

  void lowerToMVM(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());
    auto layerOp = nn_layer_match::matchSingleNNLayerOp<NNConv1DOp>(func);
    if (failed(layerOp))
      return;

    if (!nn_layer_match::hasLayerTypeMatchingBias(
            func, "conv1d", "conv1d_w_bias", (*layerOp).getHasBias()))
      return;

    mlir::FailureOr<Conv1DMatch> match =
        matchSupportedConv1D(*layerOp, rewriter);
    if (failed(match))
      return;

    Conv1DLoweringState state = buildLoweringState(*match);
    PreparedFilter preparedFilter = prepareFilter(*match);
    PreparedBias preparedBias = prepareBias(*match, state, rewriter);

    auto rewrittenOutput = emitUnrolledOutputPositions(
        rewriter, *match, preparedFilter, preparedBias, state);
    if (failed(rewrittenOutput))
      return;

    match->result.replaceAllUsesWith(*rewrittenOutput);
    eraseUnusedConv1DOps(*match, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

// Registers the Conv1D converter for both biased and bias-free layer slices.
void registerConv1DConverter(LayerToMVMConverters &converters,
                             LayerToMVMConverterMap &converterMap,
                             MLIRContext *context) {
  (void)context;

  auto converter = std::make_unique<Conv1DConverter>();
  const LayerToMVMConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));

  converterMap["conv1d"] = converterPtr;
  converterMap["conv1d_w_bias"] = converterPtr;
}

} // namespace sculptor
} // namespace mlir
