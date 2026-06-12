#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConvConversionUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/I64ArrayAttrUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/NNLayerMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/RewriteUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>
#include <string>

namespace nn_layer_match = mlir::sculptor::nn_layer_match;
namespace converter_conv = mlir::sculptor::converter_conv;
namespace i64_array_attr = mlir::sculptor::i64_array_attr;

namespace {

using mlir::sculptor::NNGroupedConv2DOp;
using mlir::arith::ConstantOp;
using mlir::tensor::EmptyOp;

struct Conv2DGroupedShapeInfo {
  int64_t n;
  int64_t groups;
  int64_t cTotal;
  int64_t cg;
  int64_t h;
  int64_t w;
  int64_t fTotal;
  int64_t fg;
  int64_t kh;
  int64_t kw;
  int64_t oh;
  int64_t ow;
};

struct Conv2DGroupedConvolutionAttrs {
  int64_t strideH;
  int64_t strideW;
  int64_t paddingH;
  int64_t paddingW;
  int64_t dilationH;
  int64_t dilationW;
};

struct Conv2DGroupedMatch {
  mlir::Operation *rootOp = nullptr;
  mlir::Value result;
  mlir::Value sourceActivation;
  ConstantOp filterRank4Const;
  ConstantOp filterRank2Const;
  mlir::Value bias;
  ConstantOp biasConstant;
  mlir::RankedTensorType sourceActivationTy;
  mlir::RankedTensorType filterRank4Ty;
  mlir::RankedTensorType filterRank2Ty;
  mlir::RankedTensorType outputTy;
  Conv2DGroupedConvolutionAttrs attrs;
  Conv2DGroupedShapeInfo shape;
  bool hasBias = false;
};

struct PreparedGroupedFilter {
  mlir::Value filterMatrix;
};

struct PreparedGroupedBias {
  mlir::Value bias;
  ConstantOp biasConstant;
};

struct Conv2DGroupedLoweringState {
  mlir::Location loc;
  mlir::Type elementType;
  mlir::RankedTensorType patchTy;
  mlir::RankedTensorType matmulResultTy;
  mlir::RankedTensorType outputTy;
  Conv2DGroupedShapeInfo shape;
  Conv2DGroupedConvolutionAttrs attrs;
  bool hasBias = false;
};

static mlir::FailureOr<Conv2DGroupedConvolutionAttrs>
getSupportedGroupedConvolutionAttrs(NNGroupedConv2DOp convOp) {
  llvm::SmallVector<int64_t> padding;
  if (!i64_array_attr::extract(convOp.getPadding(), /*expectedSize=*/2,
                               padding) ||
      !i64_array_attr::allEqual(padding, 0))
    return mlir::failure();

  llvm::SmallVector<int64_t> dilations;
  if (!i64_array_attr::extract(convOp.getDilation(), /*expectedSize=*/2,
                               dilations) ||
      !i64_array_attr::allEqual(dilations, 1))
    return mlir::failure();

  llvm::SmallVector<int64_t> strides;
  if (!i64_array_attr::extract(convOp.getStride(), /*expectedSize=*/2,
                               strides) ||
      !i64_array_attr::allPositive(strides))
    return mlir::failure();

  return Conv2DGroupedConvolutionAttrs{strides[0], strides[1],   padding[0],
                                       padding[1], dilations[0], dilations[1]};
}

static mlir::RankedTensorType
buildGroupedFlattenedTensorType(mlir::RankedTensorType filterRank4Ty,
                                int64_t cTotal) {
  auto filterShape = filterRank4Ty.getShape();
  int64_t flattenedCols = cTotal * filterShape[2] * filterShape[3];
  return mlir::RankedTensorType::get({filterShape[0], flattenedCols},
                                     filterRank4Ty.getElementType());
}

static mlir::FailureOr<Conv2DGroupedShapeInfo>
getValidatedGroupedShapeInfo(mlir::RankedTensorType sourceActivationTy,
                             std::optional<mlir::RankedTensorType> biasTy,
                             mlir::RankedTensorType filterRank4Ty,
                             mlir::RankedTensorType filterRank2Ty,
                             mlir::RankedTensorType outputTy, int64_t groups,
                             const Conv2DGroupedConvolutionAttrs &attrs) {
  auto sourceActivationShape = sourceActivationTy.getShape();
  auto filterRank4Shape = filterRank4Ty.getShape();
  auto filterRank2Shape = filterRank2Ty.getShape();
  auto outputShape = outputTy.getShape();

  if (groups <= 1)
    return mlir::failure();
  if (sourceActivationShape[1] % groups != 0 ||
      filterRank4Shape[0] % groups != 0 || outputShape[1] % groups != 0)
    return mlir::failure();

  Conv2DGroupedShapeInfo shapeInfo{
      sourceActivationShape[0], groups,
      sourceActivationShape[1], sourceActivationShape[1] / groups,
      sourceActivationShape[2], sourceActivationShape[3],
      filterRank4Shape[0],      filterRank4Shape[0] / groups,
      filterRank4Shape[2],      filterRank4Shape[3],
      outputShape[2],           outputShape[3],
  };

  if (shapeInfo.n != 1 || outputShape[0] != shapeInfo.n)
    return mlir::failure();

  if (filterRank4Shape[1] != shapeInfo.cg)
    return mlir::failure();

  if (outputShape[1] != shapeInfo.fTotal)
    return mlir::failure();

  if (filterRank2Shape[0] != shapeInfo.fTotal ||
      filterRank2Shape[1] != shapeInfo.cTotal * shapeInfo.kh * shapeInfo.kw)
    return mlir::failure();

  if (shapeInfo.kh > shapeInfo.h || shapeInfo.kw > shapeInfo.w)
    return mlir::failure();

  int64_t effectiveKh = attrs.dilationH * (shapeInfo.kh - 1) + 1;
  int64_t effectiveKw = attrs.dilationW * (shapeInfo.kw - 1) + 1;
  int64_t expectedOh =
      ((shapeInfo.h + 2 * attrs.paddingH - effectiveKh) / attrs.strideH) + 1;
  int64_t expectedOw =
      ((shapeInfo.w + 2 * attrs.paddingW - effectiveKw) / attrs.strideW) + 1;
  if (shapeInfo.oh != expectedOh || shapeInfo.ow != expectedOw)
    return mlir::failure();

  if (biasTy && biasTy->getShape()[0] != shapeInfo.fTotal)
    return mlir::failure();

  return shapeInfo;
}

static mlir::FailureOr<llvm::SmallVector<float>>
getFilterValues(ConstantOp filterConst) {
  if (auto denseAttr =
          llvm::dyn_cast<mlir::DenseFPElementsAttr>(filterConst.getValue())) {
    llvm::SmallVector<float> values;
    values.reserve(denseAttr.getNumElements());
    for (const llvm::APFloat &value : denseAttr.getValues<llvm::APFloat>())
      values.push_back(value.convertToFloat());
    return values;
  }

  if (auto denseResourceAttr =
          llvm::dyn_cast<mlir::DenseF32ResourceElementsAttr>(
              filterConst.getValue())) {
    std::optional<llvm::ArrayRef<float>> values =
        denseResourceAttr.tryGetAsArrayRef();
    if (!values)
      return mlir::failure();
    return llvm::SmallVector<float>(values->begin(), values->end());
  }

  return mlir::failure();
}

static mlir::TypedAttr
buildGroupedBlockDiagonalFilterAttr(ConstantOp filterConst,
                                    const Conv2DGroupedShapeInfo &shapeInfo,
                                    mlir::RankedTensorType flattenedTy) {
  auto maybeValues = getFilterValues(filterConst);
  if (failed(maybeValues))
    return {};

  llvm::SmallVector<float> sourceValues = *maybeValues;
  llvm::SmallVector<float> flattenedValues(flattenedTy.getNumElements(), 0.0f);
  int64_t flattenedCols = flattenedTy.getShape()[1];

  auto sourceIndex = [&](int64_t f, int64_t cgIdx, int64_t khIdx,
                         int64_t kwIdx) {
    return (((f * shapeInfo.cg + cgIdx) * shapeInfo.kh + khIdx) *
            shapeInfo.kw) +
           kwIdx;
  };

  auto destIndex = [&](int64_t f, int64_t channel, int64_t khIdx,
                       int64_t kwIdx) {
    int64_t channelOffset = channel * (shapeInfo.kh * shapeInfo.kw);
    int64_t khOffset = khIdx * shapeInfo.kw;
    int64_t flatIndex = channelOffset + khOffset + kwIdx;
    return f * flattenedCols + flatIndex;
  };

  for (int64_t group = 0; group < shapeInfo.groups; ++group) {
    for (int64_t fgIdx = 0; fgIdx < shapeInfo.fg; ++fgIdx) {
      int64_t f = group * shapeInfo.fg + fgIdx;
      for (int64_t cgIdx = 0; cgIdx < shapeInfo.cg; ++cgIdx) {
        int64_t channel = group * shapeInfo.cg + cgIdx;
        for (int64_t khIdx = 0; khIdx < shapeInfo.kh; ++khIdx) {
          for (int64_t kwIdx = 0; kwIdx < shapeInfo.kw; ++kwIdx) {
            flattenedValues[destIndex(f, channel, khIdx, kwIdx)] =
                sourceValues[sourceIndex(f, cgIdx, khIdx, kwIdx)];
          }
        }
      }
    }
  }

  if (llvm::isa<mlir::DenseF32ResourceElementsAttr>(filterConst.getValue())) {
    static uint64_t nextResourceId = 0;
    std::string resourceName =
        "analog_grouped_conv2d_filter_" + std::to_string(nextResourceId++);
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
        llvm::ArrayRef<float>(flattenedValues), /*dataIsMutable=*/false);
    return llvm::cast<mlir::TypedAttr>(mlir::DenseF32ResourceElementsAttr::get(
        flattenedTy, resourceName, std::move(blob)));
  }

  return llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(
      flattenedTy, llvm::ArrayRef<float>(flattenedValues)));
}

static mlir::FailureOr<ConstantOp> createBlockDiagonalFilter(
    ConstantOp filterConst, mlir::RankedTensorType flattenedTy,
    const Conv2DGroupedShapeInfo &shapeInfo, mlir::RewriterBase &rewriter) {
  mlir::TypedAttr flattenedAttr =
      buildGroupedBlockDiagonalFilterAttr(filterConst, shapeInfo, flattenedTy);
  if (!flattenedAttr)
    return mlir::failure();

  rewriter.setInsertionPointAfter(filterConst);
  return rewriter.create<ConstantOp>(filterConst.getLoc(), flattenedTy,
                                     flattenedAttr);
}

static mlir::FailureOr<Conv2DGroupedMatch>
matchSupportedGroupedConv2D(NNGroupedConv2DOp convOp,
                            mlir::RewriterBase &rewriter) {
  auto inputOutputTypes = converter_conv::getStaticF32InputOutputTypes(
      convOp.getInput(), convOp.getResult(), /*expectedRank=*/4);
  if (failed(inputOutputTypes))
    return mlir::failure();
  auto [sourceActivationTy, outputTy] = *inputOutputTypes;

  auto filter = converter_conv::getStaticF32FilterConstant(convOp.getWeight(),
                                                           /*expectedRank=*/4);
  if (failed(filter))
    return mlir::failure();
  auto [filterRank4Const, filterRank4Ty] = *filter;

  auto convAttrs = getSupportedGroupedConvolutionAttrs(convOp);
  if (failed(convAttrs))
    return mlir::failure();

  auto biasMatch =
      converter_conv::matchOptionalBias(convOp.getHasBias(), convOp.getBias());
  if (failed(biasMatch))
    return mlir::failure();

  int64_t groups = convOp.getGroupsAttr().getInt();
  auto inputShape = sourceActivationTy.getShape();
  mlir::RankedTensorType filterRank2Ty =
      buildGroupedFlattenedTensorType(filterRank4Ty, inputShape[1]);

  auto shapeInfo = getValidatedGroupedShapeInfo(
      sourceActivationTy, biasMatch->biasTy, filterRank4Ty, filterRank2Ty,
      outputTy, groups, *convAttrs);
  if (failed(shapeInfo))
    return mlir::failure();

  auto filterRank2Const = createBlockDiagonalFilter(
      filterRank4Const, filterRank2Ty, *shapeInfo, rewriter);
  if (failed(filterRank2Const))
    return mlir::failure();

  Conv2DGroupedMatch match;
  match.rootOp = convOp.getOperation();
  match.result = convOp.getResult();
  match.sourceActivation = convOp.getInput();
  match.filterRank4Const = filterRank4Const;
  match.filterRank2Const = *filterRank2Const;
  match.bias = biasMatch->bias;
  match.biasConstant = biasMatch->biasConstant;
  match.sourceActivationTy = sourceActivationTy;
  match.filterRank4Ty = filterRank4Ty;
  match.filterRank2Ty = filterRank2Ty;
  match.outputTy = outputTy;
  match.attrs = *convAttrs;
  match.shape = *shapeInfo;
  match.hasBias = convOp.getHasBias();
  return match;
}

static Conv2DGroupedLoweringState
buildGroupedLoweringState(const Conv2DGroupedMatch &match) {
  mlir::Location loc = match.rootOp->getLoc();
  mlir::Type elementType = match.sourceActivationTy.getElementType();
  int64_t flattenedWidth = match.shape.cTotal * match.shape.kh * match.shape.kw;
  return Conv2DGroupedLoweringState{
      .loc = loc,
      .elementType = elementType,
      .patchTy = mlir::RankedTensorType::get({1, flattenedWidth}, elementType),
      .matmulResultTy =
          mlir::RankedTensorType::get({1, match.shape.fTotal}, elementType),
      .outputTy = match.outputTy,
      .shape = match.shape,
      .attrs = match.attrs,
      .hasBias = match.hasBias,
  };
}

static PreparedGroupedFilter prepareGroupedFilter(Conv2DGroupedMatch &match) {
  return PreparedGroupedFilter{match.filterRank2Const.getResult()};
}

static PreparedGroupedBias
prepareGroupedBias(Conv2DGroupedMatch &match,
                   const Conv2DGroupedLoweringState &state,
                   mlir::OpBuilder &builder) {
  (void)state;
  (void)builder;

  PreparedGroupedBias preparedBias;
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

static mlir::Value buildGroupedFlattenedPatch(
    mlir::OpBuilder &builder, const Conv2DGroupedLoweringState &state,
    mlir::Value activation, int64_t outputH, int64_t outputW) {
  mlir::Value patchInit = builder.create<EmptyOp>(
      state.loc, state.patchTy.getShape(), state.elementType);

  mlir::Value zero = buildIndexConstant(builder, state.loc, 0);
  mlir::Value patch = patchInit;
  for (int64_t channel = 0; channel < state.shape.cTotal; ++channel) {
    mlir::Value channelIndex = buildIndexConstant(builder, state.loc, channel);
    for (int64_t kernelH = 0; kernelH < state.shape.kh; ++kernelH) {
      int64_t inputH = outputH * state.attrs.strideH +
                       kernelH * state.attrs.dilationH - state.attrs.paddingH;
      mlir::Value inputHValue = buildIndexConstant(builder, state.loc, inputH);
      for (int64_t kernelW = 0; kernelW < state.shape.kw; ++kernelW) {
        int64_t inputW = outputW * state.attrs.strideW +
                         kernelW * state.attrs.dilationW - state.attrs.paddingW;
        mlir::Value inputWValue =
            buildIndexConstant(builder, state.loc, inputW);
        mlir::Value inputValue = builder.create<mlir::tensor::ExtractOp>(
            state.loc, activation,
            mlir::ValueRange{zero, channelIndex, inputHValue, inputWValue});

        int64_t flattenedIndex = channel * state.shape.kh * state.shape.kw +
                                 kernelH * state.shape.kw + kernelW;
        mlir::Value flatIndex =
            buildIndexConstant(builder, state.loc, flattenedIndex);
        patch = builder.create<mlir::tensor::InsertOp>(
            state.loc, inputValue, patch, mlir::ValueRange{zero, flatIndex});
      }
    }
  }

  return patch;
}

static mlir::StringAttr buildOutputPositionName(mlir::OpBuilder &builder,
                                                int64_t outputH,
                                                int64_t outputW) {
  std::string name = "conv2d_grouped_oh_" + std::to_string(outputH) + "_ow_" +
                     std::to_string(outputW);
  return builder.getStringAttr(name);
}

static mlir::Value buildPatchPreparationRegion(
    mlir::OpBuilder &builder, const Conv2DGroupedMatch &match,
    const Conv2DGroupedLoweringState &state, int64_t outputH, int64_t outputW) {
  auto prepRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      state.loc, mlir::TypeRange{state.patchTy},
      mlir::ValueRange{match.sourceActivation}, "digital.conv_patch",
      buildOutputPositionName(builder, outputH, outputW));

  mlir::Block *body = new mlir::Block();
  prepRegion.getBody().push_back(body);
  body->addArgument(match.sourceActivation.getType(),
                    match.sourceActivation.getLoc());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value patch = buildGroupedFlattenedPatch(
      builder, state, body->getArgument(0), outputH, outputW);
  builder.create<mlir::sculptor::YieldOp>(state.loc, patch);
  return prepRegion.getResult(0);
}

static mlir::Value buildBiasAddRegion(mlir::OpBuilder &builder,
                                      PreparedGroupedBias &preparedBias,
                                      const Conv2DGroupedLoweringState &state,
                                      mlir::Value channelResult,
                                      int64_t outputH, int64_t outputW) {
  std::string name = "conv2d_grouped_oh_" + std::to_string(outputH) + "_ow_" +
                     std::to_string(outputW) + "_bias_add";
  auto biasAdd = builder.create<mlir::sculptor::TaskRegionOp>(
      state.loc, mlir::TypeRange{state.matmulResultTy},
      mlir::ValueRange{channelResult}, "digital.bias_add",
      builder.getStringAttr(name));

  mlir::Block *body = new mlir::Block();
  biasAdd.getBody().push_back(body);
  body->addArgument(channelResult.getType(), channelResult.getLoc());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value biasAddResult = body->getArgument(0);
  if (preparedBias.bias) {
    auto biasConstant = builder.create<ConstantOp>(
        preparedBias.biasConstant.getLoc(), preparedBias.biasConstant.getType(),
        preparedBias.biasConstant.getValue());
    biasAddResult = converter_conv::applyOptionalBias(
        state.loc, state.matmulResultTy, state.elementType, biasAddResult,
        biasConstant.getResult(), builder);
  }

  builder.create<mlir::sculptor::YieldOp>(state.loc, biasAddResult);
  return biasAdd.getResult(0);
}

static mlir::Value
buildOutputPosition(mlir::OpBuilder &builder, const Conv2DGroupedMatch &match,
                    const PreparedGroupedFilter &preparedFilter,
                    PreparedGroupedBias &preparedBias,
                    const Conv2DGroupedLoweringState &state, int64_t outputH,
                    int64_t outputW) {
  mlir::Value patch =
      buildPatchPreparationRegion(builder, match, state, outputH, outputW);
  mlir::Value channelResult =
      converter_conv::buildPatchMVM(state.loc, state.matmulResultTy, patch,
                                    preparedFilter.filterMatrix, builder);
  return buildBiasAddRegion(builder, preparedBias, state, channelResult,
                            outputH, outputW);
}

static mlir::FailureOr<mlir::Value>
assembleOutputTensor(mlir::OpBuilder &builder,
                     const Conv2DGroupedLoweringState &state,
                     llvm::ArrayRef<mlir::Value> positionResults) {
  if (positionResults.empty())
    return mlir::failure();

  auto outputAssembly = builder.create<mlir::sculptor::TaskRegionOp>(
      state.loc, mlir::TypeRange{state.outputTy},
      mlir::ValueRange(positionResults), "digital.output_recombine",
      builder.getStringAttr("conv2d_grouped_output_recombine"));

  mlir::Block *body = new mlir::Block();
  outputAssembly.getBody().push_back(body);
  for (mlir::Value positionResult : positionResults)
    body->addArgument(positionResult.getType(), positionResult.getLoc());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  mlir::RankedTensorType positionExpandedTy = mlir::RankedTensorType::get(
      {state.shape.n, state.shape.fTotal, 1, 1}, state.elementType);
  mlir::RankedTensorType rowTy = mlir::RankedTensorType::get(
      {state.shape.n, state.shape.fTotal, 1, state.shape.ow},
      state.elementType);
  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation = {{0},
                                                                    {1, 2, 3}};

  llvm::SmallVector<mlir::Value> rowResults;
  rowResults.reserve(state.shape.oh);
  int64_t positionIndex = 0;
  for (int64_t outputH = 0; outputH < state.shape.oh; ++outputH) {
    llvm::SmallVector<mlir::Value> expandedPositions;
    expandedPositions.reserve(state.shape.ow);
    for (int64_t outputW = 0; outputW < state.shape.ow; ++outputW) {
      mlir::Value positionResult = body->getArgument(positionIndex++);
      expandedPositions.push_back(
          builder
              .create<mlir::tensor::ExpandShapeOp>(
                  state.loc, positionExpandedTy, positionResult, reassociation)
              .getResult());
    }

    mlir::Value row = expandedPositions.front();
    if (expandedPositions.size() > 1) {
      row = builder
                .create<mlir::tensor::ConcatOp>(
                    state.loc, rowTy, /*dim=*/3,
                    mlir::ValueRange(expandedPositions))
                .getResult();
    }
    rowResults.push_back(row);
  }

  mlir::Value output = rowResults.front();
  if (rowResults.size() > 1) {
    output = builder
                 .create<mlir::tensor::ConcatOp>(state.loc, state.outputTy,
                                                 /*dim=*/2,
                                                 mlir::ValueRange(rowResults))
                 .getResult();
  }

  builder.create<mlir::sculptor::YieldOp>(state.loc, output);
  return outputAssembly.getResult(0);
}

static mlir::FailureOr<mlir::Value>
emitUnrolledGroupedOutputPositions(mlir::RewriterBase &rewriter,
                                   const Conv2DGroupedMatch &match,
                                   const PreparedGroupedFilter &preparedFilter,
                                   PreparedGroupedBias &preparedBias,
                                   const Conv2DGroupedLoweringState &state) {
  rewriter.setInsertionPointAfter(match.rootOp);
  llvm::SmallVector<mlir::Value> positionResults;
  positionResults.reserve(state.shape.oh * state.shape.ow);
  for (int64_t outputH = 0; outputH < state.shape.oh; ++outputH) {
    for (int64_t outputW = 0; outputW < state.shape.ow; ++outputW) {
      positionResults.push_back(
          buildOutputPosition(rewriter, match, preparedFilter, preparedBias,
                              state, outputH, outputW));
    }
  }

  return assembleOutputTensor(rewriter, state, positionResults);
}

static void eraseUnusedGroupedConv2DOps(Conv2DGroupedMatch &match,
                                        mlir::RewriterBase &rewriter) {
  mlir::sculptor::converter_rewrite::eraseIfUnused(match.rootOp, rewriter);
  mlir::sculptor::converter_rewrite::eraseIfUnused(
      match.filterRank4Const.getOperation(), rewriter);

  if (match.hasBias)
    mlir::sculptor::converter_rewrite::eraseIfUnused(
        match.biasConstant.getOperation(), rewriter);
}

// Converts extracted sculptor.nn.grouped_conv2d layer bodies into per-position
// sculptor.mvm execution.
class Conv2DGroupedConverter : public mlir::sculptor::LayerToMVMConverter {
public:
  mlir::StringRef getName() const override { return "conv2d_grouped"; }

  void lowerToMVM(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());
    auto layerOp =
        nn_layer_match::matchSingleNNLayerOp<NNGroupedConv2DOp>(func);
    if (failed(layerOp))
      return;
    if (!nn_layer_match::hasLayerTypeMatchingBias(func, "conv2d_grouped",
                                                  "conv2d_grouped_w_bias",
                                                  (*layerOp).getHasBias()))
      return;

    auto match = matchSupportedGroupedConv2D(*layerOp, rewriter);
    if (failed(match))
      return;

    Conv2DGroupedLoweringState state = buildGroupedLoweringState(*match);
    PreparedGroupedFilter preparedFilter = prepareGroupedFilter(*match);
    PreparedGroupedBias preparedBias =
        prepareGroupedBias(*match, state, rewriter);

    auto rewrittenOutput = emitUnrolledGroupedOutputPositions(
        rewriter, *match, preparedFilter, preparedBias, state);
    if (failed(rewrittenOutput))
      return;

    match->result.replaceAllUsesWith(*rewrittenOutput);
    eraseUnusedGroupedConv2DOps(*match, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

// Registers the grouped Conv2D converter for both bias forms outlined by the
// extractor.
void registerConv2DGroupedConverter(LayerToMVMConverters &converters,
                                    LayerToMVMConverterMap &converterMap,
                                    MLIRContext *context) {
  (void)context;
  auto converter = std::make_unique<Conv2DGroupedConverter>();
  const LayerToMVMConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));
  converterMap["conv2d_grouped"] = converterPtr;
  converterMap["conv2d_grouped_w_bias"] = converterPtr;
}

} // namespace sculptor
} // namespace mlir
