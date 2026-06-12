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
#include <string>

namespace nn_layer_match = mlir::sculptor::nn_layer_match;
namespace converter_constant = mlir::sculptor::converter_constant;
namespace converter_conv = mlir::sculptor::converter_conv;
namespace i64_array_attr = mlir::sculptor::i64_array_attr;

namespace {

using mlir::sculptor::NNConv3DOp;
using mlir::arith::ConstantOp;
using mlir::tensor::EmptyOp;

struct Conv3DShapeInfo {
  int64_t n;
  int64_t c;
  int64_t d;
  int64_t h;
  int64_t w;
  int64_t f;
  int64_t kd;
  int64_t kh;
  int64_t kw;
  int64_t od;
  int64_t oh;
  int64_t ow;
};

struct Conv3DConvolutionAttrs {
  int64_t strideD;
  int64_t strideH;
  int64_t strideW;
  int64_t paddingD;
  int64_t paddingH;
  int64_t paddingW;
  int64_t dilationD;
  int64_t dilationH;
  int64_t dilationW;
};

struct Conv3DMatch {
  mlir::Operation *rootOp = nullptr;
  mlir::Value result;
  mlir::Value activation;
  ConstantOp filterRank5Const;
  ConstantOp filterRank2Const;
  mlir::Value bias;
  ConstantOp biasConstant;
  mlir::RankedTensorType inputTy;
  mlir::RankedTensorType filterRank5Ty;
  mlir::RankedTensorType filterRank2Ty;
  mlir::RankedTensorType outputTy;
  Conv3DConvolutionAttrs attrs;
  Conv3DShapeInfo shape;
  bool hasBias = false;
};

struct PreparedFilter {
  mlir::Value filterMatrix;
};

struct PreparedBias {
  mlir::Value bias;
  ConstantOp biasConstant;
};

struct Conv3DLoweringState {
  mlir::Location loc;
  mlir::Type elementType;
  mlir::RankedTensorType patchTy;
  mlir::RankedTensorType matmulResultTy;
  mlir::RankedTensorType outputTy;
  Conv3DShapeInfo shape;
  Conv3DConvolutionAttrs attrs;
  bool hasBias = false;
};

static mlir::RankedTensorType
buildFlattenedTensorType(mlir::RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();
  int64_t flattenedCols = shape[1] * shape[2] * shape[3] * shape[4];
  return mlir::RankedTensorType::get({shape[0], flattenedCols},
                                     tensorTy.getElementType());
}

static mlir::FailureOr<Conv3DConvolutionAttrs>
getSupportedConvolutionAttrs(NNConv3DOp convOp) {
  llvm::SmallVector<int64_t> padding;
  if (!i64_array_attr::extract(convOp.getPadding(), /*expectedSize=*/3,
                               padding) ||
      !i64_array_attr::allEqual(padding, 0))
    return mlir::failure();

  llvm::SmallVector<int64_t> dilations;
  if (!i64_array_attr::extract(convOp.getDilation(), /*expectedSize=*/3,
                               dilations) ||
      !i64_array_attr::allEqual(dilations, 1))
    return mlir::failure();

  llvm::SmallVector<int64_t> strides;
  if (!i64_array_attr::extract(convOp.getStride(), /*expectedSize=*/3,
                               strides) ||
      !i64_array_attr::allPositive(strides))
    return mlir::failure();

  return Conv3DConvolutionAttrs{strides[0],   strides[1],   strides[2],
                                padding[0],   padding[1],   padding[2],
                                dilations[0], dilations[1], dilations[2]};
}

static mlir::FailureOr<Conv3DShapeInfo> getValidatedShapeInfo(
    mlir::RankedTensorType inputTy,
    std::optional<mlir::RankedTensorType> biasTy,
    mlir::RankedTensorType filterRank5Ty, mlir::RankedTensorType filterRank2Ty,
    mlir::RankedTensorType outputTy, const Conv3DConvolutionAttrs &attrs) {
  auto inputShape = inputTy.getShape();
  auto filterShape = filterRank5Ty.getShape();
  auto filterFlatShape = filterRank2Ty.getShape();
  auto outputShape = outputTy.getShape();

  Conv3DShapeInfo shapeInfo{
      inputShape[0],  inputShape[1],  inputShape[2],  inputShape[3],
      inputShape[4],  filterShape[0], filterShape[2], filterShape[3],
      filterShape[4], outputShape[2], outputShape[3], outputShape[4],
  };

  if (shapeInfo.n != 1)
    return mlir::failure();

  if (filterShape[1] != shapeInfo.c)
    return mlir::failure();

  if (outputShape[0] != shapeInfo.n || outputShape[1] != shapeInfo.f)
    return mlir::failure();

  if (filterFlatShape[0] != shapeInfo.f ||
      filterFlatShape[1] !=
          shapeInfo.c * shapeInfo.kd * shapeInfo.kh * shapeInfo.kw)
    return mlir::failure();

  int64_t effectiveKd = attrs.dilationD * (shapeInfo.kd - 1) + 1;
  int64_t effectiveKh = attrs.dilationH * (shapeInfo.kh - 1) + 1;
  int64_t effectiveKw = attrs.dilationW * (shapeInfo.kw - 1) + 1;
  if (effectiveKd > shapeInfo.d + 2 * attrs.paddingD ||
      effectiveKh > shapeInfo.h + 2 * attrs.paddingH ||
      effectiveKw > shapeInfo.w + 2 * attrs.paddingW)
    return mlir::failure();

  int64_t expectedOd =
      ((shapeInfo.d + 2 * attrs.paddingD - effectiveKd) / attrs.strideD) + 1;
  int64_t expectedOh =
      ((shapeInfo.h + 2 * attrs.paddingH - effectiveKh) / attrs.strideH) + 1;
  int64_t expectedOw =
      ((shapeInfo.w + 2 * attrs.paddingW - effectiveKw) / attrs.strideW) + 1;
  if (shapeInfo.od != expectedOd || shapeInfo.oh != expectedOh ||
      shapeInfo.ow != expectedOw)
    return mlir::failure();

  if (biasTy && biasTy->getShape()[0] != shapeInfo.f)
    return mlir::failure();

  return shapeInfo;
}

static mlir::FailureOr<ConstantOp>
createFlattenedFilter(ConstantOp filterConst,
                      mlir::RankedTensorType filterRank5Ty,
                      mlir::RewriterBase &rewriter) {
  mlir::RankedTensorType flattenedTy = buildFlattenedTensorType(filterRank5Ty);
  mlir::TypedAttr flattenedAttr =
      converter_constant::reshapeDenseOrResourceAttr(filterConst, flattenedTy);
  if (!flattenedAttr)
    return mlir::failure();

  rewriter.setInsertionPointAfter(filterConst);
  return rewriter.create<ConstantOp>(filterConst.getLoc(), flattenedTy,
                                     flattenedAttr);
}

static mlir::FailureOr<Conv3DMatch>
matchSupportedConv3D(NNConv3DOp convOp, mlir::RewriterBase &rewriter) {
  auto inputOutputTypes = converter_conv::getStaticF32InputOutputTypes(
      convOp.getInput(), convOp.getResult(), /*expectedRank=*/5);
  if (failed(inputOutputTypes))
    return mlir::failure();
  auto [inputTy, outputTy] = *inputOutputTypes;

  auto filterConstant =
      converter_conv::getStaticF32FilterConstant(convOp.getWeight(),
                                                 /*expectedRank=*/5);
  if (failed(filterConstant))
    return mlir::failure();
  auto [filterRank5Const, filterRank5Ty] = *filterConstant;

  auto convAttrs = getSupportedConvolutionAttrs(convOp);
  if (failed(convAttrs))
    return mlir::failure();

  auto biasMatch =
      converter_conv::matchOptionalBias(convOp.getHasBias(), convOp.getBias());
  if (failed(biasMatch))
    return mlir::failure();

  mlir::RankedTensorType filterRank2Ty =
      buildFlattenedTensorType(filterRank5Ty);
  auto shapeInfo =
      getValidatedShapeInfo(inputTy, biasMatch->biasTy, filterRank5Ty,
                            filterRank2Ty, outputTy, *convAttrs);
  if (failed(shapeInfo))
    return mlir::failure();

  auto filterRank2Const =
      createFlattenedFilter(filterRank5Const, filterRank5Ty, rewriter);
  if (failed(filterRank2Const))
    return mlir::failure();

  Conv3DMatch match;
  match.rootOp = convOp.getOperation();
  match.result = convOp.getResult();
  match.activation = convOp.getInput();
  match.filterRank5Const = filterRank5Const;
  match.filterRank2Const = *filterRank2Const;
  match.bias = biasMatch->bias;
  match.biasConstant = biasMatch->biasConstant;
  match.inputTy = inputTy;
  match.filterRank5Ty = filterRank5Ty;
  match.filterRank2Ty = filterRank2Ty;
  match.outputTy = outputTy;
  match.attrs = *convAttrs;
  match.shape = *shapeInfo;
  match.hasBias = convOp.getHasBias();
  return match;
}

static Conv3DLoweringState buildLoweringState(const Conv3DMatch &match) {
  mlir::Location loc = match.rootOp->getLoc();
  mlir::Type elementType = match.inputTy.getElementType();
  int64_t patchVolume = match.shape.kd * match.shape.kh * match.shape.kw;
  int64_t flattenedWidth = match.shape.c * patchVolume;
  return Conv3DLoweringState{
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

static PreparedFilter prepareFilter(Conv3DMatch &match) {
  return PreparedFilter{match.filterRank2Const.getResult()};
}

static PreparedBias prepareBias(Conv3DMatch &match,
                                const Conv3DLoweringState &state,
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
                                       const Conv3DLoweringState &state,
                                       mlir::Value activation, int64_t outputD,
                                       int64_t outputH, int64_t outputW) {
  mlir::Value patchInit = builder.create<EmptyOp>(
      state.loc, state.patchTy.getShape(), state.elementType);

  mlir::Value zero = buildIndexConstant(builder, state.loc, 0);
  mlir::Value patch = patchInit;
  for (int64_t channel = 0; channel < state.shape.c; ++channel) {
    mlir::Value channelIndex = buildIndexConstant(builder, state.loc, channel);
    for (int64_t kernelD = 0; kernelD < state.shape.kd; ++kernelD) {
      int64_t inputD = outputD * state.attrs.strideD +
                       kernelD * state.attrs.dilationD - state.attrs.paddingD;
      mlir::Value inputDValue = buildIndexConstant(builder, state.loc, inputD);
      for (int64_t kernelH = 0; kernelH < state.shape.kh; ++kernelH) {
        int64_t inputH = outputH * state.attrs.strideH +
                         kernelH * state.attrs.dilationH - state.attrs.paddingH;
        mlir::Value inputHValue =
            buildIndexConstant(builder, state.loc, inputH);
        for (int64_t kernelW = 0; kernelW < state.shape.kw; ++kernelW) {
          int64_t inputW = outputW * state.attrs.strideW +
                           kernelW * state.attrs.dilationW -
                           state.attrs.paddingW;
          mlir::Value inputWValue =
              buildIndexConstant(builder, state.loc, inputW);
          mlir::Value inputValue = builder.create<mlir::tensor::ExtractOp>(
              state.loc, activation,
              mlir::ValueRange{zero, channelIndex, inputDValue, inputHValue,
                               inputWValue});

          int64_t flattenedIndex =
              channel * state.shape.kd * state.shape.kh * state.shape.kw +
              kernelD * state.shape.kh * state.shape.kw +
              kernelH * state.shape.kw + kernelW;
          mlir::Value flatIndex =
              buildIndexConstant(builder, state.loc, flattenedIndex);
          patch = builder.create<mlir::tensor::InsertOp>(
              state.loc, inputValue, patch, mlir::ValueRange{zero, flatIndex});
        }
      }
    }
  }

  return patch;
}

static mlir::StringAttr buildOutputPositionName(mlir::OpBuilder &builder,
                                                int64_t outputD,
                                                int64_t outputH,
                                                int64_t outputW) {
  std::string name = "conv3d_od_" + std::to_string(outputD) + "_oh_" +
                     std::to_string(outputH) + "_ow_" + std::to_string(outputW);
  return builder.getStringAttr(name);
}

static mlir::Value buildPatchPreparationRegion(mlir::OpBuilder &builder,
                                               const Conv3DMatch &match,
                                               const Conv3DLoweringState &state,
                                               int64_t outputD, int64_t outputH,
                                               int64_t outputW) {
  auto prepRegion = builder.create<mlir::sculptor::TaskRegionOp>(
      state.loc, mlir::TypeRange{state.patchTy},
      mlir::ValueRange{match.activation}, "digital.conv_patch",
      buildOutputPositionName(builder, outputD, outputH, outputW));

  mlir::Block *body = new mlir::Block();
  prepRegion.getBody().push_back(body);
  body->addArgument(match.activation.getType(), match.activation.getLoc());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::Value patch = buildFlattenedPatch(builder, state, body->getArgument(0),
                                          outputD, outputH, outputW);
  builder.create<mlir::sculptor::YieldOp>(state.loc, patch);
  return prepRegion.getResult(0);
}

static mlir::Value
buildBiasAddRegion(mlir::OpBuilder &builder, PreparedBias &preparedBias,
                   const Conv3DLoweringState &state, mlir::Value channelResult,
                   int64_t outputD, int64_t outputH, int64_t outputW) {
  std::string name = "conv3d_od_" + std::to_string(outputD) + "_oh_" +
                     std::to_string(outputH) + "_ow_" +
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

static mlir::Value buildOutputPosition(mlir::OpBuilder &builder,
                                       const Conv3DMatch &match,
                                       const PreparedFilter &preparedFilter,
                                       PreparedBias &preparedBias,
                                       const Conv3DLoweringState &state,
                                       int64_t outputD, int64_t outputH,
                                       int64_t outputW) {
  mlir::Value patch = buildPatchPreparationRegion(builder, match, state,
                                                  outputD, outputH, outputW);
  mlir::Value channelResult =
      converter_conv::buildPatchMVM(state.loc, state.matmulResultTy, patch,
                                    preparedFilter.filterMatrix, builder);
  return buildBiasAddRegion(builder, preparedBias, state, channelResult,
                            outputD, outputH, outputW);
}

static mlir::FailureOr<mlir::Value>
assembleOutputTensor(mlir::OpBuilder &builder, const Conv3DLoweringState &state,
                     llvm::ArrayRef<mlir::Value> positionResults) {
  if (positionResults.empty())
    return mlir::failure();

  auto outputAssembly = builder.create<mlir::sculptor::TaskRegionOp>(
      state.loc, mlir::TypeRange{state.outputTy},
      mlir::ValueRange(positionResults), "digital.output_recombine",
      builder.getStringAttr("conv3d_output_recombine"));

  mlir::Block *body = new mlir::Block();
  outputAssembly.getBody().push_back(body);
  for (mlir::Value positionResult : positionResults)
    body->addArgument(positionResult.getType(), positionResult.getLoc());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  mlir::RankedTensorType positionExpandedTy = mlir::RankedTensorType::get(
      {state.shape.n, state.shape.f, 1, 1, 1}, state.elementType);
  mlir::RankedTensorType rowTy = mlir::RankedTensorType::get(
      {state.shape.n, state.shape.f, 1, 1, state.shape.ow}, state.elementType);
  mlir::RankedTensorType planeTy = mlir::RankedTensorType::get(
      {state.shape.n, state.shape.f, 1, state.shape.oh, state.shape.ow},
      state.elementType);
  llvm::SmallVector<mlir::ReassociationIndices, 2> reassociation = {
      {0}, {1, 2, 3, 4}};

  llvm::SmallVector<mlir::Value> depthResults;
  depthResults.reserve(state.shape.od);
  int64_t positionIndex = 0;
  for (int64_t outputD = 0; outputD < state.shape.od; ++outputD) {
    llvm::SmallVector<mlir::Value> rowResults;
    rowResults.reserve(state.shape.oh);
    for (int64_t outputH = 0; outputH < state.shape.oh; ++outputH) {
      llvm::SmallVector<mlir::Value> expandedPositions;
      expandedPositions.reserve(state.shape.ow);
      for (int64_t outputW = 0; outputW < state.shape.ow; ++outputW) {
        mlir::Value positionResult = body->getArgument(positionIndex++);
        expandedPositions.push_back(builder
                                        .create<mlir::tensor::ExpandShapeOp>(
                                            state.loc, positionExpandedTy,
                                            positionResult, reassociation)
                                        .getResult());
      }

      mlir::Value row = expandedPositions.front();
      if (expandedPositions.size() > 1) {
        row = builder
                  .create<mlir::tensor::ConcatOp>(
                      state.loc, rowTy, /*dim=*/4,
                      mlir::ValueRange(expandedPositions))
                  .getResult();
      }
      rowResults.push_back(row);
    }

    mlir::Value plane = rowResults.front();
    if (rowResults.size() > 1) {
      plane = builder
                  .create<mlir::tensor::ConcatOp>(state.loc, planeTy, /*dim=*/3,
                                                  mlir::ValueRange(rowResults))
                  .getResult();
    }
    depthResults.push_back(plane);
  }

  mlir::Value output = depthResults.front();
  if (depthResults.size() > 1) {
    output = builder
                 .create<mlir::tensor::ConcatOp>(state.loc, state.outputTy,
                                                 /*dim=*/2,
                                                 mlir::ValueRange(depthResults))
                 .getResult();
  }

  builder.create<mlir::sculptor::YieldOp>(state.loc, output);
  return outputAssembly.getResult(0);
}

static mlir::FailureOr<mlir::Value> emitUnrolledOutputPositions(
    mlir::RewriterBase &rewriter, const Conv3DMatch &match,
    const PreparedFilter &preparedFilter, PreparedBias &preparedBias,
    const Conv3DLoweringState &state) {
  rewriter.setInsertionPointAfter(match.rootOp);
  llvm::SmallVector<mlir::Value> positionResults;
  positionResults.reserve(state.shape.od * state.shape.oh * state.shape.ow);
  for (int64_t outputD = 0; outputD < state.shape.od; ++outputD) {
    for (int64_t outputH = 0; outputH < state.shape.oh; ++outputH) {
      for (int64_t outputW = 0; outputW < state.shape.ow; ++outputW) {
        positionResults.push_back(
            buildOutputPosition(rewriter, match, preparedFilter, preparedBias,
                                state, outputD, outputH, outputW));
      }
    }
  }

  return assembleOutputTensor(rewriter, state, positionResults);
}

static void eraseUnusedConv3DOps(Conv3DMatch &match,
                                 mlir::RewriterBase &rewriter) {
  mlir::sculptor::converter_rewrite::eraseIfUnused(match.rootOp, rewriter);
  mlir::sculptor::converter_rewrite::eraseIfUnused(
      match.filterRank5Const.getOperation(), rewriter);

  if (match.hasBias)
    mlir::sculptor::converter_rewrite::eraseIfUnused(
        match.biasConstant.getOperation(), rewriter);
}

// Converts extracted sculptor.nn.conv3d layer bodies into per-position sculptor.mvm
// execution.
class Conv3DConverter : public mlir::sculptor::LayerToMVMConverter {
public:
  mlir::StringRef getName() const override { return "conv3d"; }

  void lowerToMVM(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());
    auto layerOp = nn_layer_match::matchSingleNNLayerOp<NNConv3DOp>(func);
    if (failed(layerOp))
      return;

    if (!nn_layer_match::hasLayerTypeMatchingBias(
            func, "conv3d", "conv3d_w_bias", (*layerOp).getHasBias()))
      return;

    mlir::FailureOr<Conv3DMatch> match =
        matchSupportedConv3D(*layerOp, rewriter);
    if (failed(match))
      return;

    Conv3DLoweringState state = buildLoweringState(*match);
    PreparedFilter preparedFilter = prepareFilter(*match);
    PreparedBias preparedBias = prepareBias(*match, state, rewriter);

    auto rewrittenOutput = emitUnrolledOutputPositions(
        rewriter, *match, preparedFilter, preparedBias, state);
    if (failed(rewrittenOutput))
      return;

    match->result.replaceAllUsesWith(*rewrittenOutput);
    eraseUnusedConv3DOps(*match, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

// Registers the Conv3D converter for both biased and bias-free layer slices.
void registerConv3DConverter(LayerToMVMConverters &converters,
                             LayerToMVMConverterMap &converterMap,
                             MLIRContext *context) {
  (void)context;

  auto converter = std::make_unique<Conv3DConverter>();
  const LayerToMVMConverter *converterPtr = converter.get();
  converters.push_back(std::move(converter));

  converterMap["conv3d"] = converterPtr;
  converterMap["conv3d_w_bias"] = converterPtr;
}

} // namespace sculptor
} // namespace mlir
