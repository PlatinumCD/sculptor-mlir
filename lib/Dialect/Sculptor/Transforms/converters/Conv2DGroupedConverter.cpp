#include "sculptor-mlir/Dialect/Sculptor/Transforms/ConvertLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/ConvConversionUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/I64ArrayAttrUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/NNLayerMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/RewriteUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/SemanticOperationScope.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/SemanticOperationNames.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
namespace semantic_operation_names = mlir::sculptor::semantic_operation_names;

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
  mlir::RankedTensorType patchSequenceTy;
  mlir::RankedTensorType matmulResultTy;
  mlir::RankedTensorType mvmSequenceTy;
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
  int64_t outputPositions = match.shape.oh * match.shape.ow;
  return Conv2DGroupedLoweringState{
      .loc = loc,
      .elementType = elementType,
      .patchTy = mlir::RankedTensorType::get({1, flattenedWidth}, elementType),
      .patchSequenceTy = mlir::RankedTensorType::get(
          {outputPositions, flattenedWidth}, elementType),
      .matmulResultTy =
          mlir::RankedTensorType::get({1, match.shape.fTotal}, elementType),
      .mvmSequenceTy = mlir::RankedTensorType::get(
          {outputPositions, match.shape.fTotal}, elementType),
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

static mlir::Value buildScaledIndex(mlir::OpBuilder &builder,
                                    mlir::Location loc, mlir::Value output,
                                    int64_t stride, mlir::Value kernel,
                                    int64_t dilation, int64_t padding) {
  mlir::Value strideValue = buildIndexConstant(builder, loc, stride);
  mlir::Value dilationValue = buildIndexConstant(builder, loc, dilation);
  mlir::Value scaledOutput =
      builder.create<mlir::arith::MulIOp>(loc, output, strideValue);
  mlir::Value scaledKernel =
      builder.create<mlir::arith::MulIOp>(loc, kernel, dilationValue);
  mlir::Value inputIndex =
      builder.create<mlir::arith::AddIOp>(loc, scaledOutput, scaledKernel);
  if (padding == 0)
    return inputIndex;
  mlir::Value paddingValue = buildIndexConstant(builder, loc, padding);
  return builder.create<mlir::arith::SubIOp>(loc, inputIndex, paddingValue);
}

static mlir::Value buildGroupedPatchSequence(
    mlir::OpBuilder &builder, const Conv2DGroupedLoweringState &state,
    mlir::Value activation) {
  int64_t kernelPlane = state.shape.kh * state.shape.kw;
  mlir::Value init = builder.create<EmptyOp>(
      state.loc, state.patchSequenceTy.getShape(), state.elementType);
  mlir::AffineMap identity = builder.getMultiDimIdentityMap(2);
  mlir::AffineMap activationDependency = mlir::AffineMap::get(
      /*dimCount=*/2, /*symbolCount=*/0,
      {builder.getAffineConstantExpr(0), builder.getAffineConstantExpr(0),
       builder.getAffineConstantExpr(0), builder.getAffineConstantExpr(0)},
      builder.getContext());
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      2, mlir::utils::IteratorType::parallel);
  auto patches = builder.create<mlir::linalg::GenericOp>(
      state.loc, state.patchSequenceTy, mlir::ValueRange{activation},
      mlir::ValueRange{init},
      llvm::ArrayRef<mlir::AffineMap>{activationDependency, identity},
      iterators,
      [&](mlir::OpBuilder &bodyBuilder, mlir::Location bodyLoc,
          mlir::ValueRange) {
        mlir::Value position =
            bodyBuilder.create<mlir::linalg::IndexOp>(bodyLoc, 0);
        mlir::Value flattenedIndex =
            bodyBuilder.create<mlir::linalg::IndexOp>(bodyLoc, 1);
        mlir::Value outputWidth =
            buildIndexConstant(bodyBuilder, bodyLoc, state.shape.ow);
        mlir::Value kernelPlaneValue =
            buildIndexConstant(bodyBuilder, bodyLoc, kernelPlane);
        mlir::Value kernelWidth =
            buildIndexConstant(bodyBuilder, bodyLoc, state.shape.kw);
        mlir::Value zero = buildIndexConstant(bodyBuilder, bodyLoc, 0);
        mlir::Value outputH = bodyBuilder.create<mlir::arith::DivUIOp>(
            bodyLoc, position, outputWidth);
        mlir::Value outputW = bodyBuilder.create<mlir::arith::RemUIOp>(
            bodyLoc, position, outputWidth);
        mlir::Value channel = bodyBuilder.create<mlir::arith::DivUIOp>(
            bodyLoc, flattenedIndex, kernelPlaneValue);
        mlir::Value kernelOffset = bodyBuilder.create<mlir::arith::RemUIOp>(
            bodyLoc, flattenedIndex, kernelPlaneValue);
        mlir::Value kernelH = bodyBuilder.create<mlir::arith::DivUIOp>(
            bodyLoc, kernelOffset, kernelWidth);
        mlir::Value kernelW = bodyBuilder.create<mlir::arith::RemUIOp>(
            bodyLoc, kernelOffset, kernelWidth);
        mlir::Value inputH = buildScaledIndex(
            bodyBuilder, bodyLoc, outputH, state.attrs.strideH, kernelH,
            state.attrs.dilationH, state.attrs.paddingH);
        mlir::Value inputW = buildScaledIndex(
            bodyBuilder, bodyLoc, outputW, state.attrs.strideW, kernelW,
            state.attrs.dilationW, state.attrs.paddingW);
        mlir::Value inputValue = bodyBuilder.create<mlir::tensor::ExtractOp>(
            bodyLoc, activation,
            mlir::ValueRange{zero, channel, inputH, inputW});
        bodyBuilder.create<mlir::linalg::YieldOp>(bodyLoc, inputValue);
      });
  return patches.getResult(0);
}

static mlir::Value buildGroupedPatchPreparationStage(
    mlir::OpBuilder &builder, const Conv2DGroupedMatch &match,
    const Conv2DGroupedLoweringState &state) {
  mlir::sculptor::SemanticOperationScope scope(
      builder, semantic_operation_names::kConvPatchTaskKind,
      "conv2d_grouped_patch_sequence");
  mlir::Value patches = buildGroupedPatchSequence(
      builder, state, match.sourceActivation);
  scope.annotate();
  return patches;
}

static llvm::SmallVector<mlir::OpFoldResult>
buildRowSliceOffsets(mlir::OpBuilder &builder, mlir::Value row) {
  return {row, builder.getIndexAttr(0)};
}

static llvm::SmallVector<mlir::OpFoldResult>
buildRowSliceSizes(mlir::OpBuilder &builder, int64_t width) {
  return {builder.getIndexAttr(1), builder.getIndexAttr(width)};
}

static llvm::SmallVector<mlir::OpFoldResult>
buildUnitStrides(mlir::OpBuilder &builder) {
  return {builder.getIndexAttr(1), builder.getIndexAttr(1)};
}

static mlir::Value buildGroupedMVMSequenceStage(
    mlir::OpBuilder &builder, const PreparedGroupedFilter &preparedFilter,
    const Conv2DGroupedLoweringState &state, mlir::Value patchSequence) {
  mlir::sculptor::SemanticOperationScope scope(
      builder, semantic_operation_names::kMVMSequenceTaskKind,
      "conv2d_grouped_mvm_sequence");
  int64_t outputPositions = state.shape.oh * state.shape.ow;
  mlir::Value zero = buildIndexConstant(builder, state.loc, 0);
  mlir::Value one = buildIndexConstant(builder, state.loc, 1);
  mlir::Value upper =
      buildIndexConstant(builder, state.loc, outputPositions);
  mlir::Value init = builder.create<EmptyOp>(
      state.loc, state.mvmSequenceTy.getShape(), state.elementType);
  auto patchSizes = buildRowSliceSizes(builder, state.patchTy.getDimSize(1));
  auto strides = buildUnitStrides(builder);
  auto positionLoop = builder.create<mlir::scf::ForOp>(
      state.loc, zero, upper, one, mlir::ValueRange{init},
      [&](mlir::OpBuilder &loopBuilder, mlir::Location loopLoc,
          mlir::Value position, mlir::ValueRange iterArgs) {
        auto offsets = buildRowSliceOffsets(loopBuilder, position);
        mlir::Value patch =
            loopBuilder
                .create<mlir::tensor::ExtractSliceOp>(
                    loopLoc, state.patchTy, patchSequence, offsets,
                    patchSizes, strides)
                .getResult();
        mlir::Value channelResult = converter_conv::buildPatchMVM(
            loopLoc, state.matmulResultTy, patch,
            preparedFilter.filterMatrix, loopBuilder);
        mlir::Value updated =
            loopBuilder
                .create<mlir::tensor::InsertSliceOp>(
                    loopLoc, channelResult, iterArgs[0], offsets,
                    buildRowSliceSizes(loopBuilder, state.shape.fTotal),
                    strides)
                .getResult();
        loopBuilder.create<mlir::scf::YieldOp>(loopLoc, updated);
      });
  builder.setInsertionPointAfter(positionLoop);
  scope.annotate();
  return positionLoop.getResult(0);
}

static mlir::Value buildGroupedOutputAssemblyStage(
    mlir::OpBuilder &builder, PreparedGroupedBias &preparedBias,
    const Conv2DGroupedLoweringState &state, mlir::Value sequenceResult) {
  llvm::StringRef kind = "digital.output_recombine";
  if (preparedBias.bias)
    kind = semantic_operation_names::kBiasAddTaskKind;
  mlir::sculptor::SemanticOperationScope scope(
      builder, kind, "conv2d_grouped_output_assembly");
  mlir::Value bias;
  if (preparedBias.bias) {
    bias = builder
               .create<ConstantOp>(preparedBias.biasConstant.getLoc(),
                                   preparedBias.biasConstant.getType(),
                                   preparedBias.biasConstant.getValue())
               .getResult();
  }
  mlir::Value init = builder.create<EmptyOp>(
      state.loc, state.outputTy.getShape(), state.elementType);
  mlir::AffineExpr batch = builder.getAffineDimExpr(0);
  mlir::AffineExpr channel = builder.getAffineDimExpr(1);
  mlir::AffineExpr outputH = builder.getAffineDimExpr(2);
  mlir::AffineExpr outputW = builder.getAffineDimExpr(3);
  mlir::AffineExpr position =
      batch * (state.shape.oh * state.shape.ow) +
      outputH * state.shape.ow + outputW;
  mlir::AffineMap sequenceMap = mlir::AffineMap::get(
      /*dimCount=*/4, /*symbolCount=*/0, {position, channel},
      builder.getContext());
  mlir::AffineMap outputMap = builder.getMultiDimIdentityMap(4);
  llvm::SmallVector<mlir::AffineMap> indexingMaps{sequenceMap};
  llvm::SmallVector<mlir::Value> inputs{sequenceResult};
  if (bias) {
    indexingMaps.push_back(mlir::AffineMap::get(
        /*dimCount=*/4, /*symbolCount=*/0, {channel}, builder.getContext()));
    inputs.push_back(bias);
  }
  indexingMaps.push_back(outputMap);
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      4, mlir::utils::IteratorType::parallel);
  auto output = builder.create<mlir::linalg::GenericOp>(
      state.loc, state.outputTy, inputs, mlir::ValueRange{init}, indexingMaps,
      iterators,
      [&](mlir::OpBuilder &bodyBuilder, mlir::Location bodyLoc,
          mlir::ValueRange arguments) {
        mlir::Value value = arguments.front();
        if (bias)
          value = bodyBuilder.create<mlir::arith::AddFOp>(
              bodyLoc, value, arguments[1]);
        bodyBuilder.create<mlir::linalg::YieldOp>(bodyLoc, value);
      });
  builder.setInsertionPointAfter(output);
  scope.annotate();
  return output.getResult(0);
}

static mlir::Value emitLoopedGroupedConvolution(
    mlir::RewriterBase &rewriter, const Conv2DGroupedMatch &match,
    const PreparedGroupedFilter &preparedFilter,
    PreparedGroupedBias &preparedBias,
    const Conv2DGroupedLoweringState &state) {
  rewriter.setInsertionPointAfter(match.rootOp);
  mlir::Value patches =
      buildGroupedPatchPreparationStage(rewriter, match, state);
  mlir::Value sequence = buildGroupedMVMSequenceStage(
      rewriter, preparedFilter, state, patches);
  return buildGroupedOutputAssemblyStage(rewriter, preparedBias, state,
                                         sequence);
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

static mlir::LogicalResult
lowerGroupedConv2DOp(NNGroupedConv2DOp layerOp,
                     mlir::RewriterBase &rewriter) {
  auto match = matchSupportedGroupedConv2D(layerOp, rewriter);
  if (failed(match))
    return mlir::failure();

  Conv2DGroupedLoweringState state = buildGroupedLoweringState(*match);
  PreparedGroupedFilter preparedFilter = prepareGroupedFilter(*match);
  PreparedGroupedBias preparedBias =
      prepareGroupedBias(*match, state, rewriter);
  mlir::Value rewrittenOutput = emitLoopedGroupedConvolution(
      rewriter, *match, preparedFilter, preparedBias, state);
  match->result.replaceAllUsesWith(rewrittenOutput);
  eraseUnusedGroupedConv2DOps(*match, rewriter);
  return mlir::success();
}

// Converts extracted sculptor.nn.grouped_conv2d layer bodies into looped
// sculptor.mvm execution without statically unrolling output positions.
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

    (void)lowerGroupedConv2DOp(*layerOp, rewriter);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

LogicalResult decomposeInlineGroupedConv2DLayers(func::FuncOp func) {
  SmallVector<NNGroupedConv2DOp> layerOps;
  func.walk(
      [&](NNGroupedConv2DOp layerOp) { layerOps.push_back(layerOp); });

  SemanticLayerRewriteListener layerListener;
  IRRewriter rewriter(func.getContext(), &layerListener);
  for (NNGroupedConv2DOp layerOp : layerOps) {
    if (!layerOp || !layerOp->getBlock())
      continue;
    SemanticLayerRewriteScope layerScope(layerListener, layerOp);
    if (failed(lowerGroupedConv2DOp(layerOp, rewriter))) {
      layerOp.emitOpError("failed to decompose inline grouped Conv2D layer");
      return failure();
    }
  }
  return success();
}

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
