#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_RECURRENTLAYERCONVERSIONUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_RECURRENTLAYERCONVERSIONUTILS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/RewriteUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace converter_recurrent_layer {

inline SmallVector<OpFoldResult>
buildIndexAttrs(OpBuilder &builder, ArrayRef<int64_t> values) {
  SmallVector<OpFoldResult> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getIndexAttr(value));
  return attrs;
}

// Extracts one layer's initial state from stacked recurrent state.
inline Value extractLayerState(Location loc, Value stackedState,
                               int64_t layerIndex, int64_t batchSize,
                               int64_t hiddenSize, RankedTensorType sliceTy,
                               RankedTensorType state2DTy,
                               OpBuilder &builder) {
  auto slice = builder.create<tensor::ExtractSliceOp>(
      loc, sliceTy, stackedState, buildIndexAttrs(builder, {layerIndex, 0, 0}),
      buildIndexAttrs(builder, {1, batchSize, hiddenSize}),
      buildIndexAttrs(builder, {1, 1, 1}));

  SmallVector<ReassociationIndices, 2> reassociation = {{0, 1}, {2}};
  return builder
      .create<tensor::CollapseShapeOp>(loc, state2DTy, slice.getResult(),
                                       reassociation)
      .getResult();
}

// Slices one timestep from batch-first sequence input.
inline Value extractBatchFirstTimestep(Location loc, Value input,
                                       OpFoldResult timestep,
                                       int64_t batchSize, int64_t inputSize,
                                       RankedTensorType sliceTy,
                                       RankedTensorType input2DTy,
                                       OpBuilder &builder) {
  SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(0), timestep,
                                       builder.getIndexAttr(0)};
  auto slice = builder.create<tensor::ExtractSliceOp>(
      loc, sliceTy, input, offsets,
      buildIndexAttrs(builder, {batchSize, 1, inputSize}),
      buildIndexAttrs(builder, {1, 1, 1}));

  SmallVector<ReassociationIndices, 2> reassociation = {{0}, {1, 2}};
  return builder
      .create<tensor::CollapseShapeOp>(loc, input2DTy, slice.getResult(),
                                       reassociation)
      .getResult();
}

inline Value extractBatchRow(Location loc, Value value, RankedTensorType rowTy,
                             int64_t row, int64_t width,
                             OpBuilder &builder) {
  return builder
      .create<tensor::ExtractSliceOp>(loc, rowTy, value,
                                      buildIndexAttrs(builder, {row, 0}),
                                      buildIndexAttrs(builder, {1, width}),
                                      buildIndexAttrs(builder, {1, 1}))
      .getResult();
}

inline Value expandTimestepState(Location loc, Value timestepState,
                                 RankedTensorType resultTy,
                                 OpBuilder &builder) {
  SmallVector<ReassociationIndices, 2> reassociation = {{0}, {1, 2}};
  return builder
      .create<tensor::ExpandShapeOp>(loc, resultTy, timestepState,
                                     reassociation)
      .getResult();
}

// Inserts one timestep result into batch-first sequence output.
inline Value insertBatchFirstTimestep(Location loc, Value timestepState,
                                      Value sequenceOutput,
                                      OpFoldResult timestep,
                                      int64_t batchSize, int64_t hiddenSize,
                                      RankedTensorType timestepResultTy,
                                      OpBuilder &builder) {
  Value timestepOutput =
      expandTimestepState(loc, timestepState, timestepResultTy, builder);
  SmallVector<OpFoldResult> offsets = {builder.getIndexAttr(0), timestep,
                                       builder.getIndexAttr(0)};
  return builder
      .create<tensor::InsertSliceOp>(
          loc, timestepOutput, sequenceOutput, offsets,
          buildIndexAttrs(builder, {batchSize, 1, hiddenSize}),
          buildIndexAttrs(builder, {1, 1, 1}))
      .getResult();
}

inline Value expandFinalLayerState(Location loc, Value finalState,
                                   RankedTensorType resultTy,
                                   OpBuilder &builder) {
  SmallVector<ReassociationIndices, 2> reassociation = {{0, 1}, {2}};
  return builder
      .create<tensor::ExpandShapeOp>(loc, resultTy, finalState, reassociation)
      .getResult();
}

inline Value expandRowBias(Location loc, Value bias,
                           RankedTensorType expandedBiasTy,
                           OpBuilder &builder) {
  SmallVector<ReassociationIndices, 1> reassociation = {{0, 1}};
  return builder
      .create<tensor::ExpandShapeOp>(loc, expandedBiasTy, bias, reassociation)
      .getResult();
}

// Adds a row-broadcast bias to recurrent preactivation.
inline Value addBroadcastRowBias(Location loc, Value preActivation,
                                 Value expandedBias,
                                 RankedTensorType resultTy,
                                 OpBuilder &builder) {
  Value init =
      builder.create<tensor::EmptyOp>(loc, resultTy.getShape(),
                                      resultTy.getElementType());
  AffineMap valueMap = builder.getMultiDimIdentityMap(resultTy.getRank());
  AffineMap biasMap = AffineMap::get(
      /*dimCount=*/resultTy.getRank(), /*symbolCount=*/0,
      {builder.getAffineConstantExpr(0), builder.getAffineDimExpr(1)},
      builder.getContext());
  SmallVector<AffineMap, 3> indexingMaps = {valueMap, biasMap, valueMap};
  SmallVector<utils::IteratorType, 2> iteratorTypes(
      resultTy.getRank(), utils::IteratorType::parallel);

  return builder
      .create<linalg::GenericOp>(
          loc, resultTy, ValueRange{preActivation, expandedBias},
          ValueRange{init}, indexingMaps, iteratorTypes,
          [](OpBuilder &builder, Location nestedLoc, ValueRange args) {
            Value biased =
                builder.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
            builder.create<linalg::YieldOp>(nestedLoc, biased);
          })
      .getResult(0);
}

inline void eraseUnusedConstants(ArrayRef<arith::ConstantOp> constants,
                                 RewriterBase &rewriter) {
  SmallPtrSet<Operation *, 4> uniqueConstants;
  for (arith::ConstantOp constant : constants) {
    if (constant)
      uniqueConstants.insert(constant.getOperation());
  }

  for (Operation *constant : uniqueConstants)
    converter_rewrite::eraseIfUnused(constant, rewriter);
}

} // namespace converter_recurrent_layer
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_RECURRENTLAYERCONVERSIONUTILS_H
