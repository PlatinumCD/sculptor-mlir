#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_CONVCONVERSIONUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_CONVCONVERSIONUTILS_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Conversion/MVMBuildUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <optional>
#include <utility>

namespace mlir {
namespace sculptor {
namespace converter_conv {

struct ConvBiasMatch {
  Value bias;
  arith::ConstantOp biasConstant;
  std::optional<RankedTensorType> biasTy;
};

inline FailureOr<std::pair<RankedTensorType, RankedTensorType>>
getStaticF32InputOutputTypes(Value input, Value output, int64_t expectedRank) {
  auto inputTy = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto outputTy = llvm::dyn_cast<RankedTensorType>(output.getType());
  if (!inputTy || !outputTy || !inputTy.hasStaticShape() ||
      !outputTy.hasStaticShape())
    return failure();

  if (inputTy.getRank() != expectedRank || outputTy.getRank() != expectedRank)
    return failure();

  if (!inputTy.getElementType().isF32() || !outputTy.getElementType().isF32())
    return failure();

  return std::make_pair(inputTy, outputTy);
}

inline FailureOr<std::pair<arith::ConstantOp, RankedTensorType>>
getStaticF32FilterConstant(Value filter, int64_t expectedRank) {
  auto filterConst = filter.getDefiningOp<arith::ConstantOp>();
  if (!filterConst)
    return failure();

  auto filterTy = llvm::dyn_cast<RankedTensorType>(filterConst.getType());
  if (!filterTy || !filterTy.hasStaticShape() ||
      filterTy.getRank() != expectedRank || !filterTy.getElementType().isF32())
    return failure();

  return std::make_pair(filterConst, filterTy);
}

inline FailureOr<ConvBiasMatch> matchOptionalBias(bool hasBias, Value bias) {
  ConvBiasMatch match;
  if (!hasBias)
    return match;

  if (!bias)
    return failure();

  auto biasTy = llvm::dyn_cast<RankedTensorType>(bias.getType());
  if (!biasTy || !biasTy.hasStaticShape() || biasTy.getRank() != 1 ||
      !biasTy.getElementType().isF32())
    return failure();

  auto biasConstant = bias.getDefiningOp<arith::ConstantOp>();
  if (!biasConstant)
    return failure();

  match.bias = bias;
  match.biasConstant = biasConstant;
  match.biasTy = biasTy;
  return match;
}

inline Value buildPatchMVM(Location loc, RankedTensorType resultTy, Value patch,
                           Value filterMatrix, OpBuilder &builder) {
  return mvm_build::buildMVM(loc, resultTy, patch, filterMatrix, builder);
}

inline Value applyOptionalBias(Location loc, RankedTensorType resultTy,
                               Type elementType, Value channelResult,
                               Value bias, OpBuilder &builder) {
  if (!bias)
    return channelResult;

  SmallVector<ReassociationIndices, 2> reassociation = {{0, 1}};
  Value expandedBias =
      builder.create<tensor::ExpandShapeOp>(loc, resultTy, bias, reassociation);
  Value biasedInit =
      builder.create<tensor::EmptyOp>(loc, resultTy.getShape(), elementType);
  return builder
      .create<linalg::AddOp>(loc, ValueRange{channelResult, expandedBias},
                             ValueRange{biasedInit})
      .getResult(0);
}

} // namespace converter_conv
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_CONVCONVERSIONUTILS_H
