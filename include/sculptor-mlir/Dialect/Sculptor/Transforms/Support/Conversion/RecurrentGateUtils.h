#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_RECURRENTGATEUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_RECURRENTGATEUTILS_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace recurrent_gate {

inline RankedTensorType buildFusedInputType(RankedTensorType inputTy,
                                            RankedTensorType hiddenTy) {
  return RankedTensorType::get(
      {1, inputTy.getShape()[1] + hiddenTy.getShape()[1]},
      inputTy.getElementType());
}

// Concatenates one cell input and hidden state for the analog MVM.
inline Value buildFusedInput(Location loc, RankedTensorType fusedInputTy,
                             Value input, Value hidden,
                             RewriterBase &rewriter) {
  return rewriter
      .create<tensor::ConcatOp>(loc, fusedInputTy, /*dim=*/1,
                                ValueRange{input, hidden})
      .getResult();
}

namespace detail {

inline SmallVector<OpFoldResult>
buildIndexAttrs(OpBuilder &builder, ArrayRef<int64_t> values) {
  SmallVector<OpFoldResult> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getIndexAttr(value));
  return attrs;
}

} // namespace detail

// Extracts one hidden-size gate block from cell preactivation.
inline Value buildGateSlice(Location loc, RankedTensorType resultTy,
                            Value preActivation, int64_t gateOffset,
                            int64_t hiddenSize, RewriterBase &rewriter) {
  return rewriter
      .create<tensor::ExtractSliceOp>(
          loc, resultTy, preActivation,
          detail::buildIndexAttrs(rewriter, {0, gateOffset}),
          detail::buildIndexAttrs(rewriter, {1, hiddenSize}),
          detail::buildIndexAttrs(rewriter, {1, 1}))
      .getResult();
}

// Extracts one hidden-size gate block from batched preactivation.
inline Value extractBatchGate(Location loc, Value preActivation,
                              int64_t gateOffset, int64_t batchSize,
                              int64_t hiddenSize, RankedTensorType resultTy,
                              OpBuilder &builder) {
  return builder
      .create<tensor::ExtractSliceOp>(
          loc, resultTy, preActivation,
          detail::buildIndexAttrs(builder, {0, gateOffset}),
          detail::buildIndexAttrs(builder, {batchSize, hiddenSize}),
          detail::buildIndexAttrs(builder, {1, 1}))
      .getResult();
}

} // namespace recurrent_gate
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_RECURRENTGATEUTILS_H
