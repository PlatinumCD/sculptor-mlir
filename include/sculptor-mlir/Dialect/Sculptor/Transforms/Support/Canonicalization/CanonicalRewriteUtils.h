#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CANONICALIZATION_CANONICALREWRITEUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CANONICALIZATION_CANONICALREWRITEUTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace sculptor {
namespace canonicalizer_utils {

inline void eraseDeadMatchedOps(llvm::ArrayRef<Operation *> ops,
                                RewriterBase &rewriter) {
  for (Operation *op : llvm::reverse(ops)) {
    if (op && op->use_empty())
      rewriter.eraseOp(op);
  }
}

inline Value firstResult(Operation *op) {
  if (!op || op->getNumResults() != 1)
    return {};
  return op->getResult(0);
}

template <typename CanonicalConvOp, typename DirectConvolutionMatchT>
inline void rewriteConvolutionMatchToSculptorOp(
    const DirectConvolutionMatchT &match, RewriterBase &rewriter) {
  if (!match.root || !match.input || !match.weightConstant ||
      match.outputs.size() != 1 || match.strides.empty() ||
      match.padding.empty() || match.dilations.empty())
    return;

  Value weight = firstResult(match.weightConstant);
  if (!weight)
    return;

  Value bias;
  if (match.biasConstant) {
    bias = firstResult(match.biasConstant);
    if (!bias)
      return;
  }

  rewriter.setInsertionPoint(match.root);
  auto convOp = rewriter.create<CanonicalConvOp>(
      match.root->getLoc(), match.outputs.front().getType(), match.input,
      weight, bias, static_cast<bool>(bias),
      rewriter.getI64ArrayAttr(match.strides),
      rewriter.getI64ArrayAttr(match.padding),
      rewriter.getI64ArrayAttr(match.dilations));

  Value output = match.outputs.front();
  output.replaceAllUsesWith(convOp.getResult());
  eraseDeadMatchedOps(match.ops, rewriter);
}

} // namespace canonicalizer_utils
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CANONICALIZATION_CANONICALREWRITEUTILS_H
