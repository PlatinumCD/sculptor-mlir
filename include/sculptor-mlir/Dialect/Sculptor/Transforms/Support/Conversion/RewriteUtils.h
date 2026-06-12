#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_REWRITEUTILS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_REWRITEUTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace sculptor {
namespace converter_rewrite {

inline void eraseIfUnused(Operation *op, RewriterBase &rewriter) {
  if (op && op->use_empty())
    rewriter.eraseOp(op);
}

} // namespace converter_rewrite
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SUPPORT_CONVERSION_REWRITEUTILS_H
