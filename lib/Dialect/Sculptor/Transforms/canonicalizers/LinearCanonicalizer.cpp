#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/CommonLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/LinearLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Canonicalization/CanonicalRewriteUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include <memory>

namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace canonicalizer_utils = mlir::sculptor::canonicalizer_utils;

namespace {

using mlir::sculptor::NNLinearOp;

static void rewriteLinearMatchToSculptorOp(const layer_patterns::LinearMatch &match,
                                mlir::RewriterBase &rewriter) {
  if (!match.root || !match.weightConstant || match.inputs.size() != 1 ||
      match.outputs.size() != 1)
    return;

  mlir::Value weight = canonicalizer_utils::firstResult(match.weightConstant);
  if (!weight)
    return;

  mlir::Value bias;
  if (match.biasConstant) {
    bias = canonicalizer_utils::firstResult(match.biasConstant);
    if (!bias)
      return;
  }

  rewriter.setInsertionPoint(match.root);
  auto linearOp = rewriter.create<NNLinearOp>(
      match.root->getLoc(), match.outputs.front().getType(),
      match.inputs.front(), weight, bias, static_cast<bool>(bias));

  mlir::Value output = match.outputs.front();
  output.replaceAllUsesWith(linearOp.getResult());
  canonicalizer_utils::eraseDeadMatchedOps(match.ops, rewriter);
}

// Rewrites linalg-based linear layers to inline sculptor.nn.linear ops.
class LinearCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  // Keeps the canonicalizer interface uniform even though Linear stores no
  // state.
  explicit LinearCanonicalizer(mlir::MLIRContext *context) { (void)context; }

  // Supplies the stable layer key expected by the canonicalizer interface.
  mlir::StringRef getName() const override { return "linear"; }

  // Rewrites biased forms before matching the bias-free fallback.
  void canonicalize(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(func, rewriter,
                                      layer_patterns::matchLinearWithBias,
                                      rewriteLinearMatchToSculptorOp);
    layer_patterns::rewriteAllMatches(func, rewriter,
                                      layer_patterns::matchLinearWithoutBias,
                                      rewriteLinearMatchToSculptorOp);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

// Adds the linear canonicalizer to the layer canonicalization pipeline.
void registerLinearCanonicalizer(LayerCanonicalizers &canonicalizers,
                                 MLIRContext *context) {
  canonicalizers.push_back(std::make_unique<LinearCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
