#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/CommonLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/ConvolutionLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Canonicalization/CanonicalRewriteUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include <memory>

namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace canonicalizer_utils = mlir::sculptor::canonicalizer_utils;

namespace {

using mlir::sculptor::NNGroupedConv2DOp;

static void rewriteGroupedConv2DMatchToSculptorOp(
    const layer_patterns::Conv2DGroupedMatch &match,
    mlir::RewriterBase &rewriter) {
  if (!match.root || !match.input || !match.weightConstant ||
      match.outputs.size() != 1 || match.groups <= 1 || match.strides.empty() ||
      match.padding.empty() || match.dilations.empty())
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
  auto convOp = rewriter.create<NNGroupedConv2DOp>(
      match.root->getLoc(), match.outputs.front().getType(), match.input,
      weight, bias, static_cast<bool>(bias),
      static_cast<uint64_t>(match.groups),
      rewriter.getI64ArrayAttr(match.strides),
      rewriter.getI64ArrayAttr(match.padding),
      rewriter.getI64ArrayAttr(match.dilations));

  mlir::Value output = match.outputs.front();
  output.replaceAllUsesWith(convOp.getResult());
  canonicalizer_utils::eraseDeadMatchedOps(match.ops, rewriter);
}

// Rewrites grouped two-dimensional convolutions to inline sculptor.nn grouped
// conv2d ops.
class Conv2DGroupedCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  // Keeps the canonicalizer interface uniform even though no state is stored.
  explicit Conv2DGroupedCanonicalizer(mlir::MLIRContext *context) {
    (void)context;
  }

  // Supplies the stable layer key expected by the canonicalizer interface.
  mlir::StringRef getName() const override { return "conv2d_grouped"; }

  // Rewrites biased grouped convolutions before the bias-free fallback.
  void canonicalize(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(
        func, rewriter, layer_patterns::matchConv2DGroupedWithBias,
        rewriteGroupedConv2DMatchToSculptorOp);
    layer_patterns::rewriteAllMatches(
        func, rewriter, layer_patterns::matchConv2DGroupedWithoutBias,
        rewriteGroupedConv2DMatchToSculptorOp);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

// Adds the grouped Conv2D canonicalizer to the layer canonicalization pipeline.
void registerConv2DGroupedCanonicalizer(LayerCanonicalizers &canonicalizers,
                                        MLIRContext *context) {
  canonicalizers.push_back(
      std::make_unique<Conv2DGroupedCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
