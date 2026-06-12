#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/CommonLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/ConvolutionLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Canonicalization/CanonicalRewriteUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include <memory>
#include <optional>

namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace canonicalizer_utils = mlir::sculptor::canonicalizer_utils;

namespace {

using mlir::sculptor::NNConv2DOp;
using Conv2DMatch = layer_patterns::DirectConvolutionMatch;

// Recognizes Conv2D layers whose output init broadcasts a constant bias.
static std::optional<Conv2DMatch> matchConv2DWithBias(mlir::Operation *op) {
  return layer_patterns::matchDirectConvolutionWithBias(
      op, "linalg.conv_2d_nchw_fchw", /*spatialRank=*/2,
      layer_patterns::matchConv2DWeightInput);
}

// Recognizes Conv2D layers whose output init is a fill rather than a bias.
static std::optional<Conv2DMatch> matchConv2DWithoutBias(mlir::Operation *op) {
  return layer_patterns::matchDirectConvolutionWithoutBias(
      op, "linalg.conv_2d_nchw_fchw", /*spatialRank=*/2,
      layer_patterns::matchConv2DWeightInput);
}

// Rewrites two-dimensional convolutions to inline sculptor.nn.conv2d ops.
class Conv2DCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  // Keeps the canonicalizer interface uniform even though Conv2D stores no
  // state.
  explicit Conv2DCanonicalizer(mlir::MLIRContext *context) { (void)context; }

  // Supplies the stable layer key expected by the canonicalizer interface.
  mlir::StringRef getName() const override { return "conv2d"; }

  // Rewrites biased forms before matching the bias-free fallback.
  void canonicalize(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(
        func, rewriter, matchConv2DWithBias,
        canonicalizer_utils::rewriteConvolutionMatchToSculptorOp<NNConv2DOp,
                                                            Conv2DMatch>);
    layer_patterns::rewriteAllMatches(
        func, rewriter, matchConv2DWithoutBias,
        canonicalizer_utils::rewriteConvolutionMatchToSculptorOp<NNConv2DOp,
                                                            Conv2DMatch>);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

// Adds the Conv2D canonicalizer to the layer canonicalization pipeline.
void registerConv2DCanonicalizer(LayerCanonicalizers &canonicalizers,
                                 MLIRContext *context) {
  canonicalizers.push_back(std::make_unique<Conv2DCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
