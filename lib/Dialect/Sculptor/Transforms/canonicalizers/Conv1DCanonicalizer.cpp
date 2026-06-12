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

using mlir::sculptor::NNConv1DOp;
using Conv1DMatch = layer_patterns::DirectConvolutionMatch;

// Recognizes Conv1D layers whose output init broadcasts a constant bias.
static std::optional<Conv1DMatch> matchConv1DWithBias(mlir::Operation *op) {
  return layer_patterns::matchDirectConvolutionWithBias(
      op, "linalg.conv_1d_ncw_fcw", /*spatialRank=*/1,
      layer_patterns::matchConv1DWeightInput);
}

// Recognizes Conv1D layers whose output init is a fill rather than a bias.
static std::optional<Conv1DMatch> matchConv1DWithoutBias(mlir::Operation *op) {
  return layer_patterns::matchDirectConvolutionWithoutBias(
      op, "linalg.conv_1d_ncw_fcw", /*spatialRank=*/1,
      layer_patterns::matchConv1DWeightInput);
}

// Rewrites one-dimensional convolutions to inline sculptor.nn.conv1d ops.
class Conv1DCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  // Keeps the canonicalizer interface uniform even though Conv1D stores no
  // state.
  explicit Conv1DCanonicalizer(mlir::MLIRContext *context) { (void)context; }

  // Supplies the stable layer key expected by the canonicalizer interface.
  mlir::StringRef getName() const override { return "conv1d"; }

  // Rewrites biased forms before matching the bias-free fallback.
  void canonicalize(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(
        func, rewriter, matchConv1DWithBias,
        canonicalizer_utils::rewriteConvolutionMatchToSculptorOp<NNConv1DOp,
                                                            Conv1DMatch>);
    layer_patterns::rewriteAllMatches(
        func, rewriter, matchConv1DWithoutBias,
        canonicalizer_utils::rewriteConvolutionMatchToSculptorOp<NNConv1DOp,
                                                            Conv1DMatch>);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

// Adds the Conv1D canonicalizer to the layer canonicalization pipeline.
void registerConv1DCanonicalizer(LayerCanonicalizers &canonicalizers,
                                 MLIRContext *context) {
  canonicalizers.push_back(std::make_unique<Conv1DCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
