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

using mlir::sculptor::NNConv3DOp;
using Conv3DMatch = layer_patterns::DirectConvolutionMatch;

// Recognizes Conv3D layers whose output init broadcasts a constant bias.
static std::optional<Conv3DMatch> matchConv3DWithBias(mlir::Operation *op) {
  return layer_patterns::matchDirectConvolutionWithBias(
      op, "linalg.conv_3d_ncdhw_fcdhw", /*spatialRank=*/3,
      layer_patterns::matchConv3DWeightInput);
}

// Recognizes Conv3D layers whose output init is a fill rather than a bias.
static std::optional<Conv3DMatch> matchConv3DWithoutBias(mlir::Operation *op) {
  return layer_patterns::matchDirectConvolutionWithoutBias(
      op, "linalg.conv_3d_ncdhw_fcdhw", /*spatialRank=*/3,
      layer_patterns::matchConv3DWeightInput);
}

// Rewrites three-dimensional convolutions to inline sculptor.nn.conv3d ops.
class Conv3DCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  // Keeps the canonicalizer interface uniform even though Conv3D stores no
  // state.
  explicit Conv3DCanonicalizer(mlir::MLIRContext *context) { (void)context; }

  // Supplies the stable layer key expected by the canonicalizer interface.
  mlir::StringRef getName() const override { return "conv3d"; }

  // Rewrites biased forms before matching the bias-free fallback.
  void canonicalize(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(
        func, rewriter, matchConv3DWithBias,
        canonicalizer_utils::rewriteConvolutionMatchToSculptorOp<NNConv3DOp,
                                                            Conv3DMatch>);
    layer_patterns::rewriteAllMatches(
        func, rewriter, matchConv3DWithoutBias,
        canonicalizer_utils::rewriteConvolutionMatchToSculptorOp<NNConv3DOp,
                                                            Conv3DMatch>);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

// Adds the Conv3D canonicalizer to the layer canonicalization pipeline.
void registerConv3DCanonicalizer(LayerCanonicalizers &canonicalizers,
                                 MLIRContext *context) {
  canonicalizers.push_back(std::make_unique<Conv3DCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
