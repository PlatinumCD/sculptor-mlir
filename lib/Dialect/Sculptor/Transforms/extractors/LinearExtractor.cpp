#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Extraction/RewriteUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/CommonLayerPatterns.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <optional>

namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace rewrite_utils = mlir::sculptor::rewrite_utils;

namespace {

using mlir::sculptor::NNLinearOp;

// Recognizes a linear layer that has already been canonicalized to sculptor.nn.
static std::optional<NNLinearOp> matchCanonicalLinear(mlir::Operation *op) {
  auto linearOp = llvm::dyn_cast_or_null<NNLinearOp>(op);
  if (!linearOp)
    return std::nullopt;

  return linearOp;
}

// Outlines an existing canonical linear op into a layer function.
static void outlineLinearOpToLayerFunction(NNLinearOp linearOp,
                                            mlir::RewriterBase &rewriter) {
  bool hasBias = linearOp.getHasBias();
  mlir::StringRef layerType =
      hasBias ? mlir::StringRef("linear_w_bias") : mlir::StringRef("linear");

  rewrite_utils::extractExistingSingleResultCanonicalOpToFunction(
      linearOp, linearOp.getInput(), linearOp.getWeight(), linearOp.getBias(),
      rewriter, layerType,
      [hasBias](mlir::RewriterBase &bodyRewriter, mlir::Location loc,
                mlir::Type outputType, mlir::Value input, mlir::Value weight,
                mlir::Value bias) -> mlir::Value {
        auto outlinedLinearOp = bodyRewriter.create<NNLinearOp>(
            loc, outputType, input, weight, bias, hasBias);
        return outlinedLinearOp.getResult();
      });
}

// Finds canonical linear layers and outlines each match.
class LinearExtractor : public mlir::sculptor::LayerExtractor {
public:
  // Keeps the extractor interface uniform even though Linear stores no state.
  explicit LinearExtractor(mlir::MLIRContext *context) { (void)context; }

  // Supplies the stable layer key expected by the extractor interface.
  mlir::StringRef getName() const override { return "linear"; }

  // Extraction is intentionally canonical-only; run sculptor-canonicalize-layers
  // first to turn raw linalg patterns into sculptor.nn.linear.
  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(func, rewriter, matchCanonicalLinear,
                                      outlineLinearOpToLayerFunction);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

// Adds the linear extractor to the layer extraction pipeline.
void registerLinearExtractor(LayerExtractors &extractors, MLIRContext *context) {
  extractors.push_back(std::make_unique<LinearExtractor>(context));
}

} // namespace sculptor
} // namespace mlir
