#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"
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

using mlir::sculptor::NNGroupedConv2DOp;

// Recognizes a grouped Conv2D layer already canonicalized to sculptor.nn.
static std::optional<NNGroupedConv2DOp>
matchCanonicalConv2DGrouped(mlir::Operation *op) {
  auto convOp = llvm::dyn_cast_or_null<NNGroupedConv2DOp>(op);
  if (!convOp)
    return std::nullopt;

  return convOp;
}

// Outlines an existing canonical grouped Conv2D op into a layer function.
static void
outlineGroupedConv2DOpToLayerFunction(NNGroupedConv2DOp convOp,
                                       mlir::RewriterBase &rewriter) {
  bool hasBias = convOp.getHasBias();
  uint64_t groups = static_cast<uint64_t>(convOp.getGroupsAttr().getInt());
  mlir::ArrayAttr stride = convOp.getStride();
  mlir::ArrayAttr padding = convOp.getPadding();
  mlir::ArrayAttr dilation = convOp.getDilation();
  mlir::StringRef layerType = hasBias
                                  ? mlir::StringRef("conv2d_grouped_w_bias")
                                  : mlir::StringRef("conv2d_grouped");

  rewrite_utils::extractExistingSingleResultCanonicalOpToFunction(
      convOp, convOp.getInput(), convOp.getWeight(), convOp.getBias(), rewriter,
      layerType,
      [hasBias, groups, stride, padding,
       dilation](mlir::RewriterBase &bodyRewriter, mlir::Location loc,
                 mlir::Type outputType, mlir::Value input, mlir::Value weight,
                 mlir::Value bias) -> mlir::Value {
        auto outlinedConvOp = bodyRewriter.create<NNGroupedConv2DOp>(
            loc, outputType, input, weight, bias, hasBias, groups, stride,
            padding, dilation);
        return outlinedConvOp.getResult();
      });
}

// Finds canonical grouped two-dimensional convolution layers and outlines each
// match.
class Conv2DGroupedExtractor : public mlir::sculptor::LayerExtractor {
public:
  // Keeps the extractor interface uniform even though no state is stored.
  explicit Conv2DGroupedExtractor(mlir::MLIRContext *context) { (void)context; }

  // Supplies the stable layer key expected by the extractor interface.
  mlir::StringRef getName() const override { return "conv2d_grouped"; }

  // Extraction is intentionally canonical-only; run sculptor-canonicalize-layers
  // first to turn raw linalg patterns into sculptor.nn.grouped_conv2d.
  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(
        func, rewriter, matchCanonicalConv2DGrouped,
        outlineGroupedConv2DOpToLayerFunction);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

// Adds the grouped Conv2D extractor to the layer extraction pipeline.
void registerConv2DGroupedExtractor(LayerExtractors &extractors,
                                    MLIRContext *context) {
  extractors.push_back(std::make_unique<Conv2DGroupedExtractor>(context));
}

} // namespace sculptor
} // namespace mlir
