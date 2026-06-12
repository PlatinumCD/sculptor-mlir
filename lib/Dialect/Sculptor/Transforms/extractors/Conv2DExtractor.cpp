#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Extraction/RewriteUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/CommonLayerPatterns.h"

#include "llvm/Support/Casting.h"

#include <memory>
#include <optional>

namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace rewrite_utils = mlir::sculptor::rewrite_utils;

namespace {

using mlir::sculptor::NNConv2DOp;

// Recognizes a Conv2D layer that has already been canonicalized to sculptor.nn.
static std::optional<NNConv2DOp> matchCanonicalConv2D(mlir::Operation *op) {
  auto convOp = llvm::dyn_cast_or_null<NNConv2DOp>(op);
  if (!convOp)
    return std::nullopt;

  return convOp;
}

// Outlines an existing canonical Conv2D op into a layer function.
static void outlineConv2DOpToLayerFunction(NNConv2DOp convOp,
                                            mlir::RewriterBase &rewriter) {
  bool hasBias = convOp.getHasBias();
  mlir::ArrayAttr stride = convOp.getStride();
  mlir::ArrayAttr padding = convOp.getPadding();
  mlir::ArrayAttr dilation = convOp.getDilation();
  mlir::StringRef layerType =
      hasBias ? mlir::StringRef("conv2d_w_bias") : mlir::StringRef("conv2d");

  rewrite_utils::extractExistingSingleResultCanonicalOpToFunction(
      convOp, convOp.getInput(), convOp.getWeight(), convOp.getBias(), rewriter,
      layerType,
      [hasBias, stride, padding,
       dilation](mlir::RewriterBase &bodyRewriter, mlir::Location loc,
                 mlir::Type outputType, mlir::Value input, mlir::Value weight,
                 mlir::Value bias) -> mlir::Value {
        auto outlinedConvOp = bodyRewriter.create<NNConv2DOp>(
            loc, outputType, input, weight, bias, hasBias, stride, padding,
            dilation);
        return outlinedConvOp.getResult();
      });
}

// Finds canonical two-dimensional convolution layers and outlines each match.
class Conv2DExtractor : public mlir::sculptor::LayerExtractor {
public:
  // Keeps the extractor interface uniform even though Conv2D stores no state.
  explicit Conv2DExtractor(mlir::MLIRContext *context) { (void)context; }

  // Supplies the stable layer key expected by the extractor interface.
  mlir::StringRef getName() const override { return "conv2d"; }

  // Extraction is intentionally canonical-only; run sculptor-canonicalize-layers
  // first to turn raw linalg patterns into sculptor.nn.conv2d.
  void extract(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(func, rewriter, matchCanonicalConv2D,
                                      outlineConv2DOpToLayerFunction);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

// Adds the Conv2D extractor to the layer extraction pipeline.
void registerConv2DExtractor(LayerExtractors &extractors,
                             MLIRContext *context) {
  extractors.push_back(std::make_unique<Conv2DExtractor>(context));
}

} // namespace sculptor
} // namespace mlir
