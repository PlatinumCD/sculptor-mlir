#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExtractLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/TransformerStackStructure.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace {

using mlir::sculptor::NNTransformerBlockOp;
using mlir::sculptor::NNTransformerDecoderOp;
using mlir::sculptor::NNTransformerEncoderOp;
using mlir::sculptor::NNTransformerOp;
using mlir::sculptor::TransformerBlockKind;
using mlir::sculptor::TransformerBlockKindAttr;
using TransformerBlockDescription =
    mlir::sculptor::transformer_structure::TransformerBlockDescription;

template <typename TransformerOp>
static NNTransformerBlockOp
createInlineBlock(TransformerOp transformerOp,
                  const TransformerBlockDescription &block, mlir::Value input,
                  mlir::Value memory, mlir::Type outputType,
                  mlir::RewriterBase &rewriter) {
  TransformerBlockKind blockKind = block.isDecoder
                                       ? TransformerBlockKind::Decoder
                                       : TransformerBlockKind::Encoder;
  return rewriter.create<NNTransformerBlockOp>(
      transformerOp.getLoc(), outputType, input, memory, block.qkvWeight,
      block.qkvBias, block.attnOutputWeight, block.attnOutputBias,
      block.attnNormWeight, block.attnNormBias, block.crossQueryWeight,
      block.crossQueryBias, block.crossKeyValueWeight, block.crossKeyValueBias,
      block.crossOutputWeight, block.crossOutputBias, block.crossNormWeight,
      block.crossNormBias, block.mlpUpWeight, block.mlpUpBias,
      block.mlpDownWeight, block.mlpDownBias, block.mlpNormWeight,
      block.mlpNormBias, block.finalNormWeight, block.finalNormBias,
      transformerOp.getBatchFirstAttr(),
      transformerOp.getHasProjectionBiasAttr(),
      transformerOp.getHasLayerNormAffineAttr(),
      transformerOp.getHasLayerNormBiasAttr(),
      rewriter.getBoolAttr(block.hasFinalNorm), transformerOp.getCausalAttr(),
      rewriter.getBoolAttr(block.hasCrossAttention),
      transformerOp.getActivationAttr(),
      TransformerBlockKindAttr::get(rewriter.getContext(), blockKind),
      transformerOp.getNormModeAttr(), transformerOp.getHiddenSizeAttr(),
      transformerOp.getNumHeadsAttr(), transformerOp.getHeadDimAttr(),
      transformerOp.getMlpHiddenSizeAttr(),
      rewriter.getI64IntegerAttr(block.blockIndex),
      rewriter.getI64IntegerAttr(block.numBlocks),
      transformerOp.getLayerNormEpsAttr());
}

static mlir::LogicalResult
inlineTransformerBlocks(NNTransformerOp transformerOp,
                        mlir::RewriterBase &rewriter) {
  if (!transformerOp || transformerOp->getNumResults() != 1)
    return mlir::failure();

  auto description =
      mlir::sculptor::transformer_structure::describeTransformerStack(
          transformerOp);
  if (mlir::failed(description))
    return mlir::failure();

  rewriter.setInsertionPoint(transformerOp);
  mlir::Value encoder = transformerOp.getSrc();
  for (const TransformerBlockDescription &block : description->encoderBlocks) {
    encoder = createInlineBlock(transformerOp, block, encoder, mlir::Value{},
                                encoder.getType(), rewriter)
                  .getOutput();
  }

  mlir::Value decoder = transformerOp.getTgt();
  for (const TransformerBlockDescription &block : description->decoderBlocks) {
    decoder = createInlineBlock(transformerOp, block, decoder, encoder,
                                transformerOp.getOutput().getType(), rewriter)
                  .getOutput();
  }

  rewriter.replaceOp(transformerOp, decoder);
  return mlir::success();
}

static mlir::LogicalResult
inlineTransformerEncoderBlocks(NNTransformerEncoderOp transformerOp,
                               mlir::RewriterBase &rewriter) {
  if (!transformerOp || transformerOp->getNumResults() != 1)
    return mlir::failure();

  auto description =
      mlir::sculptor::transformer_structure::describeTransformerStack(
          transformerOp);
  if (mlir::failed(description))
    return mlir::failure();

  rewriter.setInsertionPoint(transformerOp);
  mlir::Value current = transformerOp.getInput();
  for (const TransformerBlockDescription &block : description->encoderBlocks) {
    current = createInlineBlock(transformerOp, block, current, mlir::Value{},
                                transformerOp.getOutput().getType(), rewriter)
                  .getOutput();
  }

  rewriter.replaceOp(transformerOp, current);
  return mlir::success();
}

static mlir::LogicalResult
inlineTransformerDecoderBlocks(NNTransformerDecoderOp transformerOp,
                               mlir::RewriterBase &rewriter) {
  if (!transformerOp || transformerOp->getNumResults() != 1)
    return mlir::failure();

  auto description =
      mlir::sculptor::transformer_structure::describeTransformerStack(
          transformerOp);
  if (mlir::failed(description))
    return mlir::failure();

  rewriter.setInsertionPoint(transformerOp);
  mlir::Value current = transformerOp.getInput();
  for (const TransformerBlockDescription &block : description->decoderBlocks) {
    current = createInlineBlock(transformerOp, block, current, mlir::Value{},
                                transformerOp.getOutput().getType(), rewriter)
                  .getOutput();
  }

  rewriter.replaceOp(transformerOp, current);
  return mlir::success();
}

static bool extractCanonicalTransformers(mlir::func::FuncOp func) {
  llvm::SmallVector<NNTransformerOp> matches;
  func.walk([&](NNTransformerOp transformerOp) {
    if (transformerOp)
      matches.push_back(transformerOp);
  });

  llvm::SmallVector<NNTransformerEncoderOp> encoderMatches;
  func.walk([&](NNTransformerEncoderOp transformerOp) {
    if (transformerOp)
      encoderMatches.push_back(transformerOp);
  });

  llvm::SmallVector<NNTransformerDecoderOp> decoderMatches;
  func.walk([&](NNTransformerDecoderOp transformerOp) {
    if (transformerOp)
      decoderMatches.push_back(transformerOp);
  });

  mlir::IRRewriter rewriter(func.getContext());
  bool changed = false;
  for (NNTransformerOp match : matches) {
    if (match && match->getBlock() &&
        mlir::succeeded(inlineTransformerBlocks(match, rewriter)))
      changed = true;
  }

  for (NNTransformerEncoderOp match : encoderMatches) {
    if (match && match->getBlock() &&
        mlir::succeeded(inlineTransformerEncoderBlocks(match, rewriter)))
      changed = true;
  }

  for (NNTransformerDecoderOp match : decoderMatches) {
    if (match && match->getBlock() &&
        mlir::succeeded(inlineTransformerDecoderBlocks(match, rewriter)))
      changed = true;
  }

  return changed;
}

class TransformerExtractor : public mlir::sculptor::LayerExtractor {
public:
  explicit TransformerExtractor(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "transformer"; }

  void extract(mlir::func::FuncOp func) const override {
    (void)extractCanonicalTransformers(func);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

LogicalResult decomposeInlineTransformerStacks(func::FuncOp func) {
  (void)extractCanonicalTransformers(func);

  bool hasUnsupportedStack = false;
  func.walk([&](Operation *op) {
    if (isa<NNTransformerOp, NNTransformerEncoderOp, NNTransformerDecoderOp>(
            op))
      hasUnsupportedStack = true;
  });
  if (hasUnsupportedStack) {
    func.emitError("cannot decompose supported inline Transformer stack");
    return failure();
  }
  return success();
}

void registerTransformerExtractor(LayerExtractors &extractors,
                                  MLIRContext *context) {
  extractors.push_back(std::make_unique<TransformerExtractor>(context));
}

} // namespace sculptor
} // namespace mlir
