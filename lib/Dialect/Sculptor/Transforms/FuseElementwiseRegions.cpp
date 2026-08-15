#include "sculptor-mlir/Dialect/Sculptor/Transforms/FuseElementwiseRegions.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/SemanticLayerIdentity.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

using namespace mlir;
using namespace mlir::sculptor;

bool hasOnlyParallelLoops(linalg::GenericOp operation) {
  return operation.getNumParallelLoops() == operation.getNumLoops();
}

// Missing identities may fuse with one another. Once either endpoint belongs
// to a canonical layer, both endpoints must carry the same complete identity.
bool haveCompatibleSemanticLayerIdentity(Operation *producer,
                                         Operation *consumer) {
  Attribute producerId = producer->getAttr(kSemanticLayerIdAttrName);
  Attribute producerKind = producer->getAttr(kSemanticLayerKindAttrName);
  Attribute consumerId = consumer->getAttr(kSemanticLayerIdAttrName);
  Attribute consumerKind = consumer->getAttr(kSemanticLayerKindAttrName);

  bool producerHasIdentity = producerId || producerKind;
  bool consumerHasIdentity = consumerId || consumerKind;
  if (!producerHasIdentity && !consumerHasIdentity)
    return true;
  if (!producerId || !producerKind || !consumerId || !consumerKind)
    return false;
  return producerId == consumerId && producerKind == consumerKind;
}

void copySculptorAttributes(Operation *source, Operation *target) {
  for (NamedAttribute attribute : source->getAttrs()) {
    if (attribute.getName().strref().starts_with("sculptor."))
      target->setAttr(attribute.getName(), attribute.getValue());
  }
}

class FuseSemanticElementwiseChain final
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter &rewriter) const override {
    if (!consumer.hasPureTensorSemantics() || !hasOnlyParallelLoops(consumer))
      return failure();

    for (OpOperand *operand : consumer.getDpsInputOperands()) {
      auto producerResult = dyn_cast<OpResult>(operand->get());
      if (!producerResult || !producerResult.hasOneUse())
        continue;
      auto producer = dyn_cast<linalg::GenericOp>(producerResult.getOwner());
      if (!producer || !producer.hasPureTensorSemantics() ||
          producer->getNumResults() != 1 || !hasOnlyParallelLoops(producer) ||
          !haveCompatibleSemanticLayerIdentity(producer, consumer) ||
          !linalg::areElementwiseOpsFusable(operand))
        continue;

      FailureOr<linalg::ElementwiseOpFusionResult> fusion =
          linalg::fuseElementwiseOps(rewriter, operand);
      if (failed(fusion))
        continue;

      copySculptorAttributes(consumer, fusion->fusedOp);
      for (auto [original, replacement] : fusion->replacements) {
        rewriter.replaceUsesWithIf(original, replacement, [&](OpOperand &use) {
          return use.get().getDefiningOp() != producer;
        });
      }
      rewriter.eraseOp(consumer);
      if (producer->use_empty())
        rewriter.eraseOp(producer);
      return success();
    }
    return failure();
  }
};

} // namespace

void mlir::sculptor::FuseElementwiseRegionsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<FuseSemanticElementwiseChain>(&getContext());
  GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns), config)))
    signalPassFailure();
}

void mlir::sculptor::registerFuseElementwiseRegionsPass() {
  PassRegistration<FuseElementwiseRegionsPass>();
}
