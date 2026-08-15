#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/CommonLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/LinearLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/LinalgMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/RecurrentLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Canonicalization/CanonicalRewriteUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace layer_utils = mlir::sculptor::layer_utils;
namespace linalg_match = mlir::sculptor::linalg_match;
namespace canonicalizer_utils = mlir::sculptor::canonicalizer_utils;

namespace {

using mlir::sculptor::NNLinearOp;

static bool isYieldInputGeneric(mlir::linalg::GenericOp genericOp) {
  if (!genericOp || genericOp.getInputs().size() != 1 ||
      genericOp.getOutputs().size() != 1 || genericOp.getNumResults() != 1 ||
      !genericOp.getRegion().hasOneBlock())
    return false;
  mlir::Block &body = genericOp.getRegion().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 1)
    return false;
  auto yield = llvm::dyn_cast<mlir::linalg::YieldOp>(body.front());
  return yield && yield.getValues().size() == 1 &&
         yield.getValues().front() == body.getArgument(0);
}

static mlir::arith::ConstantOp
matchBatchedLinearWeight(mlir::linalg::BatchMatmulOp batchMatmul,
                         mlir::linalg::GenericOp &broadcast,
                         mlir::linalg::TransposeOp &transpose) {
  if (!batchMatmul || batchMatmul.getInputs().size() != 2 ||
      batchMatmul.getOutputs().size() != 1)
    return {};
  auto inputType = llvm::dyn_cast<mlir::RankedTensorType>(
      batchMatmul.getInputs()[0].getType());
  auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(
      batchMatmul.getResult(0).getType());
  if (!inputType || !resultType || !inputType.hasStaticShape() ||
      !resultType.hasStaticShape() || inputType.getRank() != 3 ||
      resultType.getRank() != 3 || !inputType.getElementType().isF32() ||
      !resultType.getElementType().isF32())
    return {};

  broadcast = layer_utils::producerOfType<mlir::linalg::GenericOp>(
      batchMatmul.getInputs()[1]);
  if (!isYieldInputGeneric(broadcast) ||
      !linalg_match::hasWeightBroadcastIndexingMaps(broadcast))
    return {};

  int64_t inputWidth = inputType.getDimSize(2);
  int64_t outputWidth = resultType.getDimSize(2);
  auto weight = layer_patterns::matchProjectionWeightTranspose(
      broadcast.getInputs()[0], inputWidth, outputWidth, transpose);
  if (mlir::failed(weight))
    return {};
  return *weight;
}

static mlir::linalg::GenericOp
matchBatchedLinearBias(mlir::linalg::BatchMatmulOp batchMatmul,
                       mlir::arith::ConstantOp &bias) {
  if (!batchMatmul.getResult(0).hasOneUse())
    return {};
  auto add = llvm::dyn_cast<mlir::linalg::GenericOp>(
      *batchMatmul.getResult(0).getUsers().begin());
  if (!add || !layer_patterns::isProjectionAddfGeneric(add) ||
      add.getInputs().size() != 2)
    return {};

  unsigned biasIndex = add.getInputs()[0] == batchMatmul.getResult(0) ? 1 : 0;
  if (add.getInputs()[1 - biasIndex] != batchMatmul.getResult(0) ||
      !linalg_match::hasBiasAddIndexingMaps(add, /*rank=*/3, biasIndex))
    return {};
  bias = layer_utils::producerOfType<mlir::arith::ConstantOp>(
      add.getInputs()[biasIndex]);
  return bias ? add : mlir::linalg::GenericOp{};
}

static void canonicalizeBatchedLinearOps(mlir::func::FuncOp func,
                                         mlir::IRRewriter &rewriter) {
  llvm::SmallVector<mlir::linalg::BatchMatmulOp> candidates;
  func.walk([&](mlir::linalg::BatchMatmulOp op) { candidates.push_back(op); });
  for (mlir::linalg::BatchMatmulOp batchMatmul : candidates) {
    if (!batchMatmul || !batchMatmul->getBlock())
      continue;
    mlir::linalg::GenericOp broadcast;
    mlir::linalg::TransposeOp transpose;
    mlir::arith::ConstantOp weight =
        matchBatchedLinearWeight(batchMatmul, broadcast, transpose);
    if (!weight)
      continue;

    mlir::arith::ConstantOp bias;
    mlir::linalg::GenericOp biasAdd = matchBatchedLinearBias(batchMatmul, bias);
    mlir::Value oldResult = biasAdd ? biasAdd.getResult(0)
                                    : batchMatmul.getResult(0);
    rewriter.setInsertionPoint(batchMatmul);
    auto linear = rewriter.create<NNLinearOp>(
        batchMatmul.getLoc(), oldResult.getType(), batchMatmul.getInputs()[0],
        weight.getResult(), bias ? bias.getResult() : mlir::Value{},
        static_cast<bool>(bias));
    oldResult.replaceAllUsesWith(linear.getResult());

    // eraseDeadMatchedOps walks this producer-to-consumer list in reverse.
    // Keeping the consumer last is essential: otherwise the still-live bias
    // add prevents the batch matmul, weight broadcast, and transpose from
    // being erased, leaving a complete duplicate digital linear operation.
    llvm::SmallVector<mlir::Operation *> dead = {
        transpose.getOperation(), broadcast.getOperation(),
        batchMatmul.getOperation(),
        biasAdd ? biasAdd.getOperation() : nullptr};
    canonicalizer_utils::eraseDeadMatchedOps(dead, rewriter);
  }
}

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

    canonicalizeBatchedLinearOps(func, rewriter);

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
