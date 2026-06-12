#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Canonicalization/CanonicalRewriteUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/CommonLayerPatterns.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/LinalgMatchUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/Layers/OperandRelationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <optional>

namespace layer_patterns = mlir::sculptor::layer_patterns;
namespace layer_utils = mlir::sculptor::layer_utils;
namespace canonicalizer_utils = mlir::sculptor::canonicalizer_utils;
namespace linalg_match = mlir::sculptor::linalg_match;

namespace {

using mlir::arith::ConstantOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::TransposeOp;
using mlir::tensor::EmptyOp;

struct RNNCellBranchMatch {
  mlir::Value activation;
  mlir::Operation *weightConstant = nullptr;
  mlir::Operation *transposeEmpty = nullptr;
  mlir::Operation *transpose = nullptr;
  mlir::Operation *matmul = nullptr;
  mlir::Operation *fill = nullptr;
  mlir::Operation *fillConstant = nullptr;
  mlir::Operation *biasAdd = nullptr;
  mlir::Operation *biasConstant = nullptr;
};

struct RNNCellMatch {
  mlir::Operation *root = nullptr;
  mlir::Operation *preActivationAdd = nullptr;
  mlir::Operation *outputEmpty = nullptr;
  RNNCellBranchMatch inputBranch;
  RNNCellBranchMatch recurrentBranch;
  bool hasBias = false;
  mlir::Value output;
  llvm::SmallVector<mlir::Operation *> ops;
};

static void appendUniqueOp(llvm::SmallVectorImpl<mlir::Operation *> &ops,
                           mlir::Operation *op) {
  if (op && !llvm::is_contained(ops, op))
    ops.push_back(op);
}

static void appendBranchOps(llvm::SmallVectorImpl<mlir::Operation *> &ops,
                            const RNNCellBranchMatch &branch) {
  appendUniqueOp(ops, branch.weightConstant);
  appendUniqueOp(ops, branch.transposeEmpty);
  appendUniqueOp(ops, branch.transpose);
  appendUniqueOp(ops, branch.fillConstant);
  appendUniqueOp(ops, branch.fill);
  appendUniqueOp(ops, branch.matmul);
  appendUniqueOp(ops, branch.biasConstant);
  appendUniqueOp(ops, branch.biasAdd);
}

static void collectMatchedOps(RNNCellMatch &match) {
  match.ops.clear();
  appendUniqueOp(match.ops, match.outputEmpty);
  appendBranchOps(match.ops, match.inputBranch);
  appendBranchOps(match.ops, match.recurrentBranch);
  appendUniqueOp(match.ops, match.preActivationAdd);
  appendUniqueOp(match.ops, match.root);
}

static mlir::LogicalResult
matchRNNCellMatmulBranchCore(MatmulOp matmul,
                             mlir::Operation *sharedOutputEmpty,
                             RNNCellBranchMatch &branch) {
  if (!layer_utils::hasDpsInputsAndOperands(matmul.getOperation(), 2, 3))
    return mlir::failure();

  auto outputInit =
      layer_patterns::matchFillOutputInit(matmul.getOperation(), 2);
  if (!outputInit || outputInit->outputEmpty != sharedOutputEmpty)
    return mlir::failure();

  auto firstTranspose =
      layer_utils::operandProducerOfType<TransposeOp>(matmul.getOperation(), 0);
  auto secondTranspose =
      layer_utils::operandProducerOfType<TransposeOp>(matmul.getOperation(), 1);
  if (static_cast<bool>(firstTranspose) == static_cast<bool>(secondTranspose))
    return mlir::failure();

  auto transpose = firstTranspose ? firstTranspose : secondTranspose;
  if (!linalg_match::hasPermutation(transpose, {1, 0}) ||
      !layer_utils::hasDpsInputsAndOperands(transpose.getOperation(), 1, 2))
    return mlir::failure();

  auto weightConstant = layer_utils::operandProducerOfType<ConstantOp>(
      transpose.getOperation(), 0);
  auto transposeEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(transpose.getOperation(), 1);
  if (!weightConstant || !transposeEmpty)
    return mlir::failure();

  mlir::ValueRange matmulInputs = matmul.getInputs();
  branch.activation = firstTranspose ? matmulInputs[1] : matmulInputs[0];
  branch.weightConstant = weightConstant.getOperation();
  branch.transposeEmpty = transposeEmpty.getOperation();
  branch.transpose = transpose.getOperation();
  branch.matmul = matmul.getOperation();
  branch.fill = outputInit->outputFill;
  branch.fillConstant = outputInit->outputFillConstant;
  return mlir::success();
}

static mlir::LogicalResult
matchRNNCellBranchWithBias(mlir::Operation *branchAdd,
                           mlir::Operation *sharedOutputEmpty,
                           RNNCellBranchMatch &branch) {
  if (!layer_utils::isAddfGeneric(branchAdd) ||
      !layer_utils::hasDpsInputsAndOperands(branchAdd, 2, 3))
    return mlir::failure();

  auto branchOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(branchAdd, 2);
  if (!branchOutputEmpty ||
      branchOutputEmpty.getOperation() != sharedOutputEmpty)
    return mlir::failure();

  if (!layer_utils::operandProducersAreEither<MatmulOp, ConstantOp>(branchAdd))
    return mlir::failure();

  auto matmul = layer_utils::operandProducerOfType<MatmulOp>(branchAdd, 0);
  auto biasConstant =
      layer_utils::operandProducerOfType<ConstantOp>(branchAdd, 1);
  if (!matmul || !biasConstant) {
    matmul = layer_utils::operandProducerOfType<MatmulOp>(branchAdd, 1);
    biasConstant = layer_utils::operandProducerOfType<ConstantOp>(branchAdd, 0);
  }

  if (!matmul || !biasConstant)
    return mlir::failure();

  branch.biasAdd = branchAdd;
  branch.biasConstant = biasConstant.getOperation();
  return matchRNNCellMatmulBranchCore(matmul, sharedOutputEmpty, branch);
}

static mlir::LogicalResult
matchRNNCellBranchWithoutBias(mlir::Operation *branchMatmul,
                              mlir::Operation *sharedOutputEmpty,
                              RNNCellBranchMatch &branch) {
  auto matmul = llvm::dyn_cast_or_null<MatmulOp>(branchMatmul);
  if (!matmul)
    return mlir::failure();

  return matchRNNCellMatmulBranchCore(matmul, sharedOutputEmpty, branch);
}

static bool isRecurrentBranch(const RNNCellBranchMatch &branch,
                              mlir::RankedTensorType outputType) {
  auto activationType =
      layer_utils::getStaticRank2TensorType(branch.activation);
  auto weightType =
      layer_utils::getStaticRank2TensorType(branch.weightConstant);
  if (!activationType || !weightType)
    return false;

  int64_t batchSize = outputType.getShape()[0];
  int64_t hiddenSize = outputType.getShape()[1];
  return activationType.getShape()[0] == batchSize &&
         activationType.getShape()[1] == hiddenSize &&
         weightType.getShape()[0] == hiddenSize &&
         weightType.getShape()[1] == hiddenSize;
}

static bool isInputBranch(const RNNCellBranchMatch &branch,
                          mlir::RankedTensorType outputType) {
  auto activationType =
      layer_utils::getStaticRank2TensorType(branch.activation);
  auto weightType =
      layer_utils::getStaticRank2TensorType(branch.weightConstant);
  if (!activationType || !weightType)
    return false;

  int64_t batchSize = outputType.getShape()[0];
  int64_t hiddenSize = outputType.getShape()[1];
  int64_t inputSize = activationType.getShape()[1];
  return activationType.getShape()[0] == batchSize && inputSize != hiddenSize &&
         weightType.getShape()[0] == hiddenSize &&
         weightType.getShape()[1] == inputSize;
}

static bool hasBlockArgumentActivation(const RNNCellBranchMatch &branch) {
  return branch.activation && !layer_utils::producerOf(branch.activation);
}

static bool hasProducedActivation(const RNNCellBranchMatch &branch) {
  return branch.activation && layer_utils::producerOf(branch.activation);
}

// Classifies the two RNNCell matmul branches as input and recurrent.
static bool assignBranchRoles(const RNNCellBranchMatch &firstBranch,
                              const RNNCellBranchMatch &secondBranch,
                              mlir::RankedTensorType outputType,
                              RNNCellMatch &match) {
  bool firstIsInput = isInputBranch(firstBranch, outputType);
  bool firstIsRecurrent = isRecurrentBranch(firstBranch, outputType);
  bool secondIsInput = isInputBranch(secondBranch, outputType);
  bool secondIsRecurrent = isRecurrentBranch(secondBranch, outputType);

  if (firstIsInput && !firstIsRecurrent && secondIsRecurrent &&
      !secondIsInput) {
    match.inputBranch = firstBranch;
    match.recurrentBranch = secondBranch;
    return true;
  }

  if (secondIsInput && !secondIsRecurrent && firstIsRecurrent &&
      !firstIsInput) {
    match.inputBranch = secondBranch;
    match.recurrentBranch = firstBranch;
    return true;
  }

  if (firstIsRecurrent && secondIsRecurrent && !firstIsInput &&
      !secondIsInput) {
    if (hasProducedActivation(firstBranch) &&
        hasBlockArgumentActivation(secondBranch)) {
      match.inputBranch = firstBranch;
      match.recurrentBranch = secondBranch;
      return true;
    }

    if (hasProducedActivation(secondBranch) &&
        hasBlockArgumentActivation(firstBranch)) {
      match.inputBranch = secondBranch;
      match.recurrentBranch = firstBranch;
      return true;
    }
  }

  return false;
}

// Matches the complete RNNCell tanh pattern around two matmul branches.
template <typename BranchMatcher>
static std::optional<RNNCellMatch>
matchRNNCell(mlir::Operation *op, bool hasBias, BranchMatcher branchMatcher) {
  if (!layer_utils::isTanhGeneric(op) ||
      !layer_utils::hasDpsInputsAndOperands(op, 1, 2) ||
      op->getNumResults() != 1)
    return std::nullopt;

  auto outputType = layer_utils::getStaticRank2TensorType(op);
  if (!outputType)
    return std::nullopt;

  auto outputEmpty = layer_utils::operandProducerOfType<EmptyOp>(op, 1);
  if (!outputEmpty)
    return std::nullopt;

  mlir::Operation *preActivationAdd = layer_utils::operandProducer(op, 0);
  if (!layer_utils::isAddfGeneric(preActivationAdd) ||
      !layer_utils::hasDpsInputsAndOperands(preActivationAdd, 2, 3))
    return std::nullopt;

  auto preActivationOutputEmpty =
      layer_utils::operandProducerOfType<EmptyOp>(preActivationAdd, 2);
  if (!preActivationOutputEmpty ||
      preActivationOutputEmpty.getOperation() != outputEmpty.getOperation())
    return std::nullopt;

  RNNCellBranchMatch firstBranch;
  if (mlir::failed(
          branchMatcher(layer_utils::operandProducer(preActivationAdd, 0),
                        outputEmpty.getOperation(), firstBranch)))
    return std::nullopt;

  RNNCellBranchMatch secondBranch;
  if (mlir::failed(
          branchMatcher(layer_utils::operandProducer(preActivationAdd, 1),
                        outputEmpty.getOperation(), secondBranch)))
    return std::nullopt;

  if (firstBranch.fill != secondBranch.fill)
    return std::nullopt;

  RNNCellMatch match;
  match.root = op;
  match.preActivationAdd = preActivationAdd;
  match.outputEmpty = outputEmpty.getOperation();
  match.hasBias = hasBias;
  match.output = op->getResult(0);
  if (!assignBranchRoles(firstBranch, secondBranch, outputType, match))
    return std::nullopt;

  collectMatchedOps(match);
  return match;
}

static std::optional<RNNCellMatch> matchRNNCellWithBias(mlir::Operation *op) {
  return matchRNNCell(op, /*hasBias=*/true, matchRNNCellBranchWithBias);
}

static std::optional<RNNCellMatch>
matchRNNCellWithoutBias(mlir::Operation *op) {
  return matchRNNCell(op, /*hasBias=*/false, matchRNNCellBranchWithoutBias);
}

// Replaces the matched RNNCell dataflow with sculptor.nn.rnn_cell.
static void rewriteRNNCellMatchToSculptorOp(const RNNCellMatch &match,
                                          mlir::RewriterBase &rewriter) {
  mlir::Value wIh =
      canonicalizer_utils::firstResult(match.inputBranch.weightConstant);
  mlir::Value wHh =
      canonicalizer_utils::firstResult(match.recurrentBranch.weightConstant);
  if (!wIh || !wHh)
    return;

  mlir::Value bIh;
  mlir::Value bHh;
  if (match.hasBias) {
    bIh = canonicalizer_utils::firstResult(match.inputBranch.biasConstant);
    bHh = canonicalizer_utils::firstResult(match.recurrentBranch.biasConstant);
    if (!bIh || !bHh)
      return;
  }

  rewriter.setInsertionPoint(match.root);
  auto rnnCellOp = rewriter.create<mlir::sculptor::NNRNNCellOp>(
      match.root->getLoc(), match.output.getType(),
      match.inputBranch.activation, match.recurrentBranch.activation, wIh, wHh,
      bIh, bHh, rewriter.getBoolAttr(match.hasBias),
      rewriter.getStringAttr("tanh"));

  mlir::Value output = match.output;
  output.replaceAllUsesWith(rnnCellOp.getH());
  canonicalizer_utils::eraseDeadMatchedOps(match.ops, rewriter);
}

class RNNCellCanonicalizer : public mlir::sculptor::LayerCanonicalizer {
public:
  explicit RNNCellCanonicalizer(mlir::MLIRContext *context) { (void)context; }

  mlir::StringRef getName() const override { return "rnn_cell"; }

  void canonicalize(mlir::func::FuncOp func) const override {
    mlir::IRRewriter rewriter(func.getContext());

    layer_patterns::rewriteAllMatches(func, rewriter, matchRNNCellWithBias,
                                      rewriteRNNCellMatchToSculptorOp);
    layer_patterns::rewriteAllMatches(func, rewriter, matchRNNCellWithoutBias,
                                      rewriteRNNCellMatchToSculptorOp);
  }
};

} // namespace

namespace mlir {
namespace sculptor {

void registerRNNCellCanonicalizer(LayerCanonicalizers &canonicalizers,
                                  MLIRContext *context) {
  canonicalizers.push_back(std::make_unique<RNNCellCanonicalizer>(context));
}

} // namespace sculptor
} // namespace mlir
