#include "sculptor-mlir/Dialect/Sculptor/Transforms/FoldAttentionLayouts.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

namespace {

using namespace mlir;

struct HeadView {
  tensor::CollapseShapeOp collapse;
  linalg::TransposeOp transpose;
  tensor::ExpandShapeOp expand;
  Value source;
};

static RankedTensorType getStaticF32Type(Value value, int64_t rank) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() != rank ||
      !type.getElementType().isF32())
    return {};
  return type;
}

static bool hasPermutation(linalg::TransposeOp transpose,
                           ArrayRef<int64_t> expected) {
  if (!transpose || transpose.getPermutation().size() != expected.size())
    return false;
  return llvm::equal(transpose.getPermutation(), expected);
}

static FailureOr<HeadView> matchHeadView(Value value,
                                         ArrayRef<int64_t> permutation) {
  auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>();
  if (!collapse)
    return failure();
  auto transpose = collapse.getSrc().getDefiningOp<linalg::TransposeOp>();
  if (!transpose || !hasPermutation(transpose, permutation))
    return failure();
  auto expand = transpose.getInput().getDefiningOp<tensor::ExpandShapeOp>();
  if (!expand)
    return failure();
  return HeadView{collapse, transpose, expand, expand.getResult()};
}

static linalg::FillOp matchTensorFill(Value value) {
  auto fill = value.getDefiningOp<linalg::FillOp>();
  if (!fill || !fill.hasPureTensorSemantics() || fill.getInputs().size() != 1 ||
      fill.getOutputs().size() != 1 || fill.getNumResults() != 1)
    return {};
  return fill;
}

static void copySculptorAttributes(Operation *source, Operation *target) {
  for (NamedAttribute attribute : source->getAttrs()) {
    if (attribute.getName().strref().starts_with("sculptor."))
      target->setAttr(attribute.getName(), attribute.getValue());
  }
}

// Erase only dead layout metadata and its tensor.empty destination. This keeps
// the rewrite local and never recursively removes an arbitrary producer.
static void eraseDeadLayoutChain(Operation *operation, RewriterBase &rewriter) {
  if (!operation || !operation->use_empty() ||
      !isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, linalg::TransposeOp,
           tensor::EmptyOp>(operation))
    return;

  SmallVector<Operation *, 2> producers;
  for (Value operand : operation->getOperands()) {
    if (Operation *producer = operand.getDefiningOp())
      producers.push_back(producer);
  }
  rewriter.eraseOp(operation);
  for (Operation *producer : producers)
    eraseDeadLayoutChain(producer, rewriter);
}

static Value buildZeroedTensor(linalg::BatchMatmulOp batchMatmul,
                               RankedTensorType resultType,
                               RewriterBase &rewriter) {
  linalg::FillOp oldFill = matchTensorFill(batchMatmul.getOutputs().front());
  if (!oldFill)
    return {};
  Value empty = rewriter.create<tensor::EmptyOp>(
      batchMatmul.getLoc(), resultType.getShape(), resultType.getElementType());
  return rewriter
      .create<linalg::FillOp>(batchMatmul.getLoc(), oldFill.getInputs().front(),
                              empty)
      .getResult(0);
}

static linalg::GenericOp
buildContraction(linalg::BatchMatmulOp batchMatmul, RankedTensorType resultType,
                 ValueRange inputs, ArrayRef<AffineMap> maps,
                 ArrayRef<utils::IteratorType> iterators,
                 RewriterBase &rewriter) {
  Value init = buildZeroedTensor(batchMatmul, resultType, rewriter);
  if (!init)
    return {};

  auto contraction = rewriter.create<linalg::GenericOp>(
      batchMatmul.getLoc(), resultType, inputs, ValueRange{init}, maps,
      iterators, [](OpBuilder &builder, Location loc, ValueRange arguments) {
        Value product =
            builder.create<arith::MulFOp>(loc, arguments[0], arguments[1]);
        Value sum = builder.create<arith::AddFOp>(loc, arguments[2], product);
        builder.create<linalg::YieldOp>(loc, sum);
      });
  copySculptorAttributes(batchMatmul, contraction);
  return contraction;
}

// Fold:
//   [B,Q,H,D] -> transpose [B,H,Q,D] -> collapse [BH,Q,D]
//   [B,K,H,D] -> transpose [B,H,D,K] -> collapse [BH,D,K]
//   batch_matmul -> expand [B,H,Q,K]
// into one rank-four contraction over (B,H,Q,K,D).
static bool foldQueryKey(linalg::BatchMatmulOp batchMatmul,
                         RewriterBase &rewriter) {
  if (!batchMatmul.hasPureTensorSemantics() ||
      batchMatmul.getInputs().size() != 2 ||
      batchMatmul.getOutputs().size() != 1 ||
      batchMatmul.getNumResults() != 1 || !batchMatmul.getResult(0).hasOneUse())
    return false;

  auto outputExpand = dyn_cast<tensor::ExpandShapeOp>(
      *batchMatmul.getResult(0).getUsers().begin());
  if (!outputExpand)
    return false;
  FailureOr<HeadView> query =
      matchHeadView(batchMatmul.getInputs()[0], {0, 2, 1, 3});
  FailureOr<HeadView> key =
      matchHeadView(batchMatmul.getInputs()[1], {0, 2, 3, 1});
  if (failed(query) || failed(key))
    return false;

  RankedTensorType queryType = getStaticF32Type(query->source, 4);
  RankedTensorType keyType = getStaticF32Type(key->source, 4);
  RankedTensorType queryFlatType =
      getStaticF32Type(batchMatmul.getInputs()[0], 3);
  RankedTensorType keyFlatType =
      getStaticF32Type(batchMatmul.getInputs()[1], 3);
  RankedTensorType batchResultType =
      getStaticF32Type(batchMatmul.getResult(0), 3);
  RankedTensorType resultType = getStaticF32Type(outputExpand.getResult(), 4);
  if (!queryType || !keyType || !queryFlatType || !keyFlatType ||
      !batchResultType || !resultType)
    return false;

  auto queryShape = queryType.getShape();
  int64_t batch = queryShape[0];
  int64_t queryLength = queryShape[1];
  int64_t heads = queryShape[2];
  int64_t headDim = queryShape[3];
  auto keyShape = keyType.getShape();
  if (batch <= 0 || queryLength <= 0 || heads <= 0 || headDim <= 0 ||
      keyShape[0] != batch || keyShape[2] != heads || keyShape[3] != headDim)
    return false;
  int64_t keyLength = keyShape[1];
  if (keyLength <= 0 ||
      queryFlatType.getShape() !=
          ArrayRef<int64_t>({batch * heads, queryLength, headDim}) ||
      keyFlatType.getShape() !=
          ArrayRef<int64_t>({batch * heads, headDim, keyLength}) ||
      batchResultType.getShape() !=
          ArrayRef<int64_t>({batch * heads, queryLength, keyLength}) ||
      resultType.getShape() !=
          ArrayRef<int64_t>({batch, heads, queryLength, keyLength}))
    return false;

  rewriter.setInsertionPoint(batchMatmul);
  MLIRContext *context = rewriter.getContext();
  AffineExpr b = rewriter.getAffineDimExpr(0);
  AffineExpr h = rewriter.getAffineDimExpr(1);
  AffineExpr q = rewriter.getAffineDimExpr(2);
  AffineExpr k = rewriter.getAffineDimExpr(3);
  AffineExpr d = rewriter.getAffineDimExpr(4);
  SmallVector<AffineMap, 3> maps = {
      AffineMap::get(5, 0, {b, q, h, d}, context),
      AffineMap::get(5, 0, {b, k, h, d}, context),
      AffineMap::get(5, 0, {b, h, q, k}, context)};
  SmallVector<utils::IteratorType, 5> iterators = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};
  linalg::GenericOp contraction = buildContraction(
      batchMatmul, resultType, ValueRange{query->source, key->source}, maps,
      iterators, rewriter);
  if (!contraction)
    return false;
  copySculptorAttributes(outputExpand, contraction);

  rewriter.replaceAllUsesWith(outputExpand.getResult(),
                              contraction.getResult(0));
  rewriter.eraseOp(outputExpand);
  rewriter.eraseOp(batchMatmul);
  eraseDeadLayoutChain(query->collapse, rewriter);
  eraseDeadLayoutChain(key->collapse, rewriter);
  return true;
}

// Fold:
//   probabilities [B,H,Q,K] -> collapse [BH,Q,K]
//   value [B,K,H,D] -> transpose [B,H,K,D] -> collapse [BH,K,D]
//   batch_matmul -> expand [B,H,Q,D] -> transpose [B,Q,H,D]
// into one rank-four contraction over (B,Q,H,D,K).
static bool foldProbabilityValue(linalg::BatchMatmulOp batchMatmul,
                                 RewriterBase &rewriter) {
  if (!batchMatmul.hasPureTensorSemantics() ||
      batchMatmul.getInputs().size() != 2 ||
      batchMatmul.getOutputs().size() != 1 ||
      batchMatmul.getNumResults() != 1 || !batchMatmul.getResult(0).hasOneUse())
    return false;

  auto probabilityCollapse =
      batchMatmul.getInputs()[0].getDefiningOp<tensor::CollapseShapeOp>();
  FailureOr<HeadView> value =
      matchHeadView(batchMatmul.getInputs()[1], {0, 2, 1, 3});
  auto outputExpand = dyn_cast<tensor::ExpandShapeOp>(
      *batchMatmul.getResult(0).getUsers().begin());
  if (!probabilityCollapse || failed(value) || !outputExpand ||
      !outputExpand.getResult().hasOneUse())
    return false;
  auto outputTranspose = dyn_cast<linalg::TransposeOp>(
      *outputExpand.getResult().getUsers().begin());
  if (!outputTranspose || !hasPermutation(outputTranspose, {0, 2, 1, 3}) ||
      !outputTranspose->getResult(0).hasOneUse() ||
      !isa<tensor::CollapseShapeOp>(
          *outputTranspose->getResult(0).getUsers().begin()))
    return false;

  RankedTensorType probabilityType =
      getStaticF32Type(probabilityCollapse.getSrc(), 4);
  RankedTensorType probabilityFlatType =
      getStaticF32Type(batchMatmul.getInputs()[0], 3);
  RankedTensorType valueType = getStaticF32Type(value->source, 4);
  RankedTensorType valueFlatType =
      getStaticF32Type(batchMatmul.getInputs()[1], 3);
  RankedTensorType batchResultType =
      getStaticF32Type(batchMatmul.getResult(0), 3);
  RankedTensorType expandedResultType =
      getStaticF32Type(outputExpand.getResult(), 4);
  RankedTensorType resultType =
      getStaticF32Type(outputTranspose->getResult(0), 4);
  if (!probabilityType || !probabilityFlatType || !valueType ||
      !valueFlatType || !batchResultType || !expandedResultType || !resultType)
    return false;

  auto probabilityShape = probabilityType.getShape();
  int64_t batch = probabilityShape[0];
  int64_t heads = probabilityShape[1];
  int64_t queryLength = probabilityShape[2];
  int64_t keyLength = probabilityShape[3];
  auto valueShape = valueType.getShape();
  if (batch <= 0 || heads <= 0 || queryLength <= 0 || keyLength <= 0 ||
      valueShape[0] != batch || valueShape[1] != keyLength ||
      valueShape[2] != heads || valueShape[3] <= 0)
    return false;
  int64_t headDim = valueShape[3];
  if (probabilityFlatType.getShape() !=
          ArrayRef<int64_t>({batch * heads, queryLength, keyLength}) ||
      valueFlatType.getShape() !=
          ArrayRef<int64_t>({batch * heads, keyLength, headDim}) ||
      batchResultType.getShape() !=
          ArrayRef<int64_t>({batch * heads, queryLength, headDim}) ||
      expandedResultType.getShape() !=
          ArrayRef<int64_t>({batch, heads, queryLength, headDim}) ||
      resultType.getShape() !=
          ArrayRef<int64_t>({batch, queryLength, heads, headDim}))
    return false;

  rewriter.setInsertionPoint(batchMatmul);
  MLIRContext *context = rewriter.getContext();
  AffineExpr b = rewriter.getAffineDimExpr(0);
  AffineExpr q = rewriter.getAffineDimExpr(1);
  AffineExpr h = rewriter.getAffineDimExpr(2);
  AffineExpr d = rewriter.getAffineDimExpr(3);
  AffineExpr k = rewriter.getAffineDimExpr(4);
  SmallVector<AffineMap, 3> maps = {
      AffineMap::get(5, 0, {b, h, q, k}, context),
      AffineMap::get(5, 0, {b, k, h, d}, context),
      AffineMap::get(5, 0, {b, q, h, d}, context)};
  SmallVector<utils::IteratorType, 5> iterators = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};
  linalg::GenericOp contraction =
      buildContraction(batchMatmul, resultType,
                       ValueRange{probabilityCollapse.getSrc(), value->source},
                       maps, iterators, rewriter);
  if (!contraction)
    return false;
  copySculptorAttributes(outputTranspose, contraction);

  rewriter.replaceAllUsesWith(outputTranspose->getResult(0),
                              contraction.getResult(0));
  rewriter.eraseOp(outputTranspose);
  rewriter.eraseOp(outputExpand);
  rewriter.eraseOp(batchMatmul);
  eraseDeadLayoutChain(probabilityCollapse, rewriter);
  eraseDeadLayoutChain(value->collapse, rewriter);
  return true;
}

} // namespace

void mlir::sculptor::FoldAttentionLayoutsPass::runOnOperation() {
  IRRewriter rewriter(&getContext());
  SmallVector<linalg::BatchMatmulOp> candidates;
  getOperation().walk([&](linalg::BatchMatmulOp operation) {
    candidates.push_back(operation);
  });
  for (linalg::BatchMatmulOp operation : candidates) {
    if (operation && operation->getBlock())
      foldQueryKey(operation, rewriter);
  }

  candidates.clear();
  getOperation().walk([&](linalg::BatchMatmulOp operation) {
    candidates.push_back(operation);
  });
  for (linalg::BatchMatmulOp operation : candidates) {
    if (operation && operation->getBlock())
      foldProbabilityValue(operation, rewriter);
  }
}

void mlir::sculptor::registerFoldAttentionLayoutsPass() {
  PassRegistration<FoldAttentionLayoutsPass>();
}
