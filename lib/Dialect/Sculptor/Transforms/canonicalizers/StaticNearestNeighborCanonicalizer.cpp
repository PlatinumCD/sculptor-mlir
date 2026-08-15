#include "sculptor-mlir/Dialect/Sculptor/Transforms/CanonicalizeLayers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cmath>
#include <cstdint>
#include <optional>

namespace {

using namespace mlir;

constexpr int64_t kMaximumExactlyRepresentableF32Integer = 1LL << 24;

struct StaticNearestIndex {
  linalg::IndexOp index;
  int64_t scale = 1;
};

std::optional<int64_t> getPowerOfTwoScale(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto floating = dyn_cast<FloatAttr>(constant.getValue());
  if (!floating)
    return std::nullopt;
  double exact = floating.getValueAsDouble();
  if (!std::isfinite(exact) || exact < 1.0 || std::floor(exact) != exact ||
      exact > static_cast<double>(INT64_MAX))
    return std::nullopt;
  int64_t scale = static_cast<int64_t>(exact);
  if ((scale & (scale - 1)) != 0)
    return std::nullopt;
  return scale;
}

std::optional<StaticNearestIndex> matchStaticNearestIndex(Value value) {
  auto finalCast = value.getDefiningOp<arith::IndexCastOp>();
  if (!finalCast || !value.getType().isIndex())
    return std::nullopt;
  auto toInteger = finalCast.getIn().getDefiningOp<arith::FPToSIOp>();
  if (!toInteger)
    return std::nullopt;
  auto floor = toInteger.getIn().getDefiningOp<math::FloorOp>();
  if (!floor)
    return std::nullopt;

  Value convertedIndex = floor.getOperand();
  int64_t scale = 1;
  if (auto divide = convertedIndex.getDefiningOp<arith::DivFOp>()) {
    std::optional<int64_t> matchedScale = getPowerOfTwoScale(divide.getRhs());
    if (!matchedScale)
      return std::nullopt;
    scale = *matchedScale;
    convertedIndex = divide.getLhs();
  }

  auto toFloat = convertedIndex.getDefiningOp<arith::SIToFPOp>();
  if (!toFloat || !toFloat.getType().isF32())
    return std::nullopt;
  auto initialCast = toFloat.getIn().getDefiningOp<arith::IndexCastOp>();
  if (!initialCast || !initialCast.getIn().getType().isIndex())
    return std::nullopt;
  auto index = initialCast.getIn().getDefiningOp<linalg::IndexOp>();
  if (!index)
    return std::nullopt;
  return StaticNearestIndex{index, scale};
}

bool hasSafeStaticIndexRange(linalg::GenericOp generic,
                             const StaticNearestIndex &matched) {
  if (generic.getNumResults() != 1)
    return false;
  auto resultType = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  linalg::IndexOp index = matched.index;
  if (!resultType || !resultType.hasStaticShape() ||
      index.getDim() >= static_cast<uint64_t>(resultType.getRank()))
    return false;
  int64_t extent = resultType.getDimSize(index.getDim());
  return extent > 0 && extent <= kMaximumExactlyRepresentableF32Integer;
}

struct StrengthReduceStaticNearestIndex
    : public OpRewritePattern<arith::IndexCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp finalCast,
                                PatternRewriter &rewriter) const override {
    std::optional<StaticNearestIndex> matched =
        matchStaticNearestIndex(finalCast.getResult());
    if (!matched)
      return failure();
    auto generic = finalCast->getParentOfType<linalg::GenericOp>();
    if (!generic || !hasSafeStaticIndexRange(generic, *matched))
      return failure();

    if (matched->scale == 1) {
      rewriter.replaceOp(finalCast, matched->index.getResult());
      return success();
    }
    rewriter.setInsertionPoint(finalCast);
    Value divisor = rewriter.create<arith::ConstantIndexOp>(finalCast.getLoc(),
                                                            matched->scale);
    Value quotient = rewriter.create<arith::DivUIOp>(
        finalCast.getLoc(), matched->index.getResult(), divisor);
    rewriter.replaceOp(finalCast, quotient);
    return success();
  }
};

bool isIdentityIndex(Value value, int64_t dimension,
                     RankedTensorType tensorType) {
  if (matchPattern(value, m_Zero()))
    return tensorType.getDimSize(dimension) == 1;
  if (auto index = value.getDefiningOp<linalg::IndexOp>())
    return index.getDim() == static_cast<uint64_t>(dimension);
  std::optional<StaticNearestIndex> matched = matchStaticNearestIndex(value);
  return matched && matched->scale == 1 &&
         matched->index.getDim() == static_cast<uint64_t>(dimension);
}

struct EliminateIdentityNearestResize
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const override {
    if (generic.getNumResults() != 1 || generic.getNumDpsInputs() != 0 ||
        generic.getNumDpsInits() != 1 ||
        generic.getNumParallelLoops() != generic.getNumLoops() ||
        !generic.getRegion().hasOneBlock())
      return failure();
    auto resultType =
        dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return failure();
    ArrayRef<AffineMap> indexingMaps = generic.getIndexingMapsArray();
    if (indexingMaps.size() != 1 || !indexingMaps.front().isIdentity())
      return failure();

    auto yield =
        dyn_cast<linalg::YieldOp>(generic.getRegion().front().getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return failure();
    auto extract = yield.getOperand(0).getDefiningOp<tensor::ExtractOp>();
    if (!extract || extract->getBlock() != &generic.getRegion().front())
      return failure();
    auto sourceType = dyn_cast<RankedTensorType>(extract.getTensor().getType());
    if (!sourceType || sourceType != resultType ||
        extract.getIndices().size() !=
            static_cast<size_t>(sourceType.getRank()))
      return failure();
    for (auto [dimension, index] : llvm::enumerate(extract.getIndices())) {
      if (!isIdentityIndex(index, dimension, sourceType))
        return failure();
    }

    rewriter.replaceOp(generic, extract.getTensor());
    return success();
  }
};

} // namespace

LogicalResult
mlir::sculptor::canonicalizeStaticNearestNeighborResizes(func::FuncOp func) {
  RewritePatternSet patterns(func.getContext());
  patterns
      .add<EliminateIdentityNearestResize, StrengthReduceStaticNearestIndex>(
          func.getContext());
  GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  return applyPatternsGreedily(func, std::move(patterns), config);
}
