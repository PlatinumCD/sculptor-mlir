#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphMatmulDistributor.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTimingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphExecutionGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResources.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

enum class MatmulBodyKind {
  Matmul,
  MatmulTransposeB,
};

enum class AttentionMatmulKind {
  Scores,
  Apply,
};

struct MatmulMatch {
  TaskCreateOp task;
  func::FuncOp callee;
  MatmulBodyKind bodyKind;
  RankedTensorType lhsType;
  RankedTensorType rhsType;
  RankedTensorType resultType;
  int64_t rows;
  int64_t columns;
  int64_t reduction;
};

struct AttentionMatmulMatch {
  TaskCreateOp task;
  func::FuncOp callee;
  AttentionMatmulKind kind;
  RankedTensorType firstInputType;
  RankedTensorType secondInputType;
  RankedTensorType resultType;
  int64_t batch;
  int64_t heads;
  int64_t queryLength;
  int64_t keyLength;
  int64_t headDim;
  bool causal;
};

struct Slice {
  int64_t offset = 0;
  int64_t size = 0;
};

struct DistributionPlan {
  MatmulDistributionStrategy strategy;
  int64_t rowShards = 1;
  int64_t columnShards = 1;
  int64_t maxShardOps = 0;
  int64_t communicationBytes = 0;
  llvm::SmallVector<Slice, 8> rowSlices;
  llvm::SmallVector<Slice, 8> columnSlices;

  int64_t getShardCount() const { return rowShards * columnShards; }
};

static FailureOr<int64_t> checkedProduct(Operation *anchor,
                                         ArrayRef<int64_t> values,
                                         StringRef description) {
  int64_t product = 1;
  for (int64_t value : values) {
    if (value < 0) {
      anchor->emitError(description) << " requires non-negative dimensions";
      return failure();
    }
    std::optional<int64_t> next = llvm::checkedMul(product, value);
    if (!next) {
      anchor->emitError(description) << " overflows signed 64-bit arithmetic";
      return failure();
    }
    product = *next;
  }
  return product;
}

static FailureOr<int64_t>
checkedSum(Operation *anchor, ArrayRef<int64_t> values, StringRef description) {
  int64_t sum = 0;
  for (int64_t value : values) {
    std::optional<int64_t> next = llvm::checkedAdd(sum, value);
    if (!next) {
      anchor->emitError(description) << " overflows signed 64-bit arithmetic";
      return failure();
    }
    sum = *next;
  }
  return sum;
}

static SmallVector<OpFoldResult> getIndexAttrs(OpBuilder &builder,
                                               ArrayRef<int64_t> values) {
  SmallVector<OpFoldResult> result;
  result.reserve(values.size());
  for (int64_t value : values)
    result.push_back(builder.getIndexAttr(value));
  return result;
}

static SmallVector<Value, 8> deduplicateValues(ArrayRef<Value> values) {
  llvm::DenseSet<Value> seen;
  SmallVector<Value, 8> result;
  result.reserve(values.size());
  for (Value value : values) {
    if (seen.insert(value).second)
      result.push_back(value);
  }
  return result;
}

static SmallVector<Value, 8> selectDependenciesForInputs(
    ArrayRef<Value> dependencies, ArrayRef<Value> inputs,
    ArrayRef<Value> allDataInputs, bool includeControlOnly) {
  SmallVector<Value, 8> selected;
  for (Value dependency : dependencies) {
    auto producer = dependency.getDefiningOp<TaskCreateOp>();
    bool producesInput =
        producer && llvm::any_of(producer.getOutputs(), [&](Value output) {
          return llvm::is_contained(inputs, output);
        });
    bool producesAnyDataInput =
        producer && llvm::any_of(producer.getOutputs(), [&](Value output) {
          return llvm::is_contained(allDataInputs, output);
        });
    if (producesInput || (includeControlOnly && !producesAnyDataInput))
      selected.push_back(dependency);
  }
  return deduplicateValues(selected);
}

static llvm::StringMap<int64_t>
collectNextSourceOrdinals(func::FuncOp taskGraphFunc) {
  llvm::StringMap<int64_t> nextOrdinalByLayer;
  for (TaskCreateOp task : taskGraphFunc.getOps<TaskCreateOp>()) {
    int64_t &next = nextOrdinalByLayer[task.getSourceLayer()];
    next =
        std::max(next, static_cast<int64_t>(task.getSourceTaskOrdinal()) + 1);
  }
  return nextOrdinalByLayer;
}

static int64_t collectNextDistributionGroupId(func::FuncOp taskGraphFunc) {
  int64_t nextGroupId = 0;
  for (TaskCreateOp task : taskGraphFunc.getOps<TaskCreateOp>()) {
    auto distribution = task->getAttrOfType<TaskDistributionAttr>(
        task_graph_attrs::kTaskDistributionAttrName);
    if (distribution)
      nextGroupId =
          std::max(nextGroupId, distribution.getGroupId().getInt() + 1);
  }
  return nextGroupId;
}

static LogicalResult rejectStalePlacementMetadata(func::FuncOp taskGraphFunc) {
  if (taskGraphFunc->hasAttr(schedule_attrs::kNumCoresAttrName) ||
      taskGraphFunc->hasAttr(timing_attrs::kTimingModelAttrName)) {
    return taskGraphFunc.emitError(
        "digital matmul distribution must run before timing and scheduling");
  }

  for (TaskCreateOp task : taskGraphFunc.getOps<TaskCreateOp>()) {
    if (task->hasAttr(schedule_attrs::kIslandIdAttrName) ||
        task->hasAttr(runtime_attrs::kTaskCoreIdAttrName) ||
        task->hasAttr(runtime_attrs::kTaskPhysicalArrayIdAttrName)) {
      return task.emitError(
          "digital matmul distribution must run before island construction "
          "and placement");
    }
  }
  return success();
}

static bool isAllowedMatmulSupportOp(Operation *op) {
  return isa<arith::ConstantOp, tensor::EmptyOp, linalg::FillOp,
             linalg::MatmulOp, linalg::MatmulTransposeBOp, func::ReturnOp>(op);
}

static FailureOr<std::optional<MatmulMatch>>
matchDigitalMatmul(ModuleOp module, TaskCreateOp task) {
  if (task.getDomain() != task_graph_names::kDigitalDomain ||
      task->hasAttr(task_graph_attrs::kTaskDistributionAttrName))
    return std::optional<MatmulMatch>();

  func::FuncOp callee = module.lookupSymbol<func::FuncOp>(task.getCallee());
  if (!callee)
    return task.emitOpError("expected digital task callee to resolve");
  if (!callee.getBody().hasOneBlock())
    return std::optional<MatmulMatch>();

  Operation *matmulOperation = nullptr;
  bool unsupportedBody = false;
  for (Operation &op : callee.getBody().front()) {
    if (!isAllowedMatmulSupportOp(&op)) {
      unsupportedBody = true;
      break;
    }
    if (isa<linalg::MatmulOp, linalg::MatmulTransposeBOp>(&op)) {
      if (matmulOperation) {
        unsupportedBody = true;
        break;
      }
      matmulOperation = &op;
    }
  }
  if (unsupportedBody || !matmulOperation)
    return std::optional<MatmulMatch>();

  if (task.getInputs().size() != 2 || task.getOutputs().size() != 1 ||
      callee.getNumArguments() != 2 || callee.getNumResults() != 1)
    return std::optional<MatmulMatch>();

  auto returnOp =
      dyn_cast<func::ReturnOp>(callee.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1 ||
      returnOp.getOperand(0) != matmulOperation->getResult(0))
    return std::optional<MatmulMatch>();

  Value lhs;
  Value rhs;
  MatmulBodyKind bodyKind;
  if (auto matmul = dyn_cast<linalg::MatmulOp>(matmulOperation)) {
    lhs = matmul.getInputs()[0];
    rhs = matmul.getInputs()[1];
    bodyKind = MatmulBodyKind::Matmul;
  } else {
    auto transposedMatmul = cast<linalg::MatmulTransposeBOp>(matmulOperation);
    lhs = transposedMatmul.getInputs()[0];
    rhs = transposedMatmul.getInputs()[1];
    bodyKind = MatmulBodyKind::MatmulTransposeB;
  }
  if (lhs != callee.getArgument(0) || rhs != callee.getArgument(1))
    return std::optional<MatmulMatch>();

  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  auto resultType =
      dyn_cast<RankedTensorType>(matmulOperation->getResult(0).getType());
  if (!lhsType || !rhsType || !resultType)
    return std::optional<MatmulMatch>();

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return std::optional<MatmulMatch>();
  if (!lhsType.getElementType().isF32() || !rhsType.getElementType().isF32() ||
      !resultType.getElementType().isF32())
    return std::optional<MatmulMatch>();

  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !resultType.hasStaticShape()) {
    return task.emitOpError(
        "digital matmul distribution requires static ranked tensor shapes");
  }

  int64_t rows = lhsType.getDimSize(0);
  int64_t reduction = lhsType.getDimSize(1);
  int64_t columns = bodyKind == MatmulBodyKind::Matmul ? rhsType.getDimSize(1)
                                                       : rhsType.getDimSize(0);
  int64_t rhsReduction = bodyKind == MatmulBodyKind::Matmul
                             ? rhsType.getDimSize(0)
                             : rhsType.getDimSize(1);
  if (rows <= 0 || columns <= 0 || reduction <= 0 ||
      rhsReduction != reduction || resultType.getDimSize(0) != rows ||
      resultType.getDimSize(1) != columns) {
    return task.emitOpError(
        "digital matmul distribution found inconsistent matmul geometry");
  }

  auto lhsResource = dyn_cast<TaskResourceType>(task.getInputs()[0].getType());
  auto rhsResource = dyn_cast<TaskResourceType>(task.getInputs()[1].getType());
  auto outputResource =
      dyn_cast<TaskResourceType>(task.getOutputs()[0].getType());
  if (!lhsResource || !rhsResource || !outputResource ||
      lhsResource.getValueType() != lhsType ||
      rhsResource.getValueType() != rhsType ||
      outputResource.getValueType() != resultType) {
    return task.emitOpError(
        "digital matmul task resources must match its callee signature");
  }

  return std::optional<MatmulMatch>(MatmulMatch{task, callee, bodyKind, lhsType,
                                                rhsType, resultType, rows,
                                                columns, reduction});
}

static FailureOr<std::optional<AttentionMatmulMatch>>
matchAttentionMatmul(ModuleOp module, TaskCreateOp task) {
  if (task.getDomain() != task_graph_names::kDigitalDomain ||
      task->hasAttr(task_graph_attrs::kTaskDistributionAttrName))
    return std::optional<AttentionMatmulMatch>();

  AttentionMatmulKind kind;
  if (task.getTaskKind() == task_graph_names::kDigitalAttentionScoresTaskKind) {
    kind = AttentionMatmulKind::Scores;
  } else if (task.getTaskKind() ==
             task_graph_names::kDigitalAttentionApplyTaskKind) {
    kind = AttentionMatmulKind::Apply;
  } else {
    return std::optional<AttentionMatmulMatch>();
  }

  func::FuncOp callee = module.lookupSymbol<func::FuncOp>(task.getCallee());
  if (!callee)
    return task.emitOpError("expected attention task callee to resolve");
  if (task.getInputs().size() != 2 || task.getOutputs().size() != 1 ||
      callee.getNumArguments() != 2 || callee.getNumResults() != 1) {
    return task.emitOpError(
        "distributed attention matmul requires two inputs and one output");
  }

  FunctionType calleeType = callee.getFunctionType();
  auto firstInputType = dyn_cast<RankedTensorType>(calleeType.getInput(0));
  auto secondInputType = dyn_cast<RankedTensorType>(calleeType.getInput(1));
  auto resultType = dyn_cast<RankedTensorType>(calleeType.getResult(0));
  if (!firstInputType || !secondInputType || !resultType) {
    return task.emitOpError(
        "distributed attention matmul requires ranked tensor types");
  }
  if (!firstInputType.hasStaticShape() || !secondInputType.hasStaticShape() ||
      !resultType.hasStaticShape()) {
    return task.emitOpError(
        "digital attention matmul distribution requires static tensor shapes");
  }
  if (!firstInputType.getElementType().isF32() ||
      !secondInputType.getElementType().isF32() ||
      !resultType.getElementType().isF32()) {
    return task.emitOpError(
        "digital attention matmul distribution supports only f32 tensors");
  }

  auto headDimAttr =
      callee->getAttrOfType<IntegerAttr>(task_attrs::kAttentionHeadDimAttrName);
  if (!headDimAttr || headDimAttr.getInt() <= 0) {
    return task.emitOpError(
        "expected positive head_dim metadata on attention task callee");
  }
  int64_t headDim = headDimAttr.getInt();

  int64_t batch = 0;
  int64_t heads = 0;
  int64_t queryLength = 0;
  int64_t keyLength = 0;
  bool causal = false;
  if (kind == AttentionMatmulKind::Scores) {
    if (firstInputType.getRank() != 3 || secondInputType.getRank() != 3 ||
        resultType.getRank() != 4) {
      return task.emitOpError(
          "attention score distribution expects rank-3 query/key and a "
          "rank-4 score result");
    }
    auto causalAttr =
        callee->getAttrOfType<BoolAttr>(task_attrs::kAttentionCausalAttrName);
    if (!causalAttr) {
      return task.emitOpError(
          "expected causal metadata on attention score task callee");
    }
    causal = causalAttr.getValue();
    batch = resultType.getDimSize(0);
    heads = resultType.getDimSize(1);
    queryLength = resultType.getDimSize(2);
    keyLength = resultType.getDimSize(3);

    auto hidden = checkedProduct(task, {heads, headDim},
                                 "attention score hidden dimension");
    if (failed(hidden))
      return failure();
    if (firstInputType.getDimSize(0) != batch ||
        firstInputType.getDimSize(1) != queryLength ||
        firstInputType.getDimSize(2) != *hidden ||
        secondInputType.getDimSize(0) != batch ||
        secondInputType.getDimSize(1) != keyLength ||
        secondInputType.getDimSize(2) != *hidden) {
      return task.emitOpError(
          "attention score tensor types disagree with head metadata");
    }
  } else {
    if (firstInputType.getRank() != 4 || secondInputType.getRank() != 3 ||
        resultType.getRank() != 4) {
      return task.emitOpError(
          "attention apply distribution expects rank-4 probabilities, "
          "rank-3 values, and a rank-4 result");
    }
    batch = resultType.getDimSize(0);
    heads = resultType.getDimSize(1);
    queryLength = resultType.getDimSize(2);
    if (resultType.getDimSize(3) != headDim) {
      return task.emitOpError(
          "attention apply result disagrees with head_dim metadata");
    }
    keyLength = firstInputType.getDimSize(3);
    auto hidden = checkedProduct(task, {heads, headDim},
                                 "attention apply hidden dimension");
    if (failed(hidden))
      return failure();
    if (firstInputType.getDimSize(0) != batch ||
        firstInputType.getDimSize(1) != heads ||
        firstInputType.getDimSize(2) != queryLength ||
        secondInputType.getDimSize(0) != batch ||
        secondInputType.getDimSize(1) != keyLength ||
        secondInputType.getDimSize(2) != *hidden) {
      return task.emitOpError(
          "attention apply tensor types disagree with head metadata");
    }
  }

  if (batch <= 0 || heads <= 0 || queryLength <= 0 || keyLength <= 0) {
    return task.emitOpError(
        "attention matmul distribution requires positive static dimensions");
  }

  auto firstResource =
      dyn_cast<TaskResourceType>(task.getInputs()[0].getType());
  auto secondResource =
      dyn_cast<TaskResourceType>(task.getInputs()[1].getType());
  auto outputResource =
      dyn_cast<TaskResourceType>(task.getOutputs()[0].getType());
  if (!firstResource || !secondResource || !outputResource ||
      firstResource.getValueType() != firstInputType ||
      secondResource.getValueType() != secondInputType ||
      outputResource.getValueType() != resultType) {
    return task.emitOpError(
        "attention task resources must match its callee signature");
  }

  return std::optional<AttentionMatmulMatch>(AttentionMatmulMatch{
      task, callee, kind, firstInputType, secondInputType, resultType, batch,
      heads, queryLength, keyLength, headDim, causal});
}

static SmallVector<Slice, 8> splitDimension(int64_t dimension,
                                            int64_t shardCount) {
  SmallVector<Slice, 8> slices;
  slices.reserve(shardCount);
  int64_t base = dimension / shardCount;
  int64_t extra = dimension % shardCount;
  int64_t offset = 0;
  for (int64_t shard = 0; shard < shardCount; ++shard) {
    int64_t size = base + (shard < extra ? 1 : 0);
    slices.push_back(Slice{offset, size});
    offset += size;
  }
  return slices;
}

static FailureOr<SmallVector<Slice, 8>> scaleSlices(Operation *anchor,
                                                    ArrayRef<Slice> slices,
                                                    int64_t scale,
                                                    StringRef description) {
  SmallVector<Slice, 8> scaled;
  scaled.reserve(slices.size());
  for (const Slice &slice : slices) {
    auto offset = checkedProduct(anchor, {slice.offset, scale}, description);
    auto size = checkedProduct(anchor, {slice.size, scale}, description);
    if (failed(offset) || failed(size))
      return failure();
    scaled.push_back(Slice{*offset, *size});
  }
  return scaled;
}

static FailureOr<DistributionPlan>
buildPlan(Operation *anchor, const MatmulMatch &match, int64_t rowShards,
          int64_t columnShards, MatmulDistributionStrategy strategy) {
  DistributionPlan plan;
  plan.strategy = strategy;
  plan.rowShards = rowShards;
  plan.columnShards = columnShards;
  plan.rowSlices = splitDimension(match.rows, rowShards);
  plan.columnSlices = splitDimension(match.columns, columnShards);

  int64_t maximumRows = plan.rowSlices.front().size;
  int64_t maximumColumns = plan.columnSlices.front().size;
  auto maxOps =
      checkedProduct(anchor, {2, maximumRows, maximumColumns, match.reduction},
                     "digital matmul shard operation count");
  if (failed(maxOps))
    return failure();
  plan.maxShardOps = *maxOps;

  auto lhsBytes = checkedProduct(anchor, {match.rows, match.reduction, 4},
                                 "digital matmul left operand byte count");
  auto rhsBytes = checkedProduct(anchor, {match.columns, match.reduction, 4},
                                 "digital matmul right operand byte count");
  auto resultBytes = checkedProduct(anchor, {match.rows, match.columns, 4},
                                    "digital matmul result byte count");
  if (failed(lhsBytes) || failed(rhsBytes) || failed(resultBytes))
    return failure();

  auto broadcastLhs =
      checkedProduct(anchor, {*lhsBytes, columnShards},
                     "digital matmul left broadcast byte count");
  auto broadcastRhs =
      checkedProduct(anchor, {*rhsBytes, rowShards},
                     "digital matmul right broadcast byte count");
  if (failed(broadcastLhs) || failed(broadcastRhs))
    return failure();
  auto communication =
      checkedSum(anchor, {*broadcastLhs, *broadcastRhs, *resultBytes},
                 "digital matmul distribution communication byte count");
  if (failed(communication))
    return failure();
  plan.communicationBytes = *communication;
  return plan;
}

static bool isBetterPlan(const DistributionPlan &candidate,
                         const DistributionPlan &current) {
  return std::tie(candidate.maxShardOps, candidate.communicationBytes,
                  candidate.strategy, candidate.rowShards,
                  candidate.columnShards) <
         std::tie(current.maxShardOps, current.communicationBytes,
                  current.strategy, current.rowShards, current.columnShards);
}

static FailureOr<std::optional<DistributionPlan>>
selectPlan(const MatmulMatch &match,
           const DigitalMatmulDistributionOptions &options) {
  std::optional<DistributionPlan> best;
  int64_t maxRows = std::min(match.rows, options.maxShards);
  for (int64_t rowShards = 1; rowShards <= maxRows; ++rowShards) {
    int64_t maxColumns = std::min(match.columns, options.maxShards / rowShards);
    for (int64_t columnShards = 1; columnShards <= maxColumns; ++columnShards) {
      int64_t shardCount = rowShards * columnShards;
      if (shardCount < 2 || shardCount > options.maxShards)
        continue;

      MatmulDistributionStrategy strategy;
      if (rowShards == 1)
        strategy = MatmulDistributionStrategy::OutputColumns;
      else if (columnShards == 1)
        strategy = MatmulDistributionStrategy::OutputRows;
      else
        strategy = MatmulDistributionStrategy::TwoDimensional;
      if (options.strategy && strategy != *options.strategy)
        continue;

      int64_t smallestRows = match.rows / rowShards;
      int64_t smallestColumns = match.columns / columnShards;
      auto smallestOps = checkedProduct(
          match.task, {2, smallestRows, smallestColumns, match.reduction},
          "digital matmul minimum shard operation count");
      if (failed(smallestOps))
        return failure();
      if (*smallestOps < options.minOpsPerShard)
        continue;

      auto candidate =
          buildPlan(match.task, match, rowShards, columnShards, strategy);
      if (failed(candidate))
        return failure();
      if (!best || isBetterPlan(*candidate, *best))
        best = std::move(*candidate);
    }
  }
  return best;
}

static FailureOr<int64_t> getTensorByteSize(Operation *anchor,
                                            RankedTensorType type,
                                            StringRef description) {
  SmallVector<int64_t, 8> factors(type.getShape());
  factors.push_back(4);
  return checkedProduct(anchor, factors, description);
}

static FailureOr<std::optional<DistributionPlan>>
selectAttentionPlan(const AttentionMatmulMatch &match,
                    const DigitalMatmulDistributionOptions &options) {
  if (options.strategy &&
      *options.strategy != MatmulDistributionStrategy::AttentionHeads)
    return std::optional<DistributionPlan>();

  auto firstBytes = getTensorByteSize(match.task, match.firstInputType,
                                      "attention first-input byte count");
  auto secondBytes = getTensorByteSize(match.task, match.secondInputType,
                                       "attention second-input byte count");
  auto resultBytes = getTensorByteSize(match.task, match.resultType,
                                       "attention result byte count");
  if (failed(firstBytes) || failed(secondBytes) || failed(resultBytes))
    return failure();
  auto communication =
      checkedSum(match.task, {*firstBytes, *secondBytes, *resultBytes},
                 "attention distribution communication byte count");
  if (failed(communication))
    return failure();

  std::optional<DistributionPlan> best;
  int64_t maximumShards = std::min(match.heads, options.maxShards);
  for (int64_t headShards = 2; headShards <= maximumShards; ++headShards) {
    SmallVector<Slice, 8> headSlices = splitDimension(match.heads, headShards);
    auto minimumOps =
        checkedProduct(match.task,
                       {2, match.batch, headSlices.back().size,
                        match.queryLength, match.keyLength, match.headDim},
                       "attention minimum shard operation count");
    if (failed(minimumOps))
      return failure();
    if (*minimumOps < options.minOpsPerShard)
      continue;

    DistributionPlan candidate;
    candidate.strategy = MatmulDistributionStrategy::AttentionHeads;
    candidate.rowShards = headShards;
    candidate.columnShards = 1;
    candidate.rowSlices = std::move(headSlices);
    candidate.columnSlices.push_back(Slice{0, 1});
    auto maximumOps =
        checkedProduct(match.task,
                       {2, match.batch, candidate.rowSlices.front().size,
                        match.queryLength, match.keyLength, match.headDim},
                       "attention maximum shard operation count");
    if (failed(maximumOps))
      return failure();
    candidate.maxShardOps = *maximumOps;
    candidate.communicationBytes = *communication;
    if (!best || isBetterPlan(candidate, *best))
      best = std::move(candidate);
  }
  return best;
}

static std::string getUniqueSymbolName(ModuleOp module, StringRef base) {
  std::string name = base.str();
  unsigned suffix = 0;
  while (module.lookupSymbol(name))
    name = (base + "_" + Twine(++suffix)).str();
  return name;
}

static void attachTaskFunctionMetadata(func::FuncOp function,
                                       TaskCreateOp original, StringRef kind,
                                       StringRef name, int64_t ordinal) {
  OpBuilder builder(function.getContext());
  function->setAttr(task_attrs::kTaskDomainAttrName,
                    builder.getStringAttr(task_graph_names::kDigitalDomain));
  function->setAttr(task_attrs::kTaskKindAttrName, builder.getStringAttr(kind));
  function->setAttr(task_attrs::kTaskNameAttrName, builder.getStringAttr(name));
  function->setAttr(task_attrs::kSourceLayerAttrName,
                    builder.getStringAttr(original.getSourceLayer()));
  function->setAttr(task_attrs::kSourceTaskOrdinalAttrName,
                    builder.getI64IntegerAttr(ordinal));
}

static TaskDistributionAttr getDistributionAttr(
    OpBuilder &builder, int64_t groupId, TaskDistributionRole role,
    int64_t shardId, const DistributionPlan &plan, int64_t rowShard,
    int64_t columnShard, DistributionPlacementPolicy placement) {
  return TaskDistributionAttr::get(
      builder.getContext(), builder.getI64IntegerAttr(groupId), role,
      builder.getI64IntegerAttr(shardId),
      builder.getI64IntegerAttr(plan.getShardCount()),
      builder.getI64IntegerAttr(rowShard),
      builder.getI64IntegerAttr(plan.rowShards),
      builder.getI64IntegerAttr(columnShard),
      builder.getI64IntegerAttr(plan.columnShards), plan.strategy, placement);
}

static func::FuncOp createPartitionFunction(
    ModuleOp module, TaskCreateOp original, StringRef operandName,
    RankedTensorType inputType, int64_t axis, ArrayRef<Slice> slices,
    int64_t groupId, const DistributionPlan &plan,
    DistributionPlacementPolicy placement, int64_t ordinal) {
  SmallVector<Type, 8> resultTypes;
  resultTypes.reserve(slices.size());
  for (const Slice &slice : slices) {
    SmallVector<int64_t, 8> shape(inputType.getShape());
    shape[axis] = slice.size;
    resultTypes.push_back(
        RankedTensorType::get(shape, inputType.getElementType()));
  }

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  std::string base = ("__sculptor_distribute_" + original.getTaskName() +
                      "_group" + Twine(groupId) + "_partition_" + operandName)
                         .str();
  auto function = builder.create<func::FuncOp>(
      original.getLoc(), getUniqueSymbolName(module, base),
      builder.getFunctionType({inputType}, resultTypes));
  function.setPrivate();
  std::string taskName =
      (original.getTaskName() + ".partition." + operandName).str();
  attachTaskFunctionMetadata(function, original,
                             task_graph_names::kDigitalMatmulPartitionTaskKind,
                             taskName, ordinal);
  function->setAttr(task_graph_attrs::kTaskDistributionAttrName,
                    getDistributionAttr(builder, groupId,
                                        TaskDistributionRole::Partition, -1,
                                        plan, -1, -1, placement));

  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  SmallVector<Value, 8> results;
  for (const Slice &slice : slices) {
    SmallVector<int64_t, 8> offsets(inputType.getRank(), 0);
    SmallVector<int64_t, 8> sizes(inputType.getShape());
    SmallVector<int64_t, 8> strides(inputType.getRank(), 1);
    offsets[axis] = slice.offset;
    sizes[axis] = slice.size;
    auto resultType = cast<RankedTensorType>(resultTypes[results.size()]);
    results.push_back(
        builder
            .create<tensor::ExtractSliceOp>(
                original.getLoc(), resultType, entry->getArgument(0),
                getIndexAttrs(builder, offsets), getIndexAttrs(builder, sizes),
                getIndexAttrs(builder, strides))
            .getResult());
  }
  builder.create<func::ReturnOp>(original.getLoc(), results);
  return function;
}

static func::FuncOp
createShardFunction(ModuleOp module, TaskCreateOp original,
                    const MatmulMatch &match, RankedTensorType lhsType,
                    RankedTensorType rhsType, RankedTensorType resultType,
                    int64_t groupId, const DistributionPlan &plan,
                    int64_t rowShard, int64_t columnShard, int64_t shardId,
                    DistributionPlacementPolicy placement, int64_t ordinal,
                    int64_t digitalOps) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  std::string base = ("__sculptor_distribute_" + original.getTaskName() +
                      "_group" + Twine(groupId) + "_shard" + Twine(shardId))
                         .str();
  auto function = builder.create<func::FuncOp>(
      original.getLoc(), getUniqueSymbolName(module, base),
      builder.getFunctionType({lhsType, rhsType}, {resultType}));
  function.setPrivate();
  std::string taskName =
      (original.getTaskName() + ".shard." + Twine(shardId)).str();
  attachTaskFunctionMetadata(function, original,
                             task_graph_names::kDigitalMatmulShardTaskKind,
                             taskName, ordinal);
  auto distribution =
      getDistributionAttr(builder, groupId, TaskDistributionRole::Shard,
                          shardId, plan, rowShard, columnShard, placement);
  function->setAttr(task_graph_attrs::kTaskDistributionAttrName, distribution);
  function->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                    builder.getI64IntegerAttr(digitalOps));

  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value empty = builder.create<tensor::EmptyOp>(
      original.getLoc(), resultType.getShape(), resultType.getElementType());
  Value zero = builder.create<arith::ConstantOp>(
      original.getLoc(), builder.getF32Type(), builder.getF32FloatAttr(0.0));
  Value initialized =
      builder.create<linalg::FillOp>(original.getLoc(), zero, empty)
          .getResult(0);
  Value result;
  if (match.bodyKind == MatmulBodyKind::Matmul) {
    result = builder
                 .create<linalg::MatmulOp>(
                     original.getLoc(),
                     ValueRange{entry->getArgument(0), entry->getArgument(1)},
                     ValueRange{initialized})
                 .getResult(0);
  } else {
    result = builder
                 .create<linalg::MatmulTransposeBOp>(
                     original.getLoc(),
                     ValueRange{entry->getArgument(0), entry->getArgument(1)},
                     ValueRange{initialized})
                 .getResult(0);
  }
  builder.create<func::ReturnOp>(original.getLoc(), result);
  return function;
}

static Value buildAttentionHeadView(OpBuilder &builder, Location loc,
                                    Value sequence, int64_t batch,
                                    int64_t sequenceLength, int64_t heads,
                                    int64_t headDim) {
  auto inputType = cast<RankedTensorType>(sequence.getType());
  auto expandedType = RankedTensorType::get(
      {batch, sequenceLength, heads, headDim}, inputType.getElementType());
  SmallVector<ReassociationIndices, 3> reassociation = {{0}, {1}, {2, 3}};
  return builder
      .create<tensor::ExpandShapeOp>(loc, expandedType, sequence, reassociation)
      .getResult();
}

static Value buildAttentionScoresShard(OpBuilder &builder, Location loc,
                                       Value query, Value key,
                                       RankedTensorType resultType,
                                       int64_t batch, int64_t heads,
                                       int64_t queryLength, int64_t keyLength,
                                       int64_t headDim, bool causal) {
  Value queryHeads = buildAttentionHeadView(builder, loc, query, batch,
                                            queryLength, heads, headDim);
  Value keyHeads = buildAttentionHeadView(builder, loc, key, batch, keyLength,
                                          heads, headDim);

  MLIRContext *context = builder.getContext();
  AffineExpr b = builder.getAffineDimExpr(0);
  AffineExpr h = builder.getAffineDimExpr(1);
  AffineExpr q = builder.getAffineDimExpr(2);
  AffineExpr k = builder.getAffineDimExpr(3);
  AffineExpr d = builder.getAffineDimExpr(4);
  AffineMap queryMap = AffineMap::get(5, 0, {b, q, h, d}, context);
  AffineMap keyMap = AffineMap::get(5, 0, {b, k, h, d}, context);
  AffineMap scoreMap = AffineMap::get(5, 0, {b, h, q, k}, context);
  SmallVector<utils::IteratorType, 5> contractionIterators = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};

  Value zero = builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                                 builder.getF32FloatAttr(0.0));
  Value rawEmpty = builder.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                   resultType.getElementType());
  Value rawInit =
      builder.create<linalg::FillOp>(loc, zero, rawEmpty).getResult(0);
  Value rawScores =
      builder
          .create<linalg::GenericOp>(
              loc, resultType, ValueRange{queryHeads, keyHeads},
              ValueRange{rawInit},
              SmallVector<AffineMap, 3>{queryMap, keyMap, scoreMap},
              contractionIterators,
              [](OpBuilder &nestedBuilder, Location nestedLoc,
                 ValueRange arguments) {
                Value product = nestedBuilder.create<arith::MulFOp>(
                    nestedLoc, arguments[0], arguments[1]);
                Value sum = nestedBuilder.create<arith::AddFOp>(
                    nestedLoc, arguments[2], product);
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
              })
          .getResult(0);

  Value scaledEmpty = builder.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType());
  AffineMap identityMap = builder.getMultiDimIdentityMap(4);
  SmallVector<utils::IteratorType, 4> parallelIterators(
      4, utils::IteratorType::parallel);
  double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
  return builder
      .create<linalg::GenericOp>(
          loc, resultType, ValueRange{rawScores}, ValueRange{scaledEmpty},
          SmallVector<AffineMap, 2>{identityMap, identityMap},
          parallelIterators,
          [causal, scale](OpBuilder &nestedBuilder, Location nestedLoc,
                          ValueRange arguments) {
            Value scaleValue = nestedBuilder.create<arith::ConstantOp>(
                nestedLoc, nestedBuilder.getF32Type(),
                nestedBuilder.getF32FloatAttr(scale));
            Value result = nestedBuilder.create<arith::MulFOp>(
                nestedLoc, arguments[0], scaleValue);
            if (causal) {
              Value queryIndex =
                  nestedBuilder.create<linalg::IndexOp>(nestedLoc, 2);
              Value keyIndex =
                  nestedBuilder.create<linalg::IndexOp>(nestedLoc, 3);
              Value masked = nestedBuilder.create<arith::CmpIOp>(
                  nestedLoc, arith::CmpIPredicate::ugt, keyIndex, queryIndex);
              Value negativeInfinity = nestedBuilder.create<arith::ConstantOp>(
                  nestedLoc, nestedBuilder.getF32Type(),
                  nestedBuilder.getF32FloatAttr(
                      -std::numeric_limits<float>::infinity()));
              result = nestedBuilder.create<arith::SelectOp>(
                  nestedLoc, masked, negativeInfinity, result);
            }
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
          })
      .getResult(0);
}

static Value buildAttentionApplyShard(OpBuilder &builder, Location loc,
                                      Value probabilities, Value value,
                                      RankedTensorType resultType,
                                      int64_t batch, int64_t heads,
                                      int64_t queryLength, int64_t keyLength,
                                      int64_t headDim) {
  Value valueHeads = buildAttentionHeadView(builder, loc, value, batch,
                                            keyLength, heads, headDim);

  MLIRContext *context = builder.getContext();
  AffineExpr b = builder.getAffineDimExpr(0);
  AffineExpr h = builder.getAffineDimExpr(1);
  AffineExpr q = builder.getAffineDimExpr(2);
  AffineExpr d = builder.getAffineDimExpr(3);
  AffineExpr k = builder.getAffineDimExpr(4);
  AffineMap probabilityMap = AffineMap::get(5, 0, {b, h, q, k}, context);
  AffineMap valueMap = AffineMap::get(5, 0, {b, k, h, d}, context);
  AffineMap resultMap = AffineMap::get(5, 0, {b, h, q, d}, context);
  SmallVector<utils::IteratorType, 5> contractionIterators = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};

  Value zero = builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                                 builder.getF32FloatAttr(0.0));
  Value empty = builder.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                resultType.getElementType());
  Value initialized =
      builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);
  return builder
      .create<linalg::GenericOp>(
          loc, resultType, ValueRange{probabilities, valueHeads},
          ValueRange{initialized},
          SmallVector<AffineMap, 3>{probabilityMap, valueMap, resultMap},
          contractionIterators,
          [](OpBuilder &nestedBuilder, Location nestedLoc,
             ValueRange arguments) {
            Value product = nestedBuilder.create<arith::MulFOp>(
                nestedLoc, arguments[0], arguments[1]);
            Value sum = nestedBuilder.create<arith::AddFOp>(
                nestedLoc, arguments[2], product);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum);
          })
      .getResult(0);
}

static func::FuncOp createAttentionShardFunction(
    ModuleOp module, TaskCreateOp original, const AttentionMatmulMatch &match,
    RankedTensorType firstInputType, RankedTensorType secondInputType,
    RankedTensorType resultType, int64_t shardHeads, int64_t groupId,
    const DistributionPlan &plan, int64_t rowShard, int64_t shardId,
    DistributionPlacementPolicy placement, int64_t ordinal,
    int64_t digitalOps) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  std::string base = ("__sculptor_distribute_" + original.getTaskName() +
                      "_group" + Twine(groupId) + "_shard" + Twine(shardId))
                         .str();
  auto function = builder.create<func::FuncOp>(
      original.getLoc(), getUniqueSymbolName(module, base),
      builder.getFunctionType({firstInputType, secondInputType}, {resultType}));
  function.setPrivate();
  std::string taskName =
      (original.getTaskName() + ".shard." + Twine(shardId)).str();
  attachTaskFunctionMetadata(function, original,
                             task_graph_names::kDigitalMatmulShardTaskKind,
                             taskName, ordinal);
  auto distribution =
      getDistributionAttr(builder, groupId, TaskDistributionRole::Shard,
                          shardId, plan, rowShard, 0, placement);
  function->setAttr(task_graph_attrs::kTaskDistributionAttrName, distribution);
  function->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                    builder.getI64IntegerAttr(digitalOps));
  function->setAttr(task_attrs::kAttentionHeadDimAttrName,
                    builder.getI64IntegerAttr(match.headDim));
  if (match.kind == AttentionMatmulKind::Scores) {
    function->setAttr(task_attrs::kAttentionCausalAttrName,
                      builder.getBoolAttr(match.causal));
  }

  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value result;
  if (match.kind == AttentionMatmulKind::Scores) {
    result = buildAttentionScoresShard(
        builder, original.getLoc(), entry->getArgument(0),
        entry->getArgument(1), resultType, match.batch, shardHeads,
        match.queryLength, match.keyLength, match.headDim, match.causal);
  } else {
    result = buildAttentionApplyShard(
        builder, original.getLoc(), entry->getArgument(0),
        entry->getArgument(1), resultType, match.batch, shardHeads,
        match.queryLength, match.keyLength, match.headDim);
  }
  builder.create<func::ReturnOp>(original.getLoc(), result);
  return function;
}

static func::FuncOp createAttentionAssemblyFunction(
    ModuleOp module, TaskCreateOp original, RankedTensorType resultType,
    ArrayRef<RankedTensorType> shardTypes, int64_t groupId,
    const DistributionPlan &plan, DistributionPlacementPolicy placement,
    int64_t ordinal) {
  SmallVector<Type, 8> inputs(shardTypes.begin(), shardTypes.end());
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  std::string base = ("__sculptor_distribute_" + original.getTaskName() +
                      "_group" + Twine(groupId) + "_assemble")
                         .str();
  auto function = builder.create<func::FuncOp>(
      original.getLoc(), getUniqueSymbolName(module, base),
      builder.getFunctionType(inputs, {resultType}));
  function.setPrivate();
  std::string taskName = (original.getTaskName() + ".assemble").str();
  attachTaskFunctionMetadata(function, original,
                             task_graph_names::kDigitalMatmulAssemblyTaskKind,
                             taskName, ordinal);
  function->setAttr(task_graph_attrs::kTaskDistributionAttrName,
                    getDistributionAttr(builder, groupId,
                                        TaskDistributionRole::Assembly, -1,
                                        plan, -1, -1, placement));

  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value assembled = builder.create<tensor::EmptyOp>(
      original.getLoc(), resultType.getShape(), resultType.getElementType());
  for (auto [index, headSlice] : llvm::enumerate(plan.rowSlices)) {
    SmallVector<int64_t, 4> offsets = {0, headSlice.offset, 0, 0};
    SmallVector<int64_t, 4> sizes(
        cast<RankedTensorType>(entry->getArgument(index).getType()).getShape());
    assembled =
        builder
            .create<tensor::InsertSliceOp>(
                original.getLoc(), entry->getArgument(index), assembled,
                getIndexAttrs(builder, offsets), getIndexAttrs(builder, sizes),
                getIndexAttrs(builder, {1, 1, 1, 1}))
            .getResult();
  }
  builder.create<func::ReturnOp>(original.getLoc(), assembled);
  return function;
}

static func::FuncOp
createAssemblyFunction(ModuleOp module, TaskCreateOp original,
                       RankedTensorType resultType,
                       ArrayRef<RankedTensorType> shardTypes, int64_t groupId,
                       const DistributionPlan &plan,
                       DistributionPlacementPolicy placement, int64_t ordinal) {
  SmallVector<Type, 8> inputs(shardTypes.begin(), shardTypes.end());
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  std::string base = ("__sculptor_distribute_" + original.getTaskName() +
                      "_group" + Twine(groupId) + "_assemble")
                         .str();
  auto function = builder.create<func::FuncOp>(
      original.getLoc(), getUniqueSymbolName(module, base),
      builder.getFunctionType(inputs, {resultType}));
  function.setPrivate();
  std::string taskName = (original.getTaskName() + ".assemble").str();
  attachTaskFunctionMetadata(function, original,
                             task_graph_names::kDigitalMatmulAssemblyTaskKind,
                             taskName, ordinal);
  function->setAttr(task_graph_attrs::kTaskDistributionAttrName,
                    getDistributionAttr(builder, groupId,
                                        TaskDistributionRole::Assembly, -1,
                                        plan, -1, -1, placement));

  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value assembled = builder.create<tensor::EmptyOp>(
      original.getLoc(), resultType.getShape(), resultType.getElementType());
  unsigned argument = 0;
  for (int64_t row = 0; row < plan.rowShards; ++row) {
    for (int64_t column = 0; column < plan.columnShards; ++column) {
      const Slice &rowSlice = plan.rowSlices[row];
      const Slice &columnSlice = plan.columnSlices[column];
      assembled =
          builder
              .create<tensor::InsertSliceOp>(
                  original.getLoc(), entry->getArgument(argument++), assembled,
                  getIndexAttrs(builder, {rowSlice.offset, columnSlice.offset}),
                  getIndexAttrs(builder, {rowSlice.size, columnSlice.size}),
                  getIndexAttrs(builder, {1, 1}))
              .getResult();
    }
  }
  builder.create<func::ReturnOp>(original.getLoc(), assembled);
  return function;
}

static Value createIntermediateResource(OpBuilder &builder, Location loc,
                                        Value graph,
                                        RankedTensorType tensorType) {
  return builder
      .create<TaskGraphIntermediateOp>(
          loc, TaskResourceType::get(builder.getContext(), tensorType), graph)
      .getResult();
}

static TaskCreateOp
createTask(OpBuilder &builder, TaskCreateOp original, func::FuncOp callee,
           StringRef kind, StringRef name, int64_t ordinal,
           ArrayRef<Value> inputs, ArrayRef<Value> outputs,
           ArrayRef<Value> dependencies, TaskDistributionAttr distribution,
           std::optional<int64_t> digitalOps = std::nullopt) {
  auto task = builder.create<TaskCreateOp>(
      original.getLoc(), original.getResult().getType(), original.getGraph(),
      FlatSymbolRefAttr::get(builder.getContext(), callee.getSymName()),
      builder.getStringAttr(task_graph_names::kDigitalDomain),
      builder.getStringAttr(kind), builder.getStringAttr(name),
      builder.getStringAttr(original.getSourceLayer()),
      builder.getI64IntegerAttr(ordinal), inputs, outputs,
      deduplicateValues(dependencies));
  task->setAttr(task_graph_attrs::kTaskDistributionAttrName, distribution);
  if (!outputs.empty()) {
    SmallVector<int64_t, 8> resultIndices;
    for (auto result : llvm::enumerate(outputs))
      resultIndices.push_back(result.index());
    task->setAttr(runtime_attrs::kTaskResultIndicesAttrName,
                  builder.getI64ArrayAttr(resultIndices));
  }
  if (digitalOps) {
    task->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                  builder.getI64IntegerAttr(*digitalOps));
  }
  return task;
}

static LogicalResult
rewriteMatmul(ModuleOp module, func::FuncOp taskGraphFunc,
              const MatmulMatch &match, const DistributionPlan &plan,
              DistributionPlacementPolicy placementPolicy, int64_t groupId,
              llvm::StringMap<int64_t> &nextOrdinalByLayer,
              SmallVectorImpl<func::FuncOp> &obsoleteCallees) {
  TaskCreateOp original = match.task;
  int64_t &nextOrdinal = nextOrdinalByLayer[original.getSourceLayer()];

  auto taskOps = taskGraphFunc.getOps<TaskCreateOp>();
  auto firstTask = taskOps.begin();
  if (firstTask == taskOps.end())
    return original.emitOpError("expected task graph to contain the matmul");

  OpBuilder resourceBuilder(original.getContext());
  resourceBuilder.setInsertionPoint(*firstTask);
  OpBuilder taskBuilder(original);
  SmallVector<Value, 8> originalDependencies(original.getDependencies().begin(),
                                             original.getDependencies().end());
  SmallVector<Value, 2> originalInputs(original.getInputs().begin(),
                                       original.getInputs().end());
  SmallVector<Value, 2> originalOutputs(original.getOutputs().begin(),
                                        original.getOutputs().end());

  SmallVector<Value, 8> lhsResources;
  SmallVector<Value, 8> rhsResources;
  SmallVector<Value, 8> lhsProducerTokens;
  SmallVector<Value, 8> rhsProducerTokens;

  if (plan.rowShards > 1) {
    int64_t ordinal = nextOrdinal++;
    func::FuncOp partition = createPartitionFunction(
        module, original, "lhs", match.lhsType, /*axis=*/0, plan.rowSlices,
        groupId, plan, placementPolicy, ordinal);
    for (const Slice &slice : plan.rowSlices) {
      auto type = RankedTensorType::get({slice.size, match.reduction},
                                        match.lhsType.getElementType());
      lhsResources.push_back(createIntermediateResource(
          resourceBuilder, original.getLoc(), original.getGraph(), type));
    }
    auto distribution = getDistributionAttr(taskBuilder, groupId,
                                            TaskDistributionRole::Partition, -1,
                                            plan, -1, -1, placementPolicy);
    TaskCreateOp partitionTask =
        createTask(taskBuilder, original, partition,
                   task_graph_names::kDigitalMatmulPartitionTaskKind,
                   (original.getTaskName() + ".partition.lhs").str(), ordinal,
                   {original.getInputs()[0]}, lhsResources,
                   selectDependenciesForInputs(
                       originalDependencies, {original.getInputs()[0]},
                       originalInputs, /*includeControlOnly=*/true),
                   distribution);
    lhsProducerTokens.assign(lhsResources.size(), partitionTask.getResult());
  } else {
    lhsResources.push_back(original.getInputs()[0]);
  }

  if (plan.columnShards > 1) {
    int64_t ordinal = nextOrdinal++;
    int64_t rhsAxis = match.bodyKind == MatmulBodyKind::Matmul ? 1 : 0;
    func::FuncOp partition = createPartitionFunction(
        module, original, "rhs", match.rhsType, rhsAxis, plan.columnSlices,
        groupId, plan, placementPolicy, ordinal);
    for (const Slice &slice : plan.columnSlices) {
      RankedTensorType type;
      if (match.bodyKind == MatmulBodyKind::Matmul) {
        type = RankedTensorType::get({match.reduction, slice.size},
                                     match.rhsType.getElementType());
      } else {
        type = RankedTensorType::get({slice.size, match.reduction},
                                     match.rhsType.getElementType());
      }
      rhsResources.push_back(createIntermediateResource(
          resourceBuilder, original.getLoc(), original.getGraph(), type));
    }
    auto distribution = getDistributionAttr(taskBuilder, groupId,
                                            TaskDistributionRole::Partition, -1,
                                            plan, -1, -1, placementPolicy);
    TaskCreateOp partitionTask =
        createTask(taskBuilder, original, partition,
                   task_graph_names::kDigitalMatmulPartitionTaskKind,
                   (original.getTaskName() + ".partition.rhs").str(), ordinal,
                   {original.getInputs()[1]}, rhsResources,
                   selectDependenciesForInputs(
                       originalDependencies, {original.getInputs()[1]},
                       originalInputs, /*includeControlOnly=*/true),
                   distribution);
    rhsProducerTokens.assign(rhsResources.size(), partitionTask.getResult());
  } else {
    rhsResources.push_back(original.getInputs()[1]);
  }

  SmallVector<Value, 8> shardResources;
  SmallVector<Value, 8> shardTokens;
  SmallVector<RankedTensorType, 8> shardTypes;
  int64_t shardId = 0;
  for (int64_t row = 0; row < plan.rowShards; ++row) {
    for (int64_t column = 0; column < plan.columnShards; ++column) {
      const Slice &rowSlice = plan.rowSlices[row];
      const Slice &columnSlice = plan.columnSlices[column];
      auto lhsType = RankedTensorType::get({rowSlice.size, match.reduction},
                                           match.lhsType.getElementType());
      RankedTensorType rhsType;
      if (match.bodyKind == MatmulBodyKind::Matmul) {
        rhsType = RankedTensorType::get({match.reduction, columnSlice.size},
                                        match.rhsType.getElementType());
      } else {
        rhsType = RankedTensorType::get({columnSlice.size, match.reduction},
                                        match.rhsType.getElementType());
      }
      auto resultType = RankedTensorType::get(
          {rowSlice.size, columnSlice.size}, match.resultType.getElementType());
      auto digitalOps = checkedProduct(
          original, {2, rowSlice.size, columnSlice.size, match.reduction},
          "digital matmul shard operation count");
      if (failed(digitalOps))
        return failure();
      int64_t ordinal = nextOrdinal++;
      func::FuncOp callee = createShardFunction(
          module, original, match, lhsType, rhsType, resultType, groupId, plan,
          row, column, shardId, placementPolicy, ordinal, *digitalOps);
      Value shardResource = createIntermediateResource(
          resourceBuilder, original.getLoc(), original.getGraph(), resultType);
      SmallVector<Value, 8> dependencies;
      if (plan.rowShards > 1) {
        dependencies.push_back(lhsProducerTokens[row]);
      } else {
        llvm::append_range(dependencies,
                           selectDependenciesForInputs(
                               originalDependencies, {original.getInputs()[0]},
                               originalInputs, /*includeControlOnly=*/false));
      }
      if (plan.columnShards > 1) {
        dependencies.push_back(rhsProducerTokens[column]);
      } else {
        llvm::append_range(dependencies,
                           selectDependenciesForInputs(
                               originalDependencies, {original.getInputs()[1]},
                               originalInputs, /*includeControlOnly=*/false));
      }
      auto distribution =
          getDistributionAttr(taskBuilder, groupId, TaskDistributionRole::Shard,
                              shardId, plan, row, column, placementPolicy);
      TaskCreateOp shardTask = createTask(
          taskBuilder, original, callee,
          task_graph_names::kDigitalMatmulShardTaskKind,
          (original.getTaskName() + ".shard." + Twine(shardId)).str(), ordinal,
          {lhsResources[row], rhsResources[column]}, {shardResource},
          dependencies, distribution, *digitalOps);
      shardResources.push_back(shardResource);
      shardTokens.push_back(shardTask.getResult());
      shardTypes.push_back(resultType);
      ++shardId;
    }
  }

  int64_t assemblyOrdinal = nextOrdinal++;
  func::FuncOp assembly =
      createAssemblyFunction(module, original, match.resultType, shardTypes,
                             groupId, plan, placementPolicy, assemblyOrdinal);
  auto assemblyDistribution =
      getDistributionAttr(taskBuilder, groupId, TaskDistributionRole::Assembly,
                          -1, plan, -1, -1, placementPolicy);
  TaskCreateOp assemblyTask = createTask(
      taskBuilder, original, assembly,
      task_graph_names::kDigitalMatmulAssemblyTaskKind,
      (original.getTaskName() + ".assemble").str(), assemblyOrdinal,
      shardResources, originalOutputs, shardTokens, assemblyDistribution);

  obsoleteCallees.push_back(match.callee);
  original.getResult().replaceAllUsesWith(assemblyTask.getResult());
  original.erase();
  return success();
}

static LogicalResult rewriteAttentionMatmul(
    ModuleOp module, func::FuncOp taskGraphFunc,
    const AttentionMatmulMatch &match, const DistributionPlan &plan,
    DistributionPlacementPolicy placementPolicy, int64_t groupId,
    llvm::StringMap<int64_t> &nextOrdinalByLayer,
    SmallVectorImpl<func::FuncOp> &obsoleteCallees) {
  TaskCreateOp original = match.task;
  int64_t &nextOrdinal = nextOrdinalByLayer[original.getSourceLayer()];

  auto taskOps = taskGraphFunc.getOps<TaskCreateOp>();
  auto firstTask = taskOps.begin();
  if (firstTask == taskOps.end())
    return original.emitOpError("expected task graph to contain the attention");

  auto hiddenSlices = scaleSlices(original, plan.rowSlices, match.headDim,
                                  "attention hidden-dimension shard");
  if (failed(hiddenSlices))
    return failure();

  OpBuilder resourceBuilder(original.getContext());
  resourceBuilder.setInsertionPoint(*firstTask);
  OpBuilder taskBuilder(original);
  SmallVector<Value, 8> originalDependencies(original.getDependencies().begin(),
                                             original.getDependencies().end());
  SmallVector<Value, 2> originalInputs(original.getInputs().begin(),
                                       original.getInputs().end());
  SmallVector<Value, 2> originalOutputs(original.getOutputs().begin(),
                                        original.getOutputs().end());

  int64_t firstAxis = match.kind == AttentionMatmulKind::Scores ? 2 : 1;
  ArrayRef<Slice> firstSlices = match.kind == AttentionMatmulKind::Scores
                                    ? ArrayRef<Slice>(*hiddenSlices)
                                    : ArrayRef<Slice>(plan.rowSlices);
  StringRef firstName =
      match.kind == AttentionMatmulKind::Scores ? "query" : "probabilities";
  int64_t firstPartitionOrdinal = nextOrdinal++;
  func::FuncOp firstPartition = createPartitionFunction(
      module, original, firstName, match.firstInputType, firstAxis, firstSlices,
      groupId, plan, placementPolicy, firstPartitionOrdinal);
  SmallVector<Value, 8> firstResources;
  for (const Slice &slice : firstSlices) {
    SmallVector<int64_t, 8> shape(match.firstInputType.getShape());
    shape[firstAxis] = slice.size;
    firstResources.push_back(createIntermediateResource(
        resourceBuilder, original.getLoc(), original.getGraph(),
        RankedTensorType::get(shape, match.firstInputType.getElementType())));
  }
  auto partitionDistribution =
      getDistributionAttr(taskBuilder, groupId, TaskDistributionRole::Partition,
                          -1, plan, -1, -1, placementPolicy);
  TaskCreateOp firstPartitionTask = createTask(
      taskBuilder, original, firstPartition,
      task_graph_names::kDigitalMatmulPartitionTaskKind,
      (original.getTaskName() + ".partition." + firstName).str(),
      firstPartitionOrdinal, {original.getInputs()[0]}, firstResources,
      selectDependenciesForInputs(originalDependencies,
                                  {original.getInputs()[0]}, originalInputs,
                                  /*includeControlOnly=*/true),
      partitionDistribution);

  StringRef secondName =
      match.kind == AttentionMatmulKind::Scores ? "key" : "value";
  int64_t secondPartitionOrdinal = nextOrdinal++;
  func::FuncOp secondPartition = createPartitionFunction(
      module, original, secondName, match.secondInputType, /*axis=*/2,
      *hiddenSlices, groupId, plan, placementPolicy, secondPartitionOrdinal);
  SmallVector<Value, 8> secondResources;
  for (const Slice &slice : *hiddenSlices) {
    SmallVector<int64_t, 8> shape(match.secondInputType.getShape());
    shape[2] = slice.size;
    secondResources.push_back(createIntermediateResource(
        resourceBuilder, original.getLoc(), original.getGraph(),
        RankedTensorType::get(shape, match.secondInputType.getElementType())));
  }
  TaskCreateOp secondPartitionTask = createTask(
      taskBuilder, original, secondPartition,
      task_graph_names::kDigitalMatmulPartitionTaskKind,
      (original.getTaskName() + ".partition." + secondName).str(),
      secondPartitionOrdinal, {original.getInputs()[1]}, secondResources,
      selectDependenciesForInputs(originalDependencies,
                                  {original.getInputs()[1]}, originalInputs,
                                  /*includeControlOnly=*/true),
      partitionDistribution);

  SmallVector<Value, 8> shardResources;
  SmallVector<Value, 8> shardTokens;
  SmallVector<RankedTensorType, 8> shardTypes;
  for (auto [shardId, headSlice] : llvm::enumerate(plan.rowSlices)) {
    RankedTensorType firstType;
    if (match.kind == AttentionMatmulKind::Scores) {
      firstType = RankedTensorType::get(
          {match.batch, match.queryLength, (*hiddenSlices)[shardId].size},
          match.firstInputType.getElementType());
    } else {
      firstType = RankedTensorType::get(
          {match.batch, headSlice.size, match.queryLength, match.keyLength},
          match.firstInputType.getElementType());
    }
    auto secondType = RankedTensorType::get(
        {match.batch, match.keyLength, (*hiddenSlices)[shardId].size},
        match.secondInputType.getElementType());
    RankedTensorType resultType;
    if (match.kind == AttentionMatmulKind::Scores) {
      resultType = RankedTensorType::get(
          {match.batch, headSlice.size, match.queryLength, match.keyLength},
          match.resultType.getElementType());
    } else {
      resultType = RankedTensorType::get(
          {match.batch, headSlice.size, match.queryLength, match.headDim},
          match.resultType.getElementType());
    }
    auto digitalOps =
        checkedProduct(original,
                       {2, match.batch, headSlice.size, match.queryLength,
                        match.keyLength, match.headDim},
                       "attention shard operation count");
    if (failed(digitalOps))
      return failure();

    int64_t ordinal = nextOrdinal++;
    func::FuncOp shardFunction = createAttentionShardFunction(
        module, original, match, firstType, secondType, resultType,
        headSlice.size, groupId, plan, shardId, shardId, placementPolicy,
        ordinal, *digitalOps);
    Value shardResource = createIntermediateResource(
        resourceBuilder, original.getLoc(), original.getGraph(), resultType);
    SmallVector<Value, 8> dependencies;
    dependencies.push_back(firstPartitionTask.getResult());
    dependencies.push_back(secondPartitionTask.getResult());
    auto shardDistribution =
        getDistributionAttr(taskBuilder, groupId, TaskDistributionRole::Shard,
                            shardId, plan, shardId, 0, placementPolicy);
    TaskCreateOp shardTask = createTask(
        taskBuilder, original, shardFunction,
        task_graph_names::kDigitalMatmulShardTaskKind,
        (original.getTaskName() + ".shard." + Twine(shardId)).str(), ordinal,
        {firstResources[shardId], secondResources[shardId]}, {shardResource},
        dependencies, shardDistribution, *digitalOps);
    shardResources.push_back(shardResource);
    shardTokens.push_back(shardTask.getResult());
    shardTypes.push_back(resultType);
  }

  int64_t assemblyOrdinal = nextOrdinal++;
  func::FuncOp assembly = createAttentionAssemblyFunction(
      module, original, match.resultType, shardTypes, groupId, plan,
      placementPolicy, assemblyOrdinal);
  auto assemblyDistribution =
      getDistributionAttr(taskBuilder, groupId, TaskDistributionRole::Assembly,
                          -1, plan, -1, -1, placementPolicy);
  TaskCreateOp assemblyTask = createTask(
      taskBuilder, original, assembly,
      task_graph_names::kDigitalMatmulAssemblyTaskKind,
      (original.getTaskName() + ".assemble").str(), assemblyOrdinal,
      shardResources, originalOutputs, shardTokens, assemblyDistribution);

  obsoleteCallees.push_back(match.callee);
  original.getResult().replaceAllUsesWith(assemblyTask.getResult());
  original.erase();
  return success();
}

static LogicalResult verifyRewrittenTaskGraph(func::FuncOp taskGraphFunc) {
  auto dag = parseTaskGraphDAG(taskGraphFunc);
  if (failed(dag))
    return failure();
  if (failed(buildTaskExecutionGraph(taskGraphFunc, *dag)))
    return failure();
  llvm::DenseMap<Value, unsigned> producerByResource;
  return collectResourceProducers(*dag, producerByResource);
}

} // namespace

FailureOr<unsigned>
distributeDigitalMatmuls(ModuleOp module, func::FuncOp taskGraphFunc,
                         const DigitalMatmulDistributionOptions &options) {
  if (failed(rejectStalePlacementMetadata(taskGraphFunc)))
    return failure();

  SmallVector<MatmulMatch, 8> matmulCandidates;
  SmallVector<AttentionMatmulMatch, 8> attentionCandidates;
  bool matchRankTwo =
      !options.strategy ||
      *options.strategy != MatmulDistributionStrategy::AttentionHeads;
  bool matchAttention =
      !options.strategy ||
      *options.strategy == MatmulDistributionStrategy::AttentionHeads;
  for (TaskCreateOp task : taskGraphFunc.getOps<TaskCreateOp>()) {
    if (matchRankTwo) {
      auto matmulCandidate = matchDigitalMatmul(module, task);
      if (failed(matmulCandidate))
        return failure();
      if (*matmulCandidate)
        matmulCandidates.push_back(**matmulCandidate);
    }

    if (matchAttention) {
      auto attentionCandidate = matchAttentionMatmul(module, task);
      if (failed(attentionCandidate))
        return failure();
      if (*attentionCandidate)
        attentionCandidates.push_back(**attentionCandidate);
    }
  }
  if (matmulCandidates.empty() && attentionCandidates.empty())
    return 0u;

  llvm::StringMap<int64_t> nextOrdinalByLayer =
      collectNextSourceOrdinals(taskGraphFunc);
  int64_t nextGroupId = collectNextDistributionGroupId(taskGraphFunc);
  SmallVector<func::FuncOp, 8> obsoleteCallees;
  unsigned distributedCount = 0;
  for (const MatmulMatch &candidate : matmulCandidates) {
    auto plan = selectPlan(candidate, options);
    if (failed(plan))
      return failure();
    if (!*plan)
      continue;
    if (failed(rewriteMatmul(module, taskGraphFunc, candidate, **plan,
                             options.placement, nextGroupId++,
                             nextOrdinalByLayer, obsoleteCallees)))
      return failure();
    ++distributedCount;
  }
  for (const AttentionMatmulMatch &candidate : attentionCandidates) {
    auto plan = selectAttentionPlan(candidate, options);
    if (failed(plan))
      return failure();
    if (!*plan)
      continue;
    if (failed(rewriteAttentionMatmul(module, taskGraphFunc, candidate, **plan,
                                      options.placement, nextGroupId++,
                                      nextOrdinalByLayer, obsoleteCallees)))
      return failure();
    ++distributedCount;
  }

  if (distributedCount == 0)
    return 0u;
  if (failed(verifyRewrittenTaskGraph(taskGraphFunc))) {
    taskGraphFunc.emitError(
        "digital matmul distribution produced an invalid task graph");
    return failure();
  }

  llvm::SmallPtrSet<Operation *, 8> erased;
  for (func::FuncOp callee : obsoleteCallees) {
    if (!callee || !erased.insert(callee.getOperation()).second)
      continue;
    if (SymbolTable::symbolKnownUseEmpty(callee, module))
      callee.erase();
  }
  return distributedCount;
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
