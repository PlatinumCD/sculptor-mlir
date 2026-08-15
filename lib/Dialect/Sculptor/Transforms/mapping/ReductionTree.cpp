#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ReductionTree.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SHA256.h"

#include <array>
#include <functional>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

std::string digestToHex(const std::array<uint8_t, 32> &digest) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string value;
  value.reserve(64);
  for (uint8_t byte : digest) {
    value.push_back(hex[byte >> 4]);
    value.push_back(hex[byte & 0x0f]);
  }
  return value;
}

TaskReductionAttr getReduction(Operation *operation) {
  return operation->getAttrOfType<TaskReductionAttr>(
      task_graph_attrs::kTaskReductionAttrName);
}

bool isCompatibleAdd(linalg::AddOp candidate, TaskReductionAttr rootReduction,
                     Type resultType) {
  TaskReductionAttr reduction = getReduction(candidate);
  return reduction && reduction == rootReduction &&
         reduction.getReassociate().getValue() &&
         reduction.getKind() == TaskReductionKind::Add &&
         candidate->getNumResults() == 1 &&
         candidate.getResult(0).getType() == resultType;
}

void annotateReductionOperation(Operation *operation, Operation *source,
                                int64_t stageId, int64_t treeId, int64_t nodeId,
                                int64_t level, int64_t ordinal, int64_t width,
                                OpBuilder &builder) {
  static constexpr StringLiteral copiedPrefixes[] = {
      "sculptor.semantic.", "sculptor.source_", "sculptor.task.reduction",
      "sculptor.mapping.mvm_wave_"};
  for (NamedAttribute attribute : source->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (llvm::any_of(copiedPrefixes, [&](StringRef prefix) {
          return name.starts_with(prefix);
        }))
      operation->setAttr(attribute.getName(), attribute.getValue());
  }
  operation->setAttr(kStageIdAttrName, builder.getI64IntegerAttr(stageId));
  operation->setAttr(kStageKindAttrName,
                     builder.getStringAttr(kDigitalStageKind));
  operation->setAttr(
      kStageNameAttrName,
      builder.getStringAttr(
          (Twine("reduction_tree_") + Twine(treeId) + "_node_" + Twine(nodeId))
              .str()));
  operation->setAttr("sculptor.semantic.section",
                     builder.getStringAttr("digital.reduction"));
  operation->setAttr(kReductionTreeIdAttrName,
                     builder.getI64IntegerAttr(treeId));
  operation->setAttr(kReductionNodeIdAttrName,
                     builder.getI64IntegerAttr(nodeId));
  operation->setAttr(kReductionLevelAttrName, builder.getI64IntegerAttr(level));
  operation->setAttr(kReductionOrdinalAttrName,
                     builder.getI64IntegerAttr(ordinal));
  operation->setAttr(kReductionWidthAttrName, builder.getI64IntegerAttr(width));
  operation->setAttr(task_graph_attrs::kTaskReductionTreeIdAttrName,
                     builder.getI64IntegerAttr(treeId));
  operation->setAttr(task_graph_attrs::kTaskReductionLevelAttrName,
                     builder.getI64IntegerAttr(level));
  operation->setAttr(task_graph_attrs::kTaskReductionWidthAttrName,
                     builder.getI64IntegerAttr(width));
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<ReductionTreePolicy> parseReductionTreePolicy(StringRef value,
                                                        Operation *anchor) {
  if (value == "none")
    return ReductionTreePolicy::None;
  if (value == "balanced")
    return ReductionTreePolicy::Balanced;
  anchor->emitError("unknown reduction-tree policy '") << value << "'";
  return failure();
}

StringRef stringifyReductionTreePolicy(ReductionTreePolicy policy) {
  switch (policy) {
  case ReductionTreePolicy::None:
    return "none";
  case ReductionTreePolicy::Balanced:
    return "balanced";
  }
  llvm_unreachable("unknown reduction-tree policy");
}

FailureOr<int64_t> buildReductionTrees(func::FuncOp function,
                                       ReductionTreePolicy policy,
                                       int64_t fanIn, int64_t minimumWidth) {
  function->removeAttr(kReductionTreeFingerprintAttrName);
  function->removeAttr("sculptor.mapping.reduction_node_count");
  function->removeAttr("sculptor.mapping.maximum_reduction_fan_in");
  if (policy == ReductionTreePolicy::None) {
    OpBuilder builder(function.getContext());
    function->setAttr("sculptor.mapping.reduction_node_count",
                      builder.getI64IntegerAttr(0));
    function->setAttr("sculptor.mapping.maximum_reduction_fan_in",
                      builder.getI64IntegerAttr(0));
    return int64_t{0};
  }
  if (fanIn != 2)
    return function.emitError("balanced reduction trees require fan-in=2"),
           failure();
  if (minimumWidth < 3)
    return function.emitError("reduction-min-width must be at least three"),
           failure();

  int64_t nextStageId = 0;
  function.walk([&](Operation *operation) {
    if (auto stage = operation->getAttrOfType<IntegerAttr>(kStageIdAttrName))
      nextStageId = std::max(nextStageId, stage.getInt() + 1);
  });

  int64_t treeCount = 0;
  int64_t totalNodeCount = 0;
  auto buildTree = [&](Operation *source, ArrayRef<Value> leaves,
                       RankedTensorType tensorType) -> Value {
    OpBuilder builder(source);
    builder.setInsertionPoint(source);
    SmallVector<Value> levelValues(leaves.begin(), leaves.end());
    int64_t treeId = treeCount++;
    int64_t nodeId = 0;
    int64_t level = 0;
    while (levelValues.size() > 1) {
      SmallVector<Value> nextLevel;
      for (size_t index = 0; index < levelValues.size(); index += 2) {
        if (index + 1 == levelValues.size()) {
          nextLevel.push_back(levelValues[index]);
          continue;
        }
        auto empty = builder.create<tensor::EmptyOp>(
            source->getLoc(), tensorType.getShape(), tensorType.getElementType());
        auto add = builder.create<linalg::AddOp>(
            source->getLoc(),
            ValueRange{levelValues[index], levelValues[index + 1]},
            ValueRange{empty.getResult()});
        int64_t stageId = nextStageId++;
        annotateReductionOperation(empty, source, stageId, treeId, nodeId,
                                   level, index / 2, leaves.size(), builder);
        annotateReductionOperation(add, source, stageId, treeId, nodeId, level,
                                   index / 2, leaves.size(), builder);
        nextLevel.push_back(add.getResult(0));
        ++nodeId;
        ++totalNodeCount;
      }
      levelValues = std::move(nextLevel);
      ++level;
    }
    return levelValues.front();
  };

  // Current MVM expansion emits one N-input elementwise generic for tile
  // recombination.  The task-reduction attribute is the explicit semantic
  // contract that permits reassociation; normalize that canonical form before
  // handling already-binary linalg.add chains below.
  SmallVector<linalg::GenericOp> genericReductions;
  function.walk([&](linalg::GenericOp generic) {
    TaskReductionAttr reduction = getReduction(generic);
    if (!reduction || !reduction.getReassociate().getValue() ||
        reduction.getKind() != TaskReductionKind::Add ||
        generic->getNumResults() != 1 ||
        static_cast<int64_t>(generic.getNumDpsInputs()) < minimumWidth)
      return;
    Type resultType = generic->getResult(0).getType();
    bool uniformInputs = llvm::all_of(
        generic.getDpsInputOperands(), [&](OpOperand *operand) {
          return operand->get().getType() == resultType;
        });
    bool identityMaps = llvm::all_of(generic.getIndexingMapsArray(),
                                     [](AffineMap map) {
                                       return map.isIdentity();
                                     });
    if (uniformInputs && identityMaps)
      genericReductions.push_back(generic);
  });
  for (linalg::GenericOp generic : genericReductions) {
    auto tensorType = dyn_cast<RankedTensorType>(generic->getResult(0).getType());
    if (!tensorType || !tensorType.hasStaticShape())
      return generic.emitOpError(
                 "balanced reduction requires a static ranked tensor"),
             failure();
    SmallVector<Value> leaves;
    for (OpOperand *input : generic.getDpsInputOperands())
      leaves.push_back(input->get());
    Value result = buildTree(generic, leaves, tensorType);
    generic->getResult(0).replaceAllUsesWith(result);
    SmallVector<Operation *> oldInitializers;
    for (Value init : generic.getDpsInits())
      if (Operation *defining = init.getDefiningOp())
        oldInitializers.push_back(defining);
    generic.erase();
    for (Operation *initializer : oldInitializers)
      if (llvm::all_of(initializer->getResults(),
                       [](Value value) { return value.use_empty(); }))
        initializer->erase();
  }

  SmallVector<linalg::AddOp> candidates;
  function.walk([&](linalg::AddOp add) {
    // Nodes emitted by buildTree already form the requested balanced tree.
    // Do not rediscover and rebuild that fresh chain in the legacy binary-add
    // path below.
    if (add->hasAttr(kReductionTreeIdAttrName))
      return;
    TaskReductionAttr reduction = getReduction(add);
    if (reduction && reduction.getReassociate().getValue() &&
        reduction.getKind() == TaskReductionKind::Add)
      candidates.push_back(add);
  });

  SmallVector<linalg::AddOp> roots;
  for (linalg::AddOp candidate : candidates) {
    bool consumedByCompatibleAdd = false;
    if (candidate.getResult(0).hasOneUse()) {
      Operation *consumer = (*candidate.getResult(0).getUsers().begin());
      if (auto add = dyn_cast<linalg::AddOp>(consumer))
        consumedByCompatibleAdd = isCompatibleAdd(
            add, getReduction(candidate), candidate.getResult(0).getType());
    }
    if (!consumedByCompatibleAdd)
      roots.push_back(candidate);
  }

  llvm::SmallPtrSet<Operation *, 32> rewritten;
  for (linalg::AddOp root : roots) {
    if (rewritten.contains(root))
      continue;
    TaskReductionAttr reduction = getReduction(root);
    Type resultType = root.getResult(0).getType();
    SmallVector<Value> leaves;
    SmallVector<linalg::AddOp> oldAdds;
    llvm::SmallPtrSet<Operation *, 16> visited;
    std::function<void(Value)> flatten = [&](Value value) {
      auto add = value.getDefiningOp<linalg::AddOp>();
      if (!add || !isCompatibleAdd(add, reduction, resultType) ||
          (add != root && !add.getResult(0).hasOneUse()) ||
          !visited.insert(add).second) {
        leaves.push_back(value);
        return;
      }
      oldAdds.push_back(add);
      auto inputs = add.getDpsInputOperands();
      if (inputs.size() != 2) {
        leaves.push_back(value);
        oldAdds.pop_back();
        return;
      }
      flatten(inputs[0]->get());
      flatten(inputs[1]->get());
    };
    flatten(root.getResult(0));
    if (static_cast<int64_t>(leaves.size()) < minimumWidth)
      continue;

    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType || !tensorType.hasStaticShape())
      return root.emitOpError(
                 "balanced reduction requires a static ranked tensor"),
             failure();

    Value result = buildTree(root, leaves, tensorType);
    root.getResult(0).replaceAllUsesWith(result);
    llvm::SmallPtrSet<Operation *, 16> oldEmptyOperations;
    for (linalg::AddOp add : oldAdds) {
      for (Value init : add.getDpsInits()) {
        if (Operation *defining = init.getDefiningOp())
          oldEmptyOperations.insert(defining);
      }
      rewritten.insert(add);
    }
    for (linalg::AddOp add : oldAdds) {
      if (!add.getResult(0).use_empty())
        return add.emitOpError(
                   "reduction rewrite left an external internal-node use"),
               failure();
      add.erase();
    }
    for (Operation *empty : oldEmptyOperations) {
      if (llvm::all_of(empty->getResults(),
                       [](Value result) { return result.use_empty(); }))
        empty->erase();
    }
  }

  std::string fingerprint =
      (Twine("balanced:") + Twine(treeCount) + ":" + Twine(totalNodeCount))
          .str();
  llvm::SHA256 sha;
  sha.update(fingerprint);
  function->setAttr(
      kReductionTreeFingerprintAttrName,
      StringAttr::get(function.getContext(), digestToHex(sha.final())));
  OpBuilder builder(function.getContext());
  function->setAttr("sculptor.mapping.reduction_node_count",
                    builder.getI64IntegerAttr(totalNodeCount));
  function->setAttr("sculptor.mapping.maximum_reduction_fan_in",
                    builder.getI64IntegerAttr(totalNodeCount == 0 ? 0 : 2));
  return treeCount;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
