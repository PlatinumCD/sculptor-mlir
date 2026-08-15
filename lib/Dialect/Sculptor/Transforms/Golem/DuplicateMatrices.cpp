#include "sculptor-mlir/Dialect/Sculptor/Transforms/Golem/DuplicateMatrices.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/GolemTilingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"

#include <optional>
#include <string>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
namespace mapping = mlir::sculptor::mapping;
namespace tiling = mlir::sculptor::golem_tiling_attrs;

struct MappingStage {
  int64_t id = -1;
  StringRef kind;
  SmallVector<Operation *> members;
};

struct SetupReplicationPlan {
  MappingStage *setup = nullptr;
  SmallVector<MappingStage *> consumers;
  int64_t replicaCount = 1;
};

FailureOr<StringRef> getStageKind(Operation *operation) {
  auto kind = operation->getAttrOfType<StringAttr>(mapping::kStageKindAttrName);
  if (!kind)
    return operation->emitError("expected mapping stage kind"), failure();
  return kind.getValue();
}

FailureOr<StringRef> getSourceResource(const MappingStage &stage) {
  for (Operation *member : stage.members) {
    if (auto resource =
            member->getAttrOfType<StringAttr>(tiling::kSourceResourceAttrName))
      return resource.getValue();
  }
  return stage.members.front()->emitError(
             "matrix setup is missing sculptor.source_resource"),
         failure();
}

FailureOr<sculptor::ArraySetOp> getArraySet(const MappingStage &stage) {
  sculptor::ArraySetOp result;
  for (Operation *member : stage.members) {
    auto candidate = dyn_cast<sculptor::ArraySetOp>(member);
    if (!candidate)
      continue;
    if (result)
      return member->emitError(
                 "matrix-setup stage contains multiple array.set operations"),
             failure();
    result = candidate;
  }
  if (!result)
    return stage.members.front()->emitError(
               "matrix-setup stage contains no array.set operation"),
           failure();
  return result;
}

FailureOr<Value> getPhysicalMVMArray(const MappingStage &stage) {
  SmallVector<Value> arrays;
  int64_t loads = 0;
  int64_t executes = 0;
  int64_t stores = 0;
  auto append = [&](Value value) {
    if (!llvm::is_contained(arrays, value))
      arrays.push_back(value);
  };

  for (Operation *member : stage.members)
    member->walk([&](Operation *operation) {
      if (auto load = dyn_cast<sculptor::ArrayLoadOp>(operation)) {
        ++loads;
        append(load.getArray());
      } else if (auto execute = dyn_cast<sculptor::ArrayExecuteOp>(operation)) {
        ++executes;
        append(execute.getArray());
      } else if (auto store = dyn_cast<sculptor::ArrayStoreOp>(operation)) {
        ++stores;
        append(store.getArray());
      }
    });

  if (loads == 0 || executes == 0 || stores == 0)
    return stage.members.front()->emitError(
               "physical-MVM stage must contain array.load, array.execute, "
               "and array.store"),
           failure();
  if (arrays.size() != 1)
    return stage.members.front()->emitError(
               "physical-MVM stage must reference exactly one logical array"),
           failure();
  return arrays.front();
}

LogicalResult replacePhysicalMVMArray(const MappingStage &stage, Value oldArray,
                                      Value newArray) {
  for (Operation *member : stage.members) {
    WalkResult result = member->walk([&](Operation *operation) -> WalkResult {
      Value referencedArray;
      if (auto load = dyn_cast<sculptor::ArrayLoadOp>(operation))
        referencedArray = load.getArray();
      else if (auto execute = dyn_cast<sculptor::ArrayExecuteOp>(operation))
        referencedArray = execute.getArray();
      else if (auto store = dyn_cast<sculptor::ArrayStoreOp>(operation))
        referencedArray = store.getArray();
      else
        return WalkResult::advance();

      if (referencedArray == oldArray)
        return WalkResult::advance();
      operation->emitError(
          "physical-MVM array operation references an unexpected logical "
          "array");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      return failure();
  }

  for (Operation *member : stage.members) {
    member->walk([&](Operation *operation) {
      if (auto load = dyn_cast<sculptor::ArrayLoadOp>(operation))
        load.getArrayMutable().set(newArray);
      else if (auto execute = dyn_cast<sculptor::ArrayExecuteOp>(operation))
        execute.getArrayMutable().set(newArray);
      else if (auto store = dyn_cast<sculptor::ArrayStoreOp>(operation))
        store.getArrayMutable().set(newArray);
    });
  }
  return success();
}

FailureOr<Value> cloneMatrixSetup(const MappingStage &setup, int64_t newStageId,
                                  int64_t matrixId, int64_t replicaId,
                                  StringRef consumerName,
                                  Operation *insertBefore,
                                  IRRewriter &rewriter) {
  IRMapping valueMapping;
  auto originalArraySet = getArraySet(setup);
  if (failed(originalArraySet))
    return failure();

  std::string replicaName =
      (Twine(consumerName) + "_matrix_setup_replica_" + Twine(replicaId)).str();

  rewriter.setInsertionPoint(insertBefore);
  for (Operation *member : setup.members) {
    Operation *clone = rewriter.clone(*member, valueMapping);
    clone->setAttr(mapping::kStageIdAttrName,
                   rewriter.getI64IntegerAttr(newStageId));
    clone->setAttr(mapping::kStageNameAttrName,
                   rewriter.getStringAttr(replicaName));
    clone->setAttr(tiling::kMatrixIdAttrName,
                   rewriter.getI64IntegerAttr(matrixId));
    clone->setAttr(tiling::kMatrixReplicaIdAttrName,
                   rewriter.getI64IntegerAttr(replicaId));
  }

  Value clonedArray = valueMapping.lookupOrNull((*originalArraySet).getArray());
  if (!clonedArray)
    return insertBefore->emitError(
               "failed to clone matrix-setup logical array result"),
           failure();
  return clonedArray;
}

LogicalResult eraseUnusedOriginalSetups(ArrayRef<MappingStage *> setups,
                                        IRRewriter &rewriter) {
  for (MappingStage *setup : setups) {
    DenseSet<Operation *> members(setup->members.begin(), setup->members.end());
    for (Operation *member : setup->members) {
      for (Value result : member->getResults()) {
        for (OpOperand &use : result.getUses()) {
          if (!members.contains(use.getOwner()))
            return use.getOwner()->emitError(
                "original matrix setup still has a non-replicated user");
        }
      }
    }
    for (Operation *member : llvm::reverse(setup->members))
      rewriter.eraseOp(member);
  }
  return success();
}

int64_t getMaximumReplicaCount(const SetupReplicationPlan &plan,
                               int64_t minimumMVMsPerReplica,
                               int64_t maximumReplicasPerSetup) {
  int64_t consumerCount = static_cast<int64_t>(plan.consumers.size());
  int64_t maximum = std::max<int64_t>(1, consumerCount / minimumMVMsPerReplica);
  if (maximumReplicasPerSetup > 0)
    maximum = std::min(maximum, maximumReplicasPerSetup);
  return maximum;
}

LogicalResult allocateReplicaCounts(func::FuncOp function,
                                    MutableArrayRef<SetupReplicationPlan> plans,
                                    int64_t arrayCapacity,
                                    int64_t minimumMVMsPerReplica,
                                    int64_t maximumReplicasPerSetup) {
  if (arrayCapacity == 0) {
    for (SetupReplicationPlan &plan : plans)
      plan.replicaCount = static_cast<int64_t>(plan.consumers.size());
    return success();
  }

  int64_t replicaCount = static_cast<int64_t>(plans.size());
  if (replicaCount > arrayCapacity) {
    return function.emitError("matrix replication requires at least ")
           << replicaCount << " persistent arrays for the unreplicated "
           << "matrix setups, but array-capacity is " << arrayCapacity;
  }

  // Greedily spend each remaining physical lane where it reduces the largest
  // fanout pressure most. N/(R*(R+1)) is the marginal reduction in idealized
  // consumers per replica when a setup with N consumers grows from R to R+1.
  // Stable stage-ID tie breaking makes the policy deterministic.
  while (replicaCount < arrayCapacity) {
    SetupReplicationPlan *best = nullptr;
    long double bestBenefit = -1.0L;
    int64_t bestFanout = -1;
    for (SetupReplicationPlan &plan : plans) {
      int64_t maximum = getMaximumReplicaCount(plan, minimumMVMsPerReplica,
                                               maximumReplicasPerSetup);
      if (plan.replicaCount >= maximum)
        continue;
      int64_t fanout = static_cast<int64_t>(plan.consumers.size());
      long double replicas = static_cast<long double>(plan.replicaCount);
      long double benefit =
          static_cast<long double>(fanout) / (replicas * (replicas + 1.0L));
      if (!best || benefit > bestBenefit ||
          (benefit == bestBenefit && fanout > bestFanout) ||
          (benefit == bestBenefit && fanout == bestFanout &&
           plan.setup->id < best->setup->id)) {
        best = &plan;
        bestBenefit = benefit;
        bestFanout = fanout;
      }
    }
    if (!best)
      break;
    ++best->replicaCount;
    ++replicaCount;
  }
  return success();
}

LogicalResult duplicateMatricesInFunction(func::FuncOp function,
                                          llvm::StringMap<int64_t> &matrixIds,
                                          int64_t &nextMatrixId,
                                          int64_t arrayCapacity,
                                          int64_t minimumMVMsPerReplica,
                                          int64_t maximumReplicasPerSetup) {
  DenseMap<int64_t, MappingStage> stages;
  SmallVector<int64_t> stageOrder;
  int64_t maximumStageId = -1;

  for (Block &block : function.getBody()) {
    for (Operation &operation : block) {
      auto stageId =
          operation.getAttrOfType<IntegerAttr>(mapping::kStageIdAttrName);
      if (!stageId)
        continue;
      auto stageKind = getStageKind(&operation);
      if (failed(stageKind))
        return failure();

      int64_t id = stageId.getInt();
      maximumStageId = std::max(maximumStageId, id);
      auto [it, inserted] = stages.try_emplace(id);
      MappingStage &stage = it->second;
      if (inserted) {
        stage.id = id;
        stage.kind = *stageKind;
        stageOrder.push_back(id);
      } else if (stage.kind != *stageKind) {
        return operation.emitError(
            "one mapping stage ID has inconsistent stage kinds");
      }
      stage.members.push_back(&operation);
    }
  }

  SmallVector<MappingStage *> setupStages;
  SmallVector<MappingStage *> physicalMVMStages;
  DenseMap<Value, MappingStage *> setupByArray;
  for (int64_t stageId : stageOrder) {
    MappingStage &stage = stages[stageId];
    if (stage.kind == mapping::kMatrixSetupStageKind) {
      auto arraySet = getArraySet(stage);
      if (failed(arraySet))
        return failure();
      setupStages.push_back(&stage);
      setupByArray[(*arraySet).getArray()] = &stage;

      auto source = getSourceResource(stage);
      if (failed(source))
        return failure();
      if (!matrixIds.count(*source))
        matrixIds[*source] = nextMatrixId++;
    } else if (stage.kind == mapping::kPhysicalMVMStageKind) {
      physicalMVMStages.push_back(&stage);
    }
  }

  if (physicalMVMStages.empty())
    return success();
  if (setupStages.empty())
    return function.emitError(
        "duplicate-matrices found physical MVMs but no matrix setups");

  SmallVector<SetupReplicationPlan> replicationPlans;
  DenseMap<MappingStage *, int64_t> planBySetup;
  for (MappingStage *physicalMVM : physicalMVMStages) {
    auto oldArray = getPhysicalMVMArray(*physicalMVM);
    if (failed(oldArray))
      return failure();
    MappingStage *setup = setupByArray.lookup(*oldArray);
    if (!setup)
      return physicalMVM->members.front()->emitError(
          "physical MVM logical array has no matrix-setup stage");
    auto [entry, inserted] = planBySetup.try_emplace(
        setup, static_cast<int64_t>(replicationPlans.size()));
    if (inserted)
      replicationPlans.push_back({setup, {}, 1});
    replicationPlans[entry->second].consumers.push_back(physicalMVM);
  }

  if (failed(allocateReplicaCounts(function, replicationPlans, arrayCapacity,
                                   minimumMVMsPerReplica,
                                   maximumReplicasPerSetup)))
    return failure();

  IRRewriter rewriter(function.getContext());
  int64_t nextStageId = maximumStageId + 1;
  int64_t nextReplicaId = 0;
  for (SetupReplicationPlan &plan : replicationPlans) {
    auto source = getSourceResource(*plan.setup);
    if (failed(source))
      return failure();
    int64_t matrixId = matrixIds.lookup(*source);
    SmallVector<Value> replicaArrays;
    SmallVector<int64_t> replicaIds;
    replicaArrays.reserve(plan.replicaCount);
    replicaIds.reserve(plan.replicaCount);
    int64_t consumerCount = static_cast<int64_t>(plan.consumers.size());
    for (int64_t replica = 0; replica < plan.replicaCount; ++replica) {
      int64_t firstConsumerIndex =
          llvm::divideCeil(replica * consumerCount, plan.replicaCount);
      MappingStage *representative = plan.consumers[firstConsumerIndex];
      StringRef consumerName = "physical_mvm";
      if (auto name =
              representative->members.front()->getAttrOfType<StringAttr>(
                  mapping::kStageNameAttrName))
        consumerName = name.getValue();
      int64_t replicaId = nextReplicaId++;
      auto clonedArray = cloneMatrixSetup(
          *plan.setup, nextStageId++, matrixId, replicaId, consumerName,
          representative->members.front(), rewriter);
      if (failed(clonedArray))
        return failure();
      replicaArrays.push_back(*clonedArray);
      replicaIds.push_back(replicaId);
    }

    auto originalArraySet = getArraySet(*plan.setup);
    if (failed(originalArraySet))
      return failure();
    Value originalArray = (*originalArraySet).getArray();
    for (auto [consumerIndex, physicalMVM] : llvm::enumerate(plan.consumers)) {
      int64_t replica = static_cast<int64_t>(consumerIndex) *
                        plan.replicaCount / consumerCount;
      if (failed(replacePhysicalMVMArray(*physicalMVM, originalArray,
                                         replicaArrays[replica])))
        return failure();
      for (Operation *member : physicalMVM->members) {
        member->setAttr(tiling::kMatrixIdAttrName,
                        rewriter.getI64IntegerAttr(matrixId));
        member->setAttr(tiling::kMatrixReplicaIdAttrName,
                        rewriter.getI64IntegerAttr(replicaIds[replica]));
      }
    }
  }

  return eraseUnusedOriginalSetups(setupStages, rewriter);
}

} // namespace

namespace mlir {
namespace sculptor {

void DuplicateMatricesPass::runOnOperation() {
  if (arrayCapacity < 0) {
    getOperation().emitError("duplicate-matrices array-capacity must be "
                             "nonnegative");
    signalPassFailure();
    return;
  }
  if (minimumMVMsPerReplica <= 0) {
    getOperation().emitError(
        "duplicate-matrices minimum-mvms-per-replica must be positive");
    signalPassFailure();
    return;
  }
  if (maximumReplicasPerSetup < 0) {
    getOperation().emitError(
        "duplicate-matrices maximum-replicas-per-setup must be nonnegative");
    signalPassFailure();
    return;
  }
  llvm::StringMap<int64_t> matrixIds;
  int64_t nextMatrixId = 0;
  for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
    if (function.isExternal())
      continue;
    if (failed(duplicateMatricesInFunction(function, matrixIds, nextMatrixId,
                                           arrayCapacity, minimumMVMsPerReplica,
                                           maximumReplicasPerSetup))) {
      signalPassFailure();
      return;
    }
  }
}

void registerDuplicateMatricesPass() {
  PassRegistration<DuplicateMatricesPass>();
}

} // namespace sculptor
} // namespace mlir
