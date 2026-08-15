#include "sculptor-mlir/Dialect/Sculptor/Transforms/PlanMapping.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingCostProfile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingEvaluator.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingProblem.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/planners/MappingPlannerRegistry.h"

#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

#include <memory>
#include <optional>
#include <string>

namespace {

using namespace mlir;
using namespace mlir::sculptor;

FailureOr<mapping::MappingHardwareModel>
buildHardwareModel(PlanMappingPass &pass, ModuleOp module) {
  FailureOr<mapping::AnalogIOPolicy> analogIO =
      mapping::parseAnalogIOPolicy(pass.analogIOPolicy, module);
  FailureOr<mapping::AnalogArrayExecutionPolicy> arrayExecution =
      mapping::parseAnalogArrayExecutionPolicy(pass.analogArrayExecution,
                                               module);
  FailureOr<mapping::NetworkContentionModel> contention =
      mapping::parseNetworkContentionModel(pass.networkContentionModel, module);
  if (failed(analogIO) || failed(arrayExecution) || failed(contention))
    return failure();

  mapping::MappingHardwareModel hardware;
  hardware.meshRows = pass.meshRows;
  hardware.meshCols = pass.meshCols;
  hardware.arraysPerCore = pass.arraysPerCore;
  hardware.arrayRows = pass.arrayRows;
  hardware.arrayCols = pass.arrayCols;
  hardware.localMemoryBytesPerCore = pass.localMemoryBytesPerCore;
  hardware.clockFrequencyHz = pass.clockFrequencyHz;
  hardware.analogMVMLatencyNs = pass.analogMVMLatencyNs;
  hardware.analogIOBitsPerCycle = pass.analogIOBitsPerCycle;
  hardware.analogIOPolicy = *analogIO;
  hardware.analogArrayExecution = *arrayExecution;
  hardware.digitalIssueWidth = pass.digitalIssueWidth;
  hardware.digitalVectorBitsPerCycle = pass.digitalVectorBitsPerCycle;
  hardware.networkWordBits = pass.networkWordBits;
  hardware.networkHopCycles = pass.networkHopCycles;
  hardware.networkContention = *contention;
  if (failed(hardware.verify(module)))
    return failure();
  return hardware;
}

LogicalResult verifyOptionalIdentity(Operation *operation, StringRef attrName,
                                     std::optional<int64_t> expected,
                                     StringRef identityName) {
  auto actual = operation->getAttrOfType<IntegerAttr>(attrName);
  if (expected) {
    if (!actual || actual.getInt() != *expected) {
      operation->emitError(identityName)
          << " identity is missing or stale; rebuild the RA tree before "
             "planning";
      return failure();
    }
    return success();
  }
  if (actual) {
    operation->emitError("unbound operation carries stale ")
        << identityName << " identity; rebuild the RA tree before planning";
    return failure();
  }
  return success();
}

FailureOr<SmallVector<std::string>> parseStrategyPipeline(StringRef value,
                                                          Operation *anchor) {
  SmallVector<StringRef> pieces;
  value.split(pieces, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/true);

  SmallVector<std::string> names{"setup-first"};
  llvm::StringSet<> seen;
  seen.insert("setup-first");
  for (StringRef piece : pieces) {
    StringRef name = piece.trim();
    if (name.empty()) {
      anchor->emitError(
          "mapping strategy pipeline contains an empty strategy name");
      return failure();
    }
    // Setup-first is a compiler invariant. Accept it in legacy/user pipelines,
    // but normalize it to the one mandatory first stage.
    if (name == "setup-first")
      continue;
    if (!seen.insert(name).second) {
      anchor->emitError("mapping strategy pipeline contains duplicate '")
          << name << "'";
      return failure();
    }
    names.push_back(name.str());
  }
  return names;
}

LogicalResult verifyGlobalSetupFrontier(
    const mapping::ResourceAllocationTree &tree,
    const mapping::ComputeGraph &graph, Operation *anchor) {
  llvm::DenseSet<int64_t> expectedSetups;
  llvm::DenseSet<int64_t> expectedCompute;
  for (const mapping::ComputeOperation &operation : graph.operations) {
    if (operation.kind == mapping::ComputeOperationKind::MatrixSetup)
      expectedSetups.insert(operation.id);
    else
      expectedCompute.insert(operation.id);
  }
  // A setup frontier is meaningful only when both phases exist. Setup-first is
  // still run for setup-free functions, where it is an intentional no-op.
  if (expectedSetups.empty() || expectedCompute.empty())
    return success();

  llvm::DenseMap<int64_t, const mapping::StructuralRATreeNode *> nodes;
  for (const mapping::StructuralRATreeNode &node : tree.nodes)
    nodes[node.id] = &node;
  const mapping::StructuralRATreeNode *root = nodes.lookup(tree.rootId);
  if (!root || root->kind != RATreeNodeKind::TemporalCut ||
      root->childIds.size() != 2) {
    anchor->emitError("mapping strategy destroyed the mandatory global "
                      "setup-first temporal frontier");
    return failure();
  }

  auto collectOperations = [&](int64_t rootId,
                               llvm::DenseSet<int64_t> &operations) {
    SmallVector<int64_t> pending{rootId};
    while (!pending.empty()) {
      int64_t nodeId = pending.pop_back_val();
      const mapping::StructuralRATreeNode *node = nodes.lookup(nodeId);
      if (!node)
        continue;
      if (node->kind == RATreeNodeKind::Leaf) {
        operations.insert(node->operationId);
        continue;
      }
      pending.append(node->childIds.begin(), node->childIds.end());
    }
  };

  llvm::DenseSet<int64_t> setupPhase;
  llvm::DenseSet<int64_t> computePhase;
  collectOperations(root->childIds.front(), setupPhase);
  collectOperations(root->childIds.back(), computePhase);
  if (setupPhase != expectedSetups || computePhase != expectedCompute) {
    anchor->emitError("mapping strategy moved work across the mandatory "
                      "S(all matrix setups) -> compute phase boundary");
    return failure();
  }
  return success();
}

} // namespace

namespace mlir {
namespace sculptor {

void PlanMappingPass::runOnOperation() {
  ModuleOp module = getOperation();
  mapping::registerMappingPlanners();

  FailureOr<mapping::MappingHardwareModel> hardware =
      buildHardwareModel(*this, module);
  FailureOr<mapping::MappingObjective> parsedObjective =
      mapping::parseMappingObjective(objective, module);
  FailureOr<mapping::MVMBodyPolicy> parsedMVMBodyPolicy =
      mapping::parseMVMBodyPolicy(mvmBodyPolicy, module);
  FailureOr<mapping::SetupBindingPolicy> parsedSetupBindingPolicy =
      mapping::parseSetupBindingPolicy(setupBindingPolicy, module);
  FailureOr<SmallVector<std::string>> strategyNames =
      parseStrategyPipeline(strategies, module);
  if (failed(hardware) || failed(parsedObjective) ||
      failed(parsedMVMBodyPolicy) || failed(parsedSetupBindingPolicy) ||
      failed(strategyNames)) {
    signalPassFailure();
    return;
  }
  FailureOr<mapping::MappingCostProfile> resolvedCostProfile =
      mapping::loadMappingCostProfile(costProfile, *hardware, module);
  if (failed(resolvedCostProfile)) {
    signalPassFailure();
    return;
  }

  SmallVector<std::unique_ptr<mapping::MappingPlanner>> planners;
  planners.reserve(strategyNames->size());
  for (const std::string &name : *strategyNames) {
    FailureOr<std::unique_ptr<mapping::MappingPlanner>> planner =
        mapping::getMappingPlannerRegistry().createPlanner(name, module);
    if (failed(planner)) {
      signalPassFailure();
      return;
    }
    planners.push_back(std::move(*planner));
  }
  FailureOr<mapping::LogicalTileShape> logicalTileShape =
      mapping::buildLogicalTileShape(hardware->arraysPerCore, module);
  if (failed(logicalTileShape)) {
    signalPassFailure();
    return;
  }

  StringRef requestedDigitalPolicy = digitalSchedulingPolicy;
  if (requestedDigitalPolicy.empty())
    requestedDigitalPolicy =
        balanceDigitalWork ? StringRef("balanced") : StringRef("affinity");
  else if (balanceDigitalWork && requestedDigitalPolicy != "balanced") {
    module.emitError("balance-digital-work conflicts with explicit digital "
                     "scheduling policy '")
        << requestedDigitalPolicy << "'";
    signalPassFailure();
    return;
  }
  FailureOr<mapping::DigitalSchedulingPolicy> parsedDigitalPolicy =
      mapping::parseDigitalSchedulingPolicy(requestedDigitalPolicy, module);
  if (failed(parsedDigitalPolicy)) {
    signalPassFailure();
    return;
  }
  FailureOr<int64_t> logicalCoreCount = hardware->getCoreCount(module);
  if (failed(logicalCoreCount)) {
    signalPassFailure();
    return;
  }
  if (*parsedDigitalPolicy ==
      mapping::DigitalSchedulingPolicy::SlidingWindow) {
    if (digitalWindowSize <= 0 || digitalWindowSize > *logicalCoreCount) {
      module.emitError("sliding-window scheduling requires "
                       "digital-window-size in [1, ")
          << *logicalCoreCount << "]";
      signalPassFailure();
      return;
    }
  } else if (digitalWindowSize != 0) {
    module.emitError("digital-window-size requires "
                     "digital-scheduling-policy=sliding-window");
    signalPassFailure();
    return;
  }
  if (*parsedMVMBodyPolicy == mapping::MVMBodyPolicy::FirstUseWindow &&
      *parsedDigitalPolicy !=
          mapping::DigitalSchedulingPolicy::SlidingWindow) {
    module.emitError("mvm-body-policy=first-use-window requires "
                     "digital-scheduling-policy=sliding-window so both "
                     "policies share one logical-tile window");
    signalPassFailure();
    return;
  }
  if (*parsedMVMBodyPolicy == mapping::MVMBodyPolicy::FirstUseAdaptive &&
      *parsedDigitalPolicy !=
          mapping::DigitalSchedulingPolicy::SlidingWindow) {
    module.emitError("mvm-body-policy=first-use-adaptive requires "
                     "digital-scheduling-policy=sliding-window so matrix "
                     "locality and digital flow share one window");
    signalPassFailure();
    return;
  }

  mapping::ReferenceMappingEvaluator evaluator;
  bool plannedFunction = false;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    auto treeAttr = func->getAttrOfType<RATreeAttr>(mapping::kRATreeAttrName);
    if (!treeAttr)
      continue;

    FailureOr<mapping::ComputeGraph> graph = mapping::buildComputeGraph(func);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    FailureOr<mapping::ResourceAllocationTree> baseline =
        mapping::deserializeResourceAllocationTree(treeAttr, *graph, func);
    if (failed(baseline)) {
      signalPassFailure();
      return;
    }
    FailureOr<mapping::ResourceAllocationTree> plannerBaseline =
        mapping::stripLayerRegionNodes(*baseline, func);
    if (failed(plannerBaseline)) {
      signalPassFailure();
      return;
    }
    mapping::ResourceAllocationTree currentTree =
        std::move(*plannerBaseline);
    std::optional<mapping::MappingPlan> lastStagePlan;
    for (auto [plannerIndex, planner] : llvm::enumerate(planners)) {
      bool isFinalPlanner = plannerIndex + 1 == planners.size();
      mapping::MappingProblem problem{*graph,
                                      currentTree,
                                      *hardware,
                                      *resolvedCostProfile,
                                      *logicalTileShape,
                                      *parsedObjective,
                                      *parsedMVMBodyPolicy,
                                      *parsedSetupBindingPolicy,
                                      *parsedDigitalPolicy,
                                      digitalWindowSize,
                                      isFinalPlanner,
                                      func};
      FailureOr<mapping::MappingPlan> stagePlan =
          planner->refine(problem, evaluator);
      if (failed(stagePlan)) {
        signalPassFailure();
        return;
      }
      if (!stagePlan->evaluation.feasible) {
        func.emitError("mapping strategy '")
            << planner->getName() << "' did not produce a feasible plan: "
            << stagePlan->evaluation.infeasibilityReason;
        signalPassFailure();
        return;
      }
      if (verifyPlan && failed(mapping::verifyResourceAllocationTree(
                            stagePlan->selectedTree, *graph, func))) {
        signalPassFailure();
        return;
      }
      if (failed(verifyGlobalSetupFrontier(stagePlan->selectedTree, *graph,
                                           func))) {
        signalPassFailure();
        return;
      }
      currentTree = stagePlan->selectedTree;
      lastStagePlan = std::move(*stagePlan);
    }

    assert(lastStagePlan &&
           "a non-empty strategy pipeline must produce a plan");
    FailureOr<mapping::ResourceAllocationTree> layeredTree =
        mapping::materializeLayerRegionNodes(currentTree, *graph, func);
    if (failed(layeredTree)) {
      signalPassFailure();
      return;
    }
    currentTree = std::move(*layeredTree);
    mapping::MappingProblem finalProblem{*graph,
                                         currentTree,
                                         *hardware,
                                         *resolvedCostProfile,
                                         *logicalTileShape,
                                         *parsedObjective,
                                         *parsedMVMBodyPolicy,
                                         *parsedSetupBindingPolicy,
                                         *parsedDigitalPolicy,
                                         digitalWindowSize,
                                         /*requireRealization=*/true,
                                         func};
    FailureOr<mapping::MappingEvaluation> finalEvaluation =
        evaluator.evaluate(finalProblem, currentTree);
    if (failed(finalEvaluation) || !finalEvaluation->feasible) {
      if (succeeded(finalEvaluation))
        func.emitError("layer-annotated mapping plan is infeasible: ")
            << finalEvaluation->infeasibilityReason;
      signalPassFailure();
      return;
    }

    mapping::MappingPlan plan = std::move(*lastStagePlan);
    plan.plannerName = llvm::join(*strategyNames, ",");
    plan.mvmBodyPolicy = *parsedMVMBodyPolicy;
    plan.setupBindingPolicy = *parsedSetupBindingPolicy;
    plan.costProfileName = resolvedCostProfile->name;
    plan.costProfileHash = resolvedCostProfile->contentHash;
    plan.evaluation = std::move(*finalEvaluation);
    plan.selectedTree = std::move(currentTree);
    if (strategyNames->size() > 1) {
      plan.candidates.clear();
      plan.candidates.push_back(
          {plan.plannerName, plan.evaluation, /*selected=*/true});
    } else {
      for (mapping::MappingCandidateSummary &candidate : plan.candidates) {
        if (candidate.selected)
          candidate.evaluation = plan.evaluation;
      }
    }
    if (verifyPlan && failed(mapping::verifyResourceAllocationTree(
                          plan.selectedTree, *graph, func))) {
      signalPassFailure();
      return;
    }
    if (!plan.evaluation.realization) {
      func.emitError("selected mapping plan has no logical-tile realization");
      signalPassFailure();
      return;
    }
    FailureOr<mapping::LogicalTileGraph> logicalTileGraph =
        mapping::buildLogicalTileGraph(*graph, plan.selectedTree,
                                       *plan.evaluation.realization, *hardware,
                                       func);
    if (failed(logicalTileGraph)) {
      signalPassFailure();
      return;
    }

    for (const mapping::ComputeOperation &operation : graph->operations) {
      for (Operation *member : operation.members) {
        auto operationId = member->getAttrOfType<IntegerAttr>(
            mapping::kMappingOperationIdAttrName);
        // Stage discovery includes nested implementation operations (for
        // example, arith ops inside a linalg region). They belong to the
        // stage but are not independent mapping operations.
        if (!operationId)
          continue;
        if (operationId.getInt() != operation.id) {
          member->emitError(
              "mapping operation identity is missing or stale; rebuild the "
              "RA tree before planning");
          signalPassFailure();
          return;
        }
        auto layerRegionId = member->getAttrOfType<IntegerAttr>(
            mapping::kLayerRegionIdAttrName);
        if (layerRegionId && layerRegionId.getInt() != operation.layerRegionId) {
          member->emitError(
              "layer-region identity is stale; rebuild the RA tree before "
              "planning");
          signalPassFailure();
          return;
        }
        member->setAttr(mapping::kLayerRegionIdAttrName,
                        IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                         operation.layerRegionId));
        if (failed(verifyOptionalIdentity(
                member, mapping::kLaneBindingGroupAttrName,
                operation.laneBindingGroup, "lane-binding")) ||
            failed(verifyOptionalIdentity(member, mapping::kMVMWaveIdAttrName,
                                          operation.mvmWaveId, "MVM-wave")) ||
            failed(verifyOptionalIdentity(
                member, mapping::kMVMWaveMemberAttrName,
                operation.mvmWaveMember, "MVM-wave member")) ||
            failed(verifyOptionalIdentity(member, mapping::kMVMWaveSizeAttrName,
                                          operation.mvmWaveSize,
                                          "MVM-wave size"))) {
          signalPassFailure();
          return;
        }
        member->removeAttr(mapping::kRALeafIdAttrName);
        member->removeAttr(mapping::kMappingWorkUnitIdAttrName);
      }
    }
    Builder builder(&getContext());
    DenseMap<int64_t, SmallVector<const mapping::StructuralRATreeNode *>>
        leavesByOperation;
    for (const mapping::StructuralRATreeNode &node : plan.selectedTree.nodes) {
      if (node.kind != RATreeNodeKind::Leaf)
        continue;
      leavesByOperation[node.operationId].push_back(&node);
    }
    for (const auto &[operationId, leaves] : leavesByOperation) {
      if (leaves.size() != 1 || leaves.front()->workUnitId >= 0)
        continue;
      for (Operation *member : graph->operations[operationId].members) {
        member->setAttr(mapping::kRALeafIdAttrName,
                        builder.getI64IntegerAttr(leaves.front()->id));
      }
    }

    std::string graphFingerprint = mapping::computeGraphFingerprint(*graph);
    func->setAttr(
        mapping::kRATreeAttrName,
        mapping::serializeResourceAllocationTree(
            &getContext(), plan.selectedTree, *graph, graphFingerprint));
    func->setAttr(mapping::kMappingPlanAttrName,
                  mapping::serializeMappingPlan(&getContext(), plan));
    func->setAttr(mapping::kMappingCostProfileAttrName,
                  mapping::serializeMappingCostProfile(&getContext(),
                                                       *resolvedCostProfile));
    func->setAttr(mapping::kMappingCostProfileNameAttrName,
                  StringAttr::get(&getContext(), resolvedCostProfile->name));
    func->setAttr(
        mapping::kMappingCostProfileHashAttrName,
        StringAttr::get(&getContext(), resolvedCostProfile->contentHash));
    func->setAttr(
        mapping::kLogicalTileGraphAttrName,
        mapping::serializeLogicalTileGraph(&getContext(), *logicalTileGraph));
    func->removeAttr(mapping::kLogicalTilePlacementAttrName);
    func->removeAttr(mapping::kLogicalTileAnnealingTraceAttrName);
    func->setAttr(
        "sculptor.mapping.mvm_body_policy",
        StringAttr::get(&getContext(),
                        mapping::stringifyMVMBodyPolicy(*parsedMVMBodyPolicy)));
    func->setAttr(
        "sculptor.mapping.setup_binding_policy",
        StringAttr::get(&getContext(), mapping::stringifySetupBindingPolicy(
                                           *parsedSetupBindingPolicy)));
    func->setAttr(
        "sculptor.mapping.digital_scheduling_policy",
        StringAttr::get(&getContext(), mapping::stringifyDigitalSchedulingPolicy(
                                           *parsedDigitalPolicy)));
    if (*parsedDigitalPolicy ==
        mapping::DigitalSchedulingPolicy::SlidingWindow) {
      func->setAttr("sculptor.mapping.digital_window_size",
                    builder.getI64IntegerAttr(digitalWindowSize));
    } else {
      func->removeAttr("sculptor.mapping.digital_window_size");
    }
    if (*parsedDigitalPolicy ==
        mapping::DigitalSchedulingPolicy::Balanced) {
      func->setAttr("sculptor.mapping.digital_work_balancing",
                    BoolAttr::get(&getContext(), true));
    } else {
      func->removeAttr("sculptor.mapping.digital_work_balancing");
    }
    plannedFunction = true;
  }
  if (!plannedFunction) {
    module.emitError("expected at least one function with a structural "
                     "Resource Allocation Tree");
    signalPassFailure();
  }
}

void registerPlanMappingPass() {
  mapping::registerMappingPlanners();
  PassRegistration<PlanMappingPass>();
}

} // namespace sculptor
} // namespace mlir
