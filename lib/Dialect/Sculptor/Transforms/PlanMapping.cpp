#include "sculptor-mlir/Dialect/Sculptor/Transforms/PlanMapping.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingEvaluator.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingProblem.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/planners/MappingPlannerRegistry.h"

#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
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

  SmallVector<std::string> names;
  llvm::StringSet<> seen;
  for (StringRef piece : pieces) {
    StringRef name = piece.trim();
    if (name.empty()) {
      anchor->emitError(
          "mapping strategy pipeline contains an empty strategy name");
      return failure();
    }
    if (!seen.insert(name).second) {
      anchor->emitError("mapping strategy pipeline contains duplicate '")
          << name << "'";
      return failure();
    }
    names.push_back(name.str());
  }
  if (names.empty()) {
    anchor->emitError("mapping strategy pipeline must not be empty");
    return failure();
  }
  return names;
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
  FailureOr<SmallVector<std::string>> strategyNames =
      parseStrategyPipeline(strategies, module);
  if (failed(hardware) || failed(parsedObjective) || failed(strategyNames)) {
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
    mapping::ResourceAllocationTree currentTree = std::move(*baseline);
    std::optional<mapping::MappingPlan> lastStagePlan;
    for (auto [plannerIndex, planner] : llvm::enumerate(planners)) {
      bool isFinalPlanner = plannerIndex + 1 == planners.size();
      mapping::MappingProblem problem{*graph,
                                      currentTree,
                                      *hardware,
                                      *logicalTileShape,
                                      *parsedObjective,
                                      mvmWaveColocation,
                                      balanceDigitalWork,
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
      currentTree = stagePlan->selectedTree;
      lastStagePlan = std::move(*stagePlan);
    }

    assert(lastStagePlan &&
           "a non-empty strategy pipeline must produce a plan");
    mapping::MappingPlan plan = std::move(*lastStagePlan);
    plan.plannerName = llvm::join(*strategyNames, ",");
    plan.selectedTree = std::move(currentTree);
    if (strategyNames->size() > 1) {
      plan.candidates.clear();
      plan.candidates.push_back(
          {plan.plannerName, plan.evaluation, /*selected=*/true});
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
    func->setAttr(
        mapping::kLogicalTileGraphAttrName,
        mapping::serializeLogicalTileGraph(&getContext(), *logicalTileGraph));
    func->removeAttr(mapping::kLogicalTilePlacementAttrName);
    func->removeAttr(mapping::kLogicalTileAnnealingTraceAttrName);
    if (mvmWaveColocation) {
      func->setAttr("sculptor.mapping.mvm_wave_colocation",
                    BoolAttr::get(&getContext(), true));
    } else {
      func->removeAttr("sculptor.mapping.mvm_wave_colocation");
    }
    if (balanceDigitalWork) {
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
