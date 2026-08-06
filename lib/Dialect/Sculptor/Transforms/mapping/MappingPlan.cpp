#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingPlan.h"

#include "mlir/IR/Builders.h"

#include <cassert>
#include <tuple>

namespace mlir {
namespace sculptor {
namespace mapping {

bool isBetterMappingEvaluation(const MappingEvaluation &candidate,
                               const MappingEvaluation &incumbent,
                               MappingObjective objective) {
  if (candidate.feasible != incumbent.feasible)
    return candidate.feasible;
  if (!candidate.feasible)
    return false;

  switch (objective.kind) {
  case MappingObjectiveKind::Latency:
    return std::tie(candidate.estimatedLatencyNs, candidate.crossingBytes,
                    candidate.requiredResourceUnits) <
           std::tie(incumbent.estimatedLatencyNs, incumbent.crossingBytes,
                    incumbent.requiredResourceUnits);
  }
  llvm_unreachable("unknown mapping objective");
}

MappingPlanAttr serializeMappingPlan(MLIRContext *context,
                                     const MappingPlan &plan) {
  Builder builder(context);
  SmallVector<Attribute> nodeEvaluations;
  nodeEvaluations.reserve(plan.evaluation.nodes.size());
  for (const MappingNodeEvaluation &node : plan.evaluation.nodes) {
    nodeEvaluations.push_back(RATreeNodeEvaluationAttr::get(
        context, builder.getI64IntegerAttr(node.nodeId),
        builder.getBoolAttr(node.feasible),
        builder.getF64FloatAttr(node.estimatedLatencyNs),
        builder.getI64IntegerAttr(node.crossingBytes),
        builder.getF64FloatAttr(node.estimatedCommunicationNs),
        builder.getI64IntegerAttr(node.requiredResourceUnits),
        builder.getI64IntegerAttr(node.pipelineStages),
        builder.getStringAttr(node.infeasibilityReason)));
  }

  SmallVector<Attribute> candidates;
  candidates.reserve(plan.candidates.size());
  for (const MappingCandidateSummary &candidate : plan.candidates) {
    const MappingEvaluation &evaluation = candidate.evaluation;
    candidates.push_back(MappingCandidateEvaluationAttr::get(
        context, builder.getStringAttr(candidate.name),
        builder.getBoolAttr(candidate.selected),
        builder.getBoolAttr(evaluation.feasible),
        builder.getF64FloatAttr(evaluation.estimatedLatencyNs),
        builder.getI64IntegerAttr(evaluation.crossingBytes),
        builder.getF64FloatAttr(evaluation.estimatedCommunicationNs),
        builder.getI64IntegerAttr(evaluation.requiredResourceUnits),
        builder.getI64IntegerAttr(evaluation.pipelineStages),
        builder.getStringAttr(evaluation.infeasibilityReason)));
  }

  MappingPlanObjective objective;
  switch (plan.objective.kind) {
  case MappingObjectiveKind::Latency:
    objective = MappingPlanObjective::Latency;
    break;
  }

  assert(plan.evaluation.realization && plan.evaluation.realization->feasible &&
         "selected feasible mapping plan requires a realization");
  const MappingRealization &realization = *plan.evaluation.realization;
  SmallVector<Attribute> nodeAllocations;
  nodeAllocations.reserve(realization.nodeAllocations.size());
  for (const MappingNodeResourceAllocation &allocation :
       realization.nodeAllocations) {
    SmallVector<Attribute> digitalTileIds;
    for (int64_t tileId : allocation.digitalTileIds)
      digitalTileIds.push_back(builder.getI64IntegerAttr(tileId));
    SmallVector<Attribute> analogTileIds;
    SmallVector<Attribute> analogLaneIndices;
    for (const MappingAnalogLaneRef &lane : allocation.analogLanes) {
      analogTileIds.push_back(builder.getI64IntegerAttr(lane.tileId));
      analogLaneIndices.push_back(builder.getI64IntegerAttr(lane.laneIndex));
    }
    nodeAllocations.push_back(MappingNodeAllocationAttr::get(
        context, builder.getI64IntegerAttr(allocation.nodeId),
        builder.getArrayAttr(digitalTileIds),
        builder.getArrayAttr(analogTileIds),
        builder.getArrayAttr(analogLaneIndices)));
  }

  SmallVector<Attribute> leafAssignments;
  leafAssignments.reserve(realization.leafAssignments.size());
  for (const MappingLeafAssignment &assignment : realization.leafAssignments) {
    MappingLaneKind laneKind = assignment.laneKind == LogicalLaneKind::Digital
                                   ? MappingLaneKind::Digital
                                   : MappingLaneKind::Analog;
    leafAssignments.push_back(MappingLeafAssignmentAttr::get(
        context, builder.getI64IntegerAttr(assignment.leafId),
        builder.getI64IntegerAttr(assignment.operationId),
        builder.getI64IntegerAttr(assignment.tileId), laneKind,
        builder.getI64IntegerAttr(assignment.laneIndex),
        builder.getF64FloatAttr(assignment.startNs),
        builder.getF64FloatAttr(assignment.finishNs)));
  }
  SmallVector<Attribute> digitalWorkPerTile;
  digitalWorkPerTile.reserve(realization.digitalWorkPerTile.size());
  for (int64_t work : realization.digitalWorkPerTile)
    digitalWorkPerTile.push_back(builder.getI64IntegerAttr(work));
  MappingRealizationAttr realizationAttr = MappingRealizationAttr::get(
      context, builder.getI64IntegerAttr(2),
      builder.getI64IntegerAttr(realization.logicalTileCount),
      builder.getI64IntegerAttr(realization.analogLanesPerTile),
      builder.getArrayAttr(digitalWorkPerTile),
      builder.getArrayAttr(nodeAllocations),
      builder.getArrayAttr(leafAssignments));

  return MappingPlanAttr::get(
      context, builder.getI64IntegerAttr(2),
      builder.getStringAttr(plan.plannerName), objective,
      builder.getStringAttr(computeRATreeFingerprint(plan.selectedTree)),
      builder.getBoolAttr(plan.evaluation.feasible),
      builder.getF64FloatAttr(plan.evaluation.estimatedLatencyNs),
      builder.getI64IntegerAttr(plan.evaluation.crossingBytes),
      builder.getF64FloatAttr(plan.evaluation.estimatedCommunicationNs),
      builder.getI64IntegerAttr(plan.evaluation.requiredResourceUnits),
      builder.getI64IntegerAttr(plan.evaluation.pipelineStages),
      builder.getArrayAttr(candidates), builder.getArrayAttr(nodeEvaluations),
      realizationAttr);
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
