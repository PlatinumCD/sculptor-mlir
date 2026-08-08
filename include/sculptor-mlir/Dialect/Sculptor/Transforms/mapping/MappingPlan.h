#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGPLAN_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGPLAN_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingRealization.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace sculptor {
namespace mapping {

inline constexpr StringLiteral kMappingPlanAttrName = "sculptor.mapping.plan";

struct MappingNodeEvaluation {
  int64_t nodeId = -1;
  bool feasible = true;
  double estimatedLatencyNs = 0.0;
  int64_t crossingBytes = 0;
  double estimatedCommunicationNs = 0.0;
  int64_t requiredResourceUnits = 0;
  int64_t pipelineStages = 0;
  std::string infeasibilityReason;
};

struct MappingEvaluation {
  bool feasible = true;
  double estimatedLatencyNs = 0.0;
  int64_t crossingBytes = 0;
  double estimatedCommunicationNs = 0.0;
  int64_t requiredResourceUnits = 0;
  int64_t pipelineStages = 0;
  std::string infeasibilityReason;
  SmallVector<MappingNodeEvaluation, 0> nodes;
  std::optional<MappingRealization> realization;
};

struct MappingCandidateSummary {
  std::string name;
  MappingEvaluation evaluation;
  bool selected = false;
};

struct MappingPlan {
  std::string plannerName;
  MappingObjective objective;
  MVMBodyPolicy mvmBodyPolicy = MVMBodyPolicy::Spread;
  SetupBindingPolicy setupBindingPolicy = SetupBindingPolicy::Global;
  ResourceAllocationTree selectedTree;
  MappingEvaluation evaluation;
  SmallVector<MappingCandidateSummary, 0> candidates;
};

bool isBetterMappingEvaluation(const MappingEvaluation &candidate,
                               const MappingEvaluation &incumbent,
                               MappingObjective objective);

MappingPlanAttr serializeMappingPlan(MLIRContext *context,
                                     const MappingPlan &plan);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGPLAN_H
