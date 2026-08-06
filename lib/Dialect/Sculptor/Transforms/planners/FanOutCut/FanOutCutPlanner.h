#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_FANOUTCUT_FANOUTCUTPLANNER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_FANOUTCUT_FANOUTCUTPLANNER_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/planners/MappingPlanner.h"

namespace mlir {
namespace sculptor {
namespace mapping {

class FanOutCutPlanner final : public MappingPlanner {
public:
  StringRef getName() const override { return "fan-out-cut"; }

  FailureOr<MappingPlan>
  refine(const MappingProblem &problem,
         const MappingEvaluator &evaluator) const override;
};

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_FANOUTCUT_FANOUTCUTPLANNER_H
