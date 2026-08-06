#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_CONSUMERBOUNDFILL_CONSUMERBOUNDFILLPLANNER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_CONSUMERBOUNDFILL_CONSUMERBOUNDFILLPLANNER_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/planners/MappingPlanner.h"

namespace mlir {
namespace sculptor {
namespace mapping {

class ConsumerBoundFillPlanner final : public MappingPlanner {
public:
  StringRef getName() const override { return "consumer-bound-fill"; }

  FailureOr<MappingPlan>
  refine(const MappingProblem &problem,
         const MappingEvaluator &evaluator) const override;
};

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_CONSUMERBOUNDFILL_CONSUMERBOUNDFILLPLANNER_H
