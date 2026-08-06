#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_MVMWAVE_MVMWAVEPLANNER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_MVMWAVE_MVMWAVEPLANNER_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/planners/MappingPlanner.h"

namespace mlir {
namespace sculptor {
namespace mapping {

class MVMWavePlanner final : public MappingPlanner {
public:
  StringRef getName() const override { return "mvm-wave"; }

  FailureOr<MappingPlan>
  refine(const MappingProblem &problem,
         const MappingEvaluator &evaluator) const override;
};

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_MVMWAVE_MVMWAVEPLANNER_H
