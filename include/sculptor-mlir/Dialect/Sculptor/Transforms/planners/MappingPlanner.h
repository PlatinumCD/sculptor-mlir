#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_MAPPINGPLANNER_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_MAPPINGPLANNER_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingEvaluator.h"

namespace mlir {
namespace sculptor {
namespace mapping {

class MappingPlanner {
public:
  virtual ~MappingPlanner() = default;

  virtual StringRef getName() const = 0;

  /// Refine the RA tree provided as problem.currentTree.
  virtual FailureOr<MappingPlan>
  refine(const MappingProblem &problem,
         const MappingEvaluator &evaluator) const = 0;
};

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_MAPPINGPLANNER_H
