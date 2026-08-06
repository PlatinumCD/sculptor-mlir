#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGEVALUATOR_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGEVALUATOR_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingProblem.h"

namespace mlir {
namespace sculptor {
namespace mapping {

class MappingEvaluator {
public:
  virtual ~MappingEvaluator() = default;

  virtual FailureOr<MappingEvaluation>
  evaluate(const MappingProblem &problem,
           const ResourceAllocationTree &tree) const = 0;
};

class ReferenceMappingEvaluator final : public MappingEvaluator {
public:
  FailureOr<MappingEvaluation>
  evaluate(const MappingProblem &problem,
           const ResourceAllocationTree &tree) const override;
};

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGEVALUATOR_H
