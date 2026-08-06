#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_APPLYMAPPINGPLAN_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_APPLYMAPPINGPLAN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sculptor {

struct ApplyMappingPlanPass
    : public PassWrapper<ApplyMappingPlanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyMappingPlanPass)

  StringRef getArgument() const final { return "sculptor-apply-mapping-plan"; }
  StringRef getDescription() const final {
    return "Terminally materialize selected RA-tree work units and consume "
           "mapping metadata (outside the deployment pipeline)";
  }

  void runOnOperation() override;
};

void registerApplyMappingPlanPass();

} // namespace sculptor
} // namespace mlir

#endif
