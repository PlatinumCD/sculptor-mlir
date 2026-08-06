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
    return "Materialize selected RA-tree work units through MLIR interfaces";
  }

  void runOnOperation() override;
};

void registerApplyMappingPlanPass();

} // namespace sculptor
} // namespace mlir

#endif
