#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTORDEPLOYMENTATTRS_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTORDEPLOYMENTATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace deployment_attrs {

inline constexpr llvm::StringLiteral
    kGlobalTaskIdAttrName("sculptor.deployment.global_task_id");
inline constexpr llvm::StringLiteral
    kGlobalResourceIdAttrName("sculptor.deployment.global_resource_id");
inline constexpr llvm::StringLiteral
    kRouteIdAttrName("sculptor.deployment.route_id");
inline constexpr llvm::StringLiteral
    kRoutesAttrName("sculptor.deployment.routes");
inline constexpr llvm::StringLiteral
    kIncomingRoutesAttrName("sculptor.deployment.incoming_routes");
inline constexpr llvm::StringLiteral
    kOutgoingRoutesAttrName("sculptor.deployment.outgoing_routes");
inline constexpr llvm::StringLiteral
    kModelInputsAttrName("sculptor.deployment.model_inputs");
inline constexpr llvm::StringLiteral
    kModelOutputsAttrName("sculptor.deployment.model_outputs");

} // namespace deployment_attrs
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_IR_SCULPTORDEPLOYMENTATTRS_H
