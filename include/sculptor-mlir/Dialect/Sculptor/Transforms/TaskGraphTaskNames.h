#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKNAMES_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKNAMES_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace sculptor {
namespace task_graph_names {

inline constexpr llvm::StringLiteral kAnalogDomain("analog");
inline constexpr llvm::StringLiteral kDigitalDomain("digital");

inline constexpr llvm::StringLiteral kAnalogTaskKindPrefix("sculptor.");
inline constexpr llvm::StringLiteral
    kMatrixSetupTaskKind("sculptor.matrix_setup");
inline constexpr llvm::StringLiteral kMVMTaskKind("sculptor.mvm");

inline constexpr llvm::StringLiteral kForwardSourceLayer("forward");

} // namespace task_graph_names
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TASKGRAPHTASKNAMES_H
