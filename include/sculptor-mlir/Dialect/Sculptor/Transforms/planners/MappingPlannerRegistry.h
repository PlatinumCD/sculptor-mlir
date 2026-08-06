#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_MAPPINGPLANNERREGISTRY_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_MAPPINGPLANNERREGISTRY_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/planners/MappingPlanner.h"

#include "llvm/ADT/StringMap.h"

#include <functional>
#include <memory>
#include <string>

namespace mlir {
namespace sculptor {
namespace mapping {

using MappingPlannerFactory = std::function<std::unique_ptr<MappingPlanner>()>;

class MappingPlannerRegistry {
public:
  LogicalResult registerPlanner(StringRef name, MappingPlannerFactory factory,
                                std::string *errorMessage = nullptr);

  FailureOr<std::unique_ptr<MappingPlanner>>
  createPlanner(StringRef name, Operation *anchor) const;

  SmallVector<std::string> getRegisteredNames() const;

private:
  llvm::StringMap<MappingPlannerFactory> factories;
};

MappingPlannerRegistry &getMappingPlannerRegistry();
void registerMappingPlanners();

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_PLANNERS_MAPPINGPLANNERREGISTRY_H
