#include "sculptor-mlir/Dialect/Sculptor/Transforms/planners/MappingPlannerRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ManagedStatic.h"

namespace mlir {
namespace sculptor {
namespace mapping {

LogicalResult MappingPlannerRegistry::registerPlanner(
    StringRef name, MappingPlannerFactory factory, std::string *errorMessage) {
  auto fail = [&](const Twine &message) {
    if (errorMessage)
      *errorMessage = message.str();
    return failure();
  };
  if (name.empty())
    return fail("mapping planner name must not be empty");
  if (!factory)
    return fail(Twine("mapping planner '") + name + "' has no factory");
  if (!factories.try_emplace(name, std::move(factory)).second)
    return fail(Twine("mapping planner '") + name + "' is already registered");
  return success();
}

FailureOr<std::unique_ptr<MappingPlanner>>
MappingPlannerRegistry::createPlanner(StringRef name, Operation *anchor) const {
  auto planner = factories.find(name);
  if (planner == factories.end()) {
    SmallVector<std::string> names = getRegisteredNames();
    auto diagnostic = anchor->emitError("unknown mapping planner '")
                      << name << "'";
    if (!names.empty()) {
      diagnostic << "; registered planners are ";
      llvm::interleaveComma(names, diagnostic);
    }
    return failure();
  }
  return planner->second();
}

SmallVector<std::string> MappingPlannerRegistry::getRegisteredNames() const {
  SmallVector<std::string> names;
  names.reserve(factories.size());
  for (const auto &entry : factories)
    names.push_back(entry.getKey().str());
  llvm::sort(names);
  return names;
}

MappingPlannerRegistry &getMappingPlannerRegistry() {
  static llvm::ManagedStatic<MappingPlannerRegistry> registry;
  return *registry;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
