#include "sculptor-mlir/Dialect/Sculptor/Transforms/planners/MappingPlannerRegistry.h"

#include "ConsumerBoundFill/ConsumerBoundFillPlanner.h"
#include "FanOutCut/FanOutCutPlanner.h"
#include "MVMWave/MVMWavePlanner.h"
#include "RecursiveForkJoin/RecursiveForkJoinPlanner.h"
#include "SetupFirst/SetupFirstPlanner.h"

#include "llvm/Support/ErrorHandling.h"

#include <mutex>

namespace mlir {
namespace sculptor {
namespace mapping {

void registerMappingPlanners() {
  static std::once_flag once;
  std::call_once(once, [] {
    MappingPlannerRegistry &registry = getMappingPlannerRegistry();
    std::string error;
    if (failed(registry.registerPlanner(
            "consumer-bound-fill",
            [] { return std::make_unique<ConsumerBoundFillPlanner>(); },
            &error)))
      llvm::report_fatal_error(llvm::StringRef(error));
    if (failed(registry.registerPlanner(
            "fan-out-cut", [] { return std::make_unique<FanOutCutPlanner>(); },
            &error)))
      llvm::report_fatal_error(llvm::StringRef(error));
    if (failed(registry.registerPlanner(
            "mvm-wave", [] { return std::make_unique<MVMWavePlanner>(); },
            &error)))
      llvm::report_fatal_error(llvm::StringRef(error));
    if (failed(registry.registerPlanner(
            "recursive-fork-join",
            [] { return std::make_unique<RecursiveForkJoinPlanner>(); },
            &error)))
      llvm::report_fatal_error(llvm::StringRef(error));
    if (failed(registry.registerPlanner(
            "setup-first", [] { return std::make_unique<SetupFirstPlanner>(); },
            &error)))
      llvm::report_fatal_error(llvm::StringRef(error));
  });
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
