#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingCostModel.h"

#include <cmath>

namespace mlir {
namespace sculptor {
namespace mapping {

namespace {

LogicalResult verifyFiniteNonnegative(double value, Operation *anchor,
                                      StringRef description) {
  if (!std::isfinite(value) || value < 0.0)
    return anchor->emitError(description) << " must be finite and nonnegative";
  return success();
}

} // namespace

FailureOr<TaskCostEstimate>
estimateDigitalTaskCost(const MappingCostProfile &profile,
                        const TaskCostFeatures &features, Operation *anchor) {
  if (features.workItems < 0 || features.inputBytes < 0 ||
      features.outputBytes < 0) {
    anchor->emitError("digital task cost requires nonnegative static features");
    return failure();
  }

  const TaskCostRule *rule = &profile.digitalFallback;
  auto found = profile.digitalTaskKinds.find(features.semanticTaskKind);
  if (found != profile.digitalTaskKinds.end())
    rule = &found->second;

  TaskCostEstimate estimate;
  estimate.computeNs = rule->fixedNs + rule->nsPerWorkItem * features.workItems;
  estimate.memoryNs = rule->nsPerInputByte * features.inputBytes +
                      rule->nsPerOutputByte * features.outputBytes;
  estimate.runtimeNs = profile.runtime.taskDispatchNs;
  estimate.totalNs =
      estimate.computeNs + estimate.memoryNs + estimate.runtimeNs;
  if (failed(verifyFiniteNonnegative(estimate.computeNs, anchor,
                                     "digital compute cost")) ||
      failed(verifyFiniteNonnegative(estimate.memoryNs, anchor,
                                     "digital memory cost")) ||
      failed(verifyFiniteNonnegative(estimate.runtimeNs, anchor,
                                     "digital runtime cost")) ||
      failed(verifyFiniteNonnegative(estimate.totalNs, anchor,
                                     "digital total cost")))
    return failure();
  return estimate;
}

FailureOr<TaskCostEstimate>
estimateAnalogTaskCost(const MappingCostProfile &profile, int64_t loadBytes,
                       int64_t storeBytes, int64_t executionCount,
                       Operation *anchor) {
  if (loadBytes < 0 || storeBytes < 0 || executionCount <= 0) {
    anchor->emitError("analog task cost requires nonnegative bytes and "
                      "positive execution count");
    return failure();
  }

  TaskCostEstimate estimate;
  double load =
      profile.analog.loadFixedNs + profile.analog.loadNsPerByte * loadBytes;
  double execute = profile.analog.executeNs;
  double store =
      profile.analog.storeFixedNs + profile.analog.storeNsPerByte * storeBytes;
  estimate.computeNs = execute * executionCount;
  estimate.memoryNs = (load + store) * executionCount;
  estimate.runtimeNs = profile.runtime.taskDispatchNs;
  estimate.totalNs =
      estimate.computeNs + estimate.memoryNs + estimate.runtimeNs;
  if (failed(verifyFiniteNonnegative(estimate.computeNs, anchor,
                                     "analog compute cost")) ||
      failed(verifyFiniteNonnegative(estimate.memoryNs, anchor,
                                     "analog I/O cost")) ||
      failed(verifyFiniteNonnegative(estimate.runtimeNs, anchor,
                                     "analog runtime cost")) ||
      failed(verifyFiniteNonnegative(estimate.totalNs, anchor,
                                     "analog total cost")))
    return failure();
  return estimate;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
