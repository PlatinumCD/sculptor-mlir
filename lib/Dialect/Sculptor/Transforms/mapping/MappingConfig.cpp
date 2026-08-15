#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingConfig.h"

#include "llvm/Support/CheckedArithmetic.h"

#include <optional>

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<int64_t> MappingHardwareModel::getCoreCount(Operation *anchor) const {
  std::optional<int64_t> count = llvm::checkedMul(meshRows, meshCols);
  if (!count) {
    anchor->emitError("mapping mesh dimensions overflow core count");
    return failure();
  }
  return *count;
}

LogicalResult MappingHardwareModel::verify(Operation *anchor) const {
  if (meshRows <= 0 || meshCols <= 0)
    return anchor->emitError("expected positive mesh dimensions");
  if (failed(getCoreCount(anchor)))
    return failure();
  if (arraysPerCore <= 0)
    return anchor->emitError("expected at least one analog array per core");
  if (arrayRows <= 0 || arrayCols <= 0)
    return anchor->emitError("expected positive physical array dimensions");
  if (localMemoryBytesPerCore <= 0)
    return anchor->emitError("expected positive local memory capacity");
  if (clockFrequencyHz <= 0)
    return anchor->emitError("expected positive clock frequency");
  if (analogMVMLatencyNs < 0)
    return anchor->emitError("expected non-negative analog MVM latency");
  if (analogIOBitsPerCycle <= 0)
    return anchor->emitError("expected positive analog I/O bandwidth");
  if (digitalIssueWidth <= 0)
    return anchor->emitError("expected positive digital issue width");
  if (digitalVectorBitsPerCycle <= 0)
    return anchor->emitError("expected positive digital vector throughput");
  if (networkWordBits <= 0)
    return anchor->emitError("expected positive network word width");
  if (networkHopCycles < 0)
    return anchor->emitError("expected non-negative network hop latency");
  return success();
}

FailureOr<MappingObjective> parseMappingObjective(StringRef value,
                                                  Operation *anchor) {
  if (value == "latency")
    return MappingObjective{MappingObjectiveKind::Latency};
  anchor->emitError("unknown mapping objective '")
      << value << "'; expected 'latency'";
  return failure();
}

FailureOr<AnalogIOPolicy> parseAnalogIOPolicy(StringRef value,
                                              Operation *anchor) {
  if (value == "shared")
    return AnalogIOPolicy::Shared;
  anchor->emitError("unknown analog I/O policy '")
      << value << "'; expected 'shared'";
  return failure();
}

FailureOr<AnalogArrayExecutionPolicy>
parseAnalogArrayExecutionPolicy(StringRef value, Operation *anchor) {
  if (value == "concurrent")
    return AnalogArrayExecutionPolicy::Concurrent;
  anchor->emitError("unknown analog array execution policy '")
      << value << "'; expected 'concurrent'";
  return failure();
}

FailureOr<NetworkContentionModel>
parseNetworkContentionModel(StringRef value, Operation *anchor) {
  if (value == "none")
    return NetworkContentionModel::None;
  if (value == "link-serialized")
    return NetworkContentionModel::LinkSerialized;
  anchor->emitError("unknown network contention model '")
      << value << "'; expected 'none' or 'link-serialized'";
  return failure();
}

FailureOr<MVMBodyPolicy> parseMVMBodyPolicy(StringRef value,
                                            Operation *anchor) {
  if (value == "packed")
    return MVMBodyPolicy::Packed;
  if (value == "spread")
    return MVMBodyPolicy::Spread;
  if (value == "first-use-window")
    return MVMBodyPolicy::FirstUseWindow;
  if (value == "first-use-adaptive")
    return MVMBodyPolicy::FirstUseAdaptive;
  anchor->emitError("unknown MVM body policy '")
      << value << "'; expected 'packed', 'spread', 'first-use-window', or "
                  "'first-use-adaptive'";
  return failure();
}

FailureOr<SetupBindingPolicy> parseSetupBindingPolicy(StringRef value,
                                                      Operation *anchor) {
  if (value == "global")
    return SetupBindingPolicy::Global;
  if (value == "consumer-anchored")
    return SetupBindingPolicy::ConsumerAnchored;
  anchor->emitError("unknown setup binding policy '")
      << value << "'; expected 'global' or 'consumer-anchored'";
  return failure();
}

FailureOr<DigitalSchedulingPolicy>
parseDigitalSchedulingPolicy(StringRef value, Operation *anchor) {
  if (value == "affinity")
    return DigitalSchedulingPolicy::Affinity;
  if (value == "balanced")
    return DigitalSchedulingPolicy::Balanced;
  if (value == "earliest-finish")
    return DigitalSchedulingPolicy::EarliestFinish;
  if (value == "progressive")
    return DigitalSchedulingPolicy::Progressive;
  if (value == "sliding-window")
    return DigitalSchedulingPolicy::SlidingWindow;
  anchor->emitError("unknown digital scheduling policy '")
      << value << "'; expected 'affinity', 'balanced', 'earliest-finish', or "
                     "'progressive', or 'sliding-window'";
  return failure();
}

StringRef stringifyMappingObjective(MappingObjectiveKind objective) {
  switch (objective) {
  case MappingObjectiveKind::Latency:
    return "latency";
  }
  llvm_unreachable("unknown mapping objective");
}

StringRef stringifyNetworkContentionModel(NetworkContentionModel model) {
  switch (model) {
  case NetworkContentionModel::None:
    return "none";
  case NetworkContentionModel::LinkSerialized:
    return "link-serialized";
  }
  llvm_unreachable("unknown network contention model");
}

StringRef stringifyMVMBodyPolicy(MVMBodyPolicy policy) {
  switch (policy) {
  case MVMBodyPolicy::Packed:
    return "packed";
  case MVMBodyPolicy::Spread:
    return "spread";
  case MVMBodyPolicy::FirstUseWindow:
    return "first-use-window";
  case MVMBodyPolicy::FirstUseAdaptive:
    return "first-use-adaptive";
  }
  llvm_unreachable("unknown MVM body policy");
}

StringRef stringifySetupBindingPolicy(SetupBindingPolicy policy) {
  switch (policy) {
  case SetupBindingPolicy::Global:
    return "global";
  case SetupBindingPolicy::ConsumerAnchored:
    return "consumer-anchored";
  }
  llvm_unreachable("unknown setup binding policy");
}

StringRef stringifyDigitalSchedulingPolicy(DigitalSchedulingPolicy policy) {
  switch (policy) {
  case DigitalSchedulingPolicy::Affinity:
    return "affinity";
  case DigitalSchedulingPolicy::Balanced:
    return "balanced";
  case DigitalSchedulingPolicy::EarliestFinish:
    return "earliest-finish";
  case DigitalSchedulingPolicy::Progressive:
    return "progressive";
  case DigitalSchedulingPolicy::SlidingWindow:
    return "sliding-window";
  }
  llvm_unreachable("unknown digital scheduling policy");
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
