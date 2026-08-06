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

} // namespace mapping
} // namespace sculptor
} // namespace mlir
