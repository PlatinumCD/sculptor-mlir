#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGCONFIG_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGCONFIG_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mlir {
namespace sculptor {
namespace mapping {

enum class MappingObjectiveKind { Latency };
enum class AnalogIOPolicy { Shared };
enum class AnalogArrayExecutionPolicy { Concurrent };
enum class NetworkContentionModel { None, LinkSerialized };

struct MappingObjective {
  MappingObjectiveKind kind = MappingObjectiveKind::Latency;
};

struct MappingHardwareModel {
  int64_t meshRows = 1;
  int64_t meshCols = 1;
  int64_t arraysPerCore = 1;
  int64_t arrayRows = 1024;
  int64_t arrayCols = 512;
  int64_t localMemoryBytesPerCore = 64 * 1024 * 1024;
  int64_t clockFrequencyHz = 1000000000;
  int64_t analogMVMLatencyNs = 100;
  int64_t analogIOBitsPerCycle = 256;
  AnalogIOPolicy analogIOPolicy = AnalogIOPolicy::Shared;
  AnalogArrayExecutionPolicy analogArrayExecution =
      AnalogArrayExecutionPolicy::Concurrent;
  int64_t digitalIssueWidth = 2;
  int64_t digitalVectorBitsPerCycle = 256;
  int64_t networkWordBits = 32;
  int64_t networkHopCycles = 1;
  NetworkContentionModel networkContention =
      NetworkContentionModel::LinkSerialized;

  FailureOr<int64_t> getCoreCount(Operation *anchor) const;
  LogicalResult verify(Operation *anchor) const;
};

FailureOr<MappingObjective> parseMappingObjective(StringRef value,
                                                  Operation *anchor);
FailureOr<AnalogIOPolicy> parseAnalogIOPolicy(StringRef value,
                                              Operation *anchor);
FailureOr<AnalogArrayExecutionPolicy>
parseAnalogArrayExecutionPolicy(StringRef value, Operation *anchor);
FailureOr<NetworkContentionModel>
parseNetworkContentionModel(StringRef value, Operation *anchor);

StringRef stringifyMappingObjective(MappingObjectiveKind objective);
StringRef stringifyNetworkContentionModel(NetworkContentionModel model);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGCONFIG_H
