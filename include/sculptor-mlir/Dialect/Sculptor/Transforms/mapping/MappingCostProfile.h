#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGCOSTPROFILE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_MAPPINGCOSTPROFILE_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {
namespace mapping {

struct MappingHardwareModel;

inline constexpr StringLiteral kMappingCostProfileAttrName =
    "sculptor.mapping.cost_profile";
inline constexpr StringLiteral kMappingCostProfileNameAttrName =
    "sculptor.mapping.cost_profile_name";
inline constexpr StringLiteral kMappingCostProfileHashAttrName =
    "sculptor.mapping.cost_profile_hash";

struct TaskCostRule {
  double fixedNs = 0.0;
  double nsPerWorkItem = 0.0;
  double nsPerInputByte = 0.0;
  double nsPerOutputByte = 0.0;
};

struct AnalogCostRule {
  double loadFixedNs = 0.0;
  double loadNsPerByte = 0.0;
  double executeNs = 0.0;
  double storeFixedNs = 0.0;
  double storeNsPerByte = 0.0;
};

struct RuntimeCostRule {
  double taskDispatchNs = 0.0;
  double routeSetupNs = 0.0;
};

struct NetworkCostRule {
  int64_t wordBits = 32;
  double hopPipelineNs = 0.0;
  double injectFixedNs = 0.0;
  double ejectFixedNs = 0.0;
  double dmaNsPerByte = 0.0;
};

struct MappingCostProfile {
  int64_t schemaVersion = 1;
  std::string name = "legacy-v1";
  std::string source = "built-in";
  std::string contentHash;
  int64_t clockFrequencyHz = 1000000000;
  bool useLegacyFormula = true;
  TaskCostRule digitalFallback;
  llvm::StringMap<TaskCostRule> digitalTaskKinds;
  AnalogCostRule analog;
  RuntimeCostRule runtime;
  NetworkCostRule network;

  LogicalResult verify(const MappingHardwareModel &hardware,
                       Operation *anchor) const;
};

MappingCostProfile
getLegacyMappingCostProfile(const MappingHardwareModel &hardware);

FailureOr<MappingCostProfile>
loadMappingCostProfile(StringRef path, const MappingHardwareModel &hardware,
                       Operation *anchor);

DictionaryAttr serializeMappingCostProfile(MLIRContext *context,
                                           const MappingCostProfile &profile);

FailureOr<MappingCostProfile>
deserializeMappingCostProfile(DictionaryAttr attr,
                              const MappingHardwareModel &hardware,
                              Operation *anchor);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif
