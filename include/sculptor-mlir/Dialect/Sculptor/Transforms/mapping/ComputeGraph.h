#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_COMPUTEGRAPH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_COMPUTEGRAPH_H

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace sculptor {
namespace mapping {

inline constexpr StringLiteral kMappingOperationIdAttrName =
    "sculptor.mapping.operation_id";
inline constexpr StringLiteral kLayerRegionIdAttrName =
    "sculptor.mapping.layer_region_id";
inline constexpr StringLiteral kExpandedDigitalWorkAttrName =
    "sculptor.mapping.expanded_digital_work";
inline constexpr StringLiteral kRALeafIdAttrName =
    "sculptor.mapping.ra_leaf_id";
inline constexpr StringLiteral kLaneBindingGroupAttrName =
    "sculptor.mapping.lane_binding_group";
inline constexpr StringLiteral kMVMWaveIdAttrName =
    "sculptor.mapping.mvm_wave_id";
inline constexpr StringLiteral kMVMWaveMemberAttrName =
    "sculptor.mapping.mvm_wave_member";
inline constexpr StringLiteral kMVMWaveSizeAttrName =
    "sculptor.mapping.mvm_wave_size";
inline constexpr StringLiteral kStageIdAttrName = "sculptor.mapping.stage_id";
inline constexpr StringLiteral kStageKindAttrName =
    "sculptor.mapping.stage_kind";
inline constexpr StringLiteral kStageNameAttrName =
    "sculptor.mapping.stage_name";

inline constexpr StringLiteral kMatrixSetupStageKind = "matrix_setup";
inline constexpr StringLiteral kDigitalStageKind = "digital_stage";
inline constexpr StringLiteral kVectorTileStageKind = "vector_tile";
inline constexpr StringLiteral kPhysicalMVMStageKind = "physical_mvm";
inline constexpr StringLiteral kTileRecombineStageKind = "tile_recombine";

enum class ComputeOperationKind {
  Structured,
  LogicalMVM,
  MatrixSetup,
  DigitalStage,
  VectorTile,
  PhysicalMVM,
  TileRecombine,
};

enum class ComputeIteratorKind {
  Parallel,
  Reduction,
};

struct ComputeIterationDimension {
  int64_t loopIndex = -1;
  ComputeIteratorKind kind = ComputeIteratorKind::Parallel;
  int64_t staticExtent = ShapedType::kDynamic;
};

struct AnalogMVMGeometry {
  int64_t outputRows = ShapedType::kDynamic;
  int64_t inputColumns = ShapedType::kDynamic;
};

// This record is either one supported MLIR compute operation or one expanded
// realization stage. Each matrix setup, vector tile, physical MVM, and
// recombination stage remains an independently mappable RA leaf.
struct ComputeOperation {
  int64_t id = -1;
  Operation *operation = nullptr;
  SmallVector<Operation *> members;
  ComputeOperationKind kind = ComputeOperationKind::Structured;
  SmallVector<int64_t> inputTensors;
  SmallVector<int64_t> outputTensors;
  SmallVector<ComputeIterationDimension> iterationDomain;
  std::optional<AnalogMVMGeometry> analogMVM;
  std::optional<LogicalLaneKind> requiredLane;
  std::optional<int64_t> laneBindingGroup;
  std::optional<int64_t> mvmWaveId;
  std::optional<int64_t> mvmWaveMember;
  std::optional<int64_t> mvmWaveSize;
  std::optional<int64_t> reductionTreeId;
  std::optional<int64_t> reductionNodeId;
  std::optional<int64_t> reductionLevel;
  std::optional<int64_t> reductionOrdinal;
  std::optional<int64_t> reductionWidth;
  // Stable semantic parent assigned before layer decomposition. The dense
  // layerRegionId indexes ComputeGraph::layerRegions; semanticLayerId remains
  // the compiler-visible identity carried by the IR.
  std::optional<int64_t> semanticLayerId;
  std::string semanticLayerKind;
  int64_t layerRegionId = -1;
  std::string semanticTaskKind;
  std::string stageName;
};

// One matrix setup and every physical MVM that uses its logical array belong
// to one lane-binding group. The group identifies an abstract analog lane; it
// does not assign a physical tile, core, or lane index.
struct LaneBindingGroup {
  int64_t id = -1;
  int64_t setupOperationId = -1;
  SmallVector<int64_t> operationIds;
};

// One MVM wave captures the vector preparation, independently executable
// physical MVM tiles, optional tile recombination, and an optional terminal
// bias add for one logical MVM.
struct MVMWave {
  int64_t id = -1;
  SmallVector<int64_t> vectorTileOperationIds;
  SmallVector<int64_t> physicalMVMOperationIds;
  std::optional<int64_t> recombineOperationId;
  std::optional<int64_t> biasAddOperationId;
};

struct ComputeTensor {
  int64_t id = -1;
  Value value;
  Type type;
  SmallVector<int64_t> producerOperations;
  // Static bytes contributed by each producer operation through the support
  // operation closure that forms `value`. This vector is parallel to
  // producerOperations. For example, each input of tensor.concat contributes
  // only its own shard rather than the complete concat result.
  SmallVector<int64_t> producerByteSizes;
  SmallVector<int64_t> consumerOperations;
  int64_t byteSize = -1;
  bool isLogicalArray = false;
  bool isFunctionInput = false;
  bool isFunctionOutput = false;
};

// A semantic layer region is the strategic mapping boundary above individual
// compute operations. Tagged operations with the same stable layer ID share a
// region. An untagged, all-parallel structured epilogue may inherit the one
// semantic producer region that feeds it. One-use destination initializers
// belong to their consuming region; all other untagged operations receive
// explicit singleton fallback regions.
struct LayerRegion {
  int64_t id = -1;
  std::optional<int64_t> semanticLayerId;
  std::string semanticLayerKind;
  SmallVector<int64_t> operationIds;
  SmallVector<int64_t> inputTensors;
  SmallVector<int64_t> outputTensors;
  SmallVector<int64_t> internalTensors;
  int64_t analogLaneDemand = 0;
  int64_t estimatedDigitalWorkItems = 0;
  int64_t estimatedStaticMemoryBytes = 0;
  int64_t estimatedInputBytes = 0;
  int64_t estimatedOutputBytes = 0;
  bool isSingletonFallback = false;
};

int64_t getProducerContributionByteSize(const ComputeTensor &tensor,
                                        int64_t producerOperationId);

struct ComputeGraph {
  std::string functionSymbol;
  SmallVector<ComputeOperation, 0> operations;
  SmallVector<ComputeTensor, 0> tensors;
  SmallVector<int64_t, 0> topologicalOrder;
  SmallVector<LaneBindingGroup, 0> laneBindingGroups;
  SmallVector<MVMWave, 0> mvmWaves;
  SmallVector<LayerRegion, 0> layerRegions;
  SmallVector<int64_t, 0> topologicalLayerRegionOrder;
};

FailureOr<ComputeGraph> buildComputeGraph(func::FuncOp func);

StringRef stringifyComputeOperationKind(ComputeOperationKind kind);
StringRef stringifyComputeIteratorKind(ComputeIteratorKind kind);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_COMPUTEGRAPH_H
