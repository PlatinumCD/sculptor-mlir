#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_GOLEM_GOLEMTILEABI_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_GOLEM_GOLEMTILEABI_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir {
namespace sculptor {
namespace golem_tile_abi {

inline constexpr llvm::StringLiteral
    kBootTasksGlobalName("__golem_tile_boot_tasks");
inline constexpr llvm::StringLiteral
    kDispatchTasksGlobalName("__golem_tile_dispatch_tasks");
inline constexpr llvm::StringLiteral
    kIncomingRoutesGlobalName("__golem_tile_incoming_routes");
inline constexpr llvm::StringLiteral
    kOutgoingRoutesGlobalName("__golem_tile_outgoing_routes");
inline constexpr llvm::StringLiteral
    kModelInputsGlobalName("__golem_tile_model_inputs");
inline constexpr llvm::StringLiteral
    kModelOutputsGlobalName("__golem_tile_model_outputs");
inline constexpr llvm::StringLiteral
    kResourcesGlobalName("__golem_tile_resources");
inline constexpr llvm::StringLiteral
    kResourceDimensionsGlobalName("__golem_tile_resource_dimensions");
inline constexpr llvm::StringLiteral
    kTaskBindingsGlobalName("__golem_tile_task_bindings");
inline constexpr llvm::StringLiteral
    kTaskBindingDataGlobalName("__golem_tile_task_binding_data");
inline constexpr llvm::StringLiteral
    kDMADescriptorsGlobalName("__golem_tile_dma_descriptors");
inline constexpr llvm::StringLiteral
    kMemoryOwnersGlobalName("__golem_tile_memory_owners");
inline constexpr llvm::StringLiteral
    kMemoryViewsGlobalName("__golem_tile_memory_views");
inline constexpr llvm::StringLiteral
    kMemoryViewGeometryGlobalName("__golem_tile_memory_view_geometry");
inline constexpr llvm::StringLiteral
    kRouteViewsGlobalName("__golem_tile_route_views");
inline constexpr llvm::StringLiteral
    kAssembliesGlobalName("__golem_tile_assemblies");
inline constexpr llvm::StringLiteral
    kAssemblyContributionsGlobalName("__golem_tile_assembly_contributions");
inline constexpr llvm::StringLiteral
    kSegmentedMovementsGlobalName("__golem_tile_segmented_movements");
inline constexpr llvm::StringLiteral
    kMemorySegmentsGlobalName("__golem_tile_memory_segments");
inline constexpr llvm::StringLiteral
    kTaskIdIndexGlobalName("__golem_tile_task_id_index");
inline constexpr llvm::StringLiteral kIncomingRouteIdIndexGlobalName(
    "__golem_tile_incoming_route_id_index");
inline constexpr llvm::StringLiteral kOutgoingRouteIdIndexGlobalName(
    "__golem_tile_outgoing_route_id_index");
inline constexpr llvm::StringLiteral
    kStaticTasksGlobalName("__golem_tile_static_tasks");
inline constexpr llvm::StringLiteral
    kStaticResourcesGlobalName("__golem_tile_static_resources");
inline constexpr llvm::StringLiteral
    kStaticRuntimeDataGlobalName("__golem_tile_static_runtime_data");

inline constexpr llvm::StringLiteral
    kBootTasksAccessorName("golem_tile_boot_tasks");
inline constexpr llvm::StringLiteral
    kBootTaskCountAccessorName("golem_tile_boot_task_count");
inline constexpr llvm::StringLiteral
    kDispatchTasksAccessorName("golem_tile_dispatch_tasks");
inline constexpr llvm::StringLiteral
    kDispatchTaskCountAccessorName("golem_tile_dispatch_task_count");
inline constexpr llvm::StringLiteral
    kIncomingRoutesAccessorName("golem_tile_incoming_routes");
inline constexpr llvm::StringLiteral
    kIncomingRouteCountAccessorName("golem_tile_incoming_route_count");
inline constexpr llvm::StringLiteral
    kOutgoingRoutesAccessorName("golem_tile_outgoing_routes");
inline constexpr llvm::StringLiteral
    kOutgoingRouteCountAccessorName("golem_tile_outgoing_route_count");
inline constexpr llvm::StringLiteral
    kModelInputsAccessorName("golem_tile_model_inputs");
inline constexpr llvm::StringLiteral
    kModelInputCountAccessorName("golem_tile_model_input_count");
inline constexpr llvm::StringLiteral
    kModelOutputsAccessorName("golem_tile_model_outputs");
inline constexpr llvm::StringLiteral
    kModelOutputCountAccessorName("golem_tile_model_output_count");
inline constexpr llvm::StringLiteral
    kResourcesAccessorName("golem_tile_resources");
inline constexpr llvm::StringLiteral
    kResourceCountAccessorName("golem_tile_resource_count");
inline constexpr llvm::StringLiteral
    kResourceDimensionsAccessorName("golem_tile_resource_dimensions");
inline constexpr llvm::StringLiteral
    kResourceDimensionCountAccessorName("golem_tile_resource_dimension_count");
inline constexpr llvm::StringLiteral
    kWorkspaceSizeAccessorName("golem_tile_workspace_size");
inline constexpr llvm::StringLiteral
    kTaskBindingsAccessorName("golem_tile_task_bindings");
inline constexpr llvm::StringLiteral
    kTaskBindingCountAccessorName("golem_tile_task_binding_count");
inline constexpr llvm::StringLiteral
    kTaskBindingDataAccessorName("golem_tile_task_binding_data");
inline constexpr llvm::StringLiteral
    kTaskBindingDataCountAccessorName("golem_tile_task_binding_data_count");
inline constexpr llvm::StringLiteral kCoreIdAccessorName("golem_tile_core_id");
inline constexpr llvm::StringLiteral
    kABIFeaturesAccessorName("golem_tile_abi_features");
inline constexpr llvm::StringLiteral kScratchpadRequiredBytesAccessorName(
    "golem_tile_scratchpad_required_bytes");
inline constexpr llvm::StringLiteral
    kDMADescriptorsAccessorName("golem_tile_dma_descriptors");
inline constexpr llvm::StringLiteral
    kDMADescriptorCountAccessorName("golem_tile_dma_descriptor_count");
inline constexpr llvm::StringLiteral
    kMemoryOwnersAccessorName("golem_tile_memory_owners");
inline constexpr llvm::StringLiteral
    kMemoryOwnerCountAccessorName("golem_tile_memory_owner_count");
inline constexpr llvm::StringLiteral
    kMemoryViewsAccessorName("golem_tile_memory_views");
inline constexpr llvm::StringLiteral
    kMemoryViewCountAccessorName("golem_tile_memory_view_count");
inline constexpr llvm::StringLiteral
    kMemoryViewGeometryAccessorName("golem_tile_memory_view_geometry");
inline constexpr llvm::StringLiteral kMemoryViewGeometryCountAccessorName(
    "golem_tile_memory_view_geometry_count");
inline constexpr llvm::StringLiteral
    kRouteViewsAccessorName("golem_tile_route_views");
inline constexpr llvm::StringLiteral
    kRouteViewCountAccessorName("golem_tile_route_view_count");
inline constexpr llvm::StringLiteral
    kAssembliesAccessorName("golem_tile_assemblies");
inline constexpr llvm::StringLiteral
    kAssemblyCountAccessorName("golem_tile_assembly_count");
inline constexpr llvm::StringLiteral
    kAssemblyContributionsAccessorName("golem_tile_assembly_contributions");
inline constexpr llvm::StringLiteral kAssemblyContributionCountAccessorName(
    "golem_tile_assembly_contribution_count");
inline constexpr llvm::StringLiteral
    kSegmentedMovementsAccessorName("golem_tile_segmented_movements");
inline constexpr llvm::StringLiteral
    kSegmentedMovementCountAccessorName("golem_tile_segmented_movement_count");
inline constexpr llvm::StringLiteral
    kMemorySegmentsAccessorName("golem_tile_memory_segments");
inline constexpr llvm::StringLiteral
    kMemorySegmentCountAccessorName("golem_tile_memory_segment_count");
inline constexpr llvm::StringLiteral
    kTaskIdIndexAccessorName("golem_tile_task_id_index");
inline constexpr llvm::StringLiteral
    kTaskIdIndexCountAccessorName("golem_tile_task_id_index_count");
inline constexpr llvm::StringLiteral kIncomingRouteIdIndexAccessorName(
    "golem_tile_incoming_route_id_index");
inline constexpr llvm::StringLiteral kIncomingRouteIdIndexCountAccessorName(
    "golem_tile_incoming_route_id_index_count");
inline constexpr llvm::StringLiteral kOutgoingRouteIdIndexAccessorName(
    "golem_tile_outgoing_route_id_index");
inline constexpr llvm::StringLiteral kOutgoingRouteIdIndexCountAccessorName(
    "golem_tile_outgoing_route_id_index_count");
inline constexpr llvm::StringLiteral
    kStaticTasksAccessorName("golem_tile_static_tasks");
inline constexpr llvm::StringLiteral
    kStaticTaskCountAccessorName("golem_tile_static_task_count");
inline constexpr llvm::StringLiteral
    kStaticResourcesAccessorName("golem_tile_static_resources");
inline constexpr llvm::StringLiteral
    kStaticResourceCountAccessorName("golem_tile_static_resource_count");
inline constexpr llvm::StringLiteral
    kStaticRuntimeDataAccessorName("golem_tile_static_runtime_data");
inline constexpr llvm::StringLiteral kStaticRuntimeDataCountAccessorName(
    "golem_tile_static_runtime_data_count");

inline constexpr uint32_t kFloat32ElementType = 1;
inline constexpr uint32_t kInt8ElementType = 2;
inline constexpr uint32_t kUInt8ElementType = 3;
inline constexpr uint32_t kInt16ElementType = 4;
inline constexpr uint32_t kUInt16ElementType = 5;
inline constexpr uint32_t kInt32ElementType = 6;
inline constexpr uint32_t kUInt32ElementType = 7;
inline constexpr uint32_t kInt64ElementType = 8;
inline constexpr uint32_t kUInt64ElementType = 9;
inline constexpr uint32_t kTaskSuccess = 0;
inline constexpr uint32_t kTaskFailure = 1;
inline constexpr uint32_t kResourceWorkspaceFlag = 1U << 0;
inline constexpr uint32_t kResourceExternalFlag = 1U << 1;
inline constexpr uint32_t kResourceScratchpadFlag = 1U << 2;
inline constexpr uint32_t kResourceSpillFlag = 1U << 3;
inline constexpr uint32_t kScratchpadDMAFeature = 1U << 0;
inline constexpr uint32_t kMemoryViewFeature = 1U << 1;
inline constexpr uint32_t kAssemblyJoinFeature = 1U << 2;
inline constexpr uint32_t kSegmentedMovementFeature = 1U << 3;
inline constexpr uint32_t kStaticRuntimeFeature = 1U << 4;
inline constexpr uint32_t kInvalidABIId = UINT32_MAX;

enum class ResourceKind : uint32_t {
  ModelInput = 0,
  ModelOutput = 1,
  Intermediate = 2,
  Persistent = 3,
  RouteInput = 4,
  RouteOutput = 5,
};

struct ResourceWireLayout {
  uint32_t globalResourceId;
  uint32_t routeId;
  uint32_t localSlot;
  uint32_t kind;
  uint32_t elementType;
  uint32_t rank;
  uint32_t dimensionOffset;
  uint32_t flags;
  uint64_t byteSize;
  uint64_t localStorageOffset;
};
static_assert(sizeof(ResourceWireLayout) == 48);
static_assert(offsetof(ResourceWireLayout, localStorageOffset) == 40);

struct MemoryOwnerWireLayout {
  uint32_t id;
  uint32_t globalResourceId;
  uint32_t localSlot;
  uint32_t kind;
  uint64_t byteSize;
};
static_assert(sizeof(MemoryOwnerWireLayout) == 24);

struct MemoryViewWireLayout {
  uint32_t id;
  uint32_t ownerId;
  uint32_t ownerSlot;
  uint32_t rank;
  uint32_t geometryOffset;
  uint32_t contiguity;
  uint32_t reserved0;
  uint32_t reserved1;
  uint64_t byteOffset;
  uint64_t byteSize;
};
static_assert(sizeof(MemoryViewWireLayout) == 48);

enum RouteViewFlags : uint32_t {
  RouteViewIncoming = 1U << 0,
  RouteViewOutgoing = 1U << 1,
  RouteViewAssemblyDestination = 1U << 2,
};

struct RouteViewWireLayout {
  uint32_t routeId;
  uint32_t localResourceSlot;
  uint32_t ownerSlot;
  uint32_t viewId;
  uint32_t movementMode;
  uint32_t completionEventId;
  uint32_t assemblyId;
  uint32_t flags;
  uint64_t byteOffset;
  uint64_t byteSize;
};
static_assert(sizeof(RouteViewWireLayout) == 48);

struct AssemblyWireLayout {
  uint32_t id;
  uint32_t ownerSlot;
  uint32_t contributionOffset;
  uint32_t contributionCount;
  uint32_t readinessEventId;
  uint32_t reserved;
};
static_assert(sizeof(AssemblyWireLayout) == 24);

struct AssemblyContributionWireLayout {
  uint32_t assemblyId;
  uint32_t sourceViewId;
  uint32_t destinationViewId;
  uint32_t completionEventId;
  uint32_t routeId;
  uint32_t flags;
};
static_assert(sizeof(AssemblyContributionWireLayout) == 24);

struct SegmentedMovementWireLayout {
  uint32_t movementId;
  uint32_t routeId;
  uint32_t segmentOffset;
  uint32_t segmentCount;
  uint32_t completionEventId;
  uint32_t assemblyId;
  uint32_t flags;
  uint32_t reserved;
  uint64_t byteSize;
};
static_assert(sizeof(SegmentedMovementWireLayout) == 40);

struct MemorySegmentWireLayout {
  uint64_t sourceByteOffset;
  uint64_t destinationByteOffset;
  uint64_t byteSize;
};
static_assert(sizeof(MemorySegmentWireLayout) == 24);

struct DMADescriptorWireLayout {
  uint32_t descriptorId;
  uint32_t direction;
  uint32_t localSlot;
  uint32_t routeId;
  uint64_t scratchpadOffset;
  uint64_t byteSize;
  uint32_t completionTokenId;
  uint32_t triggerKind;
  uint32_t triggerId;
  uint32_t flags;
  uint32_t sourceStorage;
  uint32_t destinationStorage;
  uint64_t reserved;
};
static_assert(sizeof(DMADescriptorWireLayout) == 64);
static_assert(alignof(DMADescriptorWireLayout) == 8);
static_assert(offsetof(DMADescriptorWireLayout, scratchpadOffset) == 16);
static_assert(offsetof(DMADescriptorWireLayout, completionTokenId) == 32);
static_assert(offsetof(DMADescriptorWireLayout, sourceStorage) == 48);
static_assert(offsetof(DMADescriptorWireLayout, reserved) == 56);

struct ResourceModel {
  Operation *op = nullptr;
  Value value;
  ShapedType shapedType;
  uint32_t elementType = 0;
  ResourceKind kind = ResourceKind::Intermediate;
  uint32_t globalId = 0;
  std::optional<uint32_t> routeId;
  uint32_t slot = 0;
  uint32_t dimensionOffset = 0;
  uint64_t byteSize = 0;
  std::optional<uint64_t> workspaceOffset;
  bool scratchpad = false;
};

struct MemoryOwnerModel {
  uint32_t id = 0;
  uint32_t globalResourceId = kInvalidABIId;
  uint32_t localSlot = kInvalidABIId;
  uint32_t kind = 0;
  uint64_t byteSize = 0;
};

struct MemoryViewModel {
  uint32_t id = 0;
  uint32_t ownerId = 0;
  uint32_t ownerSlot = kInvalidABIId;
  uint32_t rank = 0;
  uint32_t geometryOffset = 0;
  uint32_t contiguity = 0;
  uint64_t byteOffset = 0;
  uint64_t byteSize = 0;
};

struct RouteViewModel {
  uint32_t routeId = 0;
  uint32_t localResourceSlot = 0;
  uint32_t ownerSlot = 0;
  uint32_t viewId = 0;
  uint32_t movementId = 0;
  uint32_t movementMode = 0;
  uint32_t completionEventId = 0;
  uint32_t assemblyId = kInvalidABIId;
  uint32_t flags = 0;
  uint64_t byteOffset = 0;
  uint64_t byteSize = 0;
};

struct AssemblyModel {
  uint32_t id = 0;
  uint32_t ownerSlot = 0;
  uint32_t contributionOffset = 0;
  uint32_t contributionCount = 0;
  uint32_t readinessEventId = 0;
};

struct AssemblyContributionModel {
  uint32_t assemblyId = 0;
  uint32_t sourceViewId = 0;
  uint32_t destinationViewId = 0;
  uint32_t completionEventId = 0;
  uint32_t routeId = kInvalidABIId;
  uint32_t flags = 0;
};

struct SegmentedMovementModel {
  uint32_t movementId = 0;
  uint32_t routeId = 0;
  uint32_t segmentOffset = 0;
  uint32_t segmentCount = 0;
  uint32_t completionEventId = 0;
  uint32_t assemblyId = kInvalidABIId;
  uint32_t flags = 0;
  uint64_t byteSize = 0;
};

struct MemorySegmentModel {
  uint64_t sourceByteOffset = 0;
  uint64_t destinationByteOffset = 0;
  uint64_t byteSize = 0;
};

struct TaskModel {
  TaskCreateOp op;
  LLVM::LLVMFuncOp callee;
  LLVM::LLVMFuncOp adapter;
  uint32_t globalId = 0;
  uint32_t localIndex = 0;
  uint32_t coreId = 0;
  std::optional<uint32_t> localArrayId;
  std::optional<uint32_t> physicalArrayId;
  SmallVector<ShapedType> inputTypes;
  SmallVector<ShapedType> outputTypes;
  SmallVector<uint32_t> inputSlots;
  SmallVector<uint32_t> outputSlots;
  SmallVector<uint32_t> dispatchDependencyIds;
  SmallVector<unsigned> resultIndices;
  SmallVector<unsigned> canonicalOutputIndices;
  bool usesOutputParameters = false;
  bool isBoot = false;
};

struct TaskBindingModel {
  uint32_t taskId = 0;
  uint32_t inputOffset = 0;
  uint32_t inputCount = 0;
  uint32_t outputOffset = 0;
  uint32_t outputCount = 0;
  uint32_t dependencyOffset = 0;
  uint32_t dependencyCount = 0;
};

struct IdIndexModel {
  uint32_t id = 0;
  uint32_t index = 0;
};

struct StaticTaskModel {
  uint32_t initialReadiness = 0;
  uint32_t outgoingRouteOffset = 0;
  uint32_t outgoingRouteCount = 0;
  uint32_t dependentOffset = 0;
  uint32_t dependentCount = 0;
};

struct StaticResourceModel {
  uint32_t readyConsumerOffset = 0;
  uint32_t readyConsumerCount = 0;
  uint32_t boundConsumerOffset = 0;
  uint32_t boundConsumerCount = 0;
};

struct RouteModel {
  DeploymentRouteAttr route;
  uint32_t localSlot = 0;
};

struct ModelIOModel {
  uint32_t modelIndex = 0;
  uint32_t ownerCore = 0;
  uint32_t globalResourceId = 0;
  uint32_t localSlot = 0;
  uint64_t byteSize = 0;
};

struct DMADescriptorModel {
  uint32_t descriptorId = 0;
  uint32_t direction = 0;
  uint32_t localSlot = 0;
  uint32_t routeId = UINT32_MAX;
  uint64_t scratchpadOffset = 0;
  uint64_t byteSize = 0;
  uint32_t completionTokenId = 0;
  uint32_t triggerKind = 0;
  uint32_t triggerId = UINT32_MAX;
  uint32_t flags = 0;
  uint32_t sourceStorage = 0;
  uint32_t destinationStorage = 0;
  uint64_t reserved = 0;
};

struct TileModel {
  uint32_t coreId = 0;
  func::FuncOp taskGraphFunc;
  SmallVector<ResourceModel> resources;
  DenseMap<Value, unsigned> resourceIndexByValue;
  DenseMap<uint32_t, unsigned> nonRouteResourceIndexByGlobalId;
  DenseMap<uint32_t, unsigned> routeInputResourceIndexByRouteId;
  DenseMap<uint32_t, unsigned> routeOutputResourceIndexByGlobalId;
  SmallVector<unsigned> resourceIndicesBySlot;
  SmallVector<int64_t> resourceDimensions;
  SmallVector<TaskModel, 0> tasks;
  SmallVector<unsigned> bootTaskIndices;
  SmallVector<unsigned> dispatchTaskIndices;
  SmallVector<TaskBindingModel> taskBindings;
  SmallVector<uint32_t> taskBindingData;
  SmallVector<IdIndexModel> taskIdIndex;
  SmallVector<IdIndexModel> incomingRouteIdIndex;
  SmallVector<IdIndexModel> outgoingRouteIdIndex;
  SmallVector<StaticTaskModel> staticTasks;
  SmallVector<StaticResourceModel> staticResources;
  SmallVector<uint32_t> staticRuntimeData;
  SmallVector<RouteModel> incomingRoutes;
  SmallVector<RouteModel> outgoingRoutes;
  SmallVector<ModelIOModel> modelInputs;
  SmallVector<ModelIOModel> modelOutputs;
  uint64_t workspaceSize = 0;
  uint64_t scratchpadRequiredBytes = 0;
  uint32_t abiFeatures = 0;
  SmallVector<DMADescriptorModel> dmaDescriptors;
  SmallVector<MemoryOwnerModel> memoryOwners;
  SmallVector<MemoryViewModel> memoryViews;
  SmallVector<int64_t> memoryViewGeometry;
  SmallVector<RouteViewModel> routeViews;
  SmallVector<AssemblyModel> assemblies;
  SmallVector<AssemblyContributionModel> assemblyContributions;
  SmallVector<SegmentedMovementModel> segmentedMovements;
  SmallVector<MemorySegmentModel> memorySegments;
};

LLVM::LLVMStructType getTensorType(MLIRContext *context);
LLVM::LLVMStructType getTaskType(MLIRContext *context);
LLVM::LLVMStructType getResourceType(MLIRContext *context);
LLVM::LLVMStructType getTaskBindingType(MLIRContext *context);
LLVM::LLVMStructType getIdIndexType(MLIRContext *context);
LLVM::LLVMStructType getStaticTaskType(MLIRContext *context);
LLVM::LLVMStructType getStaticResourceType(MLIRContext *context);
LLVM::LLVMStructType getRouteType(MLIRContext *context);
LLVM::LLVMStructType getModelIOType(MLIRContext *context);
LLVM::LLVMStructType getDMADescriptorType(MLIRContext *context);
LLVM::LLVMStructType getMemoryOwnerType(MLIRContext *context);
LLVM::LLVMStructType getMemoryViewType(MLIRContext *context);
LLVM::LLVMStructType getRouteViewType(MLIRContext *context);
LLVM::LLVMStructType getAssemblyType(MLIRContext *context);
LLVM::LLVMStructType getAssemblyContributionType(MLIRContext *context);
LLVM::LLVMStructType getSegmentedMovementType(MLIRContext *context);
LLVM::LLVMStructType getMemorySegmentType(MLIRContext *context);
LLVM::LLVMStructType getMemRefDescriptorType(MLIRContext *context,
                                             ShapedType shapedType);

FailureOr<TileModel> collectTileModel(ModuleOp module);
LogicalResult emitTaskAdapters(ModuleOp module, TileModel &model);
LogicalResult emitTileTables(ModuleOp module, const TileModel &model);

} // namespace golem_tile_abi
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_GOLEM_GOLEMTILEABI_H
