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
inline constexpr llvm::StringLiteral kCoreIdAccessorName("golem_tile_core_id");

inline constexpr uint32_t kFloat32ElementType = 1;
inline constexpr uint32_t kTaskSuccess = 0;
inline constexpr uint32_t kTaskFailure = 1;

enum class ResourceKind {
  ModelInput,
  ModelOutput,
  Intermediate,
  Persistent,
  RouteInput,
  RouteOutput,
};

struct ResourceModel {
  Operation *op = nullptr;
  Value value;
  ShapedType shapedType;
  ResourceKind kind = ResourceKind::Intermediate;
  uint32_t globalId = 0;
  std::optional<uint32_t> routeId;
  uint32_t slot = 0;
  uint64_t byteSize = 0;
  std::optional<uint64_t> workspaceOffset;
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
  SmallVector<unsigned> resultIndices;
  bool isBoot = false;
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

struct TileModel {
  uint32_t coreId = 0;
  func::FuncOp taskGraphFunc;
  SmallVector<ResourceModel> resources;
  DenseMap<Value, unsigned> resourceIndexByValue;
  DenseMap<uint32_t, unsigned> nonRouteResourceIndexByGlobalId;
  DenseMap<uint32_t, unsigned> routeInputResourceIndexByRouteId;
  DenseMap<uint32_t, unsigned> routeOutputResourceIndexByRouteId;
  SmallVector<TaskModel> tasks;
  SmallVector<unsigned> bootTaskIndices;
  SmallVector<unsigned> dispatchTaskIndices;
  SmallVector<RouteModel> incomingRoutes;
  SmallVector<RouteModel> outgoingRoutes;
  SmallVector<ModelIOModel> modelInputs;
  SmallVector<ModelIOModel> modelOutputs;
};

LLVM::LLVMStructType getTensorType(MLIRContext *context);
LLVM::LLVMStructType getTaskType(MLIRContext *context);
LLVM::LLVMStructType getRouteType(MLIRContext *context);
LLVM::LLVMStructType getModelIOType(MLIRContext *context);
LLVM::LLVMStructType getMemRefDescriptorType(MLIRContext *context,
                                             ShapedType shapedType);

FailureOr<TileModel> collectTileModel(ModuleOp module);
LogicalResult emitTaskAdapters(ModuleOp module, TileModel &model);
LogicalResult emitTileTables(ModuleOp module, const TileModel &model);

} // namespace golem_tile_abi
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_CONVERSION_GOLEM_GOLEMTILEABI_H
