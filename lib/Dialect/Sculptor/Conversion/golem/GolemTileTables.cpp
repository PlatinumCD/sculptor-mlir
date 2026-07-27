#include "GolemTileABI.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include "mlir/IR/SymbolTable.h"

#include <functional>

namespace mlir {
namespace sculptor {
namespace golem_tile_abi {

namespace {

Value buildI32(OpBuilder &builder, Location loc, uint32_t value) {
  return builder.create<LLVM::ConstantOp>(
      loc, builder.getI32Type(),
      builder.getI32IntegerAttr(static_cast<int32_t>(value)));
}

Value buildI64(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<LLVM::ConstantOp>(
      loc, builder.getI64Type(),
      builder.getI64IntegerAttr(static_cast<int64_t>(value)));
}

LLVM::GlobalOp emitArrayGlobal(
    ModuleOp module, StringRef name, Type elementType, unsigned count,
    const std::function<Value(OpBuilder &, unsigned)> &buildElement) {
  MLIRContext *context = module.getContext();
  Location loc = module.getLoc();
  auto arrayType = LLVM::LLVMArrayType::get(elementType, count);
  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  auto global = moduleBuilder.create<LLVM::GlobalOp>(
      loc, arrayType, /*isConstant=*/true, LLVM::Linkage::Internal, name,
      Attribute());

  global.getInitializerRegion().emplaceBlock();
  OpBuilder initializer =
      OpBuilder::atBlockBegin(&global.getInitializerRegion().front());
  Value array = initializer.create<LLVM::ZeroOp>(loc, arrayType);
  for (unsigned index = 0; index < count; ++index) {
    Value element = buildElement(initializer, index);
    array = initializer.create<LLVM::InsertValueOp>(
        loc, array, element, ArrayRef<int64_t>{static_cast<int64_t>(index)});
  }
  initializer.create<LLVM::ReturnOp>(loc, array);
  return global;
}

LogicalResult checkAvailableSymbol(ModuleOp module, StringRef name) {
  if (!module.lookupSymbol(name))
    return success();
  module.emitError("cannot emit Golem tile ABI symbol @")
      << name << ": already exists";
  return failure();
}

LogicalResult emitPointerAccessor(ModuleOp module, StringRef name,
                                  LLVM::GlobalOp global) {
  if (failed(checkAvailableSymbol(module, name)))
    return failure();

  MLIRContext *context = module.getContext();
  Location loc = module.getLoc();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  auto functionType = LLVM::LLVMFunctionType::get(ptrType, {}, false);
  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  auto function =
      moduleBuilder.create<LLVM::LLVMFuncOp>(loc, name, functionType);
  Block *entry = function.addEntryBlock(moduleBuilder);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  Value pointer;
  if (global)
    pointer = bodyBuilder.create<LLVM::AddressOfOp>(loc, global);
  else
    pointer = bodyBuilder.create<LLVM::ZeroOp>(loc, ptrType);
  bodyBuilder.create<LLVM::ReturnOp>(loc, pointer);
  return success();
}

LogicalResult emitCountAccessor(ModuleOp module, StringRef name,
                                uint32_t count) {
  if (failed(checkAvailableSymbol(module, name)))
    return failure();

  MLIRContext *context = module.getContext();
  Location loc = module.getLoc();
  Type i32Type = IntegerType::get(context, 32);
  auto functionType = LLVM::LLVMFunctionType::get(i32Type, {}, false);
  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  auto function =
      moduleBuilder.create<LLVM::LLVMFuncOp>(loc, name, functionType);
  Block *entry = function.addEntryBlock(moduleBuilder);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  bodyBuilder.create<LLVM::ReturnOp>(loc, buildI32(bodyBuilder, loc, count));
  return success();
}

LogicalResult emitCoreIdAccessor(ModuleOp module, uint32_t coreId) {
  if (failed(checkAvailableSymbol(module, kCoreIdAccessorName)))
    return failure();

  MLIRContext *context = module.getContext();
  Location loc = module.getLoc();
  Type i32Type = IntegerType::get(context, 32);
  auto functionType = LLVM::LLVMFunctionType::get(i32Type, {}, false);
  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  auto function = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, kCoreIdAccessorName, functionType);
  Block *entry = function.addEntryBlock(moduleBuilder);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  bodyBuilder.create<LLVM::ReturnOp>(loc, buildI32(bodyBuilder, loc, coreId));
  return success();
}

LogicalResult emitWorkspaceSizeAccessor(ModuleOp module,
                                        uint64_t workspaceSize) {
  if (failed(checkAvailableSymbol(module, kWorkspaceSizeAccessorName)))
    return failure();

  MLIRContext *context = module.getContext();
  Location loc = module.getLoc();
  Type i64Type = IntegerType::get(context, 64);
  auto functionType = LLVM::LLVMFunctionType::get(i64Type, {}, false);
  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  auto function = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, kWorkspaceSizeAccessorName, functionType);
  Block *entry = function.addEntryBlock(moduleBuilder);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  bodyBuilder.create<LLVM::ReturnOp>(loc,
                                     buildI64(bodyBuilder, loc, workspaceSize));
  return success();
}

uint32_t getResourceFlags(ResourceKind kind) {
  switch (kind) {
  case ResourceKind::ModelInput:
  case ResourceKind::ModelOutput:
    return kResourceExternalFlag;
  case ResourceKind::Intermediate:
  case ResourceKind::RouteInput:
  case ResourceKind::RouteOutput:
    return kResourceWorkspaceFlag;
  case ResourceKind::Persistent:
    return 0;
  }
  llvm_unreachable("unknown Golem tile resource kind");
}

LogicalResult emitResourceTable(ModuleOp module, const TileModel &model) {
  if (failed(checkAvailableSymbol(module, kResourcesGlobalName)) ||
      failed(checkAvailableSymbol(module, kResourcesAccessorName)) ||
      failed(checkAvailableSymbol(module, kResourceCountAccessorName)))
    return failure();

  LLVM::GlobalOp global;
  if (!model.resourceIndicesBySlot.empty()) {
    auto resourceType = getResourceType(module.getContext());
    global = emitArrayGlobal(
        module, kResourcesGlobalName, resourceType,
        model.resourceIndicesBySlot.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const ResourceModel &resource =
              model.resources[model.resourceIndicesBySlot[index]];
          const uint32_t values[] = {
              resource.globalId,
              resource.routeId.value_or(0),
              resource.slot,
              static_cast<uint32_t>(resource.kind),
              kFloat32ElementType,
              static_cast<uint32_t>(resource.shapedType.getRank()),
              resource.dimensionOffset,
              getResourceFlags(resource.kind),
          };

          Value entry =
              builder.create<LLVM::ZeroOp>(module.getLoc(), resourceType);
          for (auto indexedValue : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), indexedValue.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(indexedValue.index())});
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(), resource.byteSize),
              ArrayRef<int64_t>{8});
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(),
                       resource.workspaceOffset.value_or(0)),
              ArrayRef<int64_t>{9});
          return entry;
        });
  }

  if (failed(emitPointerAccessor(module, kResourcesAccessorName, global)) ||
      failed(emitCountAccessor(module, kResourceCountAccessorName,
                               model.resourceIndicesBySlot.size())))
    return failure();
  return success();
}

LogicalResult emitResourceDimensionTable(ModuleOp module,
                                         const TileModel &model) {
  if (failed(checkAvailableSymbol(module, kResourceDimensionsGlobalName)) ||
      failed(checkAvailableSymbol(module, kResourceDimensionsAccessorName)) ||
      failed(checkAvailableSymbol(module, kResourceDimensionCountAccessorName)))
    return failure();

  LLVM::GlobalOp global;
  if (!model.resourceDimensions.empty()) {
    Type i64Type = IntegerType::get(module.getContext(), 64);
    global =
        emitArrayGlobal(module, kResourceDimensionsGlobalName, i64Type,
                        model.resourceDimensions.size(),
                        [&](OpBuilder &builder, unsigned index) -> Value {
                          return buildI64(builder, module.getLoc(),
                                          static_cast<uint64_t>(
                                              model.resourceDimensions[index]));
                        });
  }

  if (failed(emitPointerAccessor(module, kResourceDimensionsAccessorName,
                                 global)) ||
      failed(emitCountAccessor(module, kResourceDimensionCountAccessorName,
                               model.resourceDimensions.size())))
    return failure();
  return success();
}

LogicalResult emitTaskBindingTable(ModuleOp module, const TileModel &model) {
  if (failed(checkAvailableSymbol(module, kTaskBindingsGlobalName)) ||
      failed(checkAvailableSymbol(module, kTaskBindingsAccessorName)) ||
      failed(checkAvailableSymbol(module, kTaskBindingCountAccessorName)))
    return failure();

  LLVM::GlobalOp global;
  if (!model.taskBindings.empty()) {
    auto bindingType = getTaskBindingType(module.getContext());
    global = emitArrayGlobal(
        module, kTaskBindingsGlobalName, bindingType, model.taskBindings.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const TaskBindingModel &binding = model.taskBindings[index];
          const uint32_t values[] = {
              binding.taskId,          binding.inputOffset,
              binding.inputCount,      binding.outputOffset,
              binding.outputCount,     binding.dependencyOffset,
              binding.dependencyCount, 0,
          };
          Value entry =
              builder.create<LLVM::ZeroOp>(module.getLoc(), bindingType);
          for (auto indexedValue : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), indexedValue.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(indexedValue.index())});
          return entry;
        });
  }

  if (failed(emitPointerAccessor(module, kTaskBindingsAccessorName, global)) ||
      failed(emitCountAccessor(module, kTaskBindingCountAccessorName,
                               model.taskBindings.size())))
    return failure();
  return success();
}

LogicalResult emitTaskBindingDataTable(ModuleOp module,
                                       const TileModel &model) {
  if (failed(checkAvailableSymbol(module, kTaskBindingDataGlobalName)) ||
      failed(checkAvailableSymbol(module, kTaskBindingDataAccessorName)) ||
      failed(checkAvailableSymbol(module, kTaskBindingDataCountAccessorName)))
    return failure();

  LLVM::GlobalOp global;
  if (!model.taskBindingData.empty()) {
    Type i32Type = IntegerType::get(module.getContext(), 32);
    global = emitArrayGlobal(module, kTaskBindingDataGlobalName, i32Type,
                             model.taskBindingData.size(),
                             [&](OpBuilder &builder, unsigned index) -> Value {
                               return buildI32(builder, module.getLoc(),
                                               model.taskBindingData[index]);
                             });
  }

  if (failed(
          emitPointerAccessor(module, kTaskBindingDataAccessorName, global)) ||
      failed(emitCountAccessor(module, kTaskBindingDataCountAccessorName,
                               model.taskBindingData.size())))
    return failure();
  return success();
}

LogicalResult emitTaskTable(ModuleOp module, const TileModel &model,
                            ArrayRef<unsigned> taskIndices,
                            StringRef globalName, StringRef tableAccessor,
                            StringRef countAccessor) {
  if (failed(checkAvailableSymbol(module, globalName)) ||
      failed(checkAvailableSymbol(module, tableAccessor)) ||
      failed(checkAvailableSymbol(module, countAccessor)))
    return failure();

  LLVM::GlobalOp global;
  if (!taskIndices.empty()) {
    auto taskType = getTaskType(module.getContext());
    global = emitArrayGlobal(
        module, globalName, taskType, taskIndices.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const TaskModel &task = model.tasks[taskIndices[index]];
          Value entry = builder.create<LLVM::ZeroOp>(module.getLoc(), taskType);
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI32(builder, module.getLoc(), task.globalId),
              ArrayRef<int64_t>{0});
          LLVM::LLVMFuncOp adapter = task.adapter;
          Value execute = builder.create<LLVM::AddressOfOp>(
              module.getLoc(), LLVM::LLVMPointerType::get(module.getContext()),
              adapter.getName());
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry, execute, ArrayRef<int64_t>{1});
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI32(builder, module.getLoc(), task.inputTypes.size()),
              ArrayRef<int64_t>{2});
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI32(builder, module.getLoc(), task.outputTypes.size()),
              ArrayRef<int64_t>{3});
          return entry;
        });
  }

  if (failed(emitPointerAccessor(module, tableAccessor, global)) ||
      failed(emitCountAccessor(module, countAccessor, taskIndices.size())))
    return failure();
  return success();
}

LogicalResult emitRouteTable(ModuleOp module, ArrayRef<RouteModel> routes,
                             StringRef globalName, StringRef tableAccessor,
                             StringRef countAccessor) {
  if (failed(checkAvailableSymbol(module, globalName)) ||
      failed(checkAvailableSymbol(module, tableAccessor)) ||
      failed(checkAvailableSymbol(module, countAccessor)))
    return failure();

  LLVM::GlobalOp global;
  if (!routes.empty()) {
    auto routeType = getRouteType(module.getContext());
    global = emitArrayGlobal(
        module, globalName, routeType, routes.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const RouteModel &model = routes[index];
          DeploymentRouteAttr route = model.route;
          const uint32_t values[] = {
              static_cast<uint32_t>(route.getId().getInt()),
              static_cast<uint32_t>(route.getSourceCore().getInt()),
              static_cast<uint32_t>(route.getSourceTask().getInt()),
              static_cast<uint32_t>(route.getSourceOutput().getInt()),
              static_cast<uint32_t>(route.getDestinationCore().getInt()),
              static_cast<uint32_t>(route.getDestinationTask().getInt()),
              static_cast<uint32_t>(route.getDestinationInput().getInt()),
              static_cast<uint32_t>(route.getResourceId().getInt()),
              model.localSlot,
          };

          Value entry =
              builder.create<LLVM::ZeroOp>(module.getLoc(), routeType);
          for (auto indexedValue : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), indexedValue.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(indexedValue.index())});
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(),
                       static_cast<uint64_t>(route.getByteSize().getInt())),
              ArrayRef<int64_t>{9});
          return entry;
        });
  }

  if (failed(emitPointerAccessor(module, tableAccessor, global)) ||
      failed(emitCountAccessor(module, countAccessor, routes.size())))
    return failure();
  return success();
}

LogicalResult emitModelIOTable(ModuleOp module, ArrayRef<ModelIOModel> entries,
                               StringRef globalName, StringRef tableAccessor,
                               StringRef countAccessor) {
  if (failed(checkAvailableSymbol(module, globalName)) ||
      failed(checkAvailableSymbol(module, tableAccessor)) ||
      failed(checkAvailableSymbol(module, countAccessor)))
    return failure();

  LLVM::GlobalOp global;
  if (!entries.empty()) {
    auto modelIOType = getModelIOType(module.getContext());
    global = emitArrayGlobal(
        module, globalName, modelIOType, entries.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const ModelIOModel &model = entries[index];
          const uint32_t values[] = {
              model.modelIndex,
              model.ownerCore,
              model.globalResourceId,
              model.localSlot,
          };
          Value entry =
              builder.create<LLVM::ZeroOp>(module.getLoc(), modelIOType);
          for (auto indexedValue : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), indexedValue.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(indexedValue.index())});
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(), model.byteSize),
              ArrayRef<int64_t>{4});
          return entry;
        });
  }

  if (failed(emitPointerAccessor(module, tableAccessor, global)) ||
      failed(emitCountAccessor(module, countAccessor, entries.size())))
    return failure();
  return success();
}

} // namespace

LogicalResult emitTileTables(ModuleOp module, const TileModel &model) {
  if (failed(emitCoreIdAccessor(module, model.coreId)) ||
      failed(emitResourceTable(module, model)) ||
      failed(emitResourceDimensionTable(module, model)) ||
      failed(emitWorkspaceSizeAccessor(module, model.workspaceSize)) ||
      failed(emitTaskTable(module, model, model.bootTaskIndices,
                           kBootTasksGlobalName, kBootTasksAccessorName,
                           kBootTaskCountAccessorName)) ||
      failed(emitTaskTable(module, model, model.dispatchTaskIndices,
                           kDispatchTasksGlobalName, kDispatchTasksAccessorName,
                           kDispatchTaskCountAccessorName)) ||
      failed(emitTaskBindingTable(module, model)) ||
      failed(emitTaskBindingDataTable(module, model)) ||
      failed(emitRouteTable(
          module, model.incomingRoutes, kIncomingRoutesGlobalName,
          kIncomingRoutesAccessorName, kIncomingRouteCountAccessorName)) ||
      failed(emitRouteTable(
          module, model.outgoingRoutes, kOutgoingRoutesGlobalName,
          kOutgoingRoutesAccessorName, kOutgoingRouteCountAccessorName)) ||
      failed(emitModelIOTable(module, model.modelInputs, kModelInputsGlobalName,
                              kModelInputsAccessorName,
                              kModelInputCountAccessorName)) ||
      failed(emitModelIOTable(
          module, model.modelOutputs, kModelOutputsGlobalName,
          kModelOutputsAccessorName, kModelOutputCountAccessorName)))
    return failure();
  return success();
}

} // namespace golem_tile_abi
} // namespace sculptor
} // namespace mlir
