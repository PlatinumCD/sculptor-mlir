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

LogicalResult emitU32Accessor(ModuleOp module, StringRef name, uint32_t value) {
  if (failed(checkAvailableSymbol(module, name)))
    return failure();
  Location loc = module.getLoc();
  auto functionType = LLVM::LLVMFunctionType::get(
      IntegerType::get(module.getContext(), 32), {}, false);
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto function = builder.create<LLVM::LLVMFuncOp>(loc, name, functionType);
  Block *entry = function.addEntryBlock(builder);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  bodyBuilder.create<LLVM::ReturnOp>(loc, buildI32(bodyBuilder, loc, value));
  return success();
}

LogicalResult emitU64Accessor(ModuleOp module, StringRef name, uint64_t value) {
  if (failed(checkAvailableSymbol(module, name)))
    return failure();
  Location loc = module.getLoc();
  auto functionType = LLVM::LLVMFunctionType::get(
      IntegerType::get(module.getContext(), 64), {}, false);
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  auto function = builder.create<LLVM::LLVMFuncOp>(loc, name, functionType);
  Block *entry = function.addEntryBlock(builder);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  bodyBuilder.create<LLVM::ReturnOp>(loc, buildI64(bodyBuilder, loc, value));
  return success();
}

uint32_t getResourceFlags(const ResourceModel &resource) {
  if (resource.scratchpad)
    return kResourceScratchpadFlag;
  switch (resource.kind) {
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
              resource.elementType,
              static_cast<uint32_t>(resource.shapedType.getRank()),
              resource.dimensionOffset,
              getResourceFlags(resource),
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

LogicalResult emitMemoryOwnerTable(ModuleOp module, const TileModel &model) {
  if (failed(checkAvailableSymbol(module, kMemoryOwnersGlobalName)) ||
      failed(checkAvailableSymbol(module, kMemoryOwnersAccessorName)) ||
      failed(checkAvailableSymbol(module, kMemoryOwnerCountAccessorName)))
    return failure();
  LLVM::GlobalOp global;
  if (!model.memoryOwners.empty()) {
    auto type = getMemoryOwnerType(module.getContext());
    global = emitArrayGlobal(
        module, kMemoryOwnersGlobalName, type, model.memoryOwners.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const MemoryOwnerModel &owner = model.memoryOwners[index];
          Value entry = builder.create<LLVM::ZeroOp>(module.getLoc(), type);
          const uint32_t values[] = {owner.id, owner.globalResourceId,
                                     owner.localSlot, owner.kind};
          for (auto value : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), value.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(value.index())});
          return builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(), owner.byteSize),
              ArrayRef<int64_t>{4});
        });
  }
  if (failed(emitPointerAccessor(module, kMemoryOwnersAccessorName, global)) ||
      failed(emitCountAccessor(module, kMemoryOwnerCountAccessorName,
                               model.memoryOwners.size())))
    return failure();
  return success();
}

LogicalResult emitMemoryViewTable(ModuleOp module, const TileModel &model) {
  if (failed(checkAvailableSymbol(module, kMemoryViewsGlobalName)) ||
      failed(checkAvailableSymbol(module, kMemoryViewsAccessorName)) ||
      failed(checkAvailableSymbol(module, kMemoryViewCountAccessorName)))
    return failure();
  LLVM::GlobalOp global;
  if (!model.memoryViews.empty()) {
    auto type = getMemoryViewType(module.getContext());
    global = emitArrayGlobal(
        module, kMemoryViewsGlobalName, type, model.memoryViews.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const MemoryViewModel &view = model.memoryViews[index];
          Value entry = builder.create<LLVM::ZeroOp>(module.getLoc(), type);
          const uint32_t values[] = {view.id,
                                     view.ownerId,
                                     view.ownerSlot,
                                     view.rank,
                                     view.geometryOffset,
                                     view.contiguity,
                                     0,
                                     0};
          for (auto value : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), value.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(value.index())});
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(), view.byteOffset),
              ArrayRef<int64_t>{8});
          return builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(), view.byteSize),
              ArrayRef<int64_t>{9});
        });
  }
  if (failed(emitPointerAccessor(module, kMemoryViewsAccessorName, global)) ||
      failed(emitCountAccessor(module, kMemoryViewCountAccessorName,
                               model.memoryViews.size())))
    return failure();
  return success();
}

LogicalResult emitMemoryViewGeometryTable(ModuleOp module,
                                          const TileModel &model) {
  if (failed(checkAvailableSymbol(module, kMemoryViewGeometryGlobalName)) ||
      failed(checkAvailableSymbol(module, kMemoryViewGeometryAccessorName)) ||
      failed(
          checkAvailableSymbol(module, kMemoryViewGeometryCountAccessorName)))
    return failure();
  LLVM::GlobalOp global;
  if (!model.memoryViewGeometry.empty()) {
    Type type = IntegerType::get(module.getContext(), 64);
    global =
        emitArrayGlobal(module, kMemoryViewGeometryGlobalName, type,
                        model.memoryViewGeometry.size(),
                        [&](OpBuilder &builder, unsigned index) -> Value {
                          return buildI64(builder, module.getLoc(),
                                          static_cast<uint64_t>(
                                              model.memoryViewGeometry[index]));
                        });
  }
  if (failed(emitPointerAccessor(module, kMemoryViewGeometryAccessorName,
                                 global)) ||
      failed(emitCountAccessor(module, kMemoryViewGeometryCountAccessorName,
                               model.memoryViewGeometry.size())))
    return failure();
  return success();
}

LogicalResult emitRouteViewTable(ModuleOp module, const TileModel &model) {
  if (failed(checkAvailableSymbol(module, kRouteViewsGlobalName)) ||
      failed(checkAvailableSymbol(module, kRouteViewsAccessorName)) ||
      failed(checkAvailableSymbol(module, kRouteViewCountAccessorName)))
    return failure();
  LLVM::GlobalOp global;
  if (!model.routeViews.empty()) {
    auto type = getRouteViewType(module.getContext());
    global = emitArrayGlobal(
        module, kRouteViewsGlobalName, type, model.routeViews.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const RouteViewModel &view = model.routeViews[index];
          Value entry = builder.create<LLVM::ZeroOp>(module.getLoc(), type);
          const uint32_t values[] = {view.routeId,      view.localResourceSlot,
                                     view.ownerSlot,    view.viewId,
                                     view.movementMode, view.completionEventId,
                                     view.assemblyId,   view.flags};
          for (auto value : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), value.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(value.index())});
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(), view.byteOffset),
              ArrayRef<int64_t>{8});
          return builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(), view.byteSize),
              ArrayRef<int64_t>{9});
        });
  }
  if (failed(emitPointerAccessor(module, kRouteViewsAccessorName, global)) ||
      failed(emitCountAccessor(module, kRouteViewCountAccessorName,
                               model.routeViews.size())))
    return failure();
  return success();
}

LogicalResult emitAssemblyTables(ModuleOp module, const TileModel &model) {
  if (failed(checkAvailableSymbol(module, kAssembliesGlobalName)) ||
      failed(checkAvailableSymbol(module, kAssembliesAccessorName)) ||
      failed(checkAvailableSymbol(module, kAssemblyCountAccessorName)) ||
      failed(checkAvailableSymbol(module, kAssemblyContributionsGlobalName)) ||
      failed(
          checkAvailableSymbol(module, kAssemblyContributionsAccessorName)) ||
      failed(
          checkAvailableSymbol(module, kAssemblyContributionCountAccessorName)))
    return failure();
  LLVM::GlobalOp assemblyGlobal;
  if (!model.assemblies.empty()) {
    auto type = getAssemblyType(module.getContext());
    assemblyGlobal = emitArrayGlobal(
        module, kAssembliesGlobalName, type, model.assemblies.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const AssemblyModel &assembly = model.assemblies[index];
          Value entry = builder.create<LLVM::ZeroOp>(module.getLoc(), type);
          const uint32_t values[] = {assembly.id,
                                     assembly.ownerSlot,
                                     assembly.contributionOffset,
                                     assembly.contributionCount,
                                     assembly.readinessEventId,
                                     0};
          for (auto value : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), value.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(value.index())});
          return entry;
        });
  }
  LLVM::GlobalOp contributionGlobal;
  if (!model.assemblyContributions.empty()) {
    auto type = getAssemblyContributionType(module.getContext());
    contributionGlobal = emitArrayGlobal(
        module, kAssemblyContributionsGlobalName, type,
        model.assemblyContributions.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const AssemblyContributionModel &contribution =
              model.assemblyContributions[index];
          Value entry = builder.create<LLVM::ZeroOp>(module.getLoc(), type);
          const uint32_t values[] = {
              contribution.assemblyId,        contribution.sourceViewId,
              contribution.destinationViewId, contribution.completionEventId,
              contribution.routeId,           contribution.flags};
          for (auto value : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), value.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(value.index())});
          return entry;
        });
  }
  if (failed(emitPointerAccessor(module, kAssembliesAccessorName,
                                 assemblyGlobal)) ||
      failed(emitCountAccessor(module, kAssemblyCountAccessorName,
                               model.assemblies.size())) ||
      failed(emitPointerAccessor(module, kAssemblyContributionsAccessorName,
                                 contributionGlobal)) ||
      failed(emitCountAccessor(module, kAssemblyContributionCountAccessorName,
                               model.assemblyContributions.size())))
    return failure();
  return success();
}

LogicalResult emitSegmentedMovementTables(ModuleOp module,
                                          const TileModel &model) {
  if (failed(checkAvailableSymbol(module, kSegmentedMovementsGlobalName)) ||
      failed(checkAvailableSymbol(module, kSegmentedMovementsAccessorName)) ||
      failed(
          checkAvailableSymbol(module, kSegmentedMovementCountAccessorName)) ||
      failed(checkAvailableSymbol(module, kMemorySegmentsGlobalName)) ||
      failed(checkAvailableSymbol(module, kMemorySegmentsAccessorName)) ||
      failed(checkAvailableSymbol(module, kMemorySegmentCountAccessorName)))
    return failure();

  LLVM::GlobalOp movementGlobal;
  if (!model.segmentedMovements.empty()) {
    auto type = getSegmentedMovementType(module.getContext());
    movementGlobal = emitArrayGlobal(
        module, kSegmentedMovementsGlobalName, type,
        model.segmentedMovements.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const SegmentedMovementModel &movement =
              model.segmentedMovements[index];
          Value entry = builder.create<LLVM::ZeroOp>(module.getLoc(), type);
          const uint32_t values[] = {
              movement.movementId,
              movement.routeId,
              movement.segmentOffset,
              movement.segmentCount,
              movement.completionEventId,
              movement.assemblyId,
              movement.flags,
              0U,
          };
          for (auto value : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), value.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(value.index())});
          return builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(), movement.byteSize),
              ArrayRef<int64_t>{8});
        });
  }
  if (failed(emitPointerAccessor(module, kSegmentedMovementsAccessorName,
                                 movementGlobal)) ||
      failed(emitCountAccessor(module, kSegmentedMovementCountAccessorName,
                               model.segmentedMovements.size())))
    return failure();

  LLVM::GlobalOp segmentGlobal;
  if (!model.memorySegments.empty()) {
    auto type = getMemorySegmentType(module.getContext());
    segmentGlobal = emitArrayGlobal(
        module, kMemorySegmentsGlobalName, type, model.memorySegments.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const MemorySegmentModel &segment = model.memorySegments[index];
          Value entry = builder.create<LLVM::ZeroOp>(module.getLoc(), type);
          const uint64_t values[] = {
              segment.sourceByteOffset,
              segment.destinationByteOffset,
              segment.byteSize,
          };
          for (auto value : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI64(builder, module.getLoc(), value.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(value.index())});
          return entry;
        });
  }
  if (failed(emitPointerAccessor(module, kMemorySegmentsAccessorName,
                                 segmentGlobal)) ||
      failed(emitCountAccessor(module, kMemorySegmentCountAccessorName,
                               model.memorySegments.size())))
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

LogicalResult emitIdIndexTable(ModuleOp module, ArrayRef<IdIndexModel> entries,
                               StringRef globalName, StringRef accessorName,
                               StringRef countAccessorName) {
  if (failed(checkAvailableSymbol(module, globalName)) ||
      failed(checkAvailableSymbol(module, accessorName)) ||
      failed(checkAvailableSymbol(module, countAccessorName)))
    return failure();
  LLVM::GlobalOp global;
  if (!entries.empty()) {
    auto type = getIdIndexType(module.getContext());
    global = emitArrayGlobal(
        module, globalName, type, entries.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          Value entry = builder.create<LLVM::ZeroOp>(module.getLoc(), type);
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI32(builder, module.getLoc(), entries[index].id),
              ArrayRef<int64_t>{0});
          return builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI32(builder, module.getLoc(), entries[index].index),
              ArrayRef<int64_t>{1});
        });
  }
  if (failed(emitPointerAccessor(module, accessorName, global)) ||
      failed(emitCountAccessor(module, countAccessorName, entries.size())))
    return failure();
  return success();
}

LogicalResult emitStaticRuntimeTables(ModuleOp module,
                                      const TileModel &model) {
  if (failed(emitIdIndexTable(
          module, model.taskIdIndex, kTaskIdIndexGlobalName,
          kTaskIdIndexAccessorName, kTaskIdIndexCountAccessorName)) ||
      failed(emitIdIndexTable(
          module, model.incomingRouteIdIndex,
          kIncomingRouteIdIndexGlobalName, kIncomingRouteIdIndexAccessorName,
          kIncomingRouteIdIndexCountAccessorName)) ||
      failed(emitIdIndexTable(
          module, model.outgoingRouteIdIndex,
          kOutgoingRouteIdIndexGlobalName, kOutgoingRouteIdIndexAccessorName,
          kOutgoingRouteIdIndexCountAccessorName)))
    return failure();

  if (failed(checkAvailableSymbol(module, kStaticTasksGlobalName)) ||
      failed(checkAvailableSymbol(module, kStaticTasksAccessorName)) ||
      failed(checkAvailableSymbol(module, kStaticTaskCountAccessorName)) ||
      failed(checkAvailableSymbol(module, kStaticResourcesGlobalName)) ||
      failed(checkAvailableSymbol(module, kStaticResourcesAccessorName)) ||
      failed(checkAvailableSymbol(module, kStaticResourceCountAccessorName)) ||
      failed(checkAvailableSymbol(module, kStaticRuntimeDataGlobalName)) ||
      failed(checkAvailableSymbol(module, kStaticRuntimeDataAccessorName)) ||
      failed(checkAvailableSymbol(module, kStaticRuntimeDataCountAccessorName)))
    return failure();

  LLVM::GlobalOp taskGlobal;
  if (!model.staticTasks.empty()) {
    auto type = getStaticTaskType(module.getContext());
    taskGlobal = emitArrayGlobal(
        module, kStaticTasksGlobalName, type, model.staticTasks.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const StaticTaskModel &task = model.staticTasks[index];
          const uint32_t values[] = {
              task.initialReadiness, task.outgoingRouteOffset,
              task.outgoingRouteCount, task.dependentOffset,
              task.dependentCount};
          Value entry = builder.create<LLVM::ZeroOp>(module.getLoc(), type);
          for (auto value : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), value.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(value.index())});
          return entry;
        });
  }
  if (failed(emitPointerAccessor(module, kStaticTasksAccessorName,
                                 taskGlobal)) ||
      failed(emitCountAccessor(module, kStaticTaskCountAccessorName,
                               model.staticTasks.size())))
    return failure();

  LLVM::GlobalOp resourceGlobal;
  if (!model.staticResources.empty()) {
    auto type = getStaticResourceType(module.getContext());
    resourceGlobal = emitArrayGlobal(
        module, kStaticResourcesGlobalName, type, model.staticResources.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const StaticResourceModel &resource = model.staticResources[index];
          const uint32_t values[] = {
              resource.readyConsumerOffset, resource.readyConsumerCount,
              resource.boundConsumerOffset, resource.boundConsumerCount};
          Value entry = builder.create<LLVM::ZeroOp>(module.getLoc(), type);
          for (auto value : llvm::enumerate(values))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), value.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(value.index())});
          return entry;
        });
  }
  if (failed(emitPointerAccessor(module, kStaticResourcesAccessorName,
                                 resourceGlobal)) ||
      failed(emitCountAccessor(module, kStaticResourceCountAccessorName,
                               model.staticResources.size())))
    return failure();

  LLVM::GlobalOp dataGlobal;
  if (!model.staticRuntimeData.empty()) {
    Type i32Type = IntegerType::get(module.getContext(), 32);
    dataGlobal = emitArrayGlobal(
        module, kStaticRuntimeDataGlobalName, i32Type,
        model.staticRuntimeData.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          return buildI32(builder, module.getLoc(),
                          model.staticRuntimeData[index]);
        });
  }
  if (failed(emitPointerAccessor(module, kStaticRuntimeDataAccessorName,
                                 dataGlobal)) ||
      failed(emitCountAccessor(module, kStaticRuntimeDataCountAccessorName,
                               model.staticRuntimeData.size())))
    return failure();
  return success();
}

LogicalResult emitDMADescriptorTable(ModuleOp module, const TileModel &model) {
  if (failed(checkAvailableSymbol(module, kDMADescriptorsGlobalName)) ||
      failed(checkAvailableSymbol(module, kDMADescriptorsAccessorName)) ||
      failed(checkAvailableSymbol(module, kDMADescriptorCountAccessorName)))
    return failure();
  LLVM::GlobalOp global;
  if (!model.dmaDescriptors.empty()) {
    auto descriptorType = getDMADescriptorType(module.getContext());
    global = emitArrayGlobal(
        module, kDMADescriptorsGlobalName, descriptorType,
        model.dmaDescriptors.size(),
        [&](OpBuilder &builder, unsigned index) -> Value {
          const DMADescriptorModel &descriptor = model.dmaDescriptors[index];
          Value entry =
              builder.create<LLVM::ZeroOp>(module.getLoc(), descriptorType);
          const uint32_t head[] = {descriptor.descriptorId,
                                   descriptor.direction, descriptor.localSlot,
                                   descriptor.routeId};
          for (auto value : llvm::enumerate(head))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), value.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(value.index())});
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(), descriptor.scratchpadOffset),
              ArrayRef<int64_t>{4});
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(), descriptor.byteSize),
              ArrayRef<int64_t>{5});
          const uint32_t tail[] = {
              descriptor.completionTokenId, descriptor.triggerKind,
              descriptor.triggerId,         descriptor.flags,
              descriptor.sourceStorage,     descriptor.destinationStorage};
          for (auto value : llvm::enumerate(tail))
            entry = builder.create<LLVM::InsertValueOp>(
                module.getLoc(), entry,
                buildI32(builder, module.getLoc(), value.value()),
                ArrayRef<int64_t>{static_cast<int64_t>(6 + value.index())});
          entry = builder.create<LLVM::InsertValueOp>(
              module.getLoc(), entry,
              buildI64(builder, module.getLoc(), descriptor.reserved),
              ArrayRef<int64_t>{12});
          return entry;
        });
  }
  if (failed(
          emitPointerAccessor(module, kDMADescriptorsAccessorName, global)) ||
      failed(emitCountAccessor(module, kDMADescriptorCountAccessorName,
                               model.dmaDescriptors.size())))
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
      failed(emitU32Accessor(module, kABIFeaturesAccessorName,
                             model.abiFeatures)) ||
      failed(emitU64Accessor(module, kScratchpadRequiredBytesAccessorName,
                             model.scratchpadRequiredBytes)) ||
      failed(emitDMADescriptorTable(module, model)) ||
      failed(emitResourceTable(module, model)) ||
      failed(emitResourceDimensionTable(module, model)) ||
      failed(emitMemoryOwnerTable(module, model)) ||
      failed(emitMemoryViewTable(module, model)) ||
      failed(emitMemoryViewGeometryTable(module, model)) ||
      failed(emitRouteViewTable(module, model)) ||
      failed(emitAssemblyTables(module, model)) ||
      failed(emitSegmentedMovementTables(module, model)) ||
      failed(emitWorkspaceSizeAccessor(module, model.workspaceSize)) ||
      failed(emitTaskTable(module, model, model.bootTaskIndices,
                           kBootTasksGlobalName, kBootTasksAccessorName,
                           kBootTaskCountAccessorName)) ||
      failed(emitTaskTable(module, model, model.dispatchTaskIndices,
                           kDispatchTasksGlobalName, kDispatchTasksAccessorName,
                           kDispatchTaskCountAccessorName)) ||
      failed(emitTaskBindingTable(module, model)) ||
      failed(emitTaskBindingDataTable(module, model)) ||
      failed(emitStaticRuntimeTables(module, model)) ||
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
