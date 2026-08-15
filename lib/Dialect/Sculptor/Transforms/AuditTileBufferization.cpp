#include "sculptor-mlir/Dialect/Sculptor/Transforms/AuditTileBufferization.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileMemoryPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
namespace tile_memory = mlir::sculptor::tile_memory;

constexpr StringLiteral kAuditAttrName = "sculptor.memory.bufferization_audit";
constexpr StringLiteral kOwnerIdAttrName = "sculptor.memory.owner_id";
constexpr StringLiteral kDestinationBoundAttrName =
    "sculptor.memory.destination_bound";

struct AllocationAudit {
  Operation *operation = nullptr;
  int64_t ordinal = -1;
  std::optional<int64_t> staticBytes;
  bool stack = false;
  bool hasDeallocation = false;
  bool escapesRoutine = false;
  bool routineLifetime = false;

  bool approved() const {
    return !escapesRoutine && (stack || hasDeallocation);
  }
};

struct CopyAudit {
  Operation *operation = nullptr;
  int64_t ordinal = -1;
  std::optional<int64_t> staticBytes;
  bool plannedAssembly = false;
  bool plannedBootStaging = false;
  bool plannedAnalogStore = false;
  bool plannedPadding = false;
  bool plannedResultWrite = false;
  bool plannedLocalAssembly = false;
  bool fullTensor = false;

  bool planned() const {
    return plannedAssembly || plannedBootStaging || plannedAnalogStore ||
           plannedPadding || plannedResultWrite || plannedLocalAssembly;
  }
};

struct AuditSummary {
  SmallVector<AllocationAudit> allocations;
  SmallVector<CopyAudit> copies;
  int64_t staticAllocationBytes = 0;
  int64_t staticCopyBytes = 0;
  int64_t approvedLocalAllocationCount = 0;
  int64_t unplannedAllocationCount = 0;
  int64_t escapingAllocationCount = 0;
  int64_t missingDeallocationCount = 0;
  int64_t routineLifetimeAllocationCount = 0;
  int64_t plannedAssemblyCopyCount = 0;
  int64_t plannedAssemblyCopyBytes = 0;
  int64_t plannedBootStagingCopyCount = 0;
  int64_t plannedBootStagingCopyBytes = 0;
  int64_t plannedAnalogStoreCopyCount = 0;
  int64_t plannedAnalogStoreCopyBytes = 0;
  int64_t plannedPaddingCopyCount = 0;
  int64_t plannedPaddingCopyBytes = 0;
  int64_t plannedResultWriteCopyCount = 0;
  int64_t plannedResultWriteCopyBytes = 0;
  int64_t plannedLocalAssemblyCopyCount = 0;
  int64_t plannedLocalAssemblyCopyBytes = 0;
  int64_t unplannedCopyCount = 0;
  int64_t unplannedFullTensorCopyCount = 0;
  int64_t pureCopyLoopCount = 0;
  int64_t subviewCount = 0;
};

bool checkedAddTo(int64_t &target, int64_t value) {
  std::optional<int64_t> sum = llvm::checkedAdd(target, value);
  if (!sum)
    return false;
  target = *sum;
  return true;
}

std::optional<int64_t> getStaticByteSize(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return std::nullopt;
  Type elementType = shaped.getElementType();
  if (!elementType.isIntOrFloat())
    return std::nullopt;
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0 || shaped.getNumElements() < 0)
    return std::nullopt;
  return llvm::checkedMul(shaped.getNumElements(),
                          static_cast<int64_t>(bitWidth / 8));
}

bool isAliasOperation(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  return name == "memref.cast" || name == "memref.subview" ||
         name == "memref.reinterpret_cast" || name == "memref.view" ||
         name == "memref.expand_shape" || name == "memref.collapse_shape";
}

BlockArgument getRootArgument(Value value) {
  while (true) {
    if (auto argument = dyn_cast<BlockArgument>(value))
      return argument;
    Operation *definition = value.getDefiningOp();
    if (!definition || !isAliasOperation(definition) ||
        definition->getNumOperands() == 0)
      return {};
    value = definition->getOperand(0);
  }
}

memref::SubViewOp getRootSubview(Value value) {
  while (Operation *definition = value.getDefiningOp()) {
    if (auto subview = dyn_cast<memref::SubViewOp>(definition))
      return subview;
    if (!isAliasOperation(definition) || definition->getNumOperands() == 0)
      break;
    value = definition->getOperand(0);
  }
  return {};
}

IntegerAttr getArgumentIntegerAttr(BlockArgument argument, StringRef name) {
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  return function ? function.getArgAttrOfType<IntegerAttr>(
                        argument.getArgNumber(), name)
                  : IntegerAttr();
}

bool isFunctionDestination(BlockArgument argument) {
  auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
  return function && function.getArgAttr(argument.getArgNumber(),
                                         kDestinationBoundAttrName) != nullptr;
}

bool subviewMatches(memref::SubViewOp subview, TileMemoryViewAttr view) {
  if (llvm::is_contained(subview.getStaticOffsets(), ShapedType::kDynamic) ||
      llvm::is_contained(subview.getStaticSizes(), ShapedType::kDynamic) ||
      llvm::is_contained(subview.getStaticStrides(), ShapedType::kDynamic))
    return false;
  auto matches = [](ArrayRef<int64_t> values, ArrayAttr attributes) {
    if (values.size() != attributes.size())
      return false;
    return llvm::all_of(llvm::zip(values, attributes), [](auto pair) {
      return std::get<0>(pair) == cast<IntegerAttr>(std::get<1>(pair)).getInt();
    });
  };
  return matches(subview.getStaticOffsets(), view.getOffsets()) &&
         matches(subview.getStaticSizes(), view.getSizes()) &&
         llvm::all_of(subview.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; });
}

template <typename AttrTy>
FailureOr<SmallVector<AttrTy>> getOptionalTypedArray(ModuleOp module,
                                                     StringRef name) {
  SmallVector<AttrTy> result;
  auto values = module->getAttrOfType<ArrayAttr>(name);
  if (!values)
    return result;
  for (Attribute value : values) {
    auto typed = dyn_cast<AttrTy>(value);
    if (!typed)
      return module.emitError("invalid typed tile memory-plan record in '")
             << name << "'";
    result.push_back(typed);
  }
  return result;
}

FailureOr<DenseMap<int64_t, SmallVector<TileMemoryViewAttr>>>
collectAssemblyDestinationViews(ModuleOp module) {
  auto views = getOptionalTypedArray<TileMemoryViewAttr>(
      module, tile_memory::kViewsAttrName);
  auto assemblies = getOptionalTypedArray<TileMemoryAssemblyAttr>(
      module, tile_memory::kAssembliesAttrName);
  if (failed(views) || failed(assemblies))
    return failure();
  DenseMap<int64_t, TileMemoryViewAttr> viewById;
  for (TileMemoryViewAttr view : *views)
    viewById[view.getId().getInt()] = view;

  DenseMap<int64_t, SmallVector<TileMemoryViewAttr>> result;
  for (TileMemoryAssemblyAttr assembly : *assemblies) {
    int64_t ownerId = assembly.getOwnerId().getInt();
    for (Attribute value : assembly.getDestinationViewIds()) {
      int64_t viewId = cast<IntegerAttr>(value).getInt();
      auto view = viewById.find(viewId);
      if (view == viewById.end() ||
          view->second.getOwnerId().getInt() != ownerId)
        return module.emitError(
            "assembly destination view is absent from the tile memory plan");
      result[ownerId].push_back(view->second);
    }
  }
  return result;
}

AllocationAudit analyzeAllocation(Operation *operation, int64_t ordinal) {
  AllocationAudit audit;
  audit.operation = operation;
  audit.ordinal = ordinal;
  audit.stack = isa<memref::AllocaOp>(operation);
  audit.staticBytes = operation->getNumResults() == 1
                          ? getStaticByteSize(operation->getResult(0).getType())
                          : std::nullopt;
  if (operation->getNumResults() != 1) {
    audit.escapesRoutine = true;
    return audit;
  }

  func::FuncOp function = operation->getParentOfType<func::FuncOp>();
  SmallVector<Value> worklist{operation->getResult(0)};
  DenseSet<Value> visited;
  Operation *deallocation = nullptr;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
        audit.hasDeallocation = true;
        deallocation = dealloc;
        continue;
      }
      if (isAliasOperation(user)) {
        for (Value result : user->getResults())
          if (isa<BaseMemRefType>(result.getType()))
            worklist.push_back(result);
        continue;
      }
      StringRef name = user->getName().getStringRef();
      if (isa<func::ReturnOp>(user) || name == "llvm.return" ||
          (name == "memref.store" && use.getOperandNumber() == 0 &&
           isa<BaseMemRefType>(value.getType())))
        audit.escapesRoutine = true;
    }
  }
  audit.routineLifetime = function && deallocation &&
                          operation->getBlock() == &function.front() &&
                          deallocation->getBlock() == &function.front();
  return audit;
}

bool isPlannedAssemblyCopy(
    memref::CopyOp copy,
    const DenseMap<int64_t, SmallVector<TileMemoryViewAttr>>
        &assemblyDestinations) {
  BlockArgument destination = getRootArgument(copy.getTarget());
  if (!destination || !isFunctionDestination(destination))
    return false;
  IntegerAttr owner = getArgumentIntegerAttr(destination, kOwnerIdAttrName);
  if (!owner)
    return false;
  auto planned = assemblyDestinations.find(owner.getInt());
  if (planned == assemblyDestinations.end())
    return false;
  memref::SubViewOp subview = getRootSubview(copy.getTarget());
  // A direct full-owner copy initializes padding before disjoint assembly
  // contributions write their valid regions.
  if (!subview)
    return true;
  return llvm::any_of(planned->second, [&](TileMemoryViewAttr view) {
    return subviewMatches(subview, view);
  });
}

bool isCopyOperation(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  return name == "memref.copy" || name == "linalg.copy" ||
         name == "bufferization.clone";
}

bool isPlannedAnalogStoreCopy(memref::CopyOp copy) {
  Value value = copy.getSource();
  while (Operation *definition = value.getDefiningOp()) {
    if (definition->hasAttr("sculptor.memory.analog_store_valid_prefix"))
      return true;
    if (!isAliasOperation(definition) || definition->getNumOperands() == 0)
      return false;
    value = definition->getOperand(0);
  }
  return false;
}

bool isPlannedResultWrite(memref::CopyOp copy) {
  BlockArgument destination = getRootArgument(copy.getTarget());
  return destination && isFunctionDestination(destination);
}

memref::AllocOp getRootAllocation(Value value) {
  while (Operation *definition = value.getDefiningOp()) {
    if (auto allocation = dyn_cast<memref::AllocOp>(definition))
      return allocation;
    if (!isAliasOperation(definition) || definition->getNumOperands() == 0)
      return {};
    value = definition->getOperand(0);
  }
  return {};
}

struct StaticSubviewRegion {
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  int64_t elementCount = 0;
};

std::optional<StaticSubviewRegion>
getStaticSubviewRegion(memref::SubViewOp subview, memref::AllocOp allocation) {
  auto allocationType = allocation.getType();
  if (!allocationType.hasStaticShape() ||
      allocationType.getRank() !=
          static_cast<int64_t>(subview.getStaticOffsets().size()) ||
      llvm::is_contained(subview.getStaticOffsets(), ShapedType::kDynamic) ||
      llvm::is_contained(subview.getStaticSizes(), ShapedType::kDynamic) ||
      llvm::is_contained(subview.getStaticStrides(), ShapedType::kDynamic) ||
      llvm::any_of(subview.getStaticStrides(),
                   [](int64_t stride) { return stride != 1; }))
    return std::nullopt;

  StaticSubviewRegion region;
  region.offsets.assign(subview.getStaticOffsets().begin(),
                        subview.getStaticOffsets().end());
  region.sizes.assign(subview.getStaticSizes().begin(),
                      subview.getStaticSizes().end());
  region.elementCount = 1;
  for (int64_t dimension = 0; dimension < allocationType.getRank();
       ++dimension) {
    int64_t offset = region.offsets[dimension];
    int64_t size = region.sizes[dimension];
    std::optional<int64_t> end = llvm::checkedAdd(offset, size);
    if (offset < 0 || size <= 0 || !end ||
        *end > allocationType.getDimSize(dimension))
      return std::nullopt;
    std::optional<int64_t> elements =
        llvm::checkedMul(region.elementCount, size);
    if (!elements)
      return std::nullopt;
    region.elementCount = *elements;
  }
  return region;
}

bool regionsAreDisjoint(const StaticSubviewRegion &left,
                        const StaticSubviewRegion &right) {
  for (auto [leftOffset, leftSize, rightOffset, rightSize] :
       llvm::zip(left.offsets, left.sizes, right.offsets, right.sizes)) {
    if (leftOffset + leftSize <= rightOffset ||
        rightOffset + rightSize <= leftOffset)
      return true;
  }
  return false;
}

bool regionsExactlyCover(ArrayRef<StaticSubviewRegion> regions,
                         int64_t elementCount) {
  int64_t coveredElements = 0;
  for (auto [index, region] : llvm::enumerate(regions)) {
    if (!checkedAddTo(coveredElements, region.elementCount))
      return false;
    for (const StaticSubviewRegion &previous : regions.take_front(index))
      if (!regionsAreDisjoint(previous, region))
        return false;
  }
  return coveredElements == elementCount;
}

bool isPlannedLocalAssemblyCopy(
    memref::CopyOp candidate,
    DenseMap<Operation *, bool> &localAssemblyClassifications) {
  memref::AllocOp allocation = getRootAllocation(candidate.getTarget());
  if (!allocation || !allocation.getType().hasStaticShape())
    return false;
  Operation *allocationOperation = allocation.getOperation();
  if (auto cached = localAssemblyClassifications.find(allocationOperation);
      cached != localAssemblyClassifications.end())
    return cached->second;

  SmallVector<StaticSubviewRegion> regions;
  bool valid = true;
  func::FuncOp function = candidate->getParentOfType<func::FuncOp>();
  if (!function) {
    localAssemblyClassifications[allocationOperation] = false;
    return false;
  }
  function.walk([&](memref::CopyOp copy) {
    if (getRootAllocation(copy.getTarget()) != allocation)
      return;
    auto subview = getRootSubview(copy.getTarget());
    std::optional<StaticSubviewRegion> region =
        subview ? getStaticSubviewRegion(subview, allocation) : std::nullopt;
    if (!region) {
      valid = false;
      return;
    }
    regions.push_back(std::move(*region));
  });
  if (!valid || regions.size() < 2) {
    localAssemblyClassifications[allocationOperation] = false;
    return false;
  }
  bool complete =
      regionsExactlyCover(regions, allocation.getType().getNumElements());
  localAssemblyClassifications[allocationOperation] = complete;
  return complete;
}

std::optional<StaticSubviewRegion>
getInitializationRegion(Operation *operation, memref::AllocOp allocation) {
  StringRef name = operation->getName().getStringRef();
  if ((name != "linalg.fill" && name != "linalg.map") ||
      operation->getNumOperands() == 0)
    return std::nullopt;
  Value output = operation->getOperand(operation->getNumOperands() - 1);
  if (getRootAllocation(output) != allocation)
    return std::nullopt;
  if (memref::SubViewOp subview = getRootSubview(output))
    return getStaticSubviewRegion(subview, allocation);
  if (output.getType() != allocation.getType())
    return std::nullopt;
  StaticSubviewRegion full;
  full.offsets.assign(allocation.getType().getRank(), 0);
  full.sizes.assign(allocation.getType().getShape().begin(),
                    allocation.getType().getShape().end());
  full.elementCount = allocation.getType().getNumElements();
  return full;
}

bool isPlannedPaddingCopy(memref::CopyOp copy) {
  memref::AllocOp allocation = getRootAllocation(copy.getTarget());
  memref::SubViewOp target = getRootSubview(copy.getTarget());
  if (!allocation || !target)
    return false;
  std::optional<StaticSubviewRegion> region =
      getStaticSubviewRegion(target, allocation);
  auto sourceType = dyn_cast<ShapedType>(copy.getSource().getType());
  if (!region || !sourceType || !sourceType.hasStaticShape() ||
      sourceType.getNumElements() != region->elementCount)
    return false;
  SmallVector<StaticSubviewRegion> initializedRegions{*region};
  for (Operation *previous = copy->getPrevNode(); previous;
       previous = previous->getPrevNode()) {
    std::optional<StaticSubviewRegion> initialized =
        getInitializationRegion(previous, allocation);
    if (!initialized)
      continue;
    if (initialized->elementCount == allocation.getType().getNumElements())
      return true;
    initializedRegions.push_back(std::move(*initialized));
  }
  return regionsExactlyCover(initializedRegions,
                             allocation.getType().getNumElements());
}

bool isBootRoutine(Operation *operation) {
  auto function = operation->getParentOfType<func::FuncOp>();
  auto kind = function ? function->getAttrOfType<StringAttr>(
                             "sculptor.deployment.routine_kind")
                       : StringAttr();
  return kind && kind.getValue() == "boot";
}

bool isPureCopyLoop(Operation *operation) {
  if (operation->getName().getStringRef() != "scf.for")
    return false;
  bool hasStore = false;
  bool pure = true;
  operation->walk([&](Operation *nested) {
    if (nested == operation ||
        nested->getName().getStringRef() != "memref.store")
      return;
    hasStore = true;
    if (nested->getNumOperands() == 0 ||
        !nested->getOperand(0).getDefiningOp<memref::LoadOp>())
      pure = false;
  });
  return hasStore && pure;
}

LogicalResult collectAudit(ModuleOp module, AuditSummary &summary) {
  auto assemblyDestinations = collectAssemblyDestinationViews(module);
  if (failed(assemblyDestinations))
    return failure();

  int64_t allocationOrdinal = 0;
  int64_t copyOrdinal = 0;
  DenseMap<Operation *, bool> localAssemblyClassifications;
  LogicalResult status = success();
  module.walk([&](Operation *operation) {
    if (failed(status))
      return WalkResult::interrupt();
    StringRef name = operation->getName().getStringRef();
    summary.subviewCount += name == "memref.subview";
    summary.pureCopyLoopCount += isPureCopyLoop(operation);

    if (isa<memref::AllocOp, memref::AllocaOp>(operation)) {
      AllocationAudit audit = analyzeAllocation(operation, allocationOrdinal++);
      if (audit.staticBytes &&
          !checkedAddTo(summary.staticAllocationBytes, *audit.staticBytes)) {
        status = operation->emitError(
            "tile bufferization audit allocation-byte overflow");
        return WalkResult::interrupt();
      }
      if (audit.approved())
        ++summary.approvedLocalAllocationCount;
      else
        ++summary.unplannedAllocationCount;
      summary.escapingAllocationCount += audit.escapesRoutine;
      summary.missingDeallocationCount +=
          !audit.stack && !audit.hasDeallocation;
      summary.routineLifetimeAllocationCount += audit.routineLifetime;
      summary.allocations.push_back(audit);
    }

    if (isCopyOperation(operation)) {
      CopyAudit audit;
      audit.operation = operation;
      audit.ordinal = copyOrdinal++;
      audit.staticBytes =
          operation->getNumOperands() == 0
              ? std::nullopt
              : getStaticByteSize(operation->getOperand(0).getType());
      if (auto copy = dyn_cast<memref::CopyOp>(operation)) {
        audit.plannedAssembly =
            isPlannedAssemblyCopy(copy, *assemblyDestinations);
        audit.plannedAnalogStore = isPlannedAnalogStoreCopy(copy);
        audit.plannedPadding = isPlannedPaddingCopy(copy);
        audit.plannedResultWrite = isPlannedResultWrite(copy);
        audit.plannedLocalAssembly =
            isPlannedLocalAssemblyCopy(copy, localAssemblyClassifications);
      }
      audit.plannedBootStaging = isBootRoutine(operation);
      std::optional<int64_t> targetBytes =
          operation->getNumOperands() < 2
              ? std::nullopt
              : getStaticByteSize(operation->getOperand(1).getType());
      audit.fullTensor = audit.staticBytes && targetBytes &&
                         *audit.staticBytes == *targetBytes;
      if (audit.staticBytes &&
          !checkedAddTo(summary.staticCopyBytes, *audit.staticBytes)) {
        status =
            operation->emitError("tile bufferization audit copy-byte overflow");
        return WalkResult::interrupt();
      }
      if (audit.plannedAssembly) {
        ++summary.plannedAssemblyCopyCount;
        if (audit.staticBytes && !checkedAddTo(summary.plannedAssemblyCopyBytes,
                                               *audit.staticBytes)) {
          status = operation->emitError(
              "tile bufferization audit assembly-copy overflow");
          return WalkResult::interrupt();
        }
      } else if (audit.plannedBootStaging) {
        ++summary.plannedBootStagingCopyCount;
        if (audit.staticBytes &&
            !checkedAddTo(summary.plannedBootStagingCopyBytes,
                          *audit.staticBytes)) {
          status = operation->emitError(
              "tile bufferization audit boot-staging copy overflow");
          return WalkResult::interrupt();
        }
      } else if (audit.plannedAnalogStore) {
        ++summary.plannedAnalogStoreCopyCount;
        if (audit.staticBytes &&
            !checkedAddTo(summary.plannedAnalogStoreCopyBytes,
                          *audit.staticBytes)) {
          status = operation->emitError(
              "tile bufferization audit analog-store-copy overflow");
          return WalkResult::interrupt();
        }
      } else if (audit.plannedPadding) {
        ++summary.plannedPaddingCopyCount;
        if (audit.staticBytes && !checkedAddTo(summary.plannedPaddingCopyBytes,
                                               *audit.staticBytes)) {
          status = operation->emitError(
              "tile bufferization audit padding-copy overflow");
          return WalkResult::interrupt();
        }
      } else if (audit.plannedResultWrite) {
        ++summary.plannedResultWriteCopyCount;
        if (audit.staticBytes &&
            !checkedAddTo(summary.plannedResultWriteCopyBytes,
                          *audit.staticBytes)) {
          status = operation->emitError(
              "tile bufferization audit result-write-copy overflow");
          return WalkResult::interrupt();
        }
      } else if (audit.plannedLocalAssembly) {
        ++summary.plannedLocalAssemblyCopyCount;
        if (audit.staticBytes &&
            !checkedAddTo(summary.plannedLocalAssemblyCopyBytes,
                          *audit.staticBytes)) {
          status = operation->emitError(
              "tile bufferization audit local-assembly-copy overflow");
          return WalkResult::interrupt();
        }
      } else {
        ++summary.unplannedCopyCount;
        summary.unplannedFullTensorCopyCount += audit.fullTensor;
      }
      summary.copies.push_back(audit);
    }
    return WalkResult::advance();
  });
  return status;
}

DictionaryAttr buildAllocationRecord(Builder &builder,
                                     const AllocationAudit &audit) {
  func::FuncOp function = audit.operation->getParentOfType<func::FuncOp>();
  StringRef classification =
      audit.approved()
          ? "approved_local_temporary"
          : (audit.escapesRoutine ? "unplanned_escape"
                                  : "unplanned_missing_deallocation");
  return builder.getDictionaryAttr({
      builder.getNamedAttr(
          "routine", builder.getStringAttr(function ? function.getName() : "")),
      builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(audit.ordinal)),
      builder.getNamedAttr("class", builder.getStringAttr(classification)),
      builder.getNamedAttr("static_bytes", builder.getI64IntegerAttr(
                                               audit.staticBytes.value_or(-1))),
      builder.getNamedAttr("stack", builder.getBoolAttr(audit.stack)),
      builder.getNamedAttr("has_deallocation",
                           builder.getBoolAttr(audit.hasDeallocation)),
      builder.getNamedAttr("escapes_routine",
                           builder.getBoolAttr(audit.escapesRoutine)),
  });
}

StringRef getCopyClassification(const CopyAudit &audit) {
  if (audit.plannedAssembly)
    return "planned_assembly";
  if (audit.plannedBootStaging)
    return "planned_boot_staging";
  if (audit.plannedAnalogStore)
    return "planned_analog_store";
  if (audit.plannedPadding)
    return "planned_padding";
  if (audit.plannedResultWrite)
    return "planned_result_write";
  if (audit.plannedLocalAssembly)
    return "planned_local_assembly";
  return "unplanned";
}

DictionaryAttr buildCopyRecord(Builder &builder, const CopyAudit &audit) {
  func::FuncOp function = audit.operation->getParentOfType<func::FuncOp>();
  return builder.getDictionaryAttr({
      builder.getNamedAttr(
          "routine", builder.getStringAttr(function ? function.getName() : "")),
      builder.getNamedAttr("ordinal", builder.getI64IntegerAttr(audit.ordinal)),
      builder.getNamedAttr("class",
                           builder.getStringAttr(getCopyClassification(audit))),
      builder.getNamedAttr("static_bytes", builder.getI64IntegerAttr(
                                               audit.staticBytes.value_or(-1))),
      builder.getNamedAttr("full_tensor",
                           builder.getBoolAttr(audit.fullTensor)),
  });
}

DictionaryAttr buildAuditReport(Builder &builder, const AuditSummary &summary,
                                bool strict, int64_t coreId) {
  SmallVector<Attribute> allocations;
  for (const AllocationAudit &audit : summary.allocations)
    allocations.push_back(buildAllocationRecord(builder, audit));
  SmallVector<Attribute> copies;
  for (const CopyAudit &audit : summary.copies)
    copies.push_back(buildCopyRecord(builder, audit));

  auto i64 = [&](int64_t value) { return builder.getI64IntegerAttr(value); };
  return builder.getDictionaryAttr({
      builder.getNamedAttr("schema_version", i64(1)),
      builder.getNamedAttr("core_id", i64(coreId)),
      builder.getNamedAttr("strict", builder.getBoolAttr(strict)),
      builder.getNamedAttr("allocation_count", i64(summary.allocations.size())),
      builder.getNamedAttr("static_allocation_bytes",
                           i64(summary.staticAllocationBytes)),
      builder.getNamedAttr("approved_local_allocation_count",
                           i64(summary.approvedLocalAllocationCount)),
      builder.getNamedAttr("unplanned_allocation_count",
                           i64(summary.unplannedAllocationCount)),
      builder.getNamedAttr("escaping_allocation_count",
                           i64(summary.escapingAllocationCount)),
      builder.getNamedAttr("missing_deallocation_count",
                           i64(summary.missingDeallocationCount)),
      builder.getNamedAttr("routine_lifetime_allocation_count",
                           i64(summary.routineLifetimeAllocationCount)),
      builder.getNamedAttr("copy_count", i64(summary.copies.size())),
      builder.getNamedAttr("static_copy_bytes", i64(summary.staticCopyBytes)),
      builder.getNamedAttr("planned_assembly_copy_count",
                           i64(summary.plannedAssemblyCopyCount)),
      builder.getNamedAttr("planned_assembly_copy_bytes",
                           i64(summary.plannedAssemblyCopyBytes)),
      builder.getNamedAttr("planned_boot_staging_copy_count",
                           i64(summary.plannedBootStagingCopyCount)),
      builder.getNamedAttr("planned_boot_staging_copy_bytes",
                           i64(summary.plannedBootStagingCopyBytes)),
      builder.getNamedAttr("planned_analog_store_copy_count",
                           i64(summary.plannedAnalogStoreCopyCount)),
      builder.getNamedAttr("planned_analog_store_copy_bytes",
                           i64(summary.plannedAnalogStoreCopyBytes)),
      builder.getNamedAttr("planned_padding_copy_count",
                           i64(summary.plannedPaddingCopyCount)),
      builder.getNamedAttr("planned_padding_copy_bytes",
                           i64(summary.plannedPaddingCopyBytes)),
      builder.getNamedAttr("planned_result_write_copy_count",
                           i64(summary.plannedResultWriteCopyCount)),
      builder.getNamedAttr("planned_result_write_copy_bytes",
                           i64(summary.plannedResultWriteCopyBytes)),
      builder.getNamedAttr("planned_local_assembly_copy_count",
                           i64(summary.plannedLocalAssemblyCopyCount)),
      builder.getNamedAttr("planned_local_assembly_copy_bytes",
                           i64(summary.plannedLocalAssemblyCopyBytes)),
      builder.getNamedAttr("unplanned_copy_count",
                           i64(summary.unplannedCopyCount)),
      builder.getNamedAttr("unplanned_full_tensor_copy_count",
                           i64(summary.unplannedFullTensorCopyCount)),
      builder.getNamedAttr("pure_copy_loop_count",
                           i64(summary.pureCopyLoopCount)),
      builder.getNamedAttr("subview_count", i64(summary.subviewCount)),
      builder.getNamedAttr("allocations", builder.getArrayAttr(allocations)),
      builder.getNamedAttr("copies", builder.getArrayAttr(copies)),
  });
}

LogicalResult appendRoutineAllocationLifetimes(ModuleOp module,
                                               const AuditSummary &summary,
                                               int64_t coreId) {
  if (!module->hasAttr(tile_memory::kPlanVersionAttrName))
    return success();
  if (coreId < 0)
    return module.emitError(
        "tile memory plan requires a nonnegative physical tile ID");

  auto lifetimeArray =
      module->getAttrOfType<ArrayAttr>(tile_memory::kLifetimesAttrName);
  auto completionArray =
      module->getAttrOfType<ArrayAttr>(tile_memory::kCompletionEventsAttrName);
  if (!lifetimeArray || !completionArray)
    return module.emitError("tile memory plan has no lifetime or event table");

  SmallVector<Attribute> lifetimes;
  int64_t nextId = 0;
  for (Attribute value : lifetimeArray) {
    auto lifetime = dyn_cast<TileMemoryLifetimeAttr>(value);
    if (!lifetime)
      return module.emitError(
          "tile memory lifetime table has an invalid entry");
    if (lifetime.getSubjectKind() !=
        MemoryLifetimeSubjectKind::RoutineAllocation) {
      lifetimes.push_back(lifetime);
      nextId = std::max(nextId, lifetime.getId().getInt() + 1);
    }
  }

  DenseMap<int64_t, int64_t> routineStart;
  DenseMap<int64_t, int64_t> routineComplete;
  for (Attribute value : completionArray) {
    auto completion = dyn_cast<TileMemoryCompletionEventAttr>(value);
    if (!completion)
      return module.emitError(
          "tile memory completion table has an invalid entry");
    if (completion.getKind() == MemoryCompletionKind::RoutineStart)
      routineStart[completion.getRoutine().getInt()] =
          completion.getId().getInt();
    else if (completion.getKind() == MemoryCompletionKind::RoutineComplete)
      routineComplete[completion.getRoutine().getInt()] =
          completion.getId().getInt();
  }

  Builder builder(module.getContext());
  for (const AllocationAudit &audit : summary.allocations) {
    if (!audit.staticBytes)
      continue;
    func::FuncOp function = audit.operation->getParentOfType<func::FuncOp>();
    auto routineId = function ? function->getAttrOfType<IntegerAttr>(
                                    "sculptor.deployment.global_routine_id")
                              : IntegerAttr();
    if (!routineId)
      continue;
    auto start = routineStart.find(routineId.getInt());
    auto complete = routineComplete.find(routineId.getInt());
    if (start == routineStart.end() || complete == routineComplete.end())
      return audit.operation->emitError(
          "routine allocation has no completion-event lifetime");
    lifetimes.push_back(TileMemoryLifetimeAttr::get(
        module.getContext(), builder.getI64IntegerAttr(nextId++),
        MemoryLifetimeSubjectKind::RoutineAllocation,
        MemoryLifetimeStorage::RoutineLocal, builder.getI64IntegerAttr(-1),
        routineId, builder.getI64IntegerAttr(audit.ordinal),
        builder.getI64IntegerAttr(coreId),
        builder.getI64IntegerAttr(*audit.staticBytes),
        builder.getI64IntegerAttr(16), builder.getI64IntegerAttr(-1),
        builder.getArrayAttr({}),
        builder.getI64ArrayAttr({start->second, complete->second})));
  }
  module->setAttr(tile_memory::kLifetimesAttrName,
                  builder.getArrayAttr(lifetimes));
  if (failed(tile_memory::rebuildTileMemoryInterference(module)))
    return failure();
  if (failed(tile_memory::rebuildTileMemoryCapacitySummary(module)))
    return failure();
  if (failed(tile_memory::validateTileMemoryCapacity(module)))
    return failure();
  return tile_memory::verifyTileMemoryPlan(module);
}

LogicalResult enforceStrictAudit(const AuditSummary &summary) {
  for (const AllocationAudit &audit : summary.allocations) {
    if (audit.escapesRoutine)
      return audit.operation->emitError(
          "strict tile-memory audit found an allocation that escapes its "
          "routine");
    if (!audit.stack && !audit.hasDeallocation)
      return audit.operation->emitError(
          "strict tile-memory audit found an allocation without a "
          "deallocation");
  }
  for (const CopyAudit &audit : summary.copies)
    if (!audit.planned() && audit.fullTensor)
      return audit.operation->emitError(
          "strict tile-memory audit found an unplanned full-tensor copy");
  return success();
}

} // namespace

namespace mlir::sculptor {

void AuditTileBufferizationPass::runOnOperation() {
  ModuleOp module = getOperation();
  AuditSummary summary;
  if (failed(collectAudit(module, summary))) {
    signalPassFailure();
    return;
  }
  Builder builder(module.getContext());
  int64_t coreId = -1;
  if (auto value =
          module->getAttrOfType<IntegerAttr>("sculptor.runtime.core_id"))
    coreId = value.getInt();
  else if (auto value = module->getAttrOfType<IntegerAttr>(
               "sculptor.deployment.physical_tile_id"))
    coreId = value.getInt();
  DictionaryAttr report = buildAuditReport(builder, summary, strict, coreId);
  module->setAttr(kAuditAttrName, report);
  if (printReport) {
    llvm::errs() << "SCULPTOR_TILE_BUFFERIZATION_AUDIT ";
    report.print(llvm::errs());
    llvm::errs() << '\n';
  }
  if (failed(appendRoutineAllocationLifetimes(module, summary, coreId)) ||
      (strict && failed(enforceStrictAudit(summary))) || failed(verify(module)))
    signalPassFailure();
}

void registerAuditTileBufferizationPass() {
  PassRegistration<AuditTileBufferizationPass>();
}

} // namespace mlir::sculptor
