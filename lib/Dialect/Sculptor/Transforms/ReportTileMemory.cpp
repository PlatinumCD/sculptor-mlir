#include "sculptor-mlir/Dialect/Sculptor/Transforms/ReportTileMemory.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDeploymentAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileMemoryReportAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileScratchpadAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeResourceUtils.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/tile_runtime/TileRuntimeTaskKinds.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace {

using namespace mlir;
using namespace mlir::sculptor;

namespace deployment_attrs = mlir::sculptor::deployment_attrs;
namespace report_attrs = mlir::sculptor::tile_memory_report_attrs;
namespace runtime_attrs = mlir::sculptor::tile_runtime_attrs;
namespace scratchpad_attrs = mlir::sculptor::scratchpad_attrs;

constexpr int64_t kReportSchemaVersion = 1;
constexpr int64_t kTaskTableEntryBytes = 24;
constexpr int64_t kRouteTableEntryBytes = 48;
constexpr int64_t kModelIOTableEntryBytes = 24;
constexpr int64_t kResourceTableEntryBytes = 48;
constexpr int64_t kTaskBindingEntryBytes = 32;
constexpr int64_t kTaskBindingValueBytes = 4;
constexpr int64_t kStaticIdIndexEntryBytes = 8;
constexpr int64_t kStaticTaskEntryBytes = 20;
constexpr int64_t kStaticResourceEntryBytes = 16;
constexpr int64_t kStaticRuntimeValueBytes = 4;
constexpr int64_t kDimensionValueBytes = 8;
constexpr int64_t kDMADescriptorBytes = 64;
constexpr int64_t kMemRefDescriptorHeaderBytes = 24;
constexpr int64_t kMemRefDescriptorRankBytes = 16;

struct ResourceMetric {
  int64_t count = 0;
  int64_t bytes = 0;
};

struct MemoryMetrics {
  int64_t coreId = -1;
  ResourceMetric modelInputs;
  ResourceMetric modelOutputs;
  ResourceMetric intermediates;
  ResourceMetric persistent;
  ResourceMetric routeInputs;
  ResourceMetric routeOutputs;
  int64_t resourceCount = 0;
  int64_t resourcePayloadBytes = 0;
  int64_t resourceDimensionCount = 0;
  int64_t taskCount = 0;
  int64_t bootTaskCount = 0;
  int64_t dispatchTaskCount = 0;
  int64_t taskBindingDataCount = 0;
  int64_t routeRecordCount = 0;
  int64_t modelIORecordCount = 0;
  int64_t workspaceBytes = 0;
  int64_t scratchpadBytes = 0;
  int64_t scratchpadDescriptorCount = 0;
  int64_t abiTableBytes = 0;
  int64_t runtimeDescriptorBytes = 0;
  int64_t staticAllocSiteCount = 0;
  int64_t dynamicAllocSiteCount = 0;
  int64_t staticAllocBytes = 0;
  int64_t maxRoutineStaticAllocBytes = 0;
  int64_t mallocCallCount = 0;
  int64_t freeCallCount = 0;
  int64_t copyOpCount = 0;
  int64_t knownCopyBytes = 0;
  int64_t unknownCopySizeCount = 0;
  int64_t subviewCount = 0;
  int64_t assemblyCount = 0;
  int64_t assemblyBytes = 0;
  int64_t viewCount = 0;
  int64_t conservativePeakLiveBytes = 0;
  bool hasTaskGraphResources = false;
  bool hasFinalizedLayout = false;
  bool hasBufferizedOperations = false;
  bool hasLLVMFunctions = false;
};

bool checkedAddTo(int64_t &target, int64_t value) {
  std::optional<int64_t> sum = llvm::checkedAdd(target, value);
  if (!sum)
    return false;
  target = *sum;
  return true;
}

std::optional<int64_t> checkedProduct(int64_t lhs, int64_t rhs) {
  return llvm::checkedMul(lhs, rhs);
}

std::optional<int64_t> getStaticByteSize(Type type) {
  if (auto resource = dyn_cast<TaskResourceType>(type))
    type = resource.getValueType();

  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return std::nullopt;

  Type elementType = shaped.getElementType();
  if (!elementType.isIntOrFloat())
    return std::nullopt;
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::nullopt;

  int64_t elements = shaped.getNumElements();
  if (elements < 0)
    return std::nullopt;
  return checkedProduct(elements, static_cast<int64_t>(bitWidth / 8));
}

int64_t getRank(Type type) {
  if (auto resource = dyn_cast<TaskResourceType>(type))
    type = resource.getValueType();
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && shaped.hasRank() ? shaped.getRank() : 0;
}

bool isAllocation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "memref.alloc" || name == "memref.alloca";
}

bool isCopy(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "memref.copy" || name == "linalg.copy" ||
         name == "bufferization.clone" || name == "tensor.insert_slice";
}

std::optional<StringRef> getCalleeName(Operation *op) {
  if (auto callee = op->getAttrOfType<FlatSymbolRefAttr>("callee"))
    return callee.getValue();
  if (auto callee = op->getAttrOfType<SymbolRefAttr>("callee"))
    return callee.getRootReference().getValue();
  return std::nullopt;
}

int64_t getArrayAttrSize(Operation *op, StringRef name) {
  auto values = op->getAttrOfType<ArrayAttr>(name);
  return values ? static_cast<int64_t>(values.size()) : 0;
}

LogicalResult recordResource(Operation *op, MemoryMetrics &metrics,
                             ResourceMetric &category) {
  if (op->getNumResults() != 1)
    return op->emitError("memory report expected one resource result");
  FailureOr<int64_t> byteSize = getTaskResourceByteSize(op->getResult(0));
  if (failed(byteSize) || *byteSize < 0)
    return op->emitError(
        "memory report requires a statically sized task resource");

  ++category.count;
  ++metrics.resourceCount;
  if (!checkedAddTo(category.bytes, *byteSize) ||
      !checkedAddTo(metrics.resourcePayloadBytes, *byteSize) ||
      !checkedAddTo(metrics.resourceDimensionCount,
                    getRank(op->getResult(0).getType()))) {
    return op->emitError("tile memory report metric overflow");
  }
  metrics.hasTaskGraphResources = true;
  return success();
}

LogicalResult collectResourceAndTaskMetrics(ModuleOp module,
                                            MemoryMetrics &metrics) {
  LogicalResult status = success();
  module.walk([&](Operation *op) {
    if (failed(status))
      return WalkResult::interrupt();

    if (isa<TaskGraphInputOp>(op))
      status = recordResource(op, metrics, metrics.modelInputs);
    else if (isa<TaskGraphOutputOp>(op))
      status = recordResource(op, metrics, metrics.modelOutputs);
    else if (isa<TaskGraphIntermediateOp>(op))
      status = recordResource(op, metrics, metrics.intermediates);
    else if (isa<TaskGraphPersistentOp>(op))
      status = recordResource(op, metrics, metrics.persistent);
    else if (isa<TaskGraphRouteInputOp>(op))
      status = recordResource(op, metrics, metrics.routeInputs);
    else if (isa<TaskGraphRouteOutputOp>(op))
      status = recordResource(op, metrics, metrics.routeOutputs);

    if (auto task = dyn_cast<TaskCreateOp>(op)) {
      ++metrics.taskCount;
      bool boot = tile_runtime::isMatrixSetupTask(task);
      if (boot) {
        ++metrics.bootTaskCount;
      } else {
        ++metrics.dispatchTaskCount;
        int64_t values = task.getInputs().size() + task.getOutputs().size();
        for (Value dependency : task.getDependencies()) {
          auto producer = dependency.getDefiningOp<TaskCreateOp>();
          if (!producer || !tile_runtime::isMatrixSetupTask(producer))
            ++values;
        }
        if (!checkedAddTo(metrics.taskBindingDataCount, values)) {
          status = task.emitError("tile memory report metric overflow");
          return WalkResult::interrupt();
        }
      }
    }

    if (auto workspace = op->getAttrOfType<IntegerAttr>(
            runtime_attrs::kTaskGraphWorkspaceSizeAttrName)) {
      if (workspace.getInt() < 0) {
        status = op->emitError("memory report found negative workspace size");
        return WalkResult::interrupt();
      }
      metrics.workspaceBytes =
          std::max(metrics.workspaceBytes, workspace.getInt());
      metrics.hasFinalizedLayout = true;
    }
    if (auto scratchpad = op->getAttrOfType<IntegerAttr>(
            scratchpad_attrs::kScratchpadRequiredBytesAttrName)) {
      if (scratchpad.getInt() < 0) {
        status = op->emitError("memory report found negative scratchpad size");
        return WalkResult::interrupt();
      }
      metrics.scratchpadBytes =
          std::max(metrics.scratchpadBytes, scratchpad.getInt());
    }
    metrics.scratchpadDescriptorCount =
        std::max(metrics.scratchpadDescriptorCount,
                 getArrayAttrSize(
                     op, scratchpad_attrs::kScratchpadDMADescriptorsAttrName));
    return WalkResult::advance();
  });
  return status;
}

LogicalResult collectOperationMetrics(ModuleOp module, MemoryMetrics &metrics) {
  LogicalResult status = success();
  module.walk([&](Operation *op) {
    if (failed(status))
      return WalkResult::interrupt();

    StringRef name = op->getName().getStringRef();
    metrics.hasLLVMFunctions |= name == "llvm.func";
    metrics.hasBufferizedOperations |= name.starts_with("memref.");

    if (isAllocation(op)) {
      std::optional<int64_t> bytes =
          op->getNumResults() == 1
              ? getStaticByteSize(op->getResult(0).getType())
              : std::nullopt;
      if (bytes) {
        ++metrics.staticAllocSiteCount;
        if (!checkedAddTo(metrics.staticAllocBytes, *bytes)) {
          status = op->emitError("tile memory report allocation overflow");
          return WalkResult::interrupt();
        }
      } else {
        ++metrics.dynamicAllocSiteCount;
      }
    }

    if (isCopy(op)) {
      ++metrics.copyOpCount;
      std::optional<int64_t> bytes =
          op->getNumOperands() > 0
              ? getStaticByteSize(op->getOperand(0).getType())
              : std::nullopt;
      if (bytes) {
        if (!checkedAddTo(metrics.knownCopyBytes, *bytes)) {
          status = op->emitError("tile memory report copy-byte overflow");
          return WalkResult::interrupt();
        }
      } else {
        ++metrics.unknownCopySizeCount;
      }
    }

    metrics.subviewCount += name == "memref.subview";
    metrics.viewCount += op->hasAttr("sculptor.memory.view_id");

    if (name == "tensor.empty" && op->getNumResults() == 1) {
      bool feedsInsertSlice =
          llvm::any_of(op->getResult(0).getUsers(), [](Operation *user) {
            return user->getName().getStringRef() == "tensor.insert_slice";
          });
      if (feedsInsertSlice) {
        ++metrics.assemblyCount;
        if (std::optional<int64_t> bytes =
                getStaticByteSize(op->getResult(0).getType())) {
          if (!checkedAddTo(metrics.assemblyBytes, *bytes)) {
            status = op->emitError("tile memory report assembly overflow");
            return WalkResult::interrupt();
          }
        }
      }
    }

    if (std::optional<StringRef> callee = getCalleeName(op)) {
      metrics.mallocCallCount += *callee == "malloc";
      metrics.freeCallCount += *callee == "free";
    }
    return WalkResult::advance();
  });

  if (failed(status))
    return failure();

  for (Operation &topLevel : module.getBody()->getOperations()) {
    StringRef name = topLevel.getName().getStringRef();
    if (name != "func.func" && name != "llvm.func")
      continue;
    int64_t routineBytes = 0;
    topLevel.walk([&](Operation *nested) {
      if (!isAllocation(nested) || nested->getNumResults() != 1)
        return;
      std::optional<int64_t> bytes =
          getStaticByteSize(nested->getResult(0).getType());
      if (!bytes || !checkedAddTo(routineBytes, *bytes))
        routineBytes = std::numeric_limits<int64_t>::max();
    });
    metrics.maxRoutineStaticAllocBytes =
        std::max(metrics.maxRoutineStaticAllocBytes, routineBytes);
  }
  return success();
}

LogicalResult calculateDerivedMetrics(ModuleOp module, MemoryMetrics &metrics) {
  metrics.routeRecordCount =
      getArrayAttrSize(module.getOperation(),
                       deployment_attrs::kIncomingRoutesAttrName) +
      getArrayAttrSize(module.getOperation(),
                       deployment_attrs::kOutgoingRoutesAttrName);
  metrics.modelIORecordCount =
      getArrayAttrSize(module.getOperation(),
                       deployment_attrs::kModelInputsAttrName) +
      getArrayAttrSize(module.getOperation(),
                       deployment_attrs::kModelOutputsAttrName);

  int64_t tableBytes = 0;
  auto addProduct = [&](int64_t count, int64_t width) {
    std::optional<int64_t> bytes = checkedProduct(count, width);
    return bytes && checkedAddTo(tableBytes, *bytes);
  };
  if (!addProduct(metrics.resourceCount, kResourceTableEntryBytes) ||
      !addProduct(metrics.resourceDimensionCount, kDimensionValueBytes) ||
      !addProduct(metrics.taskCount, kTaskTableEntryBytes) ||
      !addProduct(metrics.dispatchTaskCount, kTaskBindingEntryBytes) ||
      !addProduct(metrics.taskBindingDataCount, kTaskBindingValueBytes) ||
      !addProduct(metrics.routeRecordCount, kRouteTableEntryBytes) ||
      !addProduct(metrics.modelIORecordCount, kModelIOTableEntryBytes) ||
      !addProduct(metrics.scratchpadDescriptorCount, kDMADescriptorBytes) ||
      !addProduct(metrics.dispatchTaskCount, kStaticIdIndexEntryBytes) ||
      !addProduct(metrics.routeRecordCount, kStaticIdIndexEntryBytes) ||
      !addProduct(metrics.dispatchTaskCount, kStaticTaskEntryBytes) ||
      !addProduct(metrics.resourceCount, kStaticResourceEntryBytes) ||
      !addProduct(metrics.taskBindingDataCount,
                  kStaticRuntimeValueBytes)) {
    return module.emitError("tile memory report ABI table overflow");
  }
  metrics.abiTableBytes = tableBytes;

  int64_t descriptorBytes = 0;
  // Runtime memref descriptors are separate from immutable ABI tables.
  if (std::optional<int64_t> headers =
          checkedProduct(metrics.resourceCount, kMemRefDescriptorHeaderBytes))
    descriptorBytes = *headers;
  else
    return module.emitError("tile memory report descriptor overflow");
  std::optional<int64_t> rankBytes = checkedProduct(
      metrics.resourceDimensionCount, kMemRefDescriptorRankBytes);
  if (!rankBytes || !checkedAddTo(descriptorBytes, *rankBytes))
    return module.emitError("tile memory report descriptor overflow");
  metrics.runtimeDescriptorBytes = descriptorBytes;

  int64_t peak = metrics.workspaceBytes;
  if (!checkedAddTo(peak, metrics.runtimeDescriptorBytes) ||
      !checkedAddTo(peak, metrics.maxRoutineStaticAllocBytes))
    return module.emitError("tile memory report peak-byte overflow");
  metrics.conservativePeakLiveBytes = peak;
  return success();
}

StringRef inferStage(const MemoryMetrics &metrics) {
  if (metrics.hasLLVMFunctions)
    return "llvm";
  if (metrics.hasBufferizedOperations)
    return "bufferized";
  if (metrics.hasFinalizedLayout)
    return "finalized";
  if (metrics.hasTaskGraphResources)
    return "runtime-graph";
  return "unknown";
}

DictionaryAttr buildReport(Builder &builder, StringRef stage,
                           const MemoryMetrics &metrics) {
  SmallVector<NamedAttribute> fields;
  auto addI64 = [&](StringRef name, int64_t value) {
    fields.push_back(
        builder.getNamedAttr(name, builder.getI64IntegerAttr(value)));
  };
  auto addBool = [&](StringRef name, bool value) {
    fields.push_back(builder.getNamedAttr(name, builder.getBoolAttr(value)));
  };
  auto addResource = [&](StringRef prefix, const ResourceMetric &metric) {
    addI64((prefix + "_count").str(), metric.count);
    addI64((prefix + "_bytes").str(), metric.bytes);
  };

  addI64(report_attrs::kSchemaVersionFieldName, kReportSchemaVersion);
  fields.push_back(builder.getNamedAttr(report_attrs::kStageFieldName,
                                        builder.getStringAttr(stage)));
  addI64("core_id", metrics.coreId);
  addResource("model_input", metrics.modelInputs);
  addResource("model_output", metrics.modelOutputs);
  addResource("intermediate", metrics.intermediates);
  addResource("persistent", metrics.persistent);
  addResource("route_input", metrics.routeInputs);
  addResource("route_output", metrics.routeOutputs);
  addI64("resource_count", metrics.resourceCount);
  addI64("resource_payload_bytes", metrics.resourcePayloadBytes);
  addI64("resource_dimension_count", metrics.resourceDimensionCount);
  addI64("task_count", metrics.taskCount);
  addI64("boot_task_count", metrics.bootTaskCount);
  addI64("dispatch_task_count", metrics.dispatchTaskCount);
  addI64("task_binding_data_count", metrics.taskBindingDataCount);
  addI64("route_record_count", metrics.routeRecordCount);
  addI64("model_io_record_count", metrics.modelIORecordCount);
  addI64("workspace_bytes", metrics.workspaceBytes);
  addI64("scratchpad_bytes", metrics.scratchpadBytes);
  addI64("scratchpad_descriptor_count", metrics.scratchpadDescriptorCount);
  addI64("abi_table_bytes", metrics.abiTableBytes);
  addI64("runtime_descriptor_bytes", metrics.runtimeDescriptorBytes);
  addI64("static_alloc_site_count", metrics.staticAllocSiteCount);
  addI64("dynamic_alloc_site_count", metrics.dynamicAllocSiteCount);
  addI64("static_alloc_bytes", metrics.staticAllocBytes);
  addI64("max_routine_static_alloc_bytes", metrics.maxRoutineStaticAllocBytes);
  addI64("malloc_call_count", metrics.mallocCallCount);
  addI64("free_call_count", metrics.freeCallCount);
  addI64("copy_op_count", metrics.copyOpCount);
  addI64("known_copy_bytes", metrics.knownCopyBytes);
  addI64("unknown_copy_size_count", metrics.unknownCopySizeCount);
  addI64("subview_count", metrics.subviewCount);
  addI64("view_count", metrics.viewCount);
  addI64("assembly_count", metrics.assemblyCount);
  addI64("assembly_bytes", metrics.assemblyBytes);
  addI64("conservative_peak_live_bytes", metrics.conservativePeakLiveBytes);
  // Schema version 1 cannot prove complete peak memory until the unified
  // routine and runtime-resource interference model exists.
  addBool("peak_estimate_complete", false);
  return builder.getDictionaryAttr(fields);
}

} // namespace

namespace mlir::sculptor {

void ReportTileMemoryPass::runOnOperation() {
  ModuleOp module = getOperation();
  MemoryMetrics metrics;
  if (auto coreId = module->getAttrOfType<IntegerAttr>(
          runtime_attrs::kTaskCoreIdAttrName)) {
    metrics.coreId = coreId.getInt();
  } else if (auto tileId = module->getAttrOfType<IntegerAttr>(
                 "sculptor.deployment.physical_tile_id")) {
    metrics.coreId = tileId.getInt();
  }

  if (failed(collectResourceAndTaskMetrics(module, metrics)) ||
      failed(collectOperationMetrics(module, metrics)) ||
      failed(calculateDerivedMetrics(module, metrics))) {
    signalPassFailure();
    return;
  }

  StringRef configuredStage = stage.getValue();
  StringRef resolvedStage =
      configuredStage == "auto" ? inferStage(metrics) : configuredStage;
  if (resolvedStage.empty()) {
    module.emitError("tile memory report stage cannot be empty");
    signalPassFailure();
    return;
  }

  Builder builder(module.getContext());
  DictionaryAttr report = buildReport(builder, resolvedStage, metrics);
  SmallVector<Attribute> reports;
  if (auto existing =
          module->getAttrOfType<ArrayAttr>(report_attrs::kReportsAttrName)) {
    for (Attribute value : existing) {
      auto prior = dyn_cast<DictionaryAttr>(value);
      auto priorStage =
          prior ? prior.getAs<StringAttr>(report_attrs::kStageFieldName)
                : StringAttr();
      if (!priorStage || priorStage.getValue() != resolvedStage)
        reports.push_back(value);
    }
  }
  reports.push_back(report);
  module->setAttr(report_attrs::kReportsAttrName,
                  builder.getArrayAttr(reports));

  if (printReport) {
    llvm::errs() << "SCULPTOR_TILE_MEMORY_REPORT ";
    report.print(llvm::errs());
    llvm::errs() << '\n';
  }
}

void registerReportTileMemoryPass() {
  PassRegistration<ReportTileMemoryPass>();
}

} // namespace mlir::sculptor
