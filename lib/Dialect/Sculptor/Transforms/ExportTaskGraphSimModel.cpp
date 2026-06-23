#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExportTaskGraphSimModel.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <system_error>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;

bool isAnalogArrayOp(mlir::Operation *op) {
  llvm::StringRef opName = op->getName().getStringRef();
  return opName.starts_with("sculptor.array.") || opName.starts_with("analog.");
}

struct HardwareModel {
  int64_t numCores = 0;
  int64_t arraysPerCore = 0;
  std::string topology;
  int64_t meshRows = 0;
  int64_t meshCols = 0;
  int64_t numAnalogArrays = 0;
  llvm::SmallVector<int64_t> analogArrays;
};

struct SummaryModel {
  int64_t taskCount = 0;
  int64_t dependencyCount = 0;
  int64_t interCoreTransferBytes = 0;
  int64_t totalTransferCost = 0;
  int64_t totalDigitalOps = 0;
  int64_t numLogicalArrays = 0;
  llvm::SmallVector<int64_t> coreTransferBytes;
  llvm::SmallVector<int64_t> coreTransferCost;
  llvm::SmallVector<int64_t> logicalArrayToAnalogArray;
};

struct ResourceModel {
  mlir::Value value;
  mlir::Operation *op = nullptr;
  int64_t id = 0;
  std::string kind;
  std::string valueType;
  std::optional<int64_t> slot;
  std::optional<int64_t> byteSize;
  std::optional<int64_t> tempIndex;
  std::optional<int64_t> tempOffset;
  std::optional<int64_t> logicalArrayIndex;
  std::optional<int64_t> physicalArrayId;
  std::optional<int64_t> coreId;
  std::optional<int64_t> localArrayId;
};

struct AnalogOpModel {
  int64_t index = 0;
  std::string name;
};

struct TaskModel {
  mlir::sculptor::TaskCreateOp op;
  int64_t index = 0;
  std::string callee;
  std::string domain;
  std::string kind;
  std::string name;
  std::string sourceLayer;
  uint64_t sourceTaskOrdinal = 0;
  int64_t coreId = 0;
  int64_t digitalOps = 0;
  std::optional<int64_t> physicalArrayId;
  std::optional<int64_t> localArrayId;
  llvm::SmallVector<int64_t> inputResourceIds;
  llvm::SmallVector<int64_t> outputResourceIds;
  llvm::SmallVector<int64_t> dependencyTaskIndices;
  llvm::SmallVector<AnalogOpModel, 4> analogOps;
};

struct IoBoundaryModel {
  int64_t entryTaskId = 0;
  int64_t entryCore = 0;
  llvm::SmallVector<std::string, 4> entryEdges;
  int64_t exitTaskId = 0;
  int64_t exitCore = 0;
  llvm::SmallVector<std::string, 4> exitEdges;
  bool sharesEdge = false;
};

struct GraphModel {
  mlir::func::FuncOp func;
  std::string name;
  HardwareModel hardware;
  SummaryModel summary;
  std::optional<IoBoundaryModel> ioBoundary;
  llvm::SmallVector<ResourceModel, 0> resources;
  llvm::SmallVector<TaskModel, 0> tasks;
  llvm::DenseMap<mlir::Value, int64_t> resourceIdByValue;
  llvm::DenseMap<mlir::Value, int64_t> taskIndexByResult;
  llvm::DenseMap<mlir::Value, int64_t> producerTaskIndexByResource;
};

bool returnsTaskGraph(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         llvm::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

std::string stringifyType(mlir::Type type) {
  std::string result;
  llvm::raw_string_ostream os(result);
  type.print(os);
  return result;
}

mlir::FailureOr<int64_t> getRequiredI64Attr(mlir::Operation *op,
                                            llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr) {
    op->emitError("expected required attr '") << attrName << "'";
    return mlir::failure();
  }

  return attr.getInt();
}

std::optional<int64_t> getOptionalI64Attr(mlir::Operation *op,
                                          llvm::StringRef attrName) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName))
    return attr.getInt();
  return std::nullopt;
}

mlir::FailureOr<std::string> getRequiredStringAttr(mlir::Operation *op,
                                                   llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr) {
    op->emitError("expected required attr '") << attrName << "'";
    return mlir::failure();
  }

  return attr.getValue().str();
}

mlir::FailureOr<llvm::SmallVector<int64_t>>
getRequiredI64ArrayAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::ArrayAttr>(attrName);
  if (!attr) {
    op->emitError("expected required attr '") << attrName << "'";
    return mlir::failure();
  }

  llvm::SmallVector<int64_t> values;
  values.reserve(attr.size());
  for (mlir::Attribute element : attr) {
    auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(element);
    if (!intAttr) {
      op->emitError("expected attr '")
          << attrName << "' to contain only integer attrs";
      return mlir::failure();
    }
    values.push_back(intAttr.getInt());
  }

  return values;
}

mlir::FailureOr<HardwareModel> buildHardwareModel(mlir::func::FuncOp func) {
  HardwareModel hardware;
  auto numCores = getRequiredI64Attr(func, schedule_attrs::kNumCoresAttrName);
  auto arraysPerCore =
      getRequiredI64Attr(func, schedule_attrs::kArraysPerCoreAttrName);
  auto topology =
      getRequiredStringAttr(func, schedule_attrs::kTopologyAttrName);
  auto meshRows = getRequiredI64Attr(func, schedule_attrs::kMeshRowsAttrName);
  auto meshCols = getRequiredI64Attr(func, schedule_attrs::kMeshColsAttrName);
  auto numAnalogArrays =
      getRequiredI64Attr(func, schedule_attrs::kNumAnalogArraysAttrName);
  auto analogArrays =
      getRequiredI64ArrayAttr(func, schedule_attrs::kAnalogArraysAttrName);

  if (mlir::failed(numCores) || mlir::failed(arraysPerCore) ||
      mlir::failed(topology) || mlir::failed(meshRows) ||
      mlir::failed(meshCols) || mlir::failed(numAnalogArrays) ||
      mlir::failed(analogArrays))
    return mlir::failure();

  hardware.numCores = *numCores;
  hardware.arraysPerCore = *arraysPerCore;
  hardware.topology = std::move(*topology);
  hardware.meshRows = *meshRows;
  hardware.meshCols = *meshCols;
  hardware.numAnalogArrays = *numAnalogArrays;
  hardware.analogArrays = std::move(*analogArrays);

  if (hardware.numCores <= 0 || hardware.arraysPerCore <= 0) {
    func.emitError("expected positive scheduled core and array budgets");
    return mlir::failure();
  }
  if (hardware.topology != "mesh") {
    func.emitError("expected mesh topology for simulation model export");
    return mlir::failure();
  }
  if (hardware.meshRows <= 0 || hardware.meshCols <= 0 ||
      hardware.meshRows * hardware.meshCols != hardware.numCores) {
    func.emitError("expected mesh dimensions to match scheduled core count");
    return mlir::failure();
  }

  return hardware;
}

mlir::FailureOr<SummaryModel> buildSummaryModel(mlir::func::FuncOp func) {
  SummaryModel summary;
  auto taskCount = getRequiredI64Attr(func, schedule_attrs::kTaskCountAttrName);
  auto dependencyCount =
      getRequiredI64Attr(func, schedule_attrs::kDependencyCountAttrName);
  auto coreTransferBytes =
      getRequiredI64ArrayAttr(func, schedule_attrs::kCoreTransferBytesAttrName);
  auto interCoreTransferBytes =
      getRequiredI64Attr(func, schedule_attrs::kInterCoreTransferBytesAttrName);
  auto coreTransferCost =
      getRequiredI64ArrayAttr(func, schedule_attrs::kCoreTransferCostAttrName);
  auto totalTransferCost =
      getRequiredI64Attr(func, schedule_attrs::kTotalTransferCostAttrName);
  auto totalDigitalOps =
      getRequiredI64Attr(func, schedule_attrs::kTotalDigitalOpsAttrName);
  auto numLogicalArrays =
      getRequiredI64Attr(func, schedule_attrs::kNumLogicalArraysAttrName);
  auto logicalArrayToAnalogArray = getRequiredI64ArrayAttr(
      func, schedule_attrs::kLogicalArrayToAnalogArrayAttrName);

  if (mlir::failed(taskCount) || mlir::failed(dependencyCount) ||
      mlir::failed(coreTransferBytes) || mlir::failed(interCoreTransferBytes) ||
      mlir::failed(coreTransferCost) || mlir::failed(totalTransferCost) ||
      mlir::failed(totalDigitalOps) || mlir::failed(numLogicalArrays) ||
      mlir::failed(logicalArrayToAnalogArray))
    return mlir::failure();

  summary.taskCount = *taskCount;
  summary.dependencyCount = *dependencyCount;
  summary.coreTransferBytes = std::move(*coreTransferBytes);
  summary.interCoreTransferBytes = *interCoreTransferBytes;
  summary.coreTransferCost = std::move(*coreTransferCost);
  summary.totalTransferCost = *totalTransferCost;
  summary.totalDigitalOps = *totalDigitalOps;
  summary.numLogicalArrays = *numLogicalArrays;
  summary.logicalArrayToAnalogArray = std::move(*logicalArrayToAnalogArray);
  return summary;
}

std::optional<std::pair<mlir::Value, llvm::StringRef>>
getTaskGraphResource(mlir::Operation &op) {
  if (auto input = llvm::dyn_cast<mlir::sculptor::TaskGraphInputOp>(&op))
    return std::make_pair(input.getResult(), llvm::StringRef("input"));
  if (auto output = llvm::dyn_cast<mlir::sculptor::TaskGraphOutputOp>(&op))
    return std::make_pair(output.getResult(), llvm::StringRef("output"));
  if (auto temporary =
          llvm::dyn_cast<mlir::sculptor::TaskGraphTemporaryOp>(&op))
    return std::make_pair(temporary.getResult(), llvm::StringRef("temporary"));
  if (auto persistent =
          llvm::dyn_cast<mlir::sculptor::TaskGraphPersistentOp>(&op))
    return std::make_pair(persistent.getResult(),
                          llvm::StringRef("persistent"));
  return std::nullopt;
}

bool isLogicalArrayResource(mlir::Value resource) {
  auto resourceType =
      llvm::dyn_cast<mlir::sculptor::TaskResourceType>(resource.getType());
  return resourceType && llvm::isa<mlir::sculptor::LogicalArrayType>(
                             resourceType.getValueType());
}

std::string getResourceValueTypeString(mlir::Value resource) {
  auto resourceType =
      llvm::dyn_cast<mlir::sculptor::TaskResourceType>(resource.getType());
  if (!resourceType)
    return stringifyType(resource.getType());
  return stringifyType(resourceType.getValueType());
}

mlir::FailureOr<llvm::SmallVector<ResourceModel, 0>>
collectResources(mlir::func::FuncOp func, const HardwareModel &hardware,
                 const SummaryModel &summary,
                 llvm::DenseMap<mlir::Value, int64_t> &resourceIdByValue) {
  llvm::SmallVector<ResourceModel, 0> resources;

  for (mlir::Operation &op : func.getBody().front()) {
    std::optional<std::pair<mlir::Value, llvm::StringRef>> resourceInfo =
        getTaskGraphResource(op);
    if (!resourceInfo)
      continue;

    mlir::Value resource = resourceInfo->first;
    int64_t resourceId = static_cast<int64_t>(resources.size());
    if (!resourceIdByValue.try_emplace(resource, resourceId).second) {
      op.emitError("expected task graph resource value to be unique");
      return mlir::failure();
    }

    ResourceModel model;
    model.value = resource;
    model.op = &op;
    model.id = resourceId;
    model.kind = resourceInfo->second.str();
    model.valueType = getResourceValueTypeString(resource);
    model.slot = getOptionalI64Attr(&op, runtime_attrs::kResourceSlotAttrName);
    model.byteSize =
        getOptionalI64Attr(&op, runtime_attrs::kResourceByteSizeAttrName);
    model.tempIndex =
        getOptionalI64Attr(&op, runtime_attrs::kResourceTempIndexAttrName);
    model.tempOffset =
        getOptionalI64Attr(&op, runtime_attrs::kResourceTempOffsetAttrName);
    model.logicalArrayIndex =
        getOptionalI64Attr(&op, schedule_attrs::kLogicalArrayIndexAttrName);

    if (isLogicalArrayResource(resource) && model.logicalArrayIndex) {
      int64_t logicalIndex = *model.logicalArrayIndex;
      if (logicalIndex < 0 ||
          logicalIndex >=
              static_cast<int64_t>(summary.logicalArrayToAnalogArray.size())) {
        op.emitError("expected logical array index to reference scheduled "
                     "logical array placement");
        return mlir::failure();
      }
      int64_t physicalArrayId =
          summary.logicalArrayToAnalogArray[static_cast<size_t>(logicalIndex)];
      model.physicalArrayId = physicalArrayId;
      model.coreId = physicalArrayId / hardware.arraysPerCore;
      model.localArrayId = physicalArrayId % hardware.arraysPerCore;
    }

    resources.push_back(std::move(model));
  }

  return resources;
}

mlir::FailureOr<llvm::SmallVector<TaskModel, 0>> collectTasks(
    mlir::ModuleOp module, mlir::func::FuncOp func,
    const HardwareModel &hardware,
    const llvm::DenseMap<mlir::Value, int64_t> &resourceIdByValue,
    llvm::DenseMap<mlir::Value, int64_t> &taskIndexByResult,
    llvm::DenseMap<mlir::Value, int64_t> &producerTaskIndexByResource) {
  llvm::SmallVector<TaskModel, 0> tasks;
  llvm::DenseSet<int64_t> seenTaskIndices;

  for (mlir::Operation &op : func.getBody().front()) {
    auto taskOp = llvm::dyn_cast<mlir::sculptor::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    auto taskIndex =
        getRequiredI64Attr(taskOp, runtime_attrs::kTaskIndexAttrName);
    auto coreId =
        getRequiredI64Attr(taskOp, runtime_attrs::kTaskCoreIdAttrName);
    auto digitalOps =
        getRequiredI64Attr(taskOp, runtime_attrs::kTaskDigitalOpsAttrName);
    if (mlir::failed(taskIndex) || mlir::failed(coreId) ||
        mlir::failed(digitalOps))
      return mlir::failure();

    if (*taskIndex < 0 || !seenTaskIndices.insert(*taskIndex).second) {
      taskOp.emitError("expected unique non-negative task index");
      return mlir::failure();
    }
    if (*coreId < 0 || *coreId >= hardware.numCores) {
      taskOp.emitError("expected task core id to be inside scheduled core "
                       "budget");
      return mlir::failure();
    }

    TaskModel task;
    task.op = taskOp;
    task.index = *taskIndex;
    task.callee = taskOp.getCallee().str();
    task.domain = taskOp.getDomain().str();
    task.kind = taskOp.getTaskKind().str();
    task.name = taskOp.getTaskName().str();
    task.sourceLayer = taskOp.getSourceLayer().str();
    task.sourceTaskOrdinal = taskOp.getSourceTaskOrdinal();
    task.coreId = *coreId;
    task.digitalOps = *digitalOps;
    auto callee = module.lookupSymbol<mlir::func::FuncOp>(
        taskOp.getCalleeAttr().getValue());
    if (!callee) {
      taskOp.emitError("expected task callee '")
          << taskOp.getCalleeAttr().getValue()
          << "' to resolve to a func.func for simulation model export";
      return mlir::failure();
    }
    if (!callee.isDeclaration()) {
      int64_t analogOpIndex = 0;
      callee.walk([&](mlir::Operation *nestedOp) {
        if (!isAnalogArrayOp(nestedOp))
          return;
        task.analogOps.push_back(AnalogOpModel{
            analogOpIndex++, nestedOp->getName().getStringRef().str()});
      });
    }
    task.physicalArrayId =
        getOptionalI64Attr(taskOp, runtime_attrs::kTaskPhysicalArrayIdAttrName);
    if (task.physicalArrayId)
      task.localArrayId = *task.physicalArrayId % hardware.arraysPerCore;

    for (mlir::Value input : taskOp.getInputs()) {
      auto resourceIt = resourceIdByValue.find(input);
      if (resourceIt == resourceIdByValue.end()) {
        taskOp.emitError("expected every task input to reference a task graph "
                         "resource");
        return mlir::failure();
      }
      task.inputResourceIds.push_back(resourceIt->second);
    }

    for (mlir::Value output : taskOp.getOutputs()) {
      auto resourceIt = resourceIdByValue.find(output);
      if (resourceIt == resourceIdByValue.end()) {
        taskOp.emitError("expected every task output to reference a task graph "
                         "resource");
        return mlir::failure();
      }
      task.outputResourceIds.push_back(resourceIt->second);
      if (!producerTaskIndexByResource.try_emplace(output, *taskIndex).second) {
        taskOp.emitError("expected task graph resource to have one producer");
        return mlir::failure();
      }
    }

    taskIndexByResult.try_emplace(taskOp.getResult(), *taskIndex);
    tasks.push_back(std::move(task));
  }

  for (TaskModel &task : tasks) {
    for (mlir::Value dependency : task.op.getDependencies()) {
      auto dependencyIt = taskIndexByResult.find(dependency);
      if (dependencyIt == taskIndexByResult.end()) {
        task.op.emitError("expected task dependency to reference a task in "
                          "the same graph");
        return mlir::failure();
      }
      task.dependencyTaskIndices.push_back(dependencyIt->second);
    }
  }

  llvm::sort(tasks, [](const TaskModel &lhs, const TaskModel &rhs) {
    return lhs.index < rhs.index;
  });
  return tasks;
}

const ResourceModel *lookupResourceById(llvm::ArrayRef<ResourceModel> resources,
                                        int64_t resourceId);

bool taskTouchesResourceKind(const TaskModel &task,
                             llvm::ArrayRef<ResourceModel> resources,
                             llvm::ArrayRef<int64_t> resourceIds,
                             llvm::StringRef kind) {
  for (int64_t resourceId : resourceIds) {
    const ResourceModel *resource = lookupResourceById(resources, resourceId);
    if (resource && kind == resource->kind)
      return true;
  }
  return false;
}

llvm::SmallVector<std::string, 4>
getMeshEdgeMembership(int64_t coreId, const HardwareModel &hardware) {
  llvm::SmallVector<std::string, 4> edges;
  int64_t row = coreId / hardware.meshCols;
  int64_t col = coreId % hardware.meshCols;

  if (row == 0)
    edges.push_back("top");
  if (col == hardware.meshCols - 1)
    edges.push_back("right");
  if (row == hardware.meshRows - 1)
    edges.push_back("bottom");
  if (col == 0)
    edges.push_back("left");

  return edges;
}

bool shareMeshEdge(llvm::ArrayRef<std::string> lhs,
                   llvm::ArrayRef<std::string> rhs) {
  for (const std::string &lhsEdge : lhs) {
    for (const std::string &rhsEdge : rhs) {
      if (lhsEdge == rhsEdge)
        return true;
    }
  }
  return false;
}

mlir::FailureOr<IoBoundaryModel> buildIoBoundaryModel(mlir::func::FuncOp func,
                                                      const GraphModel &graph) {
  const TaskModel *entryTask = nullptr;
  const TaskModel *exitTask = nullptr;

  for (const TaskModel &task : graph.tasks) {
    if (!entryTask &&
        llvm::StringRef(task.kind) != task_graph_names::kMatrixSetupTaskKind &&
        taskTouchesResourceKind(task, graph.resources, task.inputResourceIds,
                                "input"))
      entryTask = &task;

    if (taskTouchesResourceKind(task, graph.resources, task.outputResourceIds,
                                "output"))
      exitTask = &task;
  }

  if (!entryTask || !exitTask) {
    func.emitError("expected task graph to expose entry and exit tasks for "
                   "I/O boundary metadata");
    return mlir::failure();
  }

  IoBoundaryModel boundary;
  boundary.entryTaskId = entryTask->index;
  boundary.entryCore = entryTask->coreId;
  boundary.entryEdges =
      getMeshEdgeMembership(entryTask->coreId, graph.hardware);
  boundary.exitTaskId = exitTask->index;
  boundary.exitCore = exitTask->coreId;
  boundary.exitEdges = getMeshEdgeMembership(exitTask->coreId, graph.hardware);
  boundary.sharesEdge = shareMeshEdge(boundary.entryEdges, boundary.exitEdges);
  return boundary;
}

mlir::FailureOr<GraphModel> buildGraphModel(mlir::ModuleOp module,
                                            mlir::func::FuncOp func) {
  if (!func.getBody().hasOneBlock()) {
    func.emitError("expected task graph function to have one block");
    return mlir::failure();
  }

  GraphModel graph;
  graph.func = func;
  graph.name = func.getName().str();
  auto hardware = buildHardwareModel(func);
  auto summary = buildSummaryModel(func);
  if (mlir::failed(hardware) || mlir::failed(summary))
    return mlir::failure();
  graph.hardware = std::move(*hardware);
  graph.summary = std::move(*summary);

  auto resources = collectResources(func, graph.hardware, graph.summary,
                                    graph.resourceIdByValue);
  if (mlir::failed(resources))
    return mlir::failure();
  graph.resources = std::move(*resources);

  auto tasks =
      collectTasks(module, func, graph.hardware, graph.resourceIdByValue,
                   graph.taskIndexByResult, graph.producerTaskIndexByResource);
  if (mlir::failed(tasks))
    return mlir::failure();
  graph.tasks = std::move(*tasks);

  auto ioBoundary = buildIoBoundaryModel(func, graph);
  if (mlir::failed(ioBoundary))
    return mlir::failure();
  graph.ioBoundary = std::move(*ioBoundary);

  return graph;
}

void emitOptionalI64Attr(llvm::json::OStream &json, llvm::StringRef key,
                         std::optional<int64_t> value) {
  if (value) {
    json.attribute(key, *value);
    return;
  }

  json.attribute(key, nullptr);
}

void emitI64ArrayAttr(llvm::json::OStream &json, llvm::StringRef key,
                      llvm::ArrayRef<int64_t> values) {
  json.attributeArray(key, [&] {
    for (int64_t value : values)
      json.value(value);
  });
}

void emitStringArrayAttr(llvm::json::OStream &json, llvm::StringRef key,
                         llvm::ArrayRef<std::string> values) {
  json.attributeArray(key, [&] {
    for (const std::string &value : values)
      json.value(value);
  });
}

void emitHardware(llvm::json::OStream &json, const HardwareModel &hardware) {
  json.attributeObject("hardware", [&] {
    json.attribute("topology", hardware.topology);
    json.attribute("num_cores", hardware.numCores);
    json.attribute("arrays_per_core", hardware.arraysPerCore);
    json.attribute("mesh_rows", hardware.meshRows);
    json.attribute("mesh_cols", hardware.meshCols);
    json.attribute("num_analog_arrays", hardware.numAnalogArrays);
    emitI64ArrayAttr(json, "analog_arrays", hardware.analogArrays);
  });
}

void emitResources(llvm::json::OStream &json,
                   llvm::ArrayRef<ResourceModel> resources) {
  json.attributeArray("resources", [&] {
    for (const ResourceModel &resource : resources) {
      json.object([&] {
        json.attribute("id", resource.id);
        json.attribute("kind", resource.kind);
        json.attribute("value_type", resource.valueType);
        emitOptionalI64Attr(json, "slot", resource.slot);
        emitOptionalI64Attr(json, "byte_size", resource.byteSize);
        emitOptionalI64Attr(json, "temp_index", resource.tempIndex);
        emitOptionalI64Attr(json, "temp_offset", resource.tempOffset);
        emitOptionalI64Attr(json, "logical_array_index",
                            resource.logicalArrayIndex);
        emitOptionalI64Attr(json, "physical_array_id",
                            resource.physicalArrayId);
        emitOptionalI64Attr(json, "core_id", resource.coreId);
        emitOptionalI64Attr(json, "local_array_id", resource.localArrayId);
      });
    }
  });
}

void emitTaskResourceIds(llvm::json::OStream &json, llvm::StringRef key,
                         llvm::ArrayRef<int64_t> resourceIds) {
  json.attributeArray(key, [&] {
    for (int64_t resourceId : resourceIds)
      json.value(resourceId);
  });
}

void emitSculptorOps(llvm::json::OStream &json,
                     llvm::ArrayRef<AnalogOpModel> analogOps) {
  json.attributeArray("analog_ops", [&] {
    for (const AnalogOpModel &op : analogOps) {
      json.object([&] {
        json.attribute("index", op.index);
        json.attribute("name", op.name);
      });
    }
  });

  llvm::SmallVector<std::pair<std::string, int64_t>, 4> counts;
  for (const AnalogOpModel &op : analogOps) {
    auto existing = llvm::find_if(
        counts, [&](const auto &entry) { return entry.first == op.name; });
    if (existing != counts.end()) {
      ++existing->second;
      continue;
    }
    counts.push_back({op.name, 1});
  }

  json.attributeArray("analog_op_counts", [&] {
    for (const auto &entry : counts) {
      json.object([&] {
        json.attribute("name", entry.first);
        json.attribute("count", entry.second);
      });
    }
  });
}

void emitTasks(llvm::json::OStream &json, llvm::ArrayRef<TaskModel> tasks) {
  json.attributeArray("tasks", [&] {
    for (const TaskModel &task : tasks) {
      json.object([&] {
        json.attribute("index", task.index);
        json.attribute("callee", task.callee);
        json.attribute("domain", task.domain);
        json.attribute("kind", task.kind);
        json.attribute("name", task.name);
        json.attribute("source_layer", task.sourceLayer);
        json.attribute("source_task_ordinal",
                       static_cast<int64_t>(task.sourceTaskOrdinal));
        json.attribute("core_id", task.coreId);
        emitOptionalI64Attr(json, "physical_array_id", task.physicalArrayId);
        emitOptionalI64Attr(json, "local_array_id", task.localArrayId);
        json.attribute("digital_ops", task.digitalOps);
        emitSculptorOps(json, task.analogOps);
        emitTaskResourceIds(json, "inputs", task.inputResourceIds);
        emitTaskResourceIds(json, "outputs", task.outputResourceIds);
        emitTaskResourceIds(json, "dependencies", task.dependencyTaskIndices);
      });
    }
  });
}

void emitIoBoundary(llvm::json::OStream &json,
                    const std::optional<IoBoundaryModel> &boundary) {
  if (!boundary)
    return;

  json.attributeObject("io_boundary", [&] {
    json.attribute("entry_task_id", boundary->entryTaskId);
    json.attribute("entry_core", boundary->entryCore);
    emitStringArrayAttr(json, "entry_edges", boundary->entryEdges);
    json.attribute("exit_task_id", boundary->exitTaskId);
    json.attribute("exit_core", boundary->exitCore);
    emitStringArrayAttr(json, "exit_edges", boundary->exitEdges);
    json.attribute("shares_edge", boundary->sharesEdge);
  });
}

int64_t getMeshDistance(int64_t sourceCore, int64_t destinationCore,
                        const HardwareModel &hardware) {
  int64_t sourceRow = sourceCore / hardware.meshCols;
  int64_t sourceCol = sourceCore % hardware.meshCols;
  int64_t destinationRow = destinationCore / hardware.meshCols;
  int64_t destinationCol = destinationCore % hardware.meshCols;
  return std::llabs(sourceRow - destinationRow) +
         std::llabs(sourceCol - destinationCol);
}

const ResourceModel *lookupResourceById(llvm::ArrayRef<ResourceModel> resources,
                                        int64_t resourceId) {
  if (resourceId < 0 || resourceId >= static_cast<int64_t>(resources.size()))
    return nullptr;
  return &resources[static_cast<size_t>(resourceId)];
}

void emitControlEdges(llvm::json::OStream &json,
                      llvm::ArrayRef<TaskModel> tasks) {
  json.attributeArray("control_edges", [&] {
    int64_t edgeId = 0;
    for (const TaskModel &consumer : tasks) {
      for (int64_t producerIndex : consumer.dependencyTaskIndices) {
        json.object([&] {
          json.attribute("id", edgeId++);
          json.attribute("producer_task", producerIndex);
          json.attribute("consumer_task", consumer.index);
        });
      }
    }
  });
}

void emitDataEdges(llvm::json::OStream &json, const GraphModel &graph) {
  json.attributeArray("data_edges", [&] {
    int64_t edgeId = 0;
    for (const TaskModel &consumer : graph.tasks) {
      for (int64_t resourceId : consumer.inputResourceIds) {
        const ResourceModel *resource =
            lookupResourceById(graph.resources, resourceId);
        if (!resource)
          continue;

        auto producerIt =
            graph.producerTaskIndexByResource.find(resource->value);
        if (producerIt == graph.producerTaskIndexByResource.end())
          continue;

        int64_t producerIndex = producerIt->second;
        auto producerItByIndex =
            llvm::find_if(graph.tasks, [&](const TaskModel &task) {
              return task.index == producerIndex;
            });
        if (producerItByIndex == graph.tasks.end())
          continue;

        int64_t sourceCore = producerItByIndex->coreId;
        int64_t destinationCore = consumer.coreId;
        int64_t byteSize = resource->byteSize.value_or(0);
        int64_t meshDistance =
            getMeshDistance(sourceCore, destinationCore, graph.hardware);
        int64_t transferCost = byteSize * meshDistance;

        json.object([&] {
          json.attribute("id", edgeId++);
          json.attribute("producer_task", producerIndex);
          json.attribute("consumer_task", consumer.index);
          json.attribute("resource", resourceId);
          json.attribute("byte_size", byteSize);
          json.attribute("source_core", sourceCore);
          json.attribute("destination_core", destinationCore);
          json.attribute("mesh_distance", meshDistance);
          json.attribute("transfer_cost", transferCost);
          json.attribute("inter_core", sourceCore != destinationCore);
        });
      }
    }
  });
}

void emitSummary(llvm::json::OStream &json, const SummaryModel &summary) {
  json.attributeObject("summary", [&] {
    json.attribute("task_count", summary.taskCount);
    json.attribute("dependency_count", summary.dependencyCount);
    json.attribute("inter_core_transfer_bytes", summary.interCoreTransferBytes);
    json.attribute("total_transfer_cost", summary.totalTransferCost);
    json.attribute("total_digital_ops", summary.totalDigitalOps);
    json.attribute("num_logical_arrays", summary.numLogicalArrays);
    emitI64ArrayAttr(json, "core_transfer_bytes", summary.coreTransferBytes);
    emitI64ArrayAttr(json, "core_transfer_cost", summary.coreTransferCost);
    emitI64ArrayAttr(json, "logical_array_to_analog_array",
                     summary.logicalArrayToAnalogArray);
  });
}

void emitGraph(llvm::json::OStream &json, const GraphModel &graph) {
  json.object([&] {
    json.attribute("name", graph.name);
    emitHardware(json, graph.hardware);
    emitResources(json, graph.resources);
    emitTasks(json, graph.tasks);
    emitIoBoundary(json, graph.ioBoundary);
    emitControlEdges(json, graph.tasks);
    emitDataEdges(json, graph);
    emitSummary(json, graph.summary);
  });
}

} // namespace

namespace mlir {
namespace sculptor {

void ExportTaskGraphSimModelPass::runOnOperation() {
  if (output.empty()) {
    getOperation().emitError("expected non-empty output path for "
                             "sculptor-export-task-graph-sim-model");
    signalPassFailure();
    return;
  }

  ModuleOp module = getOperation();
  llvm::SmallVector<func::FuncOp> graphFuncs;
  for (func::FuncOp func : module.getOps<func::FuncOp>())
    if (returnsTaskGraph(func))
      graphFuncs.push_back(func);

  if (graphFuncs.empty()) {
    module.emitError("expected at least one function returning "
                     "!sculptor.task_graph");
    signalPassFailure();
    return;
  }

  llvm::SmallVector<GraphModel, 1> graphs;
  graphs.reserve(graphFuncs.size());
  for (func::FuncOp func : graphFuncs) {
    auto graph = buildGraphModel(module, func);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    graphs.push_back(std::move(*graph));
  }

  std::error_code error;
  llvm::raw_fd_ostream os(output, error, llvm::sys::fs::OF_Text);
  if (error) {
    module.emitError("failed to open task graph simulation model output file '")
        << output << "': " << error.message();
    signalPassFailure();
    return;
  }

  llvm::json::OStream json(os, /*IndentSize=*/2);
  json.object([&] {
    json.attribute("schema_version", 1);
    json.attribute("format", "sculptor.task_graph.sim_model");
    json.attributeArray("graphs", [&] {
      for (const GraphModel &graph : graphs)
        emitGraph(json, graph);
    });
  });
  os << "\n";
}

void registerExportTaskGraphSimModelPass() {
  PassRegistration<ExportTaskGraphSimModelPass>();
}

} // namespace sculptor
} // namespace mlir
