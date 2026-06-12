#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExportTaskGraphVis.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>
#include <system_error>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;

struct ResourceModel {
  mlir::Value value;
  int64_t id = 0;
  std::optional<int64_t> byteSize;
  bool isLogicalArray = false;
  std::optional<int64_t> logicalArrayIndex;
  std::optional<int64_t> physicalArrayId;
  std::optional<int64_t> coreId;
  std::optional<int64_t> localArrayId;
};

struct TaskModel {
  mlir::sculptor::TaskCreateOp op;
  std::string id;
  int64_t order = 0;
  std::optional<int64_t> taskIndex;
  std::string callee;
  std::string domain;
  std::string kind;
  std::string name;
  std::string sourceLayer;
  uint64_t sourceTaskOrdinal = 0;
  std::optional<int64_t> coreId;
  std::optional<int64_t> physicalArrayId;
  std::optional<int64_t> localArrayId;
  std::optional<int64_t> digitalOps;
  int64_t analogOps = 0;
  llvm::SmallVector<int64_t> inputResourceIds;
  llvm::SmallVector<int64_t> outputResourceIds;
};

struct GraphModel {
  mlir::func::FuncOp func;
  std::string name;
  unsigned graphIndex = 0;
  std::optional<int64_t> arraysPerCore;
  std::optional<int64_t> meshCols;
  llvm::SmallVector<int64_t> logicalArrayToAnalogArray;
  llvm::SmallVector<ResourceModel, 0> resources;
  llvm::SmallVector<TaskModel, 0> tasks;
  llvm::DenseMap<mlir::Value, int64_t> resourceIdByValue;
  llvm::DenseMap<mlir::Value, unsigned> taskOrderByResult;
  llvm::DenseMap<mlir::Value, unsigned> producerOrderByResource;
};

bool returnsTaskGraph(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         llvm::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

std::optional<int64_t> getOptionalI64Attr(mlir::Operation *op,
                                          llvm::StringRef attrName) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName))
    return attr.getInt();
  return std::nullopt;
}

mlir::FailureOr<llvm::SmallVector<int64_t>>
getOptionalI64ArrayAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::ArrayAttr>(attrName);
  if (!attr)
    return llvm::SmallVector<int64_t>{};

  llvm::SmallVector<int64_t> values;
  values.reserve(attr.size());
  for (mlir::Attribute element : attr) {
    auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(element);
    if (!intAttr) {
      op->emitError("expected attr '") << attrName
                                       << "' to contain only integer attrs";
      return mlir::failure();
    }
    values.push_back(intAttr.getInt());
  }
  return values;
}

std::optional<std::pair<mlir::Value, llvm::StringRef>>
getTaskGraphResource(mlir::Operation &op) {
  if (auto input = llvm::dyn_cast<mlir::sculptor::TaskGraphInputOp>(&op))
    return std::make_pair(input.getResult(), llvm::StringRef("input"));
  if (auto output = llvm::dyn_cast<mlir::sculptor::TaskGraphOutputOp>(&op))
    return std::make_pair(output.getResult(), llvm::StringRef("output"));
  if (auto temporary = llvm::dyn_cast<mlir::sculptor::TaskGraphTemporaryOp>(&op))
    return std::make_pair(temporary.getResult(),
                          llvm::StringRef("temporary"));
  if (auto persistent =
          llvm::dyn_cast<mlir::sculptor::TaskGraphPersistentOp>(&op))
    return std::make_pair(persistent.getResult(),
                          llvm::StringRef("persistent"));
  return std::nullopt;
}

bool isLogicalArrayResource(mlir::Value resource) {
  auto resourceType =
      llvm::dyn_cast<mlir::sculptor::TaskResourceType>(resource.getType());
  return resourceType &&
         llvm::isa<mlir::sculptor::LogicalArrayType>(
             resourceType.getValueType());
}

std::string sanitizeId(llvm::StringRef value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value) {
    unsigned char byte = static_cast<unsigned char>(c);
    if (std::isalnum(byte) || c == '_') {
      result.push_back(c);
      continue;
    }
    result.push_back('_');
  }

  if (result.empty() ||
      !std::isalpha(static_cast<unsigned char>(result.front())))
    result.insert(result.begin(), 'n');
  return result;
}

std::string escapeDotString(llvm::StringRef value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value) {
    switch (c) {
    case '\\':
      result += "\\\\";
      break;
    case '"':
      result += "\\\"";
      break;
    case '\n':
      result += "\\n";
      break;
    default:
      result.push_back(c);
      break;
    }
  }
  return result;
}

std::string escapeXmlString(llvm::StringRef value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value) {
    switch (c) {
    case '&':
      result += "&amp;";
      break;
    case '<':
      result += "&lt;";
      break;
    case '>':
      result += "&gt;";
      break;
    case '"':
      result += "&quot;";
      break;
    case '\'':
      result += "&apos;";
      break;
    default:
      result.push_back(c);
      break;
    }
  }
  return result;
}

std::string buildNodeLabel(const TaskModel &task) {
  std::string label = task.kind;
  label += '\n';
  label += task.name;
  return escapeDotString(label);
}

void getNodeColors(const TaskModel &task, llvm::StringRef &fillColor,
                   llvm::StringRef &borderColor) {
  if (task.domain == "analog") {
    fillColor = "#fef3c7";
    borderColor = "#d97706";
    return;
  }

  if (task.domain == "digital") {
    fillColor = "#dbeafe";
    borderColor = "#2563eb";
    return;
  }

  fillColor = "#f1f5f9";
  borderColor = "#64748b";
}

std::optional<int64_t> getMeshDistance(std::optional<int64_t> sourceCore,
                                       std::optional<int64_t> destinationCore,
                                       const GraphModel &graph) {
  if (!sourceCore || !destinationCore || !graph.meshCols || *graph.meshCols <= 0)
    return std::nullopt;

  int64_t sourceRow = *sourceCore / *graph.meshCols;
  int64_t sourceCol = *sourceCore % *graph.meshCols;
  int64_t destinationRow = *destinationCore / *graph.meshCols;
  int64_t destinationCol = *destinationCore % *graph.meshCols;
  return std::llabs(sourceRow - destinationRow) +
         std::llabs(sourceCol - destinationCol);
}

const TaskModel *lookupTaskByOrder(const GraphModel &graph, unsigned order) {
  if (order >= graph.tasks.size())
    return nullptr;
  return &graph.tasks[order];
}

const ResourceModel *lookupResourceById(const GraphModel &graph,
                                        int64_t resourceId) {
  if (resourceId < 0 || resourceId >= static_cast<int64_t>(graph.resources.size()))
    return nullptr;
  return &graph.resources[static_cast<size_t>(resourceId)];
}

int64_t getTaskIndexForExport(const TaskModel &task) {
  return task.taskIndex.value_or(task.order);
}

bool isLogicalArrayDependency(const GraphModel &graph, const TaskModel &consumer,
                              const TaskModel &producer) {
  for (int64_t resourceId : consumer.inputResourceIds) {
    const ResourceModel *resource = lookupResourceById(graph, resourceId);
    if (!resource || !resource->isLogicalArray)
      continue;

    auto producerIt = graph.producerOrderByResource.find(resource->value);
    if (producerIt != graph.producerOrderByResource.end() &&
        producerIt->second == producer.order)
      return true;
  }
  return false;
}

int64_t countSculptorOps(mlir::ModuleOp module,
                       mlir::sculptor::TaskCreateOp taskOp) {
  auto callee =
      module.lookupSymbol<mlir::func::FuncOp>(
          taskOp.getCalleeAttr().getValue());
  if (!callee || callee.isDeclaration())
    return 0;

  int64_t count = 0;
  callee.walk([&](mlir::Operation *nestedOp) {
    if (nestedOp->getName().getDialectNamespace() == "analog")
      ++count;
  });
  return count;
}

mlir::LogicalResult collectResources(GraphModel &graph) {
  for (mlir::Operation &op : graph.func.getBody().front()) {
    std::optional<std::pair<mlir::Value, llvm::StringRef>> resourceInfo =
        getTaskGraphResource(op);
    if (!resourceInfo)
      continue;

    mlir::Value resource = resourceInfo->first;
    int64_t resourceId = static_cast<int64_t>(graph.resources.size());
    if (!graph.resourceIdByValue.try_emplace(resource, resourceId).second) {
      op.emitError("expected task graph resource value to be unique");
      return mlir::failure();
    }

    ResourceModel model;
    model.value = resource;
    model.id = resourceId;
    model.byteSize =
        getOptionalI64Attr(&op, runtime_attrs::kResourceByteSizeAttrName);
    model.isLogicalArray = isLogicalArrayResource(resource);
    model.logicalArrayIndex =
        getOptionalI64Attr(&op, schedule_attrs::kLogicalArrayIndexAttrName);

    if (model.isLogicalArray && model.logicalArrayIndex &&
        graph.arraysPerCore && !graph.logicalArrayToAnalogArray.empty()) {
      int64_t logicalIndex = *model.logicalArrayIndex;
      if (logicalIndex < 0 ||
          logicalIndex >=
              static_cast<int64_t>(graph.logicalArrayToAnalogArray.size())) {
        op.emitError("expected logical array index to reference scheduled "
                     "logical array placement");
        return mlir::failure();
      }
      int64_t physicalArrayId =
          graph.logicalArrayToAnalogArray[static_cast<size_t>(logicalIndex)];
      model.physicalArrayId = physicalArrayId;
      model.coreId = physicalArrayId / *graph.arraysPerCore;
      model.localArrayId = physicalArrayId % *graph.arraysPerCore;
    }

    graph.resources.push_back(std::move(model));
  }
  return mlir::success();
}

mlir::LogicalResult collectTasks(mlir::ModuleOp module, GraphModel &graph) {
  unsigned taskOrder = 0;
  for (mlir::Operation &op : graph.func.getBody().front()) {
    auto taskOp = llvm::dyn_cast<mlir::sculptor::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    TaskModel task;
    task.op = taskOp;
    task.id = "task_" + std::to_string(graph.graphIndex) + "_" +
              std::to_string(taskOrder);
    task.order = static_cast<int64_t>(taskOrder);
    task.taskIndex = getOptionalI64Attr(taskOp, runtime_attrs::kTaskIndexAttrName);
    task.callee = taskOp.getCallee().str();
    task.domain = taskOp.getDomain().str();
    task.kind = taskOp.getTaskKind().str();
    task.name = taskOp.getTaskName().str();
    task.sourceLayer = taskOp.getSourceLayer().str();
    task.sourceTaskOrdinal = taskOp.getSourceTaskOrdinal();
    task.coreId = getOptionalI64Attr(taskOp, runtime_attrs::kTaskCoreIdAttrName);
    task.physicalArrayId =
        getOptionalI64Attr(taskOp, runtime_attrs::kTaskPhysicalArrayIdAttrName);
    task.digitalOps =
        getOptionalI64Attr(taskOp, runtime_attrs::kTaskDigitalOpsAttrName);
    if (task.physicalArrayId && graph.arraysPerCore)
      task.localArrayId = *task.physicalArrayId % *graph.arraysPerCore;
    task.analogOps = countSculptorOps(module, taskOp);

    for (mlir::Value input : taskOp.getInputs()) {
      auto resourceIt = graph.resourceIdByValue.find(input);
      if (resourceIt == graph.resourceIdByValue.end()) {
        taskOp.emitError("expected every task input to reference a task graph "
                         "resource");
        return mlir::failure();
      }
      task.inputResourceIds.push_back(resourceIt->second);
    }

    for (mlir::Value output : taskOp.getOutputs()) {
      auto resourceIt = graph.resourceIdByValue.find(output);
      if (resourceIt == graph.resourceIdByValue.end()) {
        taskOp.emitError("expected every task output to reference a task graph "
                         "resource");
        return mlir::failure();
      }
      task.outputResourceIds.push_back(resourceIt->second);
      if (!graph.producerOrderByResource.try_emplace(output, taskOrder).second) {
        taskOp.emitError("expected task graph resource to have one producer");
        return mlir::failure();
      }
    }

    graph.taskOrderByResult.try_emplace(taskOp.getResult(), taskOrder);
    graph.tasks.push_back(std::move(task));
    ++taskOrder;
  }
  return mlir::success();
}

mlir::FailureOr<GraphModel> buildGraphModel(mlir::ModuleOp module,
                                            mlir::func::FuncOp func,
                                            unsigned graphIndex) {
  if (!func.getBody().hasOneBlock()) {
    func.emitError("expected task graph function to have one block");
    return mlir::failure();
  }

  GraphModel graph;
  graph.func = func;
  graph.name = func.getName().str();
  graph.graphIndex = graphIndex;
  graph.arraysPerCore =
      getOptionalI64Attr(func, schedule_attrs::kArraysPerCoreAttrName);
  graph.meshCols = getOptionalI64Attr(func, schedule_attrs::kMeshColsAttrName);
  auto logicalArrayMap =
      getOptionalI64ArrayAttr(func,
                              schedule_attrs::kLogicalArrayToAnalogArrayAttrName);
  if (mlir::failed(logicalArrayMap))
    return mlir::failure();
  graph.logicalArrayToAnalogArray = std::move(*logicalArrayMap);

  if (mlir::failed(collectResources(graph)))
    return mlir::failure();
  if (mlir::failed(collectTasks(module, graph)))
    return mlir::failure();
  return graph;
}

void emitDotTaskNode(llvm::raw_ostream &os, const TaskModel &task) {
  llvm::StringRef fillColor;
  llvm::StringRef borderColor;
  getNodeColors(task, fillColor, borderColor);

  os << "    " << task.id << " [label=\"" << buildNodeLabel(task)
     << "\", style=\"filled,rounded\", shape=\"box\", fillcolor=\"" << fillColor
     << "\", color=\"" << borderColor << "\"];\n";
}

void emitDotGraphFunc(llvm::raw_ostream &os, const GraphModel &graph) {
  llvm::StringMap<unsigned> clusterIndexBySourceLayer;
  llvm::SmallVector<llvm::SmallVector<const TaskModel *>> clusters;
  llvm::SmallVector<std::string> clusterLabels;

  for (const TaskModel &task : graph.tasks) {
    auto clusterIt = clusterIndexBySourceLayer.find(task.sourceLayer);
    if (clusterIt == clusterIndexBySourceLayer.end()) {
      unsigned clusterIndex = clusters.size();
      clusterIt =
          clusterIndexBySourceLayer.try_emplace(task.sourceLayer, clusterIndex)
              .first;
      clusters.push_back({});
      clusterLabels.push_back(task.sourceLayer);
    }
    clusters[clusterIt->second].push_back(&task);
  }

  os << "  subgraph cluster_graph_" << graph.graphIndex << " {\n";
  os << "    label=\"@" << escapeDotString(graph.name) << "\";\n";
  os << "    color=\"#cbd5e1\";\n";
  os << "    style=\"rounded\";\n";

  for (auto indexedCluster : llvm::enumerate(clusters)) {
    llvm::StringRef label = clusterLabels[indexedCluster.index()];
    os << "    subgraph cluster_" << graph.graphIndex << "_"
       << sanitizeId(label) << " {\n";
    os << "      label=\"" << escapeDotString(label) << "\";\n";
    os << "      color=\"#94a3b8\";\n";
    os << "      style=\"rounded\";\n";
    for (const TaskModel *task : indexedCluster.value())
      emitDotTaskNode(os, *task);
    os << "    }\n";
  }

  os << "  }\n";

  for (const TaskModel &consumer : graph.tasks) {
    mlir::sculptor::TaskCreateOp consumerOp = consumer.op;
    for (mlir::Value dependency : consumerOp.getDependencies()) {
      auto producerIt = graph.taskOrderByResult.find(dependency);
      if (producerIt == graph.taskOrderByResult.end())
        continue;

      const TaskModel *producer = lookupTaskByOrder(graph, producerIt->second);
      if (!producer)
        continue;

      bool logicalArrayEdge = isLogicalArrayDependency(graph, consumer, *producer);
      os << "  " << producer->id << " -> " << consumer.id;
      if (logicalArrayEdge)
        os << " [style=\"dotted\", color=\"#b45309\"]";
      else
        os << " [style=\"solid\", color=\"#334155\"]";
      os << ";\n";
    }
  }
}

void emitDot(llvm::raw_ostream &os, llvm::ArrayRef<GraphModel> graphs) {
  os << "digraph analog_task_graph {\n";
  os << "  rankdir=LR;\n";
  os << "  node [fontname=\"Helvetica\"];\n";
  os << "  edge [fontname=\"Helvetica\"];\n";
  for (const GraphModel &graph : graphs)
    emitDotGraphFunc(os, graph);
  os << "}\n";
}

void emitGraphMLKey(llvm::raw_ostream &os, llvm::StringRef id,
                    llvm::StringRef target, llvm::StringRef name,
                    llvm::StringRef type) {
  os << "  <key id=\"" << id << "\" for=\"" << target << "\" attr.name=\""
     << name << "\" attr.type=\"" << type << "\"/>\n";
}

void emitGraphMLStringData(llvm::raw_ostream &os, llvm::StringRef key,
                           llvm::StringRef value) {
  os << "      <data key=\"" << key << "\">" << escapeXmlString(value)
     << "</data>\n";
}

void emitGraphMLI64Data(llvm::raw_ostream &os, llvm::StringRef key,
                        int64_t value) {
  os << "      <data key=\"" << key << "\">" << value << "</data>\n";
}

void emitGraphMLI64Data(llvm::raw_ostream &os, llvm::StringRef key,
                        std::optional<int64_t> value) {
  if (value)
    emitGraphMLI64Data(os, key, *value);
}

void emitGraphMLBoolData(llvm::raw_ostream &os, llvm::StringRef key,
                         bool value) {
  os << "      <data key=\"" << key << "\">" << (value ? "true" : "false")
     << "</data>\n";
}

void emitGraphMLNode(llvm::raw_ostream &os, const TaskModel &task) {
  os << "    <node id=\"" << task.id << "\">\n";
  emitGraphMLI64Data(os, "task_index", getTaskIndexForExport(task));
  emitGraphMLStringData(os, "callee", task.callee);
  emitGraphMLStringData(os, "task_name", task.name);
  emitGraphMLStringData(os, "task_kind", task.kind);
  emitGraphMLStringData(os, "domain", task.domain);
  emitGraphMLStringData(os, "source_layer", task.sourceLayer);
  emitGraphMLI64Data(os, "source_task_ordinal",
                     static_cast<int64_t>(task.sourceTaskOrdinal));
  emitGraphMLI64Data(os, "core_id", task.coreId);
  emitGraphMLI64Data(os, "physical_array_id", task.physicalArrayId);
  emitGraphMLI64Data(os, "local_array_id", task.localArrayId);
  emitGraphMLI64Data(os, "digital_ops", task.digitalOps);
  emitGraphMLI64Data(os, "analog_ops", task.analogOps);
  os << "    </node>\n";
}

void emitGraphMLControlEdge(llvm::raw_ostream &os, const GraphModel &graph,
                            int64_t &edgeId, const TaskModel &producer,
                            const TaskModel &consumer) {
  os << "    <edge id=\"edge_" << graph.graphIndex << "_control_" << edgeId++
     << "\" source=\"" << producer.id << "\" target=\"" << consumer.id
     << "\">\n";
  emitGraphMLStringData(os, "edge_kind", "control");
  emitGraphMLI64Data(os, "producer_task", getTaskIndexForExport(producer));
  emitGraphMLI64Data(os, "consumer_task", getTaskIndexForExport(consumer));
  emitGraphMLBoolData(os, "logical_array_dependency",
                      isLogicalArrayDependency(graph, consumer, producer));
  os << "    </edge>\n";
}

void emitGraphMLDataEdge(llvm::raw_ostream &os, const GraphModel &graph,
                         int64_t &edgeId, const TaskModel &producer,
                         const TaskModel &consumer,
                         const ResourceModel &resource) {
  std::optional<int64_t> meshDistance =
      getMeshDistance(producer.coreId, consumer.coreId, graph);
  std::optional<int64_t> transferCost;
  if (meshDistance)
    transferCost = resource.byteSize.value_or(0) * *meshDistance;

  os << "    <edge id=\"edge_" << graph.graphIndex << "_data_" << edgeId++
     << "\" source=\"" << producer.id << "\" target=\"" << consumer.id
     << "\">\n";
  emitGraphMLStringData(os, "edge_kind", "data");
  emitGraphMLI64Data(os, "producer_task", getTaskIndexForExport(producer));
  emitGraphMLI64Data(os, "consumer_task", getTaskIndexForExport(consumer));
  emitGraphMLBoolData(os, "logical_array_dependency", resource.isLogicalArray);
  emitGraphMLI64Data(os, "resource_id", resource.id);
  emitGraphMLI64Data(os, "byte_size", resource.byteSize);
  emitGraphMLI64Data(os, "source_core", producer.coreId);
  emitGraphMLI64Data(os, "destination_core", consumer.coreId);
  emitGraphMLI64Data(os, "mesh_distance", meshDistance);
  emitGraphMLI64Data(os, "transfer_cost", transferCost);
  emitGraphMLBoolData(
      os, "inter_core",
      producer.coreId && consumer.coreId && *producer.coreId != *consumer.coreId);
  os << "    </edge>\n";
}

void emitGraphMLGraph(llvm::raw_ostream &os, const GraphModel &graph) {
  os << "  <graph id=\"" << sanitizeId(graph.name)
     << "\" edgedefault=\"directed\">\n";
  for (const TaskModel &task : graph.tasks)
    emitGraphMLNode(os, task);

  int64_t controlEdgeId = 0;
  for (const TaskModel &consumer : graph.tasks) {
    mlir::sculptor::TaskCreateOp consumerOp = consumer.op;
    for (mlir::Value dependency : consumerOp.getDependencies()) {
      auto producerIt = graph.taskOrderByResult.find(dependency);
      if (producerIt == graph.taskOrderByResult.end())
        continue;
      const TaskModel *producer = lookupTaskByOrder(graph, producerIt->second);
      if (!producer)
        continue;
      emitGraphMLControlEdge(os, graph, controlEdgeId, *producer, consumer);
    }
  }

  int64_t dataEdgeId = 0;
  for (const TaskModel &consumer : graph.tasks) {
    for (int64_t resourceId : consumer.inputResourceIds) {
      const ResourceModel *resource = lookupResourceById(graph, resourceId);
      if (!resource)
        continue;

      auto producerIt = graph.producerOrderByResource.find(resource->value);
      if (producerIt == graph.producerOrderByResource.end())
        continue;
      const TaskModel *producer = lookupTaskByOrder(graph, producerIt->second);
      if (!producer)
        continue;
      emitGraphMLDataEdge(os, graph, dataEdgeId, *producer, consumer, *resource);
    }
  }

  os << "  </graph>\n";
}

void emitGraphML(llvm::raw_ostream &os, llvm::ArrayRef<GraphModel> graphs) {
  os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  os << "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n";

  emitGraphMLKey(os, "task_index", "node", "task_index", "long");
  emitGraphMLKey(os, "callee", "node", "callee", "string");
  emitGraphMLKey(os, "task_name", "node", "task_name", "string");
  emitGraphMLKey(os, "task_kind", "node", "task_kind", "string");
  emitGraphMLKey(os, "domain", "node", "domain", "string");
  emitGraphMLKey(os, "source_layer", "node", "source_layer", "string");
  emitGraphMLKey(os, "source_task_ordinal", "node", "source_task_ordinal",
                 "long");
  emitGraphMLKey(os, "core_id", "node", "core_id", "long");
  emitGraphMLKey(os, "physical_array_id", "node", "physical_array_id",
                 "long");
  emitGraphMLKey(os, "local_array_id", "node", "local_array_id", "long");
  emitGraphMLKey(os, "digital_ops", "node", "digital_ops", "long");
  emitGraphMLKey(os, "analog_ops", "node", "analog_ops", "long");

  emitGraphMLKey(os, "edge_kind", "edge", "edge_kind", "string");
  emitGraphMLKey(os, "producer_task", "edge", "producer_task", "long");
  emitGraphMLKey(os, "consumer_task", "edge", "consumer_task", "long");
  emitGraphMLKey(os, "logical_array_dependency", "edge",
                 "logical_array_dependency", "boolean");
  emitGraphMLKey(os, "resource_id", "edge", "resource_id", "long");
  emitGraphMLKey(os, "byte_size", "edge", "byte_size", "long");
  emitGraphMLKey(os, "source_core", "edge", "source_core", "long");
  emitGraphMLKey(os, "destination_core", "edge", "destination_core", "long");
  emitGraphMLKey(os, "mesh_distance", "edge", "mesh_distance", "long");
  emitGraphMLKey(os, "transfer_cost", "edge", "transfer_cost", "long");
  emitGraphMLKey(os, "inter_core", "edge", "inter_core", "boolean");

  for (const GraphModel &graph : graphs)
    emitGraphMLGraph(os, graph);

  os << "</graphml>\n";
}

} // namespace

namespace mlir {
namespace sculptor {

void ExportTaskGraphVisPass::runOnOperation() {
  if (output.empty()) {
    getOperation().emitError("expected non-empty output path for "
                             "sculptor-export-task-graph-vis");
    signalPassFailure();
    return;
  }

  if (format != "dot" && format != "graphml") {
    getOperation().emitError("expected sculptor-export-task-graph-vis format to "
                             "be 'dot' or 'graphml', got '")
        << format << "'";
    signalPassFailure();
    return;
  }

  mlir::ModuleOp module = getOperation();
  llvm::SmallVector<mlir::func::FuncOp> graphFuncs;
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>())
    if (returnsTaskGraph(func))
      graphFuncs.push_back(func);

  if (graphFuncs.empty()) {
    module.emitError(
        "expected at least one function returning !sculptor.task_graph");
    signalPassFailure();
    return;
  }

  llvm::SmallVector<GraphModel, 1> graphs;
  graphs.reserve(graphFuncs.size());
  for (auto indexedFunc : llvm::enumerate(graphFuncs)) {
    auto graph = buildGraphModel(module, indexedFunc.value(),
                                 static_cast<unsigned>(indexedFunc.index()));
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    graphs.push_back(std::move(*graph));
  }

  std::error_code error;
  llvm::raw_fd_ostream os(output, error, llvm::sys::fs::OF_Text);
  if (error) {
    module.emitError("failed to open task graph visualization output file '")
        << output << "': " << error.message();
    signalPassFailure();
    return;
  }

  if (format == "dot")
    emitDot(os, graphs);
  else
    emitGraphML(os, graphs);
}

void registerExportTaskGraphVisPass() {
  PassRegistration<ExportTaskGraphVisPass>();
}

} // namespace sculptor
} // namespace mlir
