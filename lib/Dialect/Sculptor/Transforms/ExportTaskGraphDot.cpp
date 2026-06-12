#include "sculptor-mlir/Dialect/Sculptor/Transforms/ExportTaskGraphDot.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <memory>
#include <string>
#include <system_error>

namespace {

struct DotTask {
  mlir::sculptor::TaskCreateOp op;
  std::string id;
};

struct DotCluster {
  std::string id;
  std::string label;
  llvm::SmallVector<DotTask> tasks;
};

bool returnsTaskGraph(mlir::func::FuncOp func) {
  auto functionType = func.getFunctionType();
  return functionType.getNumResults() == 1 &&
         llvm::isa<mlir::sculptor::TaskGraphType>(functionType.getResult(0));
}

std::string sanitizeDotId(llvm::StringRef value) {
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

std::string buildNodeLabel(mlir::sculptor::TaskCreateOp taskOp) {
  std::string label = taskOp.getTaskKind().str();
  label += '\n';
  label += taskOp.getTaskName().str();
  return escapeDotString(label);
}

void getNodeColors(mlir::sculptor::TaskCreateOp taskOp,
                   llvm::StringRef &fillColor, llvm::StringRef &borderColor) {
  llvm::StringRef domain = taskOp.getDomain();
  if (domain == "analog") {
    fillColor = "#fef3c7";
    borderColor = "#d97706";
    return;
  }

  if (domain == "digital") {
    fillColor = "#dbeafe";
    borderColor = "#2563eb";
    return;
  }

  fillColor = "#f1f5f9";
  borderColor = "#64748b";
}

bool isLogicalArrayResource(mlir::Value resource) {
  auto resourceType =
      llvm::dyn_cast<mlir::sculptor::TaskResourceType>(resource.getType());
  return resourceType &&
         llvm::isa<mlir::sculptor::LogicalArrayType>(resourceType.getValueType());
}

void collectTasks(
    mlir::func::FuncOp taskGraphFunc,
    llvm::SmallVectorImpl<mlir::sculptor::TaskCreateOp> &tasks,
    llvm::DenseMap<mlir::Value, mlir::Operation *> &producerByResource,
    llvm::DenseMap<mlir::Value, std::string> &idByTaskResult,
    llvm::SmallVectorImpl<DotCluster> &clusters, unsigned graphIndex) {
  llvm::DenseMap<llvm::StringRef, unsigned> clusterIndexBySourceLayer;
  unsigned taskIndex = 0;

  for (mlir::Operation &op : taskGraphFunc.getBody().front()) {
    auto taskOp = llvm::dyn_cast<mlir::sculptor::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    std::string taskId = "task_" + std::to_string(graphIndex) + "_" +
                         std::to_string(taskIndex++);
    tasks.push_back(taskOp);
    idByTaskResult.try_emplace(taskOp.getResult(), taskId);

    llvm::StringRef sourceLayer = taskOp.getSourceLayer();
    auto clusterIt = clusterIndexBySourceLayer.find(sourceLayer);
    if (clusterIt == clusterIndexBySourceLayer.end()) {
      DotCluster cluster;
      cluster.id = "cluster_" + std::to_string(graphIndex) + "_" +
                   sanitizeDotId(sourceLayer);
      cluster.label = sourceLayer.str();
      clusterIt =
          clusterIndexBySourceLayer.try_emplace(sourceLayer, clusters.size())
              .first;
      clusters.push_back(std::move(cluster));
    }

    clusters[clusterIt->second].tasks.push_back(DotTask{taskOp, taskId});

    for (mlir::Value output : taskOp.getOutputs())
      producerByResource.try_emplace(output, taskOp.getOperation());
  }
}

bool isLogicalArrayDependency(
    mlir::sculptor::TaskCreateOp consumer, mlir::sculptor::TaskCreateOp producer,
    const llvm::DenseMap<mlir::Value, mlir::Operation *> &producerByResource) {
  for (mlir::Value input : consumer.getInputs()) {
    if (!isLogicalArrayResource(input))
      continue;

    auto producerIt = producerByResource.find(input);
    if (producerIt != producerByResource.end() &&
        producerIt->second == producer.getOperation())
      return true;
  }

  return false;
}

void emitTaskNode(llvm::raw_ostream &os, const DotTask &task) {
  llvm::StringRef fillColor;
  llvm::StringRef borderColor;
  getNodeColors(task.op, fillColor, borderColor);

  os << "    " << task.id << " [label=\"" << buildNodeLabel(task.op)
     << "\", style=\"filled,rounded\", shape=\"box\", fillcolor=\"" << fillColor
     << "\", color=\"" << borderColor << "\"];\n";
}

void emitTaskGraphFunc(llvm::raw_ostream &os, mlir::func::FuncOp taskGraphFunc,
                       unsigned graphIndex) {
  llvm::SmallVector<mlir::sculptor::TaskCreateOp> tasks;
  llvm::DenseMap<mlir::Value, mlir::Operation *> producerByResource;
  llvm::DenseMap<mlir::Value, std::string> idByTaskResult;
  llvm::SmallVector<DotCluster> clusters;
  collectTasks(taskGraphFunc, tasks, producerByResource, idByTaskResult,
               clusters, graphIndex);

  os << "  subgraph cluster_graph_" << graphIndex << " {\n";
  os << "    label=\"@" << escapeDotString(taskGraphFunc.getName()) << "\";\n";
  os << "    color=\"#cbd5e1\";\n";
  os << "    style=\"rounded\";\n";

  for (const DotCluster &cluster : clusters) {
    os << "    subgraph " << cluster.id << " {\n";
    os << "      label=\"" << escapeDotString(cluster.label) << "\";\n";
    os << "      color=\"#94a3b8\";\n";
    os << "      style=\"rounded\";\n";
    for (const DotTask &task : cluster.tasks)
      emitTaskNode(os, task);
    os << "    }\n";
  }

  os << "  }\n";

  for (mlir::sculptor::TaskCreateOp taskOp : tasks) {
    auto consumerIt = idByTaskResult.find(taskOp.getResult());
    if (consumerIt == idByTaskResult.end())
      continue;

    for (mlir::Value dependency : taskOp.getDependencies()) {
      auto producerIt = idByTaskResult.find(dependency);
      if (producerIt == idByTaskResult.end())
        continue;

      auto producer = dependency.getDefiningOp<mlir::sculptor::TaskCreateOp>();
      bool logicalArrayEdge =
          producer &&
          isLogicalArrayDependency(taskOp, producer, producerByResource);
      os << "  " << producerIt->second << " -> " << consumerIt->second;
      if (logicalArrayEdge)
        os << " [style=\"dotted\", color=\"#b45309\"]";
      else
        os << " [style=\"solid\", color=\"#334155\"]";
      os << ";\n";
    }
  }
}

} // namespace

namespace mlir {
namespace sculptor {

void ExportTaskGraphDotPass::runOnOperation() {
  if (output.empty()) {
    getOperation().emitError(
        "expected non-empty output path for sculptor-export-task-graph-dot");
    signalPassFailure();
    return;
  }

  std::error_code error;
  llvm::raw_fd_ostream os(output, error, llvm::sys::fs::OF_Text);
  if (error) {
    getOperation().emitError("failed to open task graph DOT output file '")
        << output << "': " << error.message();
    signalPassFailure();
    return;
  }

  mlir::ModuleOp module = getOperation();
  llvm::SmallVector<mlir::func::FuncOp> taskGraphFuncs;
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>())
    if (returnsTaskGraph(func))
      taskGraphFuncs.push_back(func);

  if (taskGraphFuncs.empty()) {
    module.emitError(
        "expected at least one function returning !sculptor.task_graph");
    signalPassFailure();
    return;
  }

  for (mlir::func::FuncOp func : taskGraphFuncs) {
    if (!func.getBody().hasOneBlock()) {
      func.emitError("expected task graph function to have a single block");
      signalPassFailure();
      return;
    }
  }

  os << "digraph analog_task_graph {\n";
  os << "  rankdir=LR;\n";
  os << "  node [fontname=\"Helvetica\"];\n";
  os << "  edge [fontname=\"Helvetica\"];\n";
  for (auto indexedFunc : llvm::enumerate(taskGraphFuncs))
    emitTaskGraphFunc(os, indexedFunc.value(), indexedFunc.index());
  os << "}\n";
}

void registerExportTaskGraphDotPass() {
  PassRegistration<ExportTaskGraphDotPass>();
}

} // namespace sculptor
} // namespace mlir
