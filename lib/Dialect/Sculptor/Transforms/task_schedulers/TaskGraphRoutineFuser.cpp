#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphScheduler.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <cctype>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace task_attrs = mlir::sculptor::task_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;

using TaskFusionPatternFn = LogicalResult (*)(ModuleOp module,
                                              func::FuncOp taskGraphFunc,
                                              const HardwareBudget &budget,
                                              const TaskGraphDAG &dag,
                                              bool &changed);

struct TaskFusionPattern {
  llvm::StringRef name;
  TaskFusionPatternFn apply;
};

struct ConvTileMVMChain {
  const TaskGraphNode *convPatch = nullptr;
  const TaskGraphNode *vectorTile = nullptr;
  const TaskGraphNode *mvm = nullptr;
  const TaskGraphNode *tileRecombine = nullptr;
  const TaskGraphNode *biasAdd = nullptr;
};

static bool hasTaskKind(mlir::sculptor::TaskCreateOp taskOp,
                        StringRef taskKind) {
  return taskOp.getTaskKind() == taskKind;
}

static bool hasDependency(mlir::sculptor::TaskCreateOp taskOp,
                          Value dependency) {
  for (Value candidate : taskOp.getDependencies()) {
    if (candidate == dependency)
      return true;
  }
  return false;
}

static std::optional<int64_t> getOptionalI64Attr(Operation *op,
                                                 StringRef attrName) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return std::nullopt;
}

static bool getCompatibleMVMPhysicalArray(const ConvTileMVMChain &chain,
                                          int64_t &physicalArrayId) {
  mlir::sculptor::TaskCreateOp mvmTask = chain.mvm->op;
  auto mvmCore = getOptionalI64Attr(mvmTask.getOperation(),
                                    runtime_attrs::kTaskCoreIdAttrName);
  auto mvmPhysicalArray = getOptionalI64Attr(
      mvmTask.getOperation(), runtime_attrs::kTaskPhysicalArrayIdAttrName);
  if (!mvmCore || !mvmPhysicalArray)
    return false;

  for (const TaskGraphNode *node : {chain.convPatch, chain.vectorTile,
                                    chain.tileRecombine, chain.biasAdd}) {
    mlir::sculptor::TaskCreateOp taskOp = node->op;
    auto coreId = getOptionalI64Attr(taskOp.getOperation(),
                                     runtime_attrs::kTaskCoreIdAttrName);
    if (!coreId || *coreId != *mvmCore)
      return false;
  }

  physicalArrayId = *mvmPhysicalArray;
  return true;
}

static const TaskGraphNode *
getSingleSuccessorWithKind(const TaskGraphNode &node, const TaskGraphDAG &dag,
                           StringRef taskKind) {
  if (node.successors.size() != 1)
    return nullptr;

  const TaskGraphNode &successor = dag.nodes[node.successors.front()];
  if (!hasTaskKind(successor.op, taskKind))
    return nullptr;

  return &successor;
}

static bool allSameSourceLayer(const ConvTileMVMChain &chain) {
  mlir::sculptor::TaskCreateOp convPatch = chain.convPatch->op;
  mlir::sculptor::TaskCreateOp vectorTile = chain.vectorTile->op;
  mlir::sculptor::TaskCreateOp mvm = chain.mvm->op;
  mlir::sculptor::TaskCreateOp tileRecombine = chain.tileRecombine->op;
  mlir::sculptor::TaskCreateOp biasAdd = chain.biasAdd->op;
  StringRef sourceLayer = convPatch.getSourceLayer();
  return vectorTile.getSourceLayer() == sourceLayer &&
         mvm.getSourceLayer() == sourceLayer &&
         tileRecombine.getSourceLayer() == sourceLayer &&
         biasAdd.getSourceLayer() == sourceLayer;
}

static bool hasSingleDependencyOn(mlir::sculptor::TaskCreateOp taskOp,
                                  mlir::sculptor::TaskCreateOp dependency) {
  return taskOp.getDependencies().size() == 1 &&
         hasDependency(taskOp, dependency.getResult());
}

static bool
hasExpectedMVMDependencies(mlir::sculptor::TaskCreateOp mvmTask,
                           mlir::sculptor::TaskCreateOp vectorTile) {
  if (mvmTask.getDependencies().size() != 2 ||
      !hasDependency(mvmTask, vectorTile.getResult()))
    return false;

  for (Value dependency : mvmTask.getDependencies()) {
    if (dependency == vectorTile.getResult())
      continue;
    auto dependencyTask =
        dependency.getDefiningOp<mlir::sculptor::TaskCreateOp>();
    if (dependencyTask &&
        hasTaskKind(dependencyTask, task_graph_names::kMatrixSetupTaskKind))
      return true;
  }

  return false;
}

static bool isExactConvTileMVMChain(const ConvTileMVMChain &chain) {
  return allSameSourceLayer(chain) &&
         hasSingleDependencyOn(chain.vectorTile->op, chain.convPatch->op) &&
         hasExpectedMVMDependencies(chain.mvm->op, chain.vectorTile->op) &&
         hasSingleDependencyOn(chain.tileRecombine->op, chain.mvm->op) &&
         hasSingleDependencyOn(chain.biasAdd->op, chain.tileRecombine->op);
}

static std::optional<ConvTileMVMChain>
matchConvTileMVMChainFrom(const TaskGraphNode &node, const TaskGraphDAG &dag) {
  if (!hasTaskKind(node.op, task_graph_names::kConvPatchTaskKind))
    return std::nullopt;

  const TaskGraphNode *vectorTile = getSingleSuccessorWithKind(
      node, dag, task_graph_names::kVectorTileTaskKind);
  if (!vectorTile)
    return std::nullopt;

  const TaskGraphNode *mvm = getSingleSuccessorWithKind(
      *vectorTile, dag, task_graph_names::kMVMTaskKind);
  if (!mvm)
    return std::nullopt;

  const TaskGraphNode *tileRecombine = getSingleSuccessorWithKind(
      *mvm, dag, task_graph_names::kTileRecombineTaskKind);
  if (!tileRecombine)
    return std::nullopt;

  const TaskGraphNode *biasAdd = getSingleSuccessorWithKind(
      *tileRecombine, dag, task_graph_names::kBiasAddTaskKind);
  if (!biasAdd)
    return std::nullopt;

  ConvTileMVMChain chain{&node, vectorTile, mvm, tileRecombine, biasAdd};
  if (!isExactConvTileMVMChain(chain))
    return std::nullopt;

  return chain;
}

static std::string sanitizeSymbolComponent(StringRef value) {
  std::string result;
  result.reserve(value.size());

  bool previousWasUnderscore = false;
  for (char c : value) {
    unsigned char uc = static_cast<unsigned char>(c);
    bool symbolSafe = std::isalnum(uc) || c == '_';
    char next = symbolSafe ? c : '_';

    if (next == '_' && previousWasUnderscore)
      continue;

    result.push_back(next);
    previousWasUnderscore = next == '_';
  }

  while (!result.empty() && result.back() == '_')
    result.pop_back();
  while (!result.empty() && result.front() == '_')
    result.erase(result.begin());

  if (result.empty())
    return "task";

  return result;
}

static std::string getConvTileSuffix(StringRef convPatchTaskName) {
  StringRef prefix = "conv2d_";
  if (convPatchTaskName.starts_with(prefix))
    convPatchTaskName = convPatchTaskName.drop_front(prefix.size());
  return sanitizeSymbolComponent(convPatchTaskName);
}

static std::string buildFusedConvTileMVMTaskName(const ConvTileMVMChain &chain,
                                                 const TaskGraphDAG &dag) {
  llvm::StringSet<> usedTaskNames;
  for (const TaskGraphNode &node : dag.nodes) {
    mlir::sculptor::TaskCreateOp taskOp = node.op;
    usedTaskNames.insert(taskOp.getTaskName());
  }

  mlir::sculptor::TaskCreateOp convPatch = chain.convPatch->op;
  mlir::sculptor::TaskCreateOp mvm = chain.mvm->op;
  std::string baseName = convPatch.getSourceLayer().str();
  baseName += "_conv_tile_mvm_";
  baseName += getConvTileSuffix(convPatch.getTaskName());

  if (usedTaskNames.insert(baseName).second)
    return baseName;

  std::string fallback = baseName;
  fallback += "_";
  fallback += std::to_string(mvm.getSourceTaskOrdinal());
  return fallback;
}

static std::string buildUniqueTaskFunctionName(ModuleOp module,
                                               func::FuncOp currentCallee,
                                               StringRef fusedTaskName,
                                               uint64_t sourceTaskOrdinal) {
  std::string baseName = "task_";
  baseName += sanitizeSymbolComponent(fusedTaskName);
  baseName += "_";
  baseName += std::to_string(sourceTaskOrdinal);

  auto isAvailable = [&](StringRef name) {
    auto existing = module.lookupSymbol<func::FuncOp>(name);
    return !existing || existing == currentCallee;
  };

  if (isAvailable(baseName))
    return baseName;

  unsigned index = 0;
  std::string candidate = baseName + "_" + std::to_string(index);
  while (!isAvailable(candidate)) {
    ++index;
    candidate = baseName + "_" + std::to_string(index);
  }
  return candidate;
}

static FailureOr<func::FuncOp>
lookupTaskCallee(ModuleOp module, mlir::sculptor::TaskCreateOp taskOp) {
  auto callee =
      module.lookupSymbol<func::FuncOp>(taskOp.getCalleeAttr().getValue());
  if (!callee) {
    return taskOp.emitError("expected task callee '")
           << taskOp.getCalleeAttr().getValue()
           << "' to resolve to a function for task fusion";
  }
  return callee;
}

static LogicalResult
renameFusedConvTileMVMTask(ModuleOp module,
                           mlir::sculptor::TaskCreateOp fusedTask,
                           StringRef fusedTaskName) {
  auto callee = lookupTaskCallee(module, fusedTask);
  if (failed(callee))
    return failure();

  func::FuncOp calleeFunc = *callee;
  std::string functionName = buildUniqueTaskFunctionName(
      module, calleeFunc, fusedTaskName, fusedTask.getSourceTaskOrdinal());
  Builder builder(module.getContext());
  auto functionNameAttr = builder.getStringAttr(functionName);
  if (failed(SymbolTable::replaceAllSymbolUses(calleeFunc, functionNameAttr,
                                               module.getOperation())))
    return failure();
  calleeFunc.setSymName(functionName);

  auto domain = builder.getStringAttr(task_graph_names::kAnalogDomain);
  auto kind = builder.getStringAttr(task_graph_names::kConvTileMVMTaskKind);
  auto name = builder.getStringAttr(fusedTaskName);

  fusedTask.setCalleeAttr(
      FlatSymbolRefAttr::get(builder.getContext(), functionName));
  fusedTask.setDomainAttr(domain);
  fusedTask.setTaskKindAttr(kind);
  fusedTask.setTaskNameAttr(name);

  calleeFunc->setAttr(task_attrs::kTaskDomainAttrName, domain);
  calleeFunc->setAttr(task_attrs::kTaskKindAttrName, kind);
  calleeFunc->setAttr(task_attrs::kTaskNameAttrName, name);
  return success();
}

static LogicalResult fuseConvTileMVMChains(ModuleOp module,
                                           func::FuncOp taskGraphFunc,
                                           const HardwareBudget &budget,
                                           const TaskGraphDAG &dag,
                                           bool &changed) {
  (void)taskGraphFunc;
  changed = false;

  for (const TaskGraphNode &node : dag.nodes) {
    std::optional<ConvTileMVMChain> chain =
        matchConvTileMVMChainFrom(node, dag);
    if (!chain)
      continue;

    int64_t physicalArrayId = 0;
    if (!getCompatibleMVMPhysicalArray(*chain, physicalArrayId))
      continue;

    std::string fusedTaskName = buildFusedConvTileMVMTaskName(*chain, dag);

    auto fusedTask =
        fuseTasks(module, chain->convPatch->op, chain->vectorTile->op);
    if (failed(fusedTask))
      return failure();

    fusedTask = fuseTasks(module, *fusedTask, chain->mvm->op);
    if (failed(fusedTask))
      return failure();

    fusedTask = fuseTasks(module, *fusedTask, chain->tileRecombine->op);
    if (failed(fusedTask))
      return failure();

    fusedTask = fuseTasks(module, *fusedTask, chain->biasAdd->op);
    if (failed(fusedTask))
      return failure();

    if (failed(attachTaskAnalogArrayPlacement(module, *fusedTask, budget,
                                              physicalArrayId)))
      return failure();

    if (failed(renameFusedConvTileMVMTask(module, *fusedTask, fusedTaskName)))
      return failure();

    changed = true;
    return success();
  }

  return success();
}

static LogicalResult
fuseTileRecombineBiasAddChains(ModuleOp module, func::FuncOp taskGraphFunc,
                               const HardwareBudget &budget,
                               const TaskGraphDAG &dag, bool &changed) {
  (void)module;
  (void)taskGraphFunc;
  (void)budget;
  (void)dag;
  changed = false;
  return success();
}

static LogicalResult fuseSameCoreDigitalProducerConsumerChains(
    ModuleOp module, func::FuncOp taskGraphFunc, const HardwareBudget &budget,
    const TaskGraphDAG &dag, bool &changed) {
  (void)module;
  (void)taskGraphFunc;
  (void)budget;
  (void)dag;
  changed = false;
  return success();
}

static llvm::ArrayRef<TaskFusionPattern> getDefaultTaskFusionPatterns() {
  static const TaskFusionPattern patterns[] = {
      {"conv-tile-mvm-bias", fuseConvTileMVMChains},
      {"tile-recombine-bias-add", fuseTileRecombineBiasAddChains},
      {"same-core-digital-producer-consumer",
       fuseSameCoreDigitalProducerConsumerChains},
  };
  return patterns;
}

static LogicalResult applyPatternToFixedPoint(ModuleOp module,
                                              func::FuncOp taskGraphFunc,
                                              const HardwareBudget &budget,
                                              const TaskFusionPattern &pattern,
                                              TaskGraphDAG &dag) {
  while (true) {
    bool changed = false;
    if (failed(pattern.apply(module, taskGraphFunc, budget, dag, changed))) {
      taskGraphFunc.emitError("failed task fusion pattern '")
          << pattern.name << "'";
      return failure();
    }

    if (!changed)
      return success();

    auto nextDag = parseTaskGraphDAG(taskGraphFunc);
    if (failed(nextDag))
      return failure();

    dag = std::move(*nextDag);
  }
}

} // namespace

LogicalResult fuseTaskGraphRoutines(ModuleOp module, func::FuncOp taskGraphFunc,
                                    const HardwareBudget &budget,
                                    const TaskGraphDAG &dag) {
  TaskGraphDAG workingDag = dag;
  for (const TaskFusionPattern &pattern : getDefaultTaskFusionPatterns()) {
    if (failed(applyPatternToFixedPoint(module, taskGraphFunc, budget, pattern,
                                        workingDag)))
      return failure();
  }
  return success();
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
