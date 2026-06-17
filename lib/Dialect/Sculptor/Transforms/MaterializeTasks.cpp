#include "sculptor-mlir/Dialect/Sculptor/Transforms/MaterializeTasks.h"

// MaterializeTasks is the task-region -> task-function boundary. The pass is
// built around an explicit v1 contract before the mutating materialization
// steps are added.

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskMetadata.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"

#include <cctype>
#include <cstdint>
#include <string>
#include <utility>

namespace {

constexpr llvm::StringLiteral kForwardFunctionName = "forward";

namespace task_graph_names = mlir::sculptor::task_graph_names;
namespace task_metadata = mlir::sculptor::task_metadata;

struct MaterializableTaskRegion {
  mlir::sculptor::TaskRegionOp region;
  int64_t ordinal = 0;
  mlir::StringAttr taskKind;
  mlir::StringAttr taskName;
  mlir::StringAttr sourceLayerSymbol;
  mlir::StringAttr taskFunctionName;
};

struct MaterializableLayerCall {
  mlir::func::CallOp call;
  mlir::func::FuncOp callee;
  mlir::StringAttr sourceLayerSymbol;
  llvm::SmallVector<MaterializableTaskRegion> taskRegions;
};

struct TaskFunctionNameState {
  llvm::DenseMap<mlir::Operation *, mlir::StringAttr> nameByRegion;
  llvm::StringSet<> generatedNames;
};

struct ResidualIsland {
  llvm::SmallVector<mlir::Operation *> ops;
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> outputs;
  unsigned ordinal = 0;
  mlir::StringAttr taskKind;
  mlir::StringAttr taskName;
  mlir::StringAttr taskFunctionName;
  mlir::Operation *insertionPoint = nullptr;
};

enum class CaptureHandling {
  CloneSafeConstant,
  PassExplicitly,
  RejectUnsupported
};

static bool isForwardFunction(mlir::func::FuncOp func) {
  return func.getSymName() == kForwardFunctionName;
}

static mlir::func::FuncOp resolveCallee(mlir::ModuleOp module,
                                        mlir::func::CallOp call) {
  auto calleeAttr = call.getCalleeAttr();
  if (!calleeAttr)
    return {};

  return module.lookupSymbol<mlir::func::FuncOp>(calleeAttr.getValue());
}

static CaptureHandling classifyExternalCapture(mlir::Value value) {
  if (llvm::isa<mlir::BlockArgument>(value))
    return CaptureHandling::PassExplicitly;

  mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return CaptureHandling::PassExplicitly;

  if (llvm::isa<mlir::arith::ConstantOp>(definingOp))
    return CaptureHandling::CloneSafeConstant;

  if (llvm::isa<mlir::sculptor::TaskRegionOp>(definingOp))
    return CaptureHandling::PassExplicitly;

  return CaptureHandling::RejectUnsupported;
}

static mlir::LogicalResult
validateTaskRegionInputs(mlir::sculptor::TaskRegionOp region) {
  for (mlir::Value input : region.getInputs()) {
    if (classifyExternalCapture(input) == CaptureHandling::RejectUnsupported) {
      return region.emitOpError(
          "cannot materialize task region with unsupported external capture");
    }
  }

  return mlir::success();
}

static mlir::StringAttr getTaskNameAttr(mlir::sculptor::TaskRegionOp region) {
  if (mlir::StringAttr name = region.getNameAttr())
    return name;
  return region.getKindAttr();
}

static mlir::StringAttr getTaskDomainAttr(mlir::Builder &builder,
                                          mlir::StringAttr taskKind) {
  llvm::StringRef kind = taskKind.getValue();
  if (kind.starts_with(task_graph_names::kAnalogTaskKindPrefix))
    return builder.getStringAttr(task_graph_names::kAnalogDomain);
  return builder.getStringAttr(task_graph_names::kDigitalDomain);
}

static std::string sanitizeSymbolComponent(llvm::StringRef value) {
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

static std::string
buildTaskFunctionBaseName(const MaterializableTaskRegion &taskRegion) {
  llvm::StringRef sourceLayer = taskRegion.sourceLayerSymbol.getValue();
  llvm::StringRef taskName = taskRegion.taskName.getValue();
  std::string sourcePrefix = sourceLayer.str();
  sourcePrefix += "_";
  if (taskName.starts_with(sourcePrefix))
    return "task_" + sanitizeSymbolComponent(taskName) + "_" +
           std::to_string(taskRegion.ordinal);

  return "task_" + sanitizeSymbolComponent(sourceLayer) + "_" +
         sanitizeSymbolComponent(taskName) + "_" +
         std::to_string(taskRegion.ordinal);
}

static std::string
makeUniqueTaskFunctionName(mlir::ModuleOp module, llvm::StringRef baseName,
                           llvm::StringSet<> &generatedNames) {
  if (!module.lookupSymbol(baseName) && generatedNames.insert(baseName).second)
    return baseName.str();

  unsigned index = 0;
  std::string candidate = (baseName + "_" + std::to_string(index)).str();
  while (module.lookupSymbol(candidate) ||
         !generatedNames.insert(candidate).second) {
    ++index;
    candidate = (baseName + "_" + std::to_string(index)).str();
  }

  return candidate;
}

static mlir::StringAttr
assignTaskFunctionName(mlir::ModuleOp module,
                       MaterializableTaskRegion &taskRegion,
                       TaskFunctionNameState &nameState) {
  if (mlir::StringAttr existing =
          nameState.nameByRegion.lookup(taskRegion.region.getOperation()))
    return existing;

  std::string baseName = buildTaskFunctionBaseName(taskRegion);
  std::string uniqueName =
      makeUniqueTaskFunctionName(module, baseName, nameState.generatedNames);
  mlir::StringAttr nameAttr =
      mlir::StringAttr::get(taskRegion.region.getContext(), uniqueName);
  nameState.nameByRegion[taskRegion.region.getOperation()] = nameAttr;
  return nameAttr;
}

static mlir::LogicalResult collectTaskRegionsInSourceOrder(
    mlir::ModuleOp module, mlir::func::FuncOp callee, mlir::func::CallOp call,
    mlir::StringAttr sourceLayerSymbol, TaskFunctionNameState &nameState,
    llvm::SmallVectorImpl<MaterializableTaskRegion> &taskRegions) {
  if (!callee.getBody().hasOneBlock()) {
    return call.emitOpError(
        "expected materializable layer function to have one block");
  }

  int64_t ordinal = 0;
  for (mlir::Operation &op : callee.front().without_terminator()) {
    auto region = llvm::dyn_cast<mlir::sculptor::TaskRegionOp>(&op);
    if (!region)
      continue;

    if (mlir::failed(validateTaskRegionInputs(region)))
      return mlir::failure();

    MaterializableTaskRegion taskRegion{region,
                                        ordinal++,
                                        region.getKindAttr(),
                                        getTaskNameAttr(region),
                                        sourceLayerSymbol,
                                        {}};
    taskRegion.taskFunctionName =
        assignTaskFunctionName(module, taskRegion, nameState);

    taskRegions.push_back(taskRegion);
  }

  return mlir::success();
}

static mlir::LogicalResult collectMaterializableLayerCalls(
    mlir::ModuleOp module, mlir::func::FuncOp forward,
    llvm::SmallVectorImpl<MaterializableLayerCall> &layerCalls) {
  if (!forward.getBody().hasOneBlock())
    return forward.emitError("expected forward to have one block");

  TaskFunctionNameState nameState;
  for (mlir::Operation &op : forward.front()) {
    auto call = llvm::dyn_cast<mlir::func::CallOp>(&op);
    if (!call)
      continue;

    mlir::func::FuncOp callee = resolveCallee(module, call);
    if (!callee)
      continue;

    mlir::StringAttr sourceLayerSymbol =
        mlir::StringAttr::get(callee.getContext(), callee.getSymName());

    llvm::SmallVector<MaterializableTaskRegion> taskRegions;
    if (mlir::failed(collectTaskRegionsInSourceOrder(
            module, callee, call, sourceLayerSymbol, nameState, taskRegions)))
      return mlir::failure();

    if (taskRegions.empty())
      continue;

    layerCalls.push_back(MaterializableLayerCall{
        call, callee, sourceLayerSymbol, std::move(taskRegions)});
  }

  return mlir::success();
}

static mlir::FunctionType
getTaskFunctionType(mlir::OpBuilder &builder,
                    const MaterializableTaskRegion &taskRegion) {
  mlir::sculptor::TaskRegionOp region = taskRegion.region;
  llvm::SmallVector<mlir::Type> inputTypes;
  inputTypes.reserve(region.getInputs().size());
  for (mlir::Value input : region.getInputs())
    inputTypes.push_back(input.getType());

  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.reserve(region->getNumResults());
  for (mlir::Value result : region->getResults())
    resultTypes.push_back(result.getType());

  return builder.getFunctionType(inputTypes, resultTypes);
}

static mlir::func::FuncOp
emitTaskFunctionDeclaration(mlir::ModuleOp module,
                            const MaterializableTaskRegion &taskRegion,
                            mlir::OpBuilder &builder) {
  mlir::sculptor::TaskRegionOp region = taskRegion.region;
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(
          taskRegion.taskFunctionName.getValue()))
    return existing;

  builder.setInsertionPointToEnd(module.getBody());
  auto taskFunc = builder.create<mlir::func::FuncOp>(
      region.getLoc(), taskRegion.taskFunctionName.getValue(),
      getTaskFunctionType(builder, taskRegion));
  taskFunc.setPrivate();
  task_metadata::setTaskFunctionMetadata(
      taskFunc,
      {getTaskDomainAttr(builder, taskRegion.taskKind), taskRegion.taskKind,
       taskRegion.taskName, taskRegion.sourceLayerSymbol,
       builder.getI64IntegerAttr(taskRegion.ordinal)});
  return taskFunc;
}

static mlir::LogicalResult
validateMappedOperands(mlir::sculptor::TaskRegionOp region, mlir::Operation &op,
                       const mlir::IRMapping &mapping) {
  for (mlir::Value operand : op.getOperands()) {
    if (mapping.contains(operand))
      continue;

    return region.emitOpError(
        "cannot materialize task region with unsupported external capture");
  }

  return mlir::success();
}

static mlir::LogicalResult
collectMappedYieldValues(mlir::sculptor::TaskRegionOp region,
                         mlir::sculptor::YieldOp yield,
                         const mlir::IRMapping &mapping,
                         llvm::SmallVectorImpl<mlir::Value> &mappedValues) {
  for (mlir::Value value : yield.getValues()) {
    mlir::Value mapped = mapping.lookupOrNull(value);
    if (!mapped) {
      return region.emitOpError(
          "could not map task region yield operand to cloned task value");
    }

    mappedValues.push_back(mapped);
  }

  return mlir::success();
}

static mlir::LogicalResult
materializeTaskFunctionBody(mlir::ModuleOp module,
                            const MaterializableTaskRegion &taskRegion,
                            mlir::OpBuilder &builder) {
  mlir::func::FuncOp taskFunc =
      emitTaskFunctionDeclaration(module, taskRegion, builder);
  if (!taskFunc.isDeclaration())
    return mlir::success();

  mlir::sculptor::TaskRegionOp region = taskRegion.region;
  mlir::Block &sourceBlock = region.getBody().front();
  auto yield =
      llvm::dyn_cast<mlir::sculptor::YieldOp>(sourceBlock.getTerminator());
  if (!yield)
    return region.emitOpError(
        "expected region to terminate with sculptor.yield");

  mlir::Block *entryBlock = taskFunc.addEntryBlock();
  mlir::IRMapping mapping;
  mapping.map(sourceBlock.getArguments(), entryBlock->getArguments());

  builder.setInsertionPointToStart(entryBlock);
  for (mlir::Operation &op : sourceBlock.without_terminator()) {
    if (mlir::failed(validateMappedOperands(region, op, mapping)))
      return mlir::failure();
    builder.clone(op, mapping);
  }

  llvm::SmallVector<mlir::Value> mappedReturns;
  if (mlir::failed(
          collectMappedYieldValues(region, yield, mapping, mappedReturns)))
    return mlir::failure();

  builder.create<mlir::func::ReturnOp>(yield.getLoc(), mappedReturns);
  return mlir::success();
}

static mlir::LogicalResult
materializeTaskFunctions(mlir::ModuleOp module,
                         llvm::ArrayRef<MaterializableLayerCall> layerCalls) {
  mlir::OpBuilder builder(module.getContext());
  for (const MaterializableLayerCall &layerCall : layerCalls) {
    for (const MaterializableTaskRegion &taskRegion : layerCall.taskRegions) {
      if (mlir::failed(
              materializeTaskFunctionBody(module, taskRegion, builder))) {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}

static mlir::func::FuncOp
lookupTaskFunction(mlir::ModuleOp module,
                   const MaterializableTaskRegion &taskRegion) {
  return module.lookupSymbol<mlir::func::FuncOp>(
      taskRegion.taskFunctionName.getValue());
}

static mlir::LogicalResult
collectMappedTaskCallOperands(const MaterializableTaskRegion &taskRegion,
                              const mlir::IRMapping &mapping,
                              llvm::SmallVectorImpl<mlir::Value> &operands) {
  mlir::sculptor::TaskRegionOp region = taskRegion.region;
  operands.reserve(region.getInputs().size());
  for (mlir::Value input : region.getInputs()) {
    mlir::Value mapped = mapping.lookupOrNull(input);
    if (!mapped) {
      return region.emitOpError(
          "could not map task region input to materialized task operand");
    }

    operands.push_back(mapped);
  }

  return mlir::success();
}

static void mapTaskCallResults(const MaterializableTaskRegion &taskRegion,
                               mlir::func::CallOp taskCall,
                               mlir::IRMapping &mapping) {
  mlir::sculptor::TaskRegionOp region = taskRegion.region;
  for (auto [sourceResult, materializedResult] :
       llvm::zip_equal(region->getResults(), taskCall->getResults())) {
    mapping.map(sourceResult, materializedResult);
  }
}

static mlir::LogicalResult collectMappedLayerReturnValues(
    const MaterializableLayerCall &layerCall, const mlir::IRMapping &mapping,
    llvm::SmallVectorImpl<mlir::Value> &replacements) {
  mlir::func::CallOp call = layerCall.call;
  mlir::func::FuncOp callee = layerCall.callee;
  auto returnOp =
      llvm::dyn_cast<mlir::func::ReturnOp>(callee.front().getTerminator());
  if (!returnOp) {
    return call.emitOpError(
        "expected materializable layer function to terminate with func.return");
  }

  replacements.reserve(returnOp.getNumOperands());
  for (mlir::Value value : returnOp.getOperands()) {
    mlir::Value mapped = mapping.lookupOrNull(value);
    if (!mapped) {
      return call.emitOpError(
          "could not map layer return value to materialized task result");
    }

    replacements.push_back(mapped);
  }

  if (replacements.size() != call.getNumResults()) {
    return call.emitOpError("materialized task result count does not match "
                            "layer call result count");
  }

  return mlir::success();
}

static void replaceLayerCallResults(mlir::func::CallOp layerCall,
                                    mlir::ValueRange replacements) {
  for (auto [oldResult, replacement] :
       llvm::zip_equal(layerCall->getResults(), replacements)) {
    oldResult.replaceAllUsesWith(replacement);
  }

  layerCall.erase();
}

static mlir::LogicalResult
rewriteLayerCallWithTaskCalls(mlir::ModuleOp module,
                              const MaterializableLayerCall &layerCall,
                              mlir::OpBuilder &builder) {
  mlir::func::CallOp call = layerCall.call;
  mlir::func::FuncOp callee = layerCall.callee;
  if (!callee.getBody().hasOneBlock()) {
    return call.emitOpError(
        "expected materializable layer function to have one block");
  }

  mlir::IRMapping mapping;
  mapping.map(callee.front().getArguments(), call.getOperands());

  builder.setInsertionPoint(call);
  for (const MaterializableTaskRegion &taskRegion : layerCall.taskRegions) {
    mlir::func::FuncOp taskFunc = lookupTaskFunction(module, taskRegion);
    if (!taskFunc) {
      return call.emitOpError("could not find materialized task '")
             << taskRegion.taskFunctionName.getValue() << "'";
    }

    llvm::SmallVector<mlir::Value> taskOperands;
    if (mlir::failed(
            collectMappedTaskCallOperands(taskRegion, mapping, taskOperands)))
      return mlir::failure();

    auto taskCall = builder.create<mlir::func::CallOp>(
        taskRegion.region->getLoc(), taskFunc.getSymName(),
        taskFunc.getResultTypes(), taskOperands);
    mapTaskCallResults(taskRegion, taskCall, mapping);
  }

  llvm::SmallVector<mlir::Value> replacements;
  if (mlir::failed(
          collectMappedLayerReturnValues(layerCall, mapping, replacements)))
    return mlir::failure();

  replaceLayerCallResults(call, replacements);
  return mlir::success();
}

static mlir::LogicalResult
rewriteForwardLayerCalls(mlir::ModuleOp module,
                         llvm::ArrayRef<MaterializableLayerCall> layerCalls) {
  mlir::OpBuilder builder(module.getContext());
  for (const MaterializableLayerCall &layerCall : layerCalls) {
    if (mlir::failed(
            rewriteLayerCallWithTaskCalls(module, layerCall, builder))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

static void
eraseUnusedLayerFunctions(mlir::ModuleOp module,
                          llvm::ArrayRef<MaterializableLayerCall> layerCalls) {
  llvm::SmallPtrSet<mlir::Operation *, 8> seen;
  llvm::SmallVector<mlir::func::FuncOp> candidates;
  for (const MaterializableLayerCall &layerCall : layerCalls) {
    mlir::func::FuncOp callee = layerCall.callee;
    if (seen.insert(callee.getOperation()).second)
      candidates.push_back(callee);
  }

  for (mlir::func::FuncOp callee : candidates) {
    if (mlir::SymbolTable::symbolKnownUseEmpty(callee, module))
      callee.erase();
  }
}

static bool isMaterializedTaskFunction(mlir::func::FuncOp func) {
  return task_metadata::hasTaskFunctionMetadata(func);
}

static bool hasMaterializedTaskCall(mlir::ModuleOp module,
                                    mlir::func::FuncOp forward) {
  if (!forward.getBody().hasOneBlock())
    return false;

  for (mlir::Operation &op : forward.front()) {
    auto call = llvm::dyn_cast<mlir::func::CallOp>(&op);
    if (!call)
      continue;

    mlir::func::FuncOp callee = resolveCallee(module, call);
    if (callee && isMaterializedTaskFunction(callee))
      return true;
  }

  return false;
}

static mlir::Operation *getEnclosingForwardBlockOp(mlir::Operation *op,
                                                   mlir::Block *forwardBlock) {
  while (op && op->getBlock() != forwardBlock)
    op = op->getParentOp();
  return op;
}

static bool isResidualCandidateOp(mlir::Operation &op) {
  return !llvm::isa<mlir::func::CallOp, mlir::func::ReturnOp>(&op);
}

static bool isSupportedResidualFloatingOp(mlir::Operation &op) {
  llvm::StringRef dialect = op.getName().getDialectNamespace();
  return dialect == "arith" || dialect == "tensor" || dialect == "linalg" ||
         dialect == "math";
}

static mlir::LogicalResult validateResidualCandidateOp(mlir::Operation &op) {
  if (llvm::isa<mlir::sculptor::TaskRegionOp>(&op)) {
    return op.emitError(
        "expected sculptor.task_region to be materialized before outlining "
        "residual forward islands");
  }

  if (isSupportedResidualFloatingOp(op))
    return mlir::success();

  if (!mlir::isMemoryEffectFree(&op)) {
    return op.emitError(
        "cannot outline residual forward island containing unsupported "
        "side-effecting op");
  }

  return mlir::success();
}

static mlir::LogicalResult collectResidualCandidateOps(
    mlir::func::FuncOp forward,
    llvm::SmallVectorImpl<mlir::Operation *> &candidateOps,
    llvm::SmallPtrSetImpl<mlir::Operation *> &candidateSet) {
  candidateOps.clear();
  candidateSet.clear();

  for (mlir::Operation &op : forward.front().without_terminator()) {
    if (!isResidualCandidateOp(op))
      continue;

    if (mlir::failed(validateResidualCandidateOp(op)))
      return mlir::failure();

    candidateOps.push_back(&op);
    candidateSet.insert(&op);
  }

  return mlir::success();
}

static bool isSupportedResidualInput(mlir::Value value,
                                     mlir::Block *forwardBlock) {
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value))
    return blockArg.getOwner() == forwardBlock;

  mlir::Operation *definingOp = value.getDefiningOp();
  return definingOp && definingOp->getBlock() == forwardBlock &&
         llvm::isa<mlir::func::CallOp>(definingOp);
}

static bool residualIslandContainsOpNamed(const ResidualIsland &island,
                                          llvm::StringRef opName) {
  for (mlir::Operation *op : island.ops) {
    bool found = false;
    op->walk([&](mlir::Operation *nestedOp) {
      if (nestedOp->getName().getStringRef() == opName)
        found = true;
    });
    if (found)
      return true;
  }

  return false;
}

static bool residualIslandContainsDialect(const ResidualIsland &island,
                                          llvm::StringRef dialectName) {
  for (mlir::Operation *op : island.ops) {
    bool found = false;
    op->walk([&](mlir::Operation *nestedOp) {
      if (nestedOp->getName().getDialectNamespace() == dialectName)
        found = true;
    });
    if (found)
      return true;
  }

  return false;
}

static void assignResidualIslandMetadata(mlir::ModuleOp module,
                                         ResidualIsland &island,
                                         llvm::StringSet<> &generatedNames) {
  mlir::Builder builder(module.getContext());

  llvm::StringRef taskName = "forward_compute";
  llvm::StringRef taskKind = "digital.compute";
  if (residualIslandContainsOpNamed(island, "math.exp") &&
      residualIslandContainsOpNamed(island, "arith.divf") &&
      residualIslandContainsOpNamed(island, "arith.negf")) {
    taskName = "forward_sigmoid";
    taskKind = "digital.activation";
  } else if (residualIslandContainsDialect(island, "math")) {
    taskName = "forward_activation";
    taskKind = "digital.activation";
  }

  island.taskKind = builder.getStringAttr(taskKind);
  island.taskName = builder.getStringAttr(taskName);

  std::string baseName =
      ("task_" + taskName + "_" + std::to_string(island.ordinal)).str();
  std::string uniqueName =
      makeUniqueTaskFunctionName(module, baseName, generatedNames);
  island.taskFunctionName = builder.getStringAttr(uniqueName);
}

static bool isValueDefinedInsideOp(mlir::Value value, mlir::Operation *op) {
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Operation *parentOp = blockArg.getOwner()->getParentOp();
    return parentOp && (parentOp == op || op->isAncestor(parentOp));
  }

  mlir::Operation *definingOp = value.getDefiningOp();
  return definingOp && (definingOp == op || op->isAncestor(definingOp));
}

static void
collectResidualOpExternalValues(mlir::Operation *op,
                                llvm::SetVector<mlir::Value> &externalValues) {
  for (mlir::Value operand : op->getOperands()) {
    if (!isValueDefinedInsideOp(operand, op))
      externalValues.insert(operand);
  }

  op->walk([&](mlir::Operation *nestedOp) {
    if (nestedOp == op)
      return;

    for (mlir::Value operand : nestedOp->getOperands()) {
      if (!isValueDefinedInsideOp(operand, op))
        externalValues.insert(operand);
    }
  });
}

static mlir::Operation *
getResidualCandidateDefiningOp(mlir::Value value, mlir::Block *forwardBlock) {
  mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return nullptr;

  mlir::Operation *forwardOp =
      getEnclosingForwardBlockOp(definingOp, forwardBlock);
  if (!forwardOp || !isResidualCandidateOp(*forwardOp))
    return nullptr;

  return forwardOp;
}

static bool isResidualCloneOnlyValue(mlir::Value value) {
  mlir::Operation *definingOp = value.getDefiningOp();
  return definingOp &&
         llvm::isa<mlir::arith::ConstantOp, mlir::tensor::EmptyOp>(definingOp);
}

static bool isUseInsideResidualIsland(
    mlir::OpOperand &use, mlir::Block *forwardBlock,
    const llvm::SmallPtrSetImpl<mlir::Operation *> &islandOpSet) {
  mlir::Operation *owner =
      getEnclosingForwardBlockOp(use.getOwner(), forwardBlock);
  return owner && islandOpSet.contains(owner);
}

static bool hasResidualIslandExternalUse(
    mlir::Value value, mlir::Block *forwardBlock,
    const llvm::SmallPtrSetImpl<mlir::Operation *> &islandOpSet) {
  for (mlir::OpOperand &use : value.getUses())
    if (!isUseInsideResidualIsland(use, forwardBlock, islandOpSet))
      return true;
  return false;
}

static void collectResidualIslandOutputs(
    mlir::Value root, mlir::Block *forwardBlock, const ResidualIsland &island,
    const llvm::SmallPtrSetImpl<mlir::Operation *> &islandOpSet,
    llvm::SmallVectorImpl<mlir::Value> &outputs) {
  llvm::SetVector<mlir::Value> liveOuts;
  for (mlir::Operation *op : island.ops) {
    for (mlir::Value result : op->getResults()) {
      if (isResidualCloneOnlyValue(result))
        continue;

      if (hasResidualIslandExternalUse(result, forwardBlock, islandOpSet))
        liveOuts.insert(result);
    }
  }

  liveOuts.insert(root);
  outputs.assign(liveOuts.begin(), liveOuts.end());
}

static mlir::LogicalResult
collectResidualProducerSlice(mlir::Value root, mlir::Block *forwardBlock,
                             ResidualIsland &island) {
  llvm::SetVector<mlir::Operation *> opSet;
  llvm::SetVector<mlir::Value> visitedValues;
  llvm::SmallVector<mlir::Value> worklist{root};

  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    if (!visitedValues.insert(value))
      continue;

    if (isSupportedResidualInput(value, forwardBlock))
      continue;

    mlir::Operation *producer =
        getResidualCandidateDefiningOp(value, forwardBlock);
    if (!producer) {
      return island.insertionPoint->emitError(
          "cannot outline residual forward island with unsupported external "
          "input");
    }

    if (mlir::failed(validateResidualCandidateOp(*producer)))
      return mlir::failure();

    if (!opSet.insert(producer))
      continue;

    llvm::SetVector<mlir::Value> externalValues;
    collectResidualOpExternalValues(producer, externalValues);
    for (mlir::Value externalValue : externalValues)
      worklist.push_back(externalValue);
  }

  for (mlir::Operation &op : *forwardBlock) {
    if (opSet.contains(&op))
      island.ops.push_back(&op);
  }

  if (island.ops.empty()) {
    return island.insertionPoint->emitError(
        "expected residual forward island to contain at least one operation");
  }

  llvm::SmallPtrSet<mlir::Operation *, 16> islandOpSet;
  for (mlir::Operation *op : island.ops)
    islandOpSet.insert(op);

  llvm::SetVector<mlir::Value> inputs;
  for (mlir::Operation *op : island.ops) {
    llvm::SetVector<mlir::Value> externalValues;
    collectResidualOpExternalValues(op, externalValues);
    for (mlir::Value externalValue : externalValues) {
      mlir::Operation *producer =
          getResidualCandidateDefiningOp(externalValue, forwardBlock);
      if (producer && islandOpSet.contains(producer))
        continue;

      if (!isSupportedResidualInput(externalValue, forwardBlock)) {
        return op->emitError(
            "cannot outline residual forward island with unsupported "
            "external input");
      }

      inputs.insert(externalValue);
    }
  }

  island.inputs.assign(inputs.begin(), inputs.end());
  collectResidualIslandOutputs(root, forwardBlock, island, islandOpSet,
                               island.outputs);
  return mlir::success();
}

static mlir::LogicalResult
collectNextResidualIsland(mlir::ModuleOp module, mlir::func::FuncOp forward,
                          unsigned ordinal, llvm::StringSet<> &generatedNames,
                          ResidualIsland &island, bool &found) {
  found = false;

  llvm::SmallVector<mlir::Operation *> candidateOps;
  llvm::SmallPtrSet<mlir::Operation *, 16> candidateSet;
  if (mlir::failed(
          collectResidualCandidateOps(forward, candidateOps, candidateSet)))
    return mlir::failure();

  if (candidateOps.empty())
    return mlir::success();

  mlir::Block *forwardBlock = &forward.front();
  for (mlir::Operation &op : *forwardBlock) {
    if (!llvm::isa<mlir::func::CallOp, mlir::func::ReturnOp>(&op))
      continue;

    for (mlir::OpOperand &operand : op.getOpOperands()) {
      mlir::Operation *producer =
          getResidualCandidateDefiningOp(operand.get(), forwardBlock);
      if (!producer || !candidateSet.contains(producer))
        continue;

      found = true;
      island = ResidualIsland{};
      island.ordinal = ordinal;
      island.insertionPoint = &op;
      if (mlir::failed(collectResidualProducerSlice(operand.get(), forwardBlock,
                                                    island)))
        return mlir::failure();
      assignResidualIslandMetadata(module, island, generatedNames);
      return mlir::success();
    }
  }

  return mlir::success();
}

static mlir::FunctionType
getResidualIslandFunctionType(mlir::OpBuilder &builder,
                              const ResidualIsland &island) {
  llvm::SmallVector<mlir::Type> inputTypes;
  inputTypes.reserve(island.inputs.size());
  for (mlir::Value input : island.inputs)
    inputTypes.push_back(input.getType());

  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.reserve(island.outputs.size());
  for (mlir::Value output : island.outputs)
    resultTypes.push_back(output.getType());

  return builder.getFunctionType(inputTypes, resultTypes);
}

static mlir::LogicalResult
validateResidualCloneOperands(const ResidualIsland &island, mlir::Operation &op,
                              const mlir::IRMapping &mapping) {
  llvm::SetVector<mlir::Value> externalValues;
  collectResidualOpExternalValues(&op, externalValues);
  for (mlir::Value value : externalValues) {
    if (mapping.contains(value))
      continue;

    return op.emitError("could not map residual forward island operand while "
                        "outlining '")
           << island.taskFunctionName.getValue() << "'";
  }

  return mlir::success();
}

static mlir::LogicalResult cloneResidualIslandOp(mlir::Operation &op,
                                                 mlir::OpBuilder &builder,
                                                 mlir::IRMapping &mapping) {
  mlir::IRMapping opMapping;
  for (mlir::Value operand : op.getOperands()) {
    mlir::Value mapped = mapping.lookupOrNull(operand);
    if (!mapped)
      return op.emitError("could not map residual forward island operand");
    opMapping.map(operand, mapped);
  }

  op.walk([&](mlir::Operation *nestedOp) {
    if (nestedOp == &op)
      return;
    for (mlir::Value operand : nestedOp->getOperands()) {
      if (mlir::Value mapped = mapping.lookupOrNull(operand))
        opMapping.map(operand, mapped);
    }
  });

  mlir::Operation *cloned = builder.clone(op, opMapping);
  for (auto [sourceResult, clonedResult] :
       llvm::zip_equal(op.getResults(), cloned->getResults())) {
    mapping.map(sourceResult, clonedResult);
  }

  return mlir::success();
}

static mlir::FailureOr<mlir::func::FuncOp>
createResidualIslandFunction(mlir::ModuleOp module,
                             const ResidualIsland &island,
                             mlir::OpBuilder &builder) {
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(
          island.taskFunctionName.getValue()))
    return existing;

  builder.setInsertionPointToEnd(module.getBody());
  auto taskFunc = builder.create<mlir::func::FuncOp>(
      island.ops.front()->getLoc(), island.taskFunctionName.getValue(),
      getResidualIslandFunctionType(builder, island));
  taskFunc.setPrivate();
  task_metadata::setTaskFunctionMetadata(
      taskFunc, {builder.getStringAttr(task_graph_names::kDigitalDomain),
                 island.taskKind, island.taskName,
                 builder.getStringAttr(task_graph_names::kForwardSourceLayer),
                 builder.getI64IntegerAttr(island.ordinal)});

  mlir::Block *entryBlock = taskFunc.addEntryBlock();
  mlir::IRMapping mapping;
  mapping.map(island.inputs, entryBlock->getArguments());

  builder.setInsertionPointToStart(entryBlock);
  for (mlir::Operation *op : island.ops) {
    if (mlir::failed(validateResidualCloneOperands(island, *op, mapping)))
      return mlir::failure();
    if (mlir::failed(cloneResidualIslandOp(*op, builder, mapping)))
      return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> mappedReturns;
  mappedReturns.reserve(island.outputs.size());
  for (mlir::Value output : island.outputs) {
    mlir::Value mapped = mapping.lookupOrNull(output);
    if (!mapped) {
      return output.getDefiningOp()->emitError(
                 "could not map residual forward island output while outlining "
                 "'")
             << island.taskFunctionName.getValue() << "'";
    }
    mappedReturns.push_back(mapped);
  }

  builder.create<mlir::func::ReturnOp>(island.ops.back()->getLoc(),
                                       mappedReturns);
  return taskFunc;
}

static mlir::LogicalResult
eraseResidualIslandOps(const ResidualIsland &island) {
  for (mlir::Operation *op : llvm::reverse(island.ops)) {
    if (op->use_empty())
      op->erase();
  }

  return mlir::success();
}

static void replaceResidualIslandBoundaryUses(const ResidualIsland &island,
                                              mlir::Block *forwardBlock,
                                              mlir::func::CallOp call) {
  llvm::SmallPtrSet<mlir::Operation *, 16> islandOpSet;
  for (mlir::Operation *op : island.ops)
    islandOpSet.insert(op);

  for (auto [output, replacement] :
       llvm::zip_equal(island.outputs, call->getResults())) {
    llvm::SmallVector<mlir::OpOperand *> usesToReplace;
    for (mlir::OpOperand &use : output.getUses()) {
      if (!isUseInsideResidualIsland(use, forwardBlock, islandOpSet))
        usesToReplace.push_back(&use);
    }

    for (mlir::OpOperand *use : usesToReplace)
      use->set(replacement);
  }
}

static mlir::LogicalResult outlineResidualIsland(mlir::ModuleOp module,
                                                 mlir::func::FuncOp forward,
                                                 const ResidualIsland &island,
                                                 mlir::OpBuilder &builder) {
  if (island.outputs.empty())
    return eraseResidualIslandOps(island);

  if (!island.insertionPoint) {
    return forward.emitError(
        "expected residual forward island with outputs to have an insertion "
        "point");
  }

  auto taskFunc = createResidualIslandFunction(module, island, builder);
  if (mlir::failed(taskFunc))
    return mlir::failure();

  builder.setInsertionPoint(island.insertionPoint);
  auto call = builder.create<mlir::func::CallOp>(
      island.insertionPoint->getLoc(), (*taskFunc).getSymName(),
      (*taskFunc).getResultTypes(), island.inputs);

  replaceResidualIslandBoundaryUses(island, &forward.front(), call);
  return eraseResidualIslandOps(island);
}

static mlir::LogicalResult
verifyForwardIsFullyMaterialized(mlir::func::FuncOp forward) {
  for (mlir::Operation &op : forward.front().without_terminator()) {
    if (llvm::isa<mlir::func::CallOp>(&op))
      continue;

    return op.emitError(
        "expected forward to contain only func.call operations after task "
        "materialization");
  }

  return mlir::success();
}

static mlir::LogicalResult
eraseDeadResidualCandidateOps(mlir::func::FuncOp forward) {
  bool localChanged = true;
  while (localChanged) {
    localChanged = false;
    llvm::SmallVector<mlir::Operation *> deadOps;
    llvm::SmallVector<mlir::Operation *> forwardOps;
    for (mlir::Operation &op : forward.front().without_terminator())
      forwardOps.push_back(&op);

    for (mlir::Operation *op : llvm::reverse(forwardOps)) {
      if (!isResidualCandidateOp(*op))
        continue;

      if (mlir::failed(validateResidualCandidateOp(*op)))
        return mlir::failure();

      if (!op->use_empty())
        continue;

      deadOps.push_back(op);
    }

    for (mlir::Operation *op : deadOps)
      op->erase();

    localChanged = !deadOps.empty();
  }

  return mlir::success();
}

static mlir::LogicalResult
outlineResidualForwardIslands(mlir::ModuleOp module,
                              mlir::func::FuncOp forward) {
  if (!forward.getBody().hasOneBlock())
    return forward.emitError("expected forward to have one block");

  mlir::OpBuilder builder(module.getContext());
  llvm::StringSet<> generatedNames;
  unsigned ordinal = 0;

  while (true) {
    if (mlir::failed(eraseDeadResidualCandidateOps(forward)))
      return mlir::failure();

    ResidualIsland island;
    bool found = false;
    if (mlir::failed(collectNextResidualIsland(module, forward, ordinal,
                                               generatedNames, island, found)))
      return mlir::failure();

    if (!found)
      break;

    if (mlir::failed(outlineResidualIsland(module, forward, island, builder)))
      return mlir::failure();
    ++ordinal;
  }

  return verifyForwardIsFullyMaterialized(forward);
}

} // namespace

namespace mlir {
namespace sculptor {

void MaterializeTasksPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  llvm::SmallVector<mlir::func::FuncOp> forwardFuncs;
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (isForwardFunction(func))
      forwardFuncs.push_back(func);
  }

  for (mlir::func::FuncOp func : forwardFuncs) {
    llvm::SmallVector<MaterializableLayerCall> layerCalls;
    if (mlir::failed(
            collectMaterializableLayerCalls(module, func, layerCalls))) {
      signalPassFailure();
      return;
    }

    if (!layerCalls.empty()) {
      if (mlir::failed(materializeTaskFunctions(module, layerCalls))) {
        signalPassFailure();
        return;
      }

      if (mlir::failed(rewriteForwardLayerCalls(module, layerCalls))) {
        signalPassFailure();
        return;
      }

      eraseUnusedLayerFunctions(module, layerCalls);
    }

    if (hasMaterializedTaskCall(module, func)) {
      if (mlir::failed(outlineResidualForwardIslands(module, func))) {
        signalPassFailure();
        return;
      }
    }
  }
}

void registerMaterializeTasksPass() {
  PassRegistration<MaterializeTasksPass>();
}

} // namespace sculptor
} // namespace mlir
