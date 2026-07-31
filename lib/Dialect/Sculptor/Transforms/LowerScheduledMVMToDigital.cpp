#include "sculptor-mlir/Dialect/Sculptor/Transforms/LowerScheduledMVMToDigital.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphCleanup.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDAG.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDigitalOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResources.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphTaskKinds.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskGraphTimingIRCodec.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace {

namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_attrs = mlir::sculptor::task_attrs;
namespace task_graph = mlir::sculptor::task_graph;
namespace task_graph_names = mlir::sculptor::task_graph_names;

struct MatrixSetup {
  mlir::sculptor::TaskCreateOp task;
  mlir::func::FuncOp callee;
  mlir::Value logicalResource;
  mlir::RankedTensorType weightType;
  mlir::TypedAttr weightValue;
  int64_t coreId = -1;
};

struct DigitalMVM {
  mlir::sculptor::TaskCreateOp task;
  mlir::func::FuncOp callee;
  unsigned setupIndex = 0;
  unsigned logicalInputIndex = 0;
  mlir::RankedTensorType vectorType;
  mlir::RankedTensorType resultType;
  int64_t coreId = -1;
  int64_t digitalOps = 0;
  bool needsVectorTileExtraction = false;
  int64_t vectorTile = 0;
  int64_t validCols = 0;
};

static bool returnsTaskGraph(mlir::func::FuncOp func) {
  mlir::FunctionType type = func.getFunctionType();
  return type.getNumResults() == 1 &&
         mlir::isa<mlir::sculptor::TaskGraphType>(type.getResult(0));
}

static bool isLogicalArrayType(mlir::Type type) {
  if (mlir::isa<mlir::sculptor::LogicalArrayType>(type))
    return true;
  auto resourceType = mlir::dyn_cast<mlir::sculptor::TaskResourceType>(type);
  return resourceType && mlir::isa<mlir::sculptor::LogicalArrayType>(
                             resourceType.getValueType());
}

static mlir::FailureOr<mlir::RankedTensorType>
getStaticRank2F32Tensor(mlir::Type type, mlir::Operation *anchor,
                        llvm::StringRef description) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType || !tensorType.hasStaticShape() ||
      tensorType.getRank() != 2 || !tensorType.getElementType().isF32()) {
    anchor->emitError("expected ")
        << description << " to be a static rank-2 f32 tensor";
    return mlir::failure();
  }
  for (int64_t dimension : tensorType.getShape()) {
    if (dimension <= 0) {
      anchor->emitError("expected ")
          << description << " to have positive static dimensions";
      return mlir::failure();
    }
  }
  return tensorType;
}

static mlir::FailureOr<mlir::func::FuncOp>
lookupTaskCallee(mlir::ModuleOp module, mlir::sculptor::TaskCreateOp task) {
  auto callee =
      module.lookupSymbol<mlir::func::FuncOp>(task.getCalleeAttr().getValue());
  if (!callee) {
    return task.emitOpError("expected task callee '")
           << task.getCalleeAttr().getValue() << "'";
  }
  if (callee.isDeclaration())
    return task.emitOpError("expected task callee to have a body");
  return callee;
}

static mlir::FailureOr<int64_t>
getScheduledCore(mlir::sculptor::TaskCreateOp task, mlir::func::FuncOp callee) {
  auto taskCore = task->getAttrOfType<mlir::IntegerAttr>(
      runtime_attrs::kTaskCoreIdAttrName);
  if (!taskCore) {
    return task.emitOpError("expected scheduled core placement; run "
                            "--sculptor-schedule-task-graph before "
                            "--sculptor-lower-scheduled-mvm-to-digital");
  }
  auto calleeCore = callee->getAttrOfType<mlir::IntegerAttr>(
      runtime_attrs::kTaskCoreIdAttrName);
  if (!calleeCore || calleeCore.getInt() != taskCore.getInt()) {
    return task.emitOpError(
        "expected task and callee to carry the same scheduled core ID");
  }
  return taskCore.getInt();
}

static mlir::FailureOr<std::pair<mlir::RankedTensorType, mlir::TypedAttr>>
getSetupWeight(mlir::sculptor::TaskCreateOp task, mlir::func::FuncOp callee) {
  llvm::SmallVector<mlir::sculptor::ArraySetOp, 2> arraySets;
  callee.walk([&](mlir::sculptor::ArraySetOp op) { arraySets.push_back(op); });
  if (arraySets.size() != 1) {
    return task.emitOpError(
        "expected matrix-setup callee to contain exactly one "
        "sculptor.array.set");
  }

  mlir::Value matrix = arraySets.front().getMatrix();
  auto constant = matrix.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant) {
    return task.emitOpError(
        "expected matrix-setup array value to come from arith.constant");
  }
  auto weightType = getStaticRank2F32Tensor(matrix.getType(), task,
                                            "matrix-setup weight tile");
  if (mlir::failed(weightType))
    return mlir::failure();

  auto value = mlir::dyn_cast<mlir::TypedAttr>(constant.getValue());
  if (!value || value.getType() != *weightType) {
    return task.emitOpError(
        "expected matrix-setup weight constant to match its tensor type");
  }
  return std::make_pair(*weightType, value);
}

static mlir::LogicalResult verifyUnfinalized(mlir::func::FuncOp taskGraphFunc) {
  if (taskGraphFunc->hasAttr(runtime_attrs::kTaskGraphResourceCountAttrName)) {
    return taskGraphFunc.emitError(
        "--sculptor-lower-scheduled-mvm-to-digital must run before "
        "--sculptor-finalize-task-graph-resources");
  }
  for (mlir::sculptor::TaskCreateOp task :
       taskGraphFunc.getOps<mlir::sculptor::TaskCreateOp>()) {
    if (task->hasAttr(runtime_attrs::kTaskIndexAttrName) ||
        task->hasAttr(runtime_attrs::kTaskInputSlotsAttrName) ||
        task->hasAttr(runtime_attrs::kTaskOutputSlotsAttrName)) {
      return task.emitOpError(
          "expected an unfinalized task graph without local task indices or "
          "resource slots");
    }
  }
  return mlir::success();
}

static mlir::FailureOr<llvm::SmallVector<MatrixSetup>>
collectMatrixSetups(mlir::ModuleOp module, mlir::func::FuncOp taskGraphFunc) {
  llvm::SmallVector<MatrixSetup> setups;
  llvm::SmallPtrSet<mlir::Operation *, 8> seenCallees;

  for (mlir::sculptor::TaskCreateOp task :
       taskGraphFunc.getOps<mlir::sculptor::TaskCreateOp>()) {
    if (!task_graph::isMatrixSetupTask(task))
      continue;

    auto callee = lookupTaskCallee(module, task);
    if (mlir::failed(callee))
      return mlir::failure();
    if (!seenCallees.insert(callee->getOperation()).second) {
      task.emitOpError(
          "expected each matrix-setup task to have a unique callee");
      return mlir::failure();
    }
    auto coreId = getScheduledCore(task, *callee);
    if (mlir::failed(coreId))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 2> logicalOutputs;
    for (mlir::Value output : task.getOutputs()) {
      if (isLogicalArrayType(output.getType()))
        logicalOutputs.push_back(output);
    }
    if (logicalOutputs.size() != 1 ||
        task.getOutputs().size() != logicalOutputs.size()) {
      task.emitOpError(
          "expected matrix-setup task to produce exactly one logical array");
      return mlir::failure();
    }
    if (!task.getInputs().empty()) {
      task.emitOpError("expected matrix-setup task to have no inputs");
      return mlir::failure();
    }
    if ((*callee).getNumArguments() != 0 || (*callee).getNumResults() != 1 ||
        !isLogicalArrayType((*callee).getResultTypes().front())) {
      task.emitOpError(
          "expected matrix-setup callee type () -> !sculptor.logical.array");
      return mlir::failure();
    }

    auto weight = getSetupWeight(task, *callee);
    if (mlir::failed(weight))
      return mlir::failure();
    setups.push_back(MatrixSetup{task, *callee, logicalOutputs.front(),
                                 weight->first, weight->second, *coreId});
  }
  return setups;
}

static mlir::FailureOr<DigitalMVM>
matchDigitalMVM(mlir::ModuleOp module, mlir::sculptor::TaskCreateOp task,
                llvm::ArrayRef<MatrixSetup> setups,
                const llvm::DenseMap<mlir::Operation *, unsigned> &setupByTask,
                const task_graph::ResourceProducerMap &producerByResource) {
  auto geometry = task_graph::resolveDigitalMatmulGeometry(module, task,
                                                           producerByResource);
  if (mlir::failed(geometry))
    return mlir::failure();
  auto coreId = getScheduledCore(task, geometry->mvmCallee);
  if (mlir::failed(coreId))
    return mlir::failure();

  auto setupIt = setupByTask.find(geometry->matrixSetupTask.getOperation());
  if (setupIt == setupByTask.end()) {
    return task.emitOpError(
        "expected logical-array input to be produced by a matrix-setup task");
  }
  const MatrixSetup &setup = setups[setupIt->second];
  if (setup.coreId != *coreId) {
    return task.emitOpError(
        "expected MVM and matrix-setup tasks to occupy the same core");
  }

  auto replacementOps = task_graph::computeDigitalMatmulScalarOps(
      task, geometry->executionRows, geometry->physicalRows,
      geometry->physicalColumns);
  auto existingDigitalOps = task_graph::estimateTaskDigitalOps(module, task);
  if (mlir::failed(replacementOps) || mlir::failed(existingDigitalOps))
    return mlir::failure();
  std::optional<int64_t> digitalOps =
      llvm::checkedAdd(*existingDigitalOps, *replacementOps);
  if (!digitalOps)
    return task.emitOpError("digital operation count overflow");

  bool hasArrayLoad = false;
  bool hasArrayExecute = false;
  bool hasArrayStore = false;
  geometry->mvmCallee.walk(
      [&](mlir::sculptor::ArrayLoadOp) { hasArrayLoad = true; });
  geometry->mvmCallee.walk(
      [&](mlir::sculptor::ArrayExecuteOp) { hasArrayExecute = true; });
  geometry->mvmCallee.walk(
      [&](mlir::sculptor::ArrayStoreOp) { hasArrayStore = true; });
  if (!hasArrayLoad || !hasArrayExecute || !hasArrayStore) {
    return task.emitOpError(
        "expected scheduled MVM callee to contain load, execute, and store");
  }

  return DigitalMVM{task,
                    geometry->mvmCallee,
                    setupIt->second,
                    geometry->logicalArrayInputIndex,
                    geometry->inputType,
                    geometry->resultType,
                    *coreId,
                    *digitalOps,
                    geometry->needsVectorTileExtraction,
                    geometry->vectorTile,
                    geometry->validColumns};
}

static llvm::SmallVector<mlir::OpFoldResult>
getIndexAttrs(mlir::OpBuilder &builder, llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::OpFoldResult> result;
  result.reserve(values.size());
  for (int64_t value : values)
    result.push_back(builder.getIndexAttr(value));
  return result;
}

static mlir::Value buildConvolutionVectorTile(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::RankedTensorType sourceType, int64_t vectorTile, int64_t validCols,
    int64_t physicalCols) {
  int64_t rows = sourceType.getDimSize(0);
  int64_t columnOffset = vectorTile * physicalCols;
  auto sliceType = mlir::RankedTensorType::get({rows, validCols},
                                               sourceType.getElementType());
  mlir::Value slice =
      builder
          .create<mlir::tensor::ExtractSliceOp>(
              loc, sliceType, source, getIndexAttrs(builder, {0, columnOffset}),
              getIndexAttrs(builder, {rows, validCols}),
              getIndexAttrs(builder, {1, 1}))
          .getResult();
  if (validCols == physicalCols)
    return slice;

  auto physicalType = mlir::RankedTensorType::get({rows, physicalCols},
                                                  sourceType.getElementType());
  auto zeroAttr =
      mlir::cast<mlir::TypedAttr>(builder.getZeroAttr(physicalType));
  mlir::Value zero =
      builder.create<mlir::arith::ConstantOp>(loc, physicalType, zeroAttr)
          .getResult();
  return builder
      .create<mlir::tensor::InsertSliceOp>(
          loc, slice, zero, getIndexAttrs(builder, {0, 0}),
          getIndexAttrs(builder, {rows, validCols}),
          getIndexAttrs(builder, {1, 1}))
      .getResult();
}

static void clearAnalogExecutionAttrs(mlir::Operation *op) {
  for (llvm::StringRef name :
       {llvm::StringRef(runtime_attrs::kTaskAnalogExecutionCountAttrName),
        llvm::StringRef(runtime_attrs::kTaskAnalogExecutionCountsAttrName),
        llvm::StringRef(runtime_attrs::kTaskAnalogLoadBytesAttrName),
        llvm::StringRef(runtime_attrs::kTaskAnalogStoreBytesAttrName),
        llvm::StringRef(runtime_attrs::kTaskArrayBindingsAttrName)})
    op->removeAttr(name);
}

static mlir::LogicalResult rewriteMVMFunction(const DigitalMVM &mvm,
                                              const MatrixSetup &setup) {
  mlir::func::FuncOp func = mvm.callee;
  mlir::Location loc = func.getLoc();
  mlir::MLIRContext *context = func.getContext();
  mlir::OpBuilder builder(context);

  llvm::SmallVector<mlir::Type, 2> retainedInputs;
  for (auto indexedType : llvm::enumerate(func.getArgumentTypes())) {
    if (indexedType.index() != mvm.logicalInputIndex)
      retainedInputs.push_back(indexedType.value());
  }
  if (retainedInputs.size() != 1)
    return func.emitError("expected one retained digital MVM input");

  func.getBody().getBlocks().clear();
  func.setType(builder.getFunctionType(retainedInputs, {mvm.resultType}));
  mlir::Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  mlir::Value vector = entry->getArgument(0);
  int64_t physicalRows = setup.weightType.getDimSize(0);
  int64_t physicalCols = setup.weightType.getDimSize(1);
  if (mvm.needsVectorTileExtraction) {
    vector =
        buildConvolutionVectorTile(builder, loc, vector, mvm.vectorType,
                                   mvm.vectorTile, mvm.validCols, physicalCols);
  }

  mlir::Value weight = builder
                           .create<mlir::arith::ConstantOp>(
                               loc, setup.weightType, setup.weightValue)
                           .getResult();
  int64_t rows = mvm.resultType.getDimSize(0);
  auto physicalResultType = mlir::RankedTensorType::get(
      {rows, physicalRows}, mvm.resultType.getElementType());
  mlir::Value empty =
      builder
          .create<mlir::tensor::EmptyOp>(loc, physicalResultType.getShape(),
                                         physicalResultType.getElementType())
          .getResult();
  mlir::Value zero =
      builder
          .create<mlir::arith::ConstantOp>(loc, builder.getF32Type(),
                                           builder.getF32FloatAttr(0.0))
          .getResult();
  mlir::Value initialized =
      builder.create<mlir::linalg::FillOp>(loc, zero, empty).getResult(0);
  mlir::Value product = builder
                            .create<mlir::linalg::MatmulTransposeBOp>(
                                loc, mlir::ValueRange{vector, weight},
                                mlir::ValueRange{initialized})
                            .getResult(0);

  mlir::Value result = product;
  int64_t validRows = mvm.resultType.getDimSize(1);
  if (validRows != physicalRows) {
    result =
        builder
            .create<mlir::tensor::ExtractSliceOp>(
                loc, mvm.resultType, product, getIndexAttrs(builder, {0, 0}),
                getIndexAttrs(builder, {rows, validRows}),
                getIndexAttrs(builder, {1, 1}))
            .getResult();
  }
  builder.create<mlir::func::ReturnOp>(loc, result);

  func->setAttr(task_attrs::kTaskDomainAttrName,
                builder.getStringAttr(task_graph_names::kDigitalDomain));
  func->setAttr(
      task_attrs::kTaskKindAttrName,
      builder.getStringAttr(task_graph_names::kDigitalMatmulTaskKind));
  func->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                builder.getI64IntegerAttr(mvm.digitalOps));
  clearAnalogExecutionAttrs(func);
  return mlir::success();
}

static mlir::LogicalResult rewriteCallsToMVM(mlir::ModuleOp module,
                                             DigitalMVM &mvm) {
  mlir::func::FuncOp callee = mvm.callee;
  llvm::SmallVector<mlir::func::CallOp, 4> calls;
  module.walk([&](mlir::func::CallOp call) {
    if (call.getCallee() == callee.getSymName())
      calls.push_back(call);
  });

  for (mlir::func::CallOp call : calls) {
    if (call.getNumOperands() <= mvm.logicalInputIndex ||
        call.getNumOperands() - 1 != callee.getNumArguments()) {
      return call.emitOpError(
          "expected call operands to match pre-digital MVM signature");
    }
    llvm::SmallVector<mlir::Value, 2> operands;
    for (auto indexedOperand : llvm::enumerate(call.getOperands())) {
      if (indexedOperand.index() != mvm.logicalInputIndex)
        operands.push_back(indexedOperand.value());
    }

    mlir::OpBuilder builder(call);
    auto replacement = builder.create<mlir::func::CallOp>(
        call.getLoc(), callee.getSymName(), callee.getResultTypes(), operands);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(call.getResults(), replacement.getResults()))
      oldResult.replaceAllUsesWith(newResult);
    call.erase();
  }
  return mlir::success();
}

static mlir::LogicalResult rewriteSetupFunction(MatrixSetup &setup) {
  mlir::func::FuncOp func = setup.callee;
  mlir::OpBuilder builder(func.getContext());
  mlir::Location loc = func.getLoc();
  func.getBody().getBlocks().clear();
  func.setType(builder.getFunctionType({}, {}));
  mlir::Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  builder.create<mlir::func::ReturnOp>(loc);
  func->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                builder.getI64IntegerAttr(0));
  clearAnalogExecutionAttrs(func);
  return mlir::success();
}

static mlir::LogicalResult rewriteCallsToSetup(mlir::ModuleOp module,
                                               MatrixSetup &setup) {
  mlir::func::FuncOp callee = setup.callee;
  llvm::SmallVector<mlir::func::CallOp, 4> calls;
  module.walk([&](mlir::func::CallOp call) {
    if (call.getCallee() == callee.getSymName())
      calls.push_back(call);
  });

  for (mlir::func::CallOp call : calls) {
    if (call.getNumOperands() != 0 || call.getNumResults() != 1 ||
        !isLogicalArrayType(call.getResult(0).getType())) {
      return call.emitOpError(
          "expected matrix-setup call type () -> !sculptor.logical.array");
    }
    if (!call.getResult(0).use_empty()) {
      return call.emitOpError(
          "matrix-setup result still has users after MVM digitalization");
    }
    mlir::OpBuilder builder(call);
    builder.create<mlir::func::CallOp>(call.getLoc(), callee.getSymName(),
                                       mlir::TypeRange{}, mlir::ValueRange{});
    call.erase();
  }
  return mlir::success();
}

static mlir::LogicalResult
refreshStructuralMetadata(mlir::func::FuncOp taskGraphFunc) {
  auto dag = task_graph::parseTaskGraphDAG(taskGraphFunc);
  if (mlir::failed(dag))
    return mlir::failure();

  mlir::Builder builder(taskGraphFunc.getContext());
  taskGraphFunc->setAttr(
      schedule_attrs::kTaskCountAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(dag->nodes.size())));
  taskGraphFunc->setAttr(
      schedule_attrs::kDependencyCountAttrName,
      builder.getI64IntegerAttr(static_cast<int64_t>(dag->dependencyCount)));

  int64_t totalDigitalOps = 0;
  for (const task_graph::TaskGraphNode &node : dag->nodes) {
    mlir::sculptor::TaskCreateOp task = node.op;
    auto digitalOps = task->getAttrOfType<mlir::IntegerAttr>(
        runtime_attrs::kTaskDigitalOpsAttrName);
    if (!digitalOps)
      continue;
    if (digitalOps.getInt() < 0 ||
        totalDigitalOps >
            std::numeric_limits<int64_t>::max() - digitalOps.getInt()) {
      return task.emitOpError("invalid or overflowing digital operation count");
    }
    totalDigitalOps += digitalOps.getInt();
  }
  taskGraphFunc->setAttr(schedule_attrs::kTotalDigitalOpsAttrName,
                         builder.getI64IntegerAttr(totalDigitalOps));
  return mlir::success();
}

static mlir::LogicalResult verifyNoAnalogArrayIR(mlir::ModuleOp module) {
  bool foundArrayOp = false;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef().starts_with("sculptor.array."))
      foundArrayOp = true;
  });
  if (foundArrayOp) {
    return module.emitError(
        "expected all scheduled analog array operations to be digitalized");
  }

  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    for (mlir::Type type : func.getArgumentTypes()) {
      if (isLogicalArrayType(type))
        return func.emitError(
            "logical-array argument remains after MVM digitalization");
    }
    for (mlir::Type type : func.getResultTypes()) {
      if (isLogicalArrayType(type))
        return func.emitError(
            "logical-array result remains after MVM digitalization");
    }
  }
  return mlir::success();
}

static mlir::LogicalResult lowerTaskGraphMVMs(mlir::ModuleOp module,
                                              mlir::func::FuncOp taskGraphFunc,
                                              bool &changed) {
  if (mlir::failed(verifyUnfinalized(taskGraphFunc)))
    return mlir::failure();

  auto setups = collectMatrixSetups(module, taskGraphFunc);
  if (mlir::failed(setups))
    return mlir::failure();

  auto dag = task_graph::parseTaskGraphDAG(taskGraphFunc);
  if (mlir::failed(dag))
    return mlir::failure();
  task_graph::ResourceProducerMap producerByResource;
  if (mlir::failed(
          task_graph::collectResourceProducers(*dag, producerByResource)))
    return mlir::failure();

  llvm::DenseMap<mlir::Operation *, unsigned> setupByTask;
  for (auto indexedSetup : llvm::enumerate(*setups)) {
    if (!setupByTask
             .try_emplace(indexedSetup.value().task.getOperation(),
                          indexedSetup.index())
             .second) {
      return indexedSetup.value().task.emitOpError(
          "duplicate matrix-setup task");
    }
  }

  llvm::SmallVector<DigitalMVM> mvms;
  llvm::SmallPtrSet<mlir::Operation *, 16> seenMVMCallees;
  for (mlir::sculptor::TaskCreateOp task :
       taskGraphFunc.getOps<mlir::sculptor::TaskCreateOp>()) {
    if (!task_graph::isAnalogComputeTask(task))
      continue;
    auto mvm =
        matchDigitalMVM(module, task, *setups, setupByTask, producerByResource);
    if (mlir::failed(mvm))
      return mlir::failure();
    if (!seenMVMCallees.insert(mvm->callee.getOperation()).second) {
      return task.emitOpError(
          "expected each scheduled MVM task to have a unique callee");
    }
    mvms.push_back(*mvm);
  }
  if (mvms.empty())
    return taskGraphFunc.emitError("expected at least one scheduled MVM task");

  llvm::SmallVector<bool> setupUsed(setups->size(), false);
  for (DigitalMVM &mvm : mvms) {
    MatrixSetup &setup = (*setups)[mvm.setupIndex];
    setupUsed[mvm.setupIndex] = true;
    if (mlir::failed(rewriteMVMFunction(mvm, setup)) ||
        mlir::failed(rewriteCallsToMVM(module, mvm)))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> retainedInputs;
    for (auto indexedInput : llvm::enumerate(mvm.task.getInputs())) {
      if (indexedInput.index() != mvm.logicalInputIndex)
        retainedInputs.push_back(indexedInput.value());
    }
    mvm.task.getInputsMutable().assign(retainedInputs);
    mvm.task.setDomain(task_graph_names::kDigitalDomain);
    mvm.task.setTaskKind(task_graph_names::kDigitalMatmulTaskKind);
    mvm.task->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                      mlir::Builder(module).getI64IntegerAttr(mvm.digitalOps));
    clearAnalogExecutionAttrs(mvm.task);
  }

  for (auto indexedSetup : llvm::enumerate(*setups)) {
    MatrixSetup &setup = indexedSetup.value();
    if (!setupUsed[indexedSetup.index()]) {
      return setup.task.emitOpError(
          "matrix-setup task is not consumed by a scheduled MVM");
    }
    setup.task.getOutputsMutable().clear();
    setup.task->removeAttr(runtime_attrs::kTaskResultIndicesAttrName);
    setup.task->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                        mlir::Builder(module).getI64IntegerAttr(0));
    clearAnalogExecutionAttrs(setup.task);
    if (mlir::failed(rewriteSetupFunction(setup)) ||
        mlir::failed(rewriteCallsToSetup(module, setup)))
      return mlir::failure();
  }

  if (mlir::failed(task_graph::eraseUnusedTaskGraphIntermediateResources(
          taskGraphFunc))) {
    return taskGraphFunc.emitError(
        "failed to erase logical-array resources after MVM digitalization");
  }
  if (mlir::failed(refreshStructuralMetadata(taskGraphFunc)))
    return mlir::failure();
  mlir::sculptor::task_timing::invalidateTaskGraphStructure(taskGraphFunc);
  changed = true;
  return mlir::success();
}

} // namespace

namespace mlir {
namespace sculptor {

void LowerScheduledMVMToDigitalPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::sculptor::SculptorDialect,
                  mlir::tensor::TensorDialect>();
}

void LowerScheduledMVMToDigitalPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  bool foundTaskGraph = false;
  bool changed = false;
  for (mlir::func::FuncOp func : module.getOps<mlir::func::FuncOp>()) {
    if (!returnsTaskGraph(func))
      continue;
    foundTaskGraph = true;
    if (mlir::failed(lowerTaskGraphMVMs(module, func, changed))) {
      signalPassFailure();
      return;
    }
  }
  if (!foundTaskGraph) {
    module.emitError("expected a scheduled task graph function");
    signalPassFailure();
    return;
  }
  if (!changed || mlir::failed(verifyNoAnalogArrayIR(module)))
    signalPassFailure();
}

void registerLowerScheduledMVMToDigitalPass() {
  mlir::PassRegistration<LowerScheduledMVMToDigitalPass>();
}

} // namespace sculptor
} // namespace mlir
