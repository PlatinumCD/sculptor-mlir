#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDigitalOps.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTilingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphResources.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphTaskKinds.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <limits>

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

namespace task_graph_names = mlir::sculptor::task_graph_names;
namespace tiling_attrs = mlir::sculptor::tiling_attrs;

static bool shouldInferDigitalOpsFromCallee(sculptor::TaskCreateOp taskOp) {
  return taskOp.getDomain() != task_graph_names::kAnalogDomain ||
         taskOp.getTaskKind() == task_graph_names::kConvTileMVMTaskKind;
}

static bool isLogicalArrayType(Type type) {
  if (isa<sculptor::LogicalArrayType>(type))
    return true;
  auto resourceType = dyn_cast<sculptor::TaskResourceType>(type);
  return resourceType &&
         isa<sculptor::LogicalArrayType>(resourceType.getValueType());
}

static Type getTaskResourceValueType(Value resource) {
  auto resourceType = dyn_cast<sculptor::TaskResourceType>(resource.getType());
  return resourceType ? resourceType.getValueType() : Type{};
}

static FailureOr<RankedTensorType>
getStaticRank2F32Tensor(Type type, Operation *anchor, StringRef description) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.hasStaticShape() ||
      tensorType.getRank() != 2 || !tensorType.getElementType().isF32()) {
    anchor->emitError("expected ")
        << description << " to be a static rank-2 f32 tensor";
    return failure();
  }
  for (int64_t dimension : tensorType.getShape()) {
    if (dimension <= 0) {
      anchor->emitError("expected ")
          << description << " to have positive static dimensions";
      return failure();
    }
  }
  return tensorType;
}

static FailureOr<func::FuncOp> lookupTaskCallee(ModuleOp module,
                                                sculptor::TaskCreateOp taskOp) {
  auto callee =
      module.lookupSymbol<func::FuncOp>(taskOp.getCalleeAttr().getValue());
  if (!callee) {
    return taskOp.emitOpError("expected task callee '")
           << taskOp.getCalleeAttr().getValue() << "'";
  }
  if (callee.isDeclaration())
    return taskOp.emitOpError("expected task callee to have a body");
  return callee;
}

static FailureOr<RankedTensorType>
getSetupWeightType(sculptor::TaskCreateOp taskOp, func::FuncOp callee) {
  llvm::SmallVector<sculptor::ArraySetOp, 2> arraySets;
  callee.walk([&](sculptor::ArraySetOp op) { arraySets.push_back(op); });
  if (arraySets.size() != 1) {
    return taskOp.emitOpError(
        "expected matrix-setup callee to contain exactly one "
        "sculptor.array.set");
  }
  return getStaticRank2F32Tensor(arraySets.front().getMatrix().getType(),
                                 taskOp, "matrix-setup weight tile");
}

static FailureOr<int64_t> getRequiredI64Attr(Operation *metadataOwner,
                                             Operation *diagnosticAnchor,
                                             StringRef name,
                                             StringRef description) {
  auto attr = metadataOwner->getAttrOfType<IntegerAttr>(name);
  if (!attr) {
    diagnosticAnchor->emitError("expected ")
        << description << " attribute '" << name << "'";
    return failure();
  }
  return attr.getInt();
}

static int64_t getStaticElementCount(Type type) {
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType || !shapedType.hasStaticShape())
    return 0;
  return shapedType.getNumElements();
}

static int64_t getStaticElementCount(Operation *op) {
  for (Type resultType : op->getResultTypes()) {
    int64_t elementCount = getStaticElementCount(resultType);
    if (elementCount > 0)
      return elementCount;
  }

  for (Value operand : llvm::reverse(op->getOperands())) {
    int64_t elementCount = getStaticElementCount(operand.getType());
    if (elementCount > 0)
      return elementCount;
  }
  return 0;
}

static bool isScalarDigitalOp(Operation *op) {
  StringRef dialectNamespace = op->getName().getDialectNamespace();
  return dialectNamespace == "arith" || dialectNamespace == "math";
}

static int64_t countScalarDigitalOps(Operation *linalgOp) {
  int64_t scalarOps = 0;
  for (Region &region : linalgOp->getRegions()) {
    region.walk([&](Operation *nestedOp) {
      if (nestedOp == linalgOp || nestedOp->hasTrait<OpTrait::IsTerminator>())
        return;
      if (isScalarDigitalOp(nestedOp))
        ++scalarOps;
    });
  }
  return scalarOps;
}

static bool isSingleScalarOpLinalg(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return opName == "linalg.add" || opName == "linalg.sub" ||
         opName == "linalg.mul" || opName == "linalg.div" ||
         opName == "linalg.max" || opName == "linalg.min";
}

static FailureOr<int64_t> getStaticIterationCount(linalg::LinalgOp linalgOp) {
  int64_t iterationCount = 1;
  for (int64_t loopRange : linalgOp.getStaticLoopRanges()) {
    if (ShapedType::isDynamic(loopRange)) {
      linalgOp.emitError(
          "cannot infer digital operations from a dynamic loop range");
      return failure();
    }
    if (loopRange < 0) {
      linalgOp.emitError(
          "cannot infer digital operations from a negative loop range");
      return failure();
    }

    std::optional<int64_t> product =
        llvm::checkedMul(iterationCount, loopRange);
    if (!product) {
      linalgOp.emitError("digital operation iteration count overflow");
      return failure();
    }
    iterationCount = *product;
  }
  return iterationCount;
}

static LogicalResult addDigitalOps(Operation *anchor, int64_t amount,
                                   int64_t &total) {
  std::optional<int64_t> sum = llvm::checkedAdd(total, amount);
  if (!sum) {
    anchor->emitError("digital operation count overflow");
    return failure();
  }
  total = *sum;
  return success();
}

static FailureOr<int64_t> inferDigitalOpsFromCallee(func::FuncOp callee) {
  if (!callee || callee.isDeclaration() || !callee.getBody().hasOneBlock())
    return int64_t{0};

  int64_t digitalOps = 0;
  for (Operation &op : callee.getBody().front().without_terminator()) {
    if (op.getName().getDialectNamespace() != "linalg")
      continue;

    if (auto genericOp = dyn_cast<linalg::GenericOp>(&op)) {
      FailureOr<int64_t> iterationCount = getStaticIterationCount(
          cast<linalg::LinalgOp>(genericOp.getOperation()));
      if (failed(iterationCount))
        return failure();
      std::optional<int64_t> genericOps =
          llvm::checkedMul(*iterationCount, countScalarDigitalOps(&op));
      if (!genericOps) {
        op.emitError("digital operation count overflow");
        return failure();
      }
      if (failed(addDigitalOps(&op, *genericOps, digitalOps)))
        return failure();
      continue;
    }

    int64_t elementCount = getStaticElementCount(&op);
    if (elementCount <= 0)
      continue;
    if (isSingleScalarOpLinalg(&op) &&
        failed(addDigitalOps(&op, elementCount, digitalOps)))
      return failure();
  }
  return digitalOps;
}

} // namespace

FailureOr<int64_t> estimateTaskDigitalOps(ModuleOp module,
                                          sculptor::TaskCreateOp taskOp) {
  if (auto attr = taskOp->getAttrOfType<IntegerAttr>(
          runtime_attrs::kTaskDigitalOpsAttrName)) {
    if (attr.getInt() < 0)
      return taskOp.emitOpError(
          "expected non-negative digital operation count");
    return attr.getInt();
  }

  if (!shouldInferDigitalOpsFromCallee(taskOp))
    return int64_t{0};

  auto callee =
      module.lookupSymbol<func::FuncOp>(taskOp.getCalleeAttr().getValue());
  if (!callee) {
    return taskOp.emitError("expected task callee '")
           << taskOp.getCalleeAttr().getValue()
           << "' to resolve to a function for digital op accounting";
  }
  return inferDigitalOpsFromCallee(callee);
}

FailureOr<DigitalMatmulGeometry>
resolveDigitalMatmulGeometry(ModuleOp module, sculptor::TaskCreateOp taskOp,
                             const ResourceProducerMap &producerByResource) {
  if (!isAnalogComputeTask(taskOp)) {
    return taskOp.emitOpError(
        "digital replacement geometry is only defined for static MVM tasks");
  }

  auto mvmCallee = lookupTaskCallee(module, taskOp);
  if (failed(mvmCallee))
    return failure();

  unsigned logicalArrayInputIndex = 0;
  unsigned logicalArrayInputCount = 0;
  llvm::SmallVector<unsigned, 2> tensorInputIndices;
  for (auto indexedInput : llvm::enumerate(taskOp.getInputs())) {
    if (isLogicalArrayType(indexedInput.value().getType())) {
      logicalArrayInputIndex = indexedInput.index();
      ++logicalArrayInputCount;
    } else {
      tensorInputIndices.push_back(indexedInput.index());
    }
  }
  if (logicalArrayInputCount != 1 || tensorInputIndices.size() != 1 ||
      taskOp.getOutputs().size() != 1) {
    return taskOp.emitOpError(
        "expected an MVM task to have one tensor input, one logical-array "
        "input, and one tensor output");
  }

  Value logicalResource = taskOp.getInputs()[logicalArrayInputIndex];
  auto setupIt = producerByResource.find(logicalResource);
  if (setupIt == producerByResource.end() ||
      !isMatrixSetupTask(setupIt->second->op)) {
    return taskOp.emitOpError(
        "expected logical-array input to be produced by a matrix-setup task");
  }
  sculptor::TaskCreateOp setupTask = setupIt->second->op;
  auto setupCallee = lookupTaskCallee(module, setupTask);
  if (failed(setupCallee))
    return failure();

  if ((*mvmCallee).getNumArguments() != taskOp.getInputs().size() ||
      (*mvmCallee).getNumResults() != taskOp.getOutputs().size() ||
      logicalArrayInputIndex >= (*mvmCallee).getNumArguments() ||
      !isLogicalArrayType(
          (*mvmCallee).getArgument(logicalArrayInputIndex).getType())) {
    return taskOp.emitOpError(
        "expected MVM task resources to match its callee signature");
  }

  unsigned tensorInputIndex = tensorInputIndices.front();
  auto inputType = getStaticRank2F32Tensor(
      getTaskResourceValueType(taskOp.getInputs()[tensorInputIndex]), taskOp,
      "MVM vector input resource");
  auto resultType = getStaticRank2F32Tensor(
      getTaskResourceValueType(taskOp.getOutputs().front()), taskOp,
      "MVM result resource");
  auto weightType = getSetupWeightType(setupTask, *setupCallee);
  if (failed(inputType) || failed(resultType) || failed(weightType))
    return failure();
  if ((*mvmCallee).getArgument(tensorInputIndex).getType() != *inputType ||
      (*mvmCallee).getResultTypes().front() != *resultType) {
    return taskOp.emitOpError(
        "expected MVM callee tensor types to match task resources");
  }

  int64_t executionRows = inputType->getDimSize(0);
  int64_t physicalRows = weightType->getDimSize(0);
  int64_t physicalColumns = weightType->getDimSize(1);
  int64_t validRows = resultType->getDimSize(1);
  if (resultType->getDimSize(0) != executionRows || validRows <= 0 ||
      validRows > physicalRows) {
    return taskOp.emitOpError(
        "expected MVM result shape to fit the physical weight rows");
  }

  bool needsVectorTileExtraction =
      taskOp.getTaskKind() == task_graph_names::kConvTileMVMTaskKind;
  int64_t vectorTile = 0;
  int64_t validColumns = physicalColumns;
  if (needsVectorTileExtraction) {
    auto vectorTileValue = getRequiredI64Attr(*mvmCallee, taskOp,
                                              tiling_attrs::kVectorTileAttrName,
                                              "convolution MVM vector tile");
    auto validColumnsValue = getRequiredI64Attr(
        *mvmCallee, taskOp, tiling_attrs::kVectorTileValidColsAttrName,
        "convolution MVM valid column count");
    auto physicalColumnsValue = getRequiredI64Attr(
        *mvmCallee, taskOp, tiling_attrs::kVectorTilePhysicalColsAttrName,
        "convolution MVM physical column count");
    if (failed(vectorTileValue) || failed(validColumnsValue) ||
        failed(physicalColumnsValue))
      return failure();
    vectorTile = *vectorTileValue;
    validColumns = *validColumnsValue;
    int64_t sourceColumns = inputType->getDimSize(1);
    if (vectorTile < 0 || validColumns <= 0 || validColumns > physicalColumns ||
        *physicalColumnsValue != physicalColumns ||
        vectorTile > std::numeric_limits<int64_t>::max() / physicalColumns ||
        vectorTile * physicalColumns > sourceColumns - validColumns) {
      return taskOp.emitOpError("invalid convolution MVM vector tile geometry");
    }
  } else if (inputType->getDimSize(1) != physicalColumns) {
    return taskOp.emitOpError(
        "expected MVM vector width to equal physical weight columns");
  }

  return DigitalMatmulGeometry{setupTask,        *setupCallee,
                               *mvmCallee,       logicalArrayInputIndex,
                               tensorInputIndex, *inputType,
                               *resultType,      *weightType,
                               executionRows,    physicalRows,
                               physicalColumns,  needsVectorTileExtraction,
                               vectorTile,       validColumns};
}

FailureOr<int64_t> computeDigitalMatmulScalarOps(Operation *anchor,
                                                 int64_t executionRows,
                                                 int64_t physicalRows,
                                                 int64_t physicalColumns) {
  if (executionRows <= 0 || physicalRows <= 0 || physicalColumns <= 0) {
    anchor->emitError("expected positive static digital matmul dimensions");
    return failure();
  }
  std::optional<int64_t> outputElements =
      llvm::checkedMul(executionRows, physicalRows);
  if (!outputElements) {
    anchor->emitError("digital matmul operation count overflow");
    return failure();
  }
  std::optional<int64_t> multiplyAccumulates =
      llvm::checkedMul(*outputElements, physicalColumns);
  if (!multiplyAccumulates) {
    anchor->emitError("digital matmul operation count overflow");
    return failure();
  }
  std::optional<int64_t> scalarOps =
      llvm::checkedMul(*multiplyAccumulates, int64_t{2});
  if (!scalarOps) {
    anchor->emitError("digital matmul operation count overflow");
    return failure();
  }
  return *scalarOps;
}

FailureOr<int64_t>
estimateDigitalReplacementOps(ModuleOp module, sculptor::TaskCreateOp taskOp,
                              const ResourceProducerMap &producerByResource) {
  auto geometry =
      resolveDigitalMatmulGeometry(module, taskOp, producerByResource);
  if (failed(geometry))
    return failure();
  return computeDigitalMatmulScalarOps(taskOp, geometry->executionRows,
                                       geometry->physicalRows,
                                       geometry->physicalColumns);
}

FailureOr<int64_t>
estimateDigitalReplacementOps(ModuleOp module, sculptor::TaskCreateOp taskOp) {
  func::FuncOp taskGraphFunc = taskOp->getParentOfType<func::FuncOp>();
  if (!taskGraphFunc)
    return taskOp.emitOpError("expected MVM task inside a task graph function");
  auto dag = parseTaskGraphDAG(taskGraphFunc);
  if (failed(dag))
    return failure();
  ResourceProducerMap producerByResource;
  if (failed(collectResourceProducers(*dag, producerByResource)))
    return failure();
  return estimateDigitalReplacementOps(module, taskOp, producerByResource);
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
