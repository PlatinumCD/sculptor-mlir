#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDigitalOps.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

static bool shouldInferDigitalOpsFromCallee(sculptor::TaskCreateOp taskOp) {
  return taskOp.getDomain() != task_graph_names::kAnalogDomain ||
         taskOp.getTaskKind() == task_graph_names::kConvTileMVMTaskKind;
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
          runtime_attrs::kTaskDigitalOpsAttrName))
    return attr.getInt();

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

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
