#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphDigitalOps.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"

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

static int64_t inferDigitalOpsFromCallee(func::FuncOp callee) {
  if (!callee || callee.isDeclaration() || !callee.getBody().hasOneBlock())
    return 0;

  int64_t digitalOps = 0;
  for (Operation &op : callee.getBody().front().without_terminator()) {
    if (op.getName().getDialectNamespace() != "linalg")
      continue;

    int64_t elementCount = getStaticElementCount(&op);
    if (elementCount <= 0)
      continue;

    if (op.getName().getStringRef() == "linalg.generic") {
      digitalOps += elementCount * countScalarDigitalOps(&op);
      continue;
    }
    if (isSingleScalarOpLinalg(&op))
      digitalOps += elementCount;
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
