#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_schedulers/TaskGraphHardwareConfig.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace mlir {
namespace sculptor {
namespace task_schedulers {
namespace {

static ArrayAttr buildI64ArrayAttr(Builder &builder,
                                   llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

} // namespace

FailureOr<HardwareBudget>
buildHardwareBudget(ModuleOp module, int64_t numCores,
                    int64_t arraysPerCore, llvm::StringRef topology,
                    int64_t meshRows, int64_t meshCols) {
  if (numCores <= 0) {
    module.emitError("expected Sculptor scheduling budget to have at least one "
                     "core");
    return failure();
  }
  if (arraysPerCore <= 0) {
    module.emitError("expected Sculptor scheduling budget to have at least one "
                     "array per core");
    return failure();
  }
  if (topology != "mesh") {
    module.emitError("unknown Sculptor scheduling topology '")
        << topology << "'";
    return failure();
  }
  if (meshRows <= 0) {
    module.emitError("expected mesh topology to have at least one row");
    return failure();
  }
  if (meshCols < 0) {
    module.emitError("expected mesh topology column count to be non-negative");
    return failure();
  }
  if (meshCols == 0) {
    if (numCores % meshRows != 0) {
      module.emitError("expected mesh topology rows to evenly divide the "
                       "number of cores when mesh-cols is inferred");
      return failure();
    }
    meshCols = numCores / meshRows;
  }
  if (meshCols <= 0) {
    module.emitError("expected mesh topology to have at least one column");
    return failure();
  }
  if (meshRows > std::numeric_limits<int64_t>::max() / meshCols) {
    module.emitError("Sculptor scheduling mesh topology overflows core count");
    return failure();
  }
  if (meshRows * meshCols != numCores) {
    module.emitError("expected mesh topology dimensions to match the number "
                     "of cores");
    return failure();
  }
  if (numCores > std::numeric_limits<int64_t>::max() / arraysPerCore) {
    module.emitError("Sculptor scheduling budget overflows total array count");
    return failure();
  }

  HardwareBudget budget;
  budget.numCores = numCores;
  budget.arraysPerCore = arraysPerCore;
  budget.topology = topology.str();
  budget.meshRows = meshRows;
  budget.meshCols = meshCols;
  budget.numAnalogArrays = numCores * arraysPerCore;
  budget.analogArrays.reserve(static_cast<size_t>(budget.numAnalogArrays));
  for (int64_t analogArray = 0; analogArray < budget.numAnalogArrays;
       ++analogArray)
    budget.analogArrays.push_back(analogArray);
  return budget;
}

void attachHardwareBudgetAttrs(Operation *op, Builder &builder,
                               const HardwareBudget &budget) {
  op->setAttr(schedule_attrs::kNumCoresAttrName,
              builder.getI64IntegerAttr(budget.numCores));
  op->setAttr(schedule_attrs::kArraysPerCoreAttrName,
              builder.getI64IntegerAttr(budget.arraysPerCore));
  op->setAttr(schedule_attrs::kTopologyAttrName,
              builder.getStringAttr(budget.topology));
  op->setAttr(schedule_attrs::kMeshRowsAttrName,
              builder.getI64IntegerAttr(budget.meshRows));
  op->setAttr(schedule_attrs::kMeshColsAttrName,
              builder.getI64IntegerAttr(budget.meshCols));
  op->setAttr(schedule_attrs::kNumAnalogArraysAttrName,
              builder.getI64IntegerAttr(budget.numAnalogArrays));
  op->setAttr(schedule_attrs::kAnalogArraysAttrName,
              buildI64ArrayAttr(builder, budget.analogArrays));
}

} // namespace task_schedulers
} // namespace sculptor
} // namespace mlir
