#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_timing/TaskCostAnalysis.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTilingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/TaskGraphTaskKinds.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mlir {
namespace sculptor {
namespace task_timing {
namespace {

struct StaticCostAnalysis {
  TaskCost cost;
  uint64_t staticElements = 0;
  uint64_t localBytesRead = 0;
  uint64_t localBytesWritten = 0;
  uint64_t loopIterations = 0;
  uint64_t analogLoadBytes = 0;
  uint64_t analogExecutionCount = 0;
  uint64_t analogStoreBytes = 0;
  bool foundExecutableOperation = false;
  bool foundUnsupportedOperation = false;
};

static LogicalResult checkedAdd(Operation *anchor, uint64_t amount,
                                uint64_t &total, StringRef description) {
  if (amount > std::numeric_limits<uint64_t>::max() - total) {
    anchor->emitError("task cost overflow while counting ") << description;
    return failure();
  }
  total += amount;
  return success();
}

static FailureOr<uint64_t> checkedMultiply(Operation *anchor, uint64_t lhs,
                                           uint64_t rhs,
                                           StringRef description) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    anchor->emitError("task cost overflow while counting ") << description;
    return failure();
  }
  return lhs * rhs;
}

static uint64_t getElementByteWidth(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    type = shaped.getElementType();
  if (type.isF32() || type.isInteger(32) || type.isIndex())
    return 4;
  if (type.isF64() || type.isInteger(64))
    return 8;
  if (type.isF16() || type.isBF16() || type.isInteger(16))
    return 2;
  if (type.isInteger(8) || type.isInteger(1))
    return 1;
  return 0;
}

static uint64_t getStaticElementCount(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  int64_t count = shaped.getNumElements();
  return count > 0 ? static_cast<uint64_t>(count) : 0;
}

static uint64_t getLargestStaticElementCount(Operation *op) {
  uint64_t result = 0;
  for (Type type : op->getResultTypes())
    result = std::max(result, getStaticElementCount(type));
  for (Value operand : op->getOperands())
    result = std::max(result, getStaticElementCount(operand.getType()));
  return result;
}

static FailureOr<uint64_t>
getStaticShapedByteCount(Operation *anchor, Type type, StringRef description) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape() || shaped.getNumElements() <= 0) {
    anchor->emitError("cannot estimate ")
        << description << " for a non-positive or dynamic shape";
    return failure();
  }
  uint64_t elementByteWidth = getElementByteWidth(shaped.getElementType());
  if (elementByteWidth == 0) {
    anchor->emitError("cannot estimate ")
        << description << " for an unsupported element type";
    return failure();
  }
  return checkedMultiply(anchor, static_cast<uint64_t>(shaped.getNumElements()),
                         elementByteWidth, description);
}

static FailureOr<uint64_t> getArrayStoreByteCount(Operation *op,
                                                  StringRef description) {
  if (op->getNumResults() != 1) {
    op->emitError("expected array store to have one logical tensor result");
    return failure();
  }
  auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!outputType || outputType.getRank() != 2 ||
      !outputType.hasStaticShape()) {
    op->emitError(
        "cannot estimate analog store bytes for a non-static rank-2 result");
    return failure();
  }

  auto physicalShape =
      op->getAttrOfType<ArrayAttr>(tiling_attrs::kTilePhysicalShapeAttrName);
  if (!physicalShape)
    return getStaticShapedByteCount(op, outputType, description);
  if (physicalShape.size() != 2) {
    op->emitError(
        "expected physical array shape to contain two integer dimensions");
    return failure();
  }
  auto physicalRows = dyn_cast<IntegerAttr>(physicalShape[0]);
  auto physicalColumns = dyn_cast<IntegerAttr>(physicalShape[1]);
  if (!physicalRows || !physicalColumns || physicalRows.getInt() <= 0 ||
      physicalColumns.getInt() <= 0) {
    op->emitError("expected positive integer physical array dimensions");
    return failure();
  }
  int64_t validRows = outputType.getDimSize(1);
  if (validRows <= 0 || validRows > physicalRows.getInt()) {
    op->emitError(
        "logical store width must be between one and physical array rows");
    return failure();
  }

  uint64_t elementByteWidth = getElementByteWidth(outputType.getElementType());
  if (elementByteWidth == 0) {
    op->emitError(
        "cannot estimate analog store bytes for an unsupported element type");
    return failure();
  }
  auto physicalElements =
      checkedMultiply(op, static_cast<uint64_t>(outputType.getDimSize(0)),
                      static_cast<uint64_t>(physicalRows.getInt()),
                      "physical analog store elements");
  if (failed(physicalElements))
    return failure();
  return checkedMultiply(op, *physicalElements, elementByteWidth, description);
}

static bool hasVectorOperandOrResult(Operation *op) {
  return llvm::any_of(op->getOperandTypes(),
                      [](Type type) { return isa<mlir::VectorType>(type); }) ||
         llvm::any_of(op->getResultTypes(),
                      [](Type type) { return isa<mlir::VectorType>(type); });
}

static FailureOr<uint64_t> getLinalgIterationCount(linalg::LinalgOp op) {
  uint64_t iterations = 1;
  for (int64_t extent : op.getStaticLoopRanges()) {
    if (ShapedType::isDynamic(extent) || extent < 0) {
      op.emitError("cannot estimate target instructions for dynamic linalg "
                   "iteration space");
      return failure();
    }
    auto product = checkedMultiply(op, iterations, extent, "linalg iterations");
    if (failed(product))
      return failure();
    iterations = *product;
  }
  return iterations;
}

static FailureOr<uint64_t> getStaticScfForTripCount(Operation *op) {
  if (op->getNumOperands() < 3)
    return failure();
  llvm::APInt lowerValue;
  llvm::APInt upperValue;
  llvm::APInt stepValue;
  if (!matchPattern(op->getOperand(0), m_ConstantInt(&lowerValue)) ||
      !matchPattern(op->getOperand(1), m_ConstantInt(&upperValue)) ||
      !matchPattern(op->getOperand(2), m_ConstantInt(&stepValue)) ||
      lowerValue.getBitWidth() > 64 || upperValue.getBitWidth() > 64 ||
      stepValue.getBitWidth() > 64) {
    op->emitError("cannot estimate target instructions for a dynamic scf.for");
    return failure();
  }
  int64_t lower = lowerValue.getSExtValue();
  int64_t upper = upperValue.getSExtValue();
  int64_t step = stepValue.getSExtValue();
  if (step <= 0) {
    op->emitError("cannot estimate target instructions for a non-positive "
                  "scf.for step");
    return failure();
  }
  if (upper <= lower)
    return uint64_t{0};
  return static_cast<uint64_t>((upper - lower + step - 1) / step);
}

class TaskFunctionAnalyzer {
public:
  TaskFunctionAnalyzer(ModuleOp module, StaticCostAnalysis &analysis)
      : module(module), analysis(analysis) {}

  LogicalResult analyze(func::FuncOp function, uint64_t multiplier = 1) {
    if (!function || function.isDeclaration()) {
      analysis.foundUnsupportedOperation = true;
      return success();
    }
    if (!activeFunctions.insert(function.getOperation()).second) {
      function.emitError("recursive task helper is unsupported by timing cost "
                         "analysis");
      return failure();
    }
    LogicalResult result = analyzeRegion(function.getBody(), multiplier);
    activeFunctions.erase(function.getOperation());
    return result;
  }

private:
  LogicalResult analyzeRegion(Region &region, uint64_t multiplier) {
    for (Block &block : region) {
      for (Operation &operation : block.without_terminator()) {
        if (failed(analyzeOperation(&operation, multiplier)))
          return failure();
      }
    }
    return success();
  }

  LogicalResult addInstruction(Operation *op, uint64_t multiplier,
                               uint64_t &field, StringRef description) {
    analysis.foundExecutableOperation = true;
    return checkedAdd(op, multiplier, field, description);
  }

  LogicalResult addMemoryMovement(Operation *op, uint64_t elements,
                                  uint64_t multiplier, bool reads,
                                  bool writes) {
    auto instances =
        checkedMultiply(op, elements, multiplier, "memory operation instances");
    if (failed(instances))
      return failure();
    uint64_t byteWidth = 4;
    for (Type type : op->getResultTypes()) {
      if (uint64_t candidate = getElementByteWidth(type)) {
        byteWidth = candidate;
        break;
      }
    }
    if (op->getResultTypes().empty()) {
      for (Value operand : op->getOperands()) {
        if (uint64_t candidate = getElementByteWidth(operand.getType())) {
          byteWidth = candidate;
          break;
        }
      }
    }
    auto bytes =
        checkedMultiply(op, *instances, byteWidth, "local memory bytes");
    if (failed(bytes))
      return failure();
    if (reads) {
      if (failed(checkedAdd(op, *instances, analysis.cost.loadInstructions,
                            "load instructions")) ||
          failed(checkedAdd(op, *bytes, analysis.localBytesRead,
                            "local bytes read")))
        return failure();
    }
    if (writes) {
      if (failed(checkedAdd(op, *instances, analysis.cost.storeInstructions,
                            "store instructions")) ||
          failed(checkedAdd(op, *bytes, analysis.localBytesWritten,
                            "local bytes written")))
        return failure();
    }
    analysis.staticElements = std::max(analysis.staticElements, elements);
    analysis.foundExecutableOperation = true;
    return success();
  }

  LogicalResult analyzeLinalg(Operation *op, uint64_t multiplier) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    auto iterations = getLinalgIterationCount(linalgOp);
    if (failed(iterations))
      return failure();
    auto dynamicIterations =
        checkedMultiply(op, *iterations, multiplier, "linalg dynamic work");
    if (failed(dynamicIterations))
      return failure();

    uint64_t scalarOps = 0;
    for (Region &region : op->getRegions()) {
      region.walk([&](Operation *nested) {
        if (nested == op || nested->hasTrait<OpTrait::IsTerminator>())
          return;
        StringRef dialect = nested->getName().getDialectNamespace();
        if (dialect == "arith" || dialect == "math")
          ++scalarOps;
      });
    }
    auto scalarInstances =
        checkedMultiply(op, scalarOps, *dynamicIterations, "scalar operations");
    if (failed(scalarInstances) ||
        failed(checkedAdd(op, *scalarInstances,
                          analysis.cost.scalarInstructions,
                          "scalar instructions")))
      return failure();

    unsigned inputCount = linalgOp.getDpsInputs().size();
    unsigned outputCount = linalgOp.getDpsInits().size();
    auto loads =
        checkedMultiply(op, *dynamicIterations, inputCount, "linalg loads");
    auto stores =
        checkedMultiply(op, *dynamicIterations, outputCount, "linalg stores");
    if (failed(loads) || failed(stores) ||
        failed(checkedAdd(op, *loads, analysis.cost.loadInstructions,
                          "load instructions")) ||
        failed(checkedAdd(op, *stores, analysis.cost.storeInstructions,
                          "store instructions")) ||
        failed(checkedAdd(op, *dynamicIterations,
                          analysis.cost.controlInstructions,
                          "loop control instructions")) ||
        failed(checkedAdd(op, *dynamicIterations, analysis.loopIterations,
                          "loop iterations")))
      return failure();

    uint64_t byteWidth = 4;
    auto readBytes =
        checkedMultiply(op, *loads, byteWidth, "linalg bytes read");
    auto writtenBytes =
        checkedMultiply(op, *stores, byteWidth, "linalg bytes written");
    if (failed(readBytes) || failed(writtenBytes) ||
        failed(checkedAdd(op, *readBytes, analysis.localBytesRead,
                          "local bytes read")) ||
        failed(checkedAdd(op, *writtenBytes, analysis.localBytesWritten,
                          "local bytes written")))
      return failure();
    analysis.staticElements =
        std::max(analysis.staticElements, getLargestStaticElementCount(op));
    analysis.foundExecutableOperation = true;
    return success();
  }

  LogicalResult analyzeOperation(Operation *op, uint64_t multiplier) {
    StringRef name = op->getName().getStringRef();
    StringRef dialect = op->getName().getDialectNamespace();

    if (dialect == "linalg")
      return analyzeLinalg(op, multiplier);

    if (name == "scf.for") {
      auto tripCount = getStaticScfForTripCount(op);
      if (failed(tripCount))
        return failure();
      auto dynamicIterations =
          checkedMultiply(op, *tripCount, multiplier, "scf.for iterations");
      if (failed(dynamicIterations) ||
          failed(checkedAdd(op, *dynamicIterations, analysis.loopIterations,
                            "loop iterations")))
        return failure();
      auto control =
          checkedMultiply(op, *dynamicIterations, 2, "scf.for control");
      if (failed(control) ||
          failed(checkedAdd(op, *control, analysis.cost.controlInstructions,
                            "loop control instructions")))
        return failure();
      for (Region &region : op->getRegions()) {
        if (failed(analyzeRegion(region, *dynamicIterations)))
          return failure();
      }
      return success();
    }

    if (name == "scf.if") {
      if (failed(addInstruction(op, multiplier,
                                analysis.cost.controlInstructions,
                                "branch instructions")))
        return failure();
      analysis.cost.confidence = TaskCostConfidence::Medium;
      for (Region &region : op->getRegions()) {
        if (failed(analyzeRegion(region, multiplier)))
          return failure();
      }
      return success();
    }

    if (auto call = dyn_cast<func::CallOp>(op)) {
      if (failed(addInstruction(op, multiplier,
                                analysis.cost.controlInstructions,
                                "call instructions")))
        return failure();
      func::FuncOp callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
      if (!callee) {
        analysis.foundUnsupportedOperation = true;
        return success();
      }
      return analyze(callee, multiplier);
    }

    if (name == "tensor.extract" || name == "memref.load" ||
        name == "vector.load" || name == "vector.transfer_read") {
      return addMemoryMovement(op, 1, multiplier, true, false);
    }
    if (name == "tensor.insert" || name == "memref.store" ||
        name == "vector.store" || name == "vector.transfer_write") {
      return addMemoryMovement(op, 1, multiplier, false, true);
    }

    if (name == "tensor.extract_slice" || name == "tensor.insert_slice" ||
        name == "tensor.pad" || name == "tensor.pack" ||
        name == "tensor.unpack" || name == "memref.copy" ||
        name == "bufferization.clone") {
      uint64_t elements = getLargestStaticElementCount(op);
      if (elements == 0) {
        analysis.foundUnsupportedOperation = true;
        return success();
      }
      return addMemoryMovement(op, elements, multiplier, true, true);
    }

    if (name == "tensor.cast" || name == "tensor.reshape" ||
        name == "tensor.collapse_shape" || name == "tensor.expand_shape" ||
        name == "memref.cast" || name == "memref.reinterpret_cast" ||
        name == "bufferization.to_tensor" ||
        name == "bufferization.to_memref") {
      uint64_t rank = 1;
      if (!op->getResultTypes().empty()) {
        if (auto shaped = dyn_cast<ShapedType>(op->getResult(0).getType()))
          rank = std::max<int64_t>(1, shaped.getRank());
      }
      auto descriptorInstructions =
          checkedMultiply(op, multiplier, rank, "descriptor instructions");
      if (failed(descriptorInstructions))
        return failure();
      return addInstruction(op, *descriptorInstructions,
                            analysis.cost.scalarInstructions,
                            "descriptor instructions");
    }

    if (name == "tensor.empty" || name == "memref.alloc" ||
        name == "memref.alloca" || name == "bufferization.alloc_tensor") {
      uint64_t rank = 1;
      uint64_t elements = getLargestStaticElementCount(op);
      if (!op->getResultTypes().empty()) {
        if (auto shaped = dyn_cast<ShapedType>(op->getResult(0).getType())) {
          if (!shaped.hasStaticShape()) {
            analysis.foundUnsupportedOperation = true;
            return success();
          }
          rank = std::max<int64_t>(1, shaped.getRank());
        }
      }
      analysis.staticElements = std::max(analysis.staticElements, elements);
      auto allocationInstructions =
          checkedMultiply(op, multiplier, rank + 4, "allocation instructions");
      if (failed(allocationInstructions))
        return failure();
      return addInstruction(op, *allocationInstructions,
                            analysis.cost.controlInstructions,
                            "allocation instructions");
    }

    if (name == "memref.dealloc")
      return addInstruction(op, multiplier, analysis.cost.controlInstructions,
                            "deallocation instructions");

    if (dialect == "vector") {
      return addInstruction(op, multiplier, analysis.cost.vectorInstructions,
                            "vector instructions");
    }
    if (dialect == "arith" || dialect == "math") {
      if (name == "arith.constant")
        return success();
      if (hasVectorOperandOrResult(op))
        return addInstruction(op, multiplier, analysis.cost.vectorInstructions,
                              "vector arithmetic instructions");
      return addInstruction(op, multiplier, analysis.cost.scalarInstructions,
                            "scalar instructions");
    }

    if (dialect == "sculptor" && name == "sculptor.array.load") {
      if (op->getNumOperands() < 1) {
        op->emitError("expected array load to have a vector operand");
        return failure();
      }
      auto bytes = getStaticShapedByteCount(op, op->getOperand(0).getType(),
                                            "analog load bytes");
      if (failed(bytes))
        return failure();
      auto dynamicBytes =
          checkedMultiply(op, *bytes, multiplier, "analog load bytes");
      if (failed(dynamicBytes))
        return failure();
      return checkedAdd(op, *dynamicBytes, analysis.analogLoadBytes,
                        "analog load bytes");
    }

    if (dialect == "sculptor" && name == "sculptor.array.execute")
      return checkedAdd(op, multiplier, analysis.analogExecutionCount,
                        "analog executions");

    if (dialect == "sculptor" && name == "sculptor.array.store") {
      auto bytes = getArrayStoreByteCount(op, "analog store bytes");
      if (failed(bytes))
        return failure();
      auto dynamicBytes =
          checkedMultiply(op, *bytes, multiplier, "analog store bytes");
      if (failed(dynamicBytes))
        return failure();
      return checkedAdd(op, *dynamicBytes, analysis.analogStoreBytes,
                        "analog store bytes");
    }

    if (dialect == "sculptor" && name == "sculptor.array.set")
      return success();

    if (dialect == "func" || op->hasTrait<OpTrait::IsTerminator>())
      return success();

    if (!op->getRegions().empty()) {
      analysis.cost.confidence = TaskCostConfidence::Medium;
      analysis.foundUnsupportedOperation = true;
      for (Region &region : op->getRegions()) {
        if (failed(analyzeRegion(region, multiplier)))
          return failure();
      }
      return success();
    }

    analysis.foundUnsupportedOperation = true;
    return success();
  }

  ModuleOp module;
  StaticCostAnalysis &analysis;
  llvm::DenseSet<Operation *> activeFunctions;
};

enum class FallbackFamily {
  Fused,
  Activation,
  LayerNorm,
  Reduction,
  Recombine,
  Movement,
  MVMWrapper,
  Generic,
};

static FallbackFamily
classifyFallbackFamily(sculptor::TaskCreateOp task, func::FuncOp callee,
                       const StaticCostAnalysis &analysis) {
  StringRef kind = task.getTaskKind();
  if (kind == "mixed.fused")
    return FallbackFamily::Fused;
  if (task_graph::isAnalogComputeTask(task) ||
      kind == "mixed.streaming_conv_mvm")
    return FallbackFamily::MVMWrapper;
  if (task_graph::isReductionTask(task) || kind.contains("reduction"))
    return FallbackFamily::Reduction;
  if (kind.contains("recombine"))
    return FallbackFamily::Recombine;
  if (kind.contains("activation"))
    return FallbackFamily::Activation;

  bool hasLayerNormMarker = false;
  bool hasActivationMarker = false;
  if (callee) {
    callee.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      hasLayerNormMarker |= name.contains("rsqrt") || name.contains("variance");
      hasActivationMarker |= name.contains("erf") || name.contains("tanh") ||
                             name.contains("gelu");
    });
  }
  if (hasLayerNormMarker)
    return FallbackFamily::LayerNorm;
  if (hasActivationMarker)
    return FallbackFamily::Activation;
  if (analysis.localBytesRead > 0 || analysis.localBytesWritten > 0)
    return FallbackFamily::Movement;
  return FallbackFamily::Generic;
}

static LogicalResult applyVersionedFallback(sculptor::TaskCreateOp task,
                                            func::FuncOp callee,
                                            StaticCostAnalysis &analysis,
                                            const TimingModel &model) {
  struct Coefficients {
    uint64_t fixed;
    uint64_t instructionsPerElement;
    uint64_t instructionsPerWord;
    uint64_t calibratedElementLimit;
  };

  Coefficients coefficients;
  switch (classifyFallbackFamily(task, callee, analysis)) {
  case FallbackFamily::Fused:
    coefficients = {64, 3, 1, 1ULL << 28};
    break;
  case FallbackFamily::Activation:
    coefficients = {24, 4, 1, 1ULL << 28};
    break;
  case FallbackFamily::LayerNorm:
    coefficients = {48, 8, 2, 1ULL << 28};
    break;
  case FallbackFamily::Reduction:
    coefficients = {32, 3, 2, 1ULL << 28};
    break;
  case FallbackFamily::Recombine:
    coefficients = {24, 2, 2, 1ULL << 28};
    break;
  case FallbackFamily::Movement:
    coefficients = {16, 0, 2, 1ULL << 30};
    break;
  case FallbackFamily::MVMWrapper:
    coefficients = {24, 0, 1, 1ULL << 28};
    break;
  case FallbackFamily::Generic:
    coefficients = {32, 1, 1, 1ULL << 26};
    break;
  }

  uint64_t elements = analysis.staticElements;
  uint64_t localBytes = 0;
  if (failed(checkedAdd(task, analysis.localBytesRead, localBytes,
                        "fallback local bytes")) ||
      failed(checkedAdd(task, analysis.localBytesWritten, localBytes,
                        "fallback local bytes")) ||
      localBytes > std::numeric_limits<uint64_t>::max() - 3) {
    task.emitError("task cost overflow while rounding fallback local bytes");
    return failure();
  }
  uint64_t words = (localBytes + 3) / 4;
  auto elementInstructions =
      checkedMultiply(task, elements, coefficients.instructionsPerElement,
                      "fallback element instructions");
  auto wordInstructions =
      checkedMultiply(task, words, coefficients.instructionsPerWord,
                      "fallback movement instructions");
  if (failed(elementInstructions) || failed(wordInstructions))
    return failure();
  uint64_t fallback = coefficients.fixed;
  if (failed(checkedAdd(task, *elementInstructions, fallback,
                        "fallback instructions")) ||
      failed(checkedAdd(task, *wordInstructions, fallback,
                        "fallback instructions")))
    return failure();

  uint64_t current = 0;
  for (uint64_t count :
       {analysis.cost.scalarInstructions, analysis.cost.vectorInstructions,
        analysis.cost.loadInstructions, analysis.cost.storeInstructions,
        analysis.cost.controlInstructions}) {
    if (failed(
            checkedAdd(task, count, current, "fallback current instructions")))
      return failure();
  }
  if (fallback > current) {
    if (failed(checkedAdd(task, fallback - current,
                          analysis.cost.scalarInstructions,
                          "fallback scalar instructions")))
      return failure();
  }
  analysis.cost.source = TaskCostSource::CalibratedFallback;
  analysis.cost.confidence = elements > coefficients.calibratedElementLimit
                                 ? TaskCostConfidence::Low
                                 : TaskCostConfidence::Medium;
  task.emitWarning("task body requires conservative fallback cost model '")
      << model.costModel << "' revision " << model.costModelRevision;
  if (analysis.cost.confidence == TaskCostConfidence::Low) {
    task.emitWarning("task feature vector is outside the calibrated range for "
                     "cost model ")
        << model.costModel << "; predicted cycles are low-confidence";
  }
  return success();
}

static LogicalResult addTaskAdapterCost(sculptor::TaskCreateOp task,
                                        StaticCostAnalysis &analysis) {
  uint64_t descriptorInstructions = 0;
  auto addDescriptorInstructions = [&](Value value) -> LogicalResult {
    auto resourceType = dyn_cast<sculptor::TaskResourceType>(value.getType());
    ShapedType shaped;
    if (resourceType)
      shaped = dyn_cast<ShapedType>(resourceType.getValueType());
    uint64_t amount = shaped && shaped.hasRank() ? 4 + shaped.getRank() : 2;
    return checkedAdd(task, amount, descriptorInstructions,
                      "task adapter descriptor instructions");
  };

  for (Value input : task.getInputs()) {
    if (failed(addDescriptorInstructions(input)))
      return failure();
  }

  for (Value output : task.getOutputs()) {
    if (failed(addDescriptorInstructions(output)))
      return failure();

    auto resourceType = dyn_cast<sculptor::TaskResourceType>(output.getType());
    ShapedType shaped;
    if (resourceType)
      shaped = dyn_cast<ShapedType>(resourceType.getValueType());
    if (!shaped)
      continue;
    if (!shaped.hasStaticShape()) {
      task.emitError(
          "cannot estimate task-adapter output copy for dynamic shape");
      return failure();
    }

    uint64_t elements = getStaticElementCount(shaped);
    uint64_t byteWidth = getElementByteWidth(shaped.getElementType());
    if (elements == 0 || byteWidth == 0) {
      task.emitError(
          "cannot estimate task-adapter output copy for unsupported type");
      return failure();
    }
    auto bytes =
        checkedMultiply(task, elements, byteWidth, "task adapter output bytes");
    if (failed(bytes) ||
        failed(checkedAdd(task, elements, analysis.cost.loadInstructions,
                          "task adapter output loads")) ||
        failed(checkedAdd(task, elements, analysis.cost.storeInstructions,
                          "task adapter output stores")) ||
        failed(checkedAdd(task, *bytes, analysis.localBytesRead,
                          "task adapter output bytes read")) ||
        failed(checkedAdd(task, *bytes, analysis.localBytesWritten,
                          "task adapter output bytes written")))
      return failure();
    analysis.staticElements = std::max(analysis.staticElements, elements);
  }
  return checkedAdd(task, descriptorInstructions,
                    analysis.cost.controlInstructions,
                    "task adapter descriptor instructions");
}

} // namespace

FailureOr<TaskCost>
estimateTaskCost(ModuleOp module, sculptor::TaskCreateOp task,
                 TaskWorkloadFeatures &workload, int64_t effectiveDigitalOps,
                 int64_t digitalReplacementOps, const TimingModel &model) {
  if (effectiveDigitalOps < 0 || digitalReplacementOps < 0) {
    task.emitError("expected non-negative digital operation counts");
    return failure();
  }

  StaticCostAnalysis analysis;
  analysis.cost.runtimeDispatchCycles =
      static_cast<uint64_t>(model.fixedRuntimeDispatchCycles);
  analysis.cost.taskEntryCycles =
      static_cast<uint64_t>(model.fixedTaskEntryCycles);
  analysis.cost.taskExitCycles =
      static_cast<uint64_t>(model.fixedTaskExitCycles);

  func::FuncOp callee =
      module.lookupSymbol<func::FuncOp>(task.getCalleeAttr().getValue());
  if (!callee) {
    task.emitError("expected task callee '")
        << task.getCalleeAttr().getValue() << "' for task cost analysis";
    return failure();
  }

  TaskFunctionAnalyzer analyzer(module, analysis);
  if (failed(analyzer.analyze(callee)) ||
      failed(addTaskAdapterCost(task, analysis)))
    return failure();

  if (digitalReplacementOps > 0) {
    if (failed(checkedAdd(task, static_cast<uint64_t>(digitalReplacementOps),
                          analysis.cost.scalarInstructions,
                          "digital replacement instructions")))
      return failure();
  }

  uint64_t instructionCount = 0;
  for (uint64_t count :
       {analysis.cost.scalarInstructions, analysis.cost.vectorInstructions,
        analysis.cost.loadInstructions, analysis.cost.storeInstructions,
        analysis.cost.controlInstructions}) {
    if (failed(checkedAdd(task, count, instructionCount, "total instructions")))
      return failure();
  }
  uint64_t explicitInstructionFloor =
      static_cast<uint64_t>(effectiveDigitalOps);
  if (instructionCount < explicitInstructionFloor) {
    if (failed(checkedAdd(task, explicitInstructionFloor - instructionCount,
                          analysis.cost.scalarInstructions,
                          "explicit digital instructions")))
      return failure();
    analysis.cost.source = TaskCostSource::ExplicitMetadata;
    analysis.cost.confidence = TaskCostConfidence::Medium;
  }

  instructionCount = 0;
  for (uint64_t count :
       {analysis.cost.scalarInstructions, analysis.cost.vectorInstructions,
        analysis.cost.loadInstructions, analysis.cost.storeInstructions,
        analysis.cost.controlInstructions}) {
    if (failed(checkedAdd(task, count, instructionCount, "total instructions")))
      return failure();
  }
  bool dispatchTask = !task_graph::isMatrixSetupTask(task);
  if ((analysis.foundUnsupportedOperation ||
       (dispatchTask && instructionCount == 0)) &&
      failed(applyVersionedFallback(task, callee, analysis, model)))
    return failure();

  instructionCount = 0;
  for (uint64_t count :
       {analysis.cost.scalarInstructions, analysis.cost.vectorInstructions,
        analysis.cost.loadInstructions, analysis.cost.storeInstructions,
        analysis.cost.controlInstructions}) {
    if (failed(checkedAdd(task, count, instructionCount, "total instructions")))
      return failure();
  }
  if (dispatchTask && instructionCount == 0) {
    task.emitError("nontrivial dispatch task has no executable cost after "
                   "static analysis and fallback estimation");
    return failure();
  }

  analysis.cost.predictedCpuCycles =
      std::ceil(static_cast<double>(instructionCount) /
                static_cast<double>(model.digitalIssueWidth)) +
      static_cast<double>(analysis.cost.runtimeDispatchCycles) +
      static_cast<double>(analysis.cost.taskEntryCycles) +
      static_cast<double>(analysis.cost.taskExitCycles);

  for (uint64_t value :
       {analysis.cost.scalarInstructions, analysis.cost.vectorInstructions,
        analysis.cost.loadInstructions, analysis.cost.storeInstructions,
        analysis.cost.controlInstructions, analysis.cost.runtimeDispatchCycles,
        analysis.cost.taskEntryCycles, analysis.cost.taskExitCycles,
        analysis.staticElements, analysis.localBytesRead,
        analysis.localBytesWritten, analysis.loopIterations,
        analysis.analogLoadBytes, analysis.analogExecutionCount,
        analysis.analogStoreBytes}) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      task.emitError("task cost or workload value exceeds signed i64 IR "
                     "attribute capacity");
      return failure();
    }
  }

  workload.staticElements = analysis.staticElements;
  workload.localBytesRead = analysis.localBytesRead;
  workload.localBytesWritten = analysis.localBytesWritten;
  workload.loopIterations = analysis.loopIterations;
  workload.analogLoadBytes = analysis.analogLoadBytes;
  workload.analogExecutionCount = analysis.analogExecutionCount;
  workload.analogStoreBytes = analysis.analogStoreBytes;
  return analysis.cost;
}

} // namespace task_timing
} // namespace sculptor
} // namespace mlir
