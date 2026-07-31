#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/StreamingConvolutionOptimizer.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphOptimizationAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphScheduleAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskNames.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTilingAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

namespace optimization_attrs = mlir::sculptor::optimization_attrs;
namespace runtime_attrs = mlir::sculptor::runtime_attrs;
namespace schedule_attrs = mlir::sculptor::schedule_attrs;
namespace task_attrs = mlir::sculptor::task_attrs;
namespace task_graph_names = mlir::sculptor::task_graph_names;
namespace tiling_attrs = mlir::sculptor::tiling_attrs;

constexpr llvm::StringLiteral
    kOutputRecombineTaskKind("digital.output_recombine");

struct ConvolutionDescriptor {
  int64_t batch = 0;
  int64_t inputChannels = 0;
  int64_t inputHeight = 0;
  int64_t inputWidth = 0;
  int64_t outputChannels = 0;
  int64_t outputHeight = 0;
  int64_t outputWidth = 0;
  int64_t kernelHeight = 0;
  int64_t kernelWidth = 0;
  int64_t strideHeight = 0;
  int64_t strideWidth = 0;
  int64_t paddingHeight = 0;
  int64_t paddingWidth = 0;
  int64_t dilationHeight = 0;
  int64_t dilationWidth = 0;
  bool hasBias = false;
  TypedAttr bias;
};

struct ArrayStage {
  const TaskGraphNode *mvmNode = nullptr;
  sculptor::TaskCreateOp setupTask;
  Value logicalArrayResource;
  int64_t tileColumn = 0;
  int64_t arrayRows = 0;
  int64_t arrayCols = 0;
  int64_t validRows = 0;
  int64_t validCols = 0;
  int64_t physicalArrayId = 0;
  int64_t localArrayId = 0;
};

struct StreamingConvolutionMatch {
  const TaskGraphNode *patchNode = nullptr;
  const TaskGraphNode *recombineNode = nullptr;
  const TaskGraphNode *outputNode = nullptr;
  SmallVector<ArrayStage, 4> arrays;
  SmallVector<const TaskGraphNode *, 8> componentNodes;
  SmallVector<Value, 8> externalDependencies;
  SmallVector<int64_t, 4> sourceIslandIds;
  Value activationResource;
  Value outputResource;
  ConvolutionDescriptor descriptor;
  int64_t coreId = 0;
  int64_t homeIslandId = 0;
};

static std::optional<int64_t> getI64Attr(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static std::optional<SmallVector<int64_t, 4>> getI64Array(Attribute attr,
                                                          size_t expectedSize) {
  auto array = dyn_cast_or_null<ArrayAttr>(attr);
  if (!array || array.size() != expectedSize)
    return std::nullopt;

  SmallVector<int64_t, 4> values;
  values.reserve(array.size());
  for (Attribute element : array) {
    auto integer = dyn_cast<IntegerAttr>(element);
    if (!integer)
      return std::nullopt;
    values.push_back(integer.getInt());
  }
  return values;
}

static std::optional<SmallVector<int64_t, 4>>
getI64Array(Operation *op, StringRef name, size_t expectedSize) {
  return getI64Array(op->getAttr(name), expectedSize);
}

static std::optional<SmallVector<int64_t, 4>>
getI64Array(DictionaryAttr dictionary, StringRef name, size_t expectedSize) {
  return getI64Array(dictionary.get(name), expectedSize);
}

static FailureOr<func::FuncOp> lookupTaskCallee(ModuleOp module,
                                                sculptor::TaskCreateOp taskOp) {
  auto callee =
      module.lookupSymbol<func::FuncOp>(taskOp.getCalleeAttr().getValue());
  if (!callee)
    return taskOp.emitError("expected task callee '")
           << taskOp.getCalleeAttr().getValue() << "'";
  return callee;
}

static bool isLogicalArrayResource(Value resource) {
  auto resourceType = dyn_cast<sculptor::TaskResourceType>(resource.getType());
  return resourceType &&
         isa<sculptor::LogicalArrayType>(resourceType.getValueType());
}

static FailureOr<Type> getTaskResourceValueType(Value resource) {
  auto resourceType = dyn_cast<sculptor::TaskResourceType>(resource.getType());
  if (!resourceType) {
    if (Operation *definingOp = resource.getDefiningOp())
      definingOp->emitError("expected task resource value");
    return failure();
  }
  return resourceType.getValueType();
}

static std::optional<ConvolutionDescriptor>
parseConvolutionDescriptor(func::FuncOp patchCallee) {
  auto descriptorAttr = patchCallee->getAttrOfType<DictionaryAttr>(
      optimization_attrs::kStreamingConvolutionAttrName);
  if (!descriptorAttr)
    return std::nullopt;

  auto inputShape =
      getI64Array(descriptorAttr, optimization_attrs::kInputShapeFieldName, 4);
  auto outputShape =
      getI64Array(descriptorAttr, optimization_attrs::kOutputShapeFieldName, 4);
  auto kernelShape =
      getI64Array(descriptorAttr, optimization_attrs::kKernelShapeFieldName, 2);
  auto stride =
      getI64Array(descriptorAttr, optimization_attrs::kStrideFieldName, 2);
  auto padding =
      getI64Array(descriptorAttr, optimization_attrs::kPaddingFieldName, 2);
  auto dilation =
      getI64Array(descriptorAttr, optimization_attrs::kDilationFieldName, 2);
  auto hasBias =
      descriptorAttr.getAs<BoolAttr>(optimization_attrs::kHasBiasFieldName);
  if (!inputShape || !outputShape || !kernelShape || !stride || !padding ||
      !dilation || !hasBias)
    return std::nullopt;

  ConvolutionDescriptor descriptor{
      .batch = (*inputShape)[0],
      .inputChannels = (*inputShape)[1],
      .inputHeight = (*inputShape)[2],
      .inputWidth = (*inputShape)[3],
      .outputChannels = (*outputShape)[1],
      .outputHeight = (*outputShape)[2],
      .outputWidth = (*outputShape)[3],
      .kernelHeight = (*kernelShape)[0],
      .kernelWidth = (*kernelShape)[1],
      .strideHeight = (*stride)[0],
      .strideWidth = (*stride)[1],
      .paddingHeight = (*padding)[0],
      .paddingWidth = (*padding)[1],
      .dilationHeight = (*dilation)[0],
      .dilationWidth = (*dilation)[1],
      .hasBias = hasBias.getValue(),
  };

  if (descriptor.hasBias) {
    descriptor.bias = dyn_cast_or_null<TypedAttr>(
        descriptorAttr.get(optimization_attrs::kBiasFieldName));
    if (!descriptor.bias)
      return std::nullopt;
  }
  return descriptor;
}

static bool isSupportedDescriptor(const ConvolutionDescriptor &descriptor) {
  return descriptor.batch == 1 && descriptor.inputChannels > 0 &&
         descriptor.inputHeight > 0 && descriptor.inputWidth > 0 &&
         descriptor.outputChannels > 0 && descriptor.outputHeight > 0 &&
         descriptor.outputWidth > 0 && descriptor.kernelHeight > 0 &&
         descriptor.kernelWidth > 0 && descriptor.strideHeight > 0 &&
         descriptor.strideWidth > 0 && descriptor.paddingHeight == 0 &&
         descriptor.paddingWidth == 0 && descriptor.dilationHeight == 1 &&
         descriptor.dilationWidth == 1;
}

static bool hasExactStaticTensorType(Type type, ArrayRef<int64_t> shape) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.hasStaticShape() &&
         tensorType.getElementType().isF32() && tensorType.getShape() == shape;
}

static bool appendUniqueValue(SmallVectorImpl<Value> &values, Value value) {
  if (llvm::is_contained(values, value))
    return false;
  values.push_back(value);
  return true;
}

static void
collectResourceConsumers(const TaskGraphDAG &dag,
                         DenseMap<Value, SmallVector<const TaskGraphNode *, 4>>
                             &consumersByResource) {
  for (const TaskGraphNode &node : dag.nodes) {
    sculptor::TaskCreateOp taskOp = node.op;
    for (Value input : taskOp.getInputs())
      consumersByResource[input].push_back(&node);
  }
}

static void collectResourceProducers(
    const TaskGraphDAG &dag,
    DenseMap<Value, const TaskGraphNode *> &producerByResource) {
  for (const TaskGraphNode &node : dag.nodes) {
    sculptor::TaskCreateOp taskOp = node.op;
    for (Value output : taskOp.getOutputs())
      producerByResource.try_emplace(output, &node);
  }
}

static std::optional<ArrayStage> parseArrayStage(
    ModuleOp module, const TaskGraphNode *mvmNode, Value patchResource,
    const DenseMap<Value, const TaskGraphNode *> &producerByResource,
    int64_t expectedCore, int64_t expectedTileColumn, int64_t expectedTileCount,
    int64_t outputChannels) {
  sculptor::TaskCreateOp mvmTask = mvmNode->op;
  if (mvmTask.getTaskKind() != task_graph_names::kConvTileMVMTaskKind ||
      mvmTask.getInputs().size() != 2 || mvmTask.getOutputs().size() != 1 ||
      mvmTask.getInputs().front() != patchResource)
    return std::nullopt;

  Value logicalArray = mvmTask.getInputs()[1];
  if (!isLogicalArrayResource(logicalArray))
    return std::nullopt;

  auto coreId = getI64Attr(mvmTask, runtime_attrs::kTaskCoreIdAttrName);
  auto physicalArrayId =
      getI64Attr(mvmTask, runtime_attrs::kTaskPhysicalArrayIdAttrName);
  auto localArrayId =
      getI64Attr(mvmTask, runtime_attrs::kTaskLocalArrayIdAttrName);
  if (!coreId || !physicalArrayId || !localArrayId || *coreId != expectedCore)
    return std::nullopt;

  auto callee = lookupTaskCallee(module, mvmTask);
  if (failed(callee))
    return std::nullopt;
  auto tile = getI64Array(*callee, tiling_attrs::kTileAttrName, 2);
  auto grid = getI64Array(*callee, tiling_attrs::kTileGridAttrName, 2);
  auto physicalShape =
      getI64Array(*callee, tiling_attrs::kTilePhysicalShapeAttrName, 2);
  auto validShape =
      getI64Array(*callee, tiling_attrs::kTileValidShapeAttrName, 2);
  auto vectorTile = getI64Attr(*callee, tiling_attrs::kVectorTileAttrName);
  if (!tile || !grid || !physicalShape || !validShape || !vectorTile ||
      (*tile)[0] != 0 || (*tile)[1] != expectedTileColumn || (*grid)[0] != 1 ||
      (*grid)[1] != expectedTileCount || *vectorTile != expectedTileColumn ||
      (*physicalShape)[0] <= 0 || (*physicalShape)[1] <= 0 ||
      (*validShape)[0] != outputChannels || (*validShape)[1] <= 0 ||
      (*validShape)[1] > (*physicalShape)[1])
    return std::nullopt;

  auto producerIt = producerByResource.find(logicalArray);
  if (producerIt == producerByResource.end())
    return std::nullopt;
  sculptor::TaskCreateOp setupTask = producerIt->second->op;
  if (setupTask.getTaskKind() != task_graph_names::kMatrixSetupTaskKind)
    return std::nullopt;
  auto setupCore = getI64Attr(setupTask, runtime_attrs::kTaskCoreIdAttrName);
  auto setupPhysical =
      getI64Attr(setupTask, runtime_attrs::kTaskPhysicalArrayIdAttrName);
  auto setupLocal =
      getI64Attr(setupTask, runtime_attrs::kTaskLocalArrayIdAttrName);
  if (!setupCore || !setupPhysical || !setupLocal ||
      *setupCore != expectedCore || *setupPhysical != *physicalArrayId ||
      *setupLocal != *localArrayId)
    return std::nullopt;

  return ArrayStage{
      .mvmNode = mvmNode,
      .setupTask = setupTask,
      .logicalArrayResource = logicalArray,
      .tileColumn = expectedTileColumn,
      .arrayRows = (*physicalShape)[0],
      .arrayCols = (*physicalShape)[1],
      .validRows = (*validShape)[0],
      .validCols = (*validShape)[1],
      .physicalArrayId = *physicalArrayId,
      .localArrayId = *localArrayId,
  };
}

static bool sameSourceLayer(ArrayRef<const TaskGraphNode *> nodes) {
  if (nodes.empty())
    return false;
  sculptor::TaskCreateOp first = nodes.front()->op;
  StringRef sourceLayer = first.getSourceLayer();
  for (const TaskGraphNode *node : nodes) {
    sculptor::TaskCreateOp task = node->op;
    if (task.getSourceLayer() != sourceLayer)
      return false;
  }
  return true;
}

static void collectExternalDependencies(StreamingConvolutionMatch &match,
                                        const TaskGraphDAG &dag) {
  SmallPtrSet<Operation *, 16> componentOps;
  for (const TaskGraphNode *node : match.componentNodes) {
    sculptor::TaskCreateOp task = node->op;
    componentOps.insert(task);
  }

  DenseMap<Value, unsigned> indexByTask;
  for (const TaskGraphNode &node : dag.nodes) {
    sculptor::TaskCreateOp task = node.op;
    indexByTask.try_emplace(task.getResult(), node.index);
  }

  for (const TaskGraphNode *node : match.componentNodes) {
    sculptor::TaskCreateOp task = node->op;
    for (Value dependency : task.getDependencies()) {
      auto producer = dependency.getDefiningOp<sculptor::TaskCreateOp>();
      if (producer && componentOps.contains(producer))
        continue;
      appendUniqueValue(match.externalDependencies, dependency);
    }
  }
  llvm::sort(match.externalDependencies, [&](Value lhs, Value rhs) {
    return indexByTask.lookup(lhs) < indexByTask.lookup(rhs);
  });
}

static void collectSourceIslands(StreamingConvolutionMatch &match) {
  llvm::SmallSet<int64_t, 8> seen;
  for (const TaskGraphNode *node : match.componentNodes) {
    sculptor::TaskCreateOp task = node->op;
    auto island = getI64Attr(task, schedule_attrs::kIslandIdAttrName);
    if (island && seen.insert(*island).second)
      match.sourceIslandIds.push_back(*island);
  }
  llvm::sort(match.sourceIslandIds);
}

static std::optional<StreamingConvolutionMatch> matchStreamingConvolution(
    ModuleOp module, const TaskGraphDAG &dag, const TaskGraphNode &patchNode,
    const DenseMap<Value, const TaskGraphNode *> &producerByResource,
    const DenseMap<Value, SmallVector<const TaskGraphNode *, 4>>
        &consumersByResource) {
  sculptor::TaskCreateOp patchTask = patchNode.op;
  if (patchTask.getTaskKind() != task_graph_names::kConvPatchTaskKind ||
      patchTask.getInputs().size() != 1 || patchTask.getOutputs().size() != 1)
    return std::nullopt;

  auto patchCallee = lookupTaskCallee(module, patchTask);
  if (failed(patchCallee))
    return std::nullopt;
  auto descriptor = parseConvolutionDescriptor(*patchCallee);
  if (!descriptor || !isSupportedDescriptor(*descriptor))
    return std::nullopt;

  auto coreId = getI64Attr(patchTask, runtime_attrs::kTaskCoreIdAttrName);
  if (!coreId)
    return std::nullopt;

  FailureOr<Type> activationType =
      getTaskResourceValueType(patchTask.getInputs().front());
  FailureOr<Type> patchType =
      getTaskResourceValueType(patchTask.getOutputs().front());
  if (failed(activationType) || failed(patchType))
    return std::nullopt;

  int64_t flattenedWidth = descriptor->inputChannels *
                           descriptor->kernelHeight * descriptor->kernelWidth;
  int64_t outputPositions = descriptor->outputHeight * descriptor->outputWidth;
  if (!hasExactStaticTensorType(
          *activationType, {descriptor->batch, descriptor->inputChannels,
                            descriptor->inputHeight, descriptor->inputWidth}) ||
      !hasExactStaticTensorType(*patchType, {outputPositions, flattenedWidth}))
    return std::nullopt;

  Value patchResource = patchTask.getOutputs().front();
  auto patchConsumers = consumersByResource.find(patchResource);
  if (patchConsumers == consumersByResource.end() ||
      patchConsumers->second.empty())
    return std::nullopt;

  const TaskGraphNode *recombineNode = nullptr;
  llvm::SmallPtrSet<Operation *, 8> mvmOps;
  for (const TaskGraphNode *candidate : patchConsumers->second) {
    sculptor::TaskCreateOp mvmTask = candidate->op;
    if (mvmTask.getTaskKind() != task_graph_names::kConvTileMVMTaskKind ||
        mvmTask.getOutputs().size() != 1)
      return std::nullopt;

    auto partialConsumers =
        consumersByResource.find(mvmTask.getOutputs().front());
    if (partialConsumers == consumersByResource.end() ||
        partialConsumers->second.size() != 1)
      return std::nullopt;
    const TaskGraphNode *candidateRecombine = partialConsumers->second.front();
    if (!recombineNode)
      recombineNode = candidateRecombine;
    else if (recombineNode != candidateRecombine)
      return std::nullopt;
    mvmOps.insert(mvmTask);
  }

  if (!recombineNode)
    return std::nullopt;
  sculptor::TaskCreateOp recombineTask = recombineNode->op;
  if (recombineTask.getTaskKind() != task_graph_names::kTileRecombineTaskKind ||
      recombineTask.getInputs().size() != patchConsumers->second.size() ||
      recombineTask.getOutputs().size() != 1)
    return std::nullopt;

  SmallVector<ArrayStage, 4> arrays;
  arrays.reserve(recombineTask.getInputs().size());
  for (auto indexedInput : llvm::enumerate(recombineTask.getInputs())) {
    auto partialProducerIt = producerByResource.find(indexedInput.value());
    if (partialProducerIt == producerByResource.end())
      return std::nullopt;
    const TaskGraphNode *mvmNode = partialProducerIt->second;
    sculptor::TaskCreateOp mvmTask = mvmNode->op;
    if (!mvmOps.contains(mvmTask))
      return std::nullopt;

    auto stage =
        parseArrayStage(module, mvmNode, patchResource, producerByResource,
                        *coreId, static_cast<int64_t>(indexedInput.index()),
                        static_cast<int64_t>(recombineTask.getInputs().size()),
                        descriptor->outputChannels);
    if (!stage)
      return std::nullopt;
    arrays.push_back(*stage);
  }

  int64_t commonArrayCols = arrays.front().arrayCols;
  for (const ArrayStage &stage : arrays) {
    if (stage.arrayCols != commonArrayCols ||
        stage.validRows != descriptor->outputChannels)
      return std::nullopt;
  }
  int64_t coveredColumns = 0;
  for (const ArrayStage &stage : arrays)
    coveredColumns += stage.validCols;
  if (coveredColumns != flattenedWidth)
    return std::nullopt;

  Value recombinedResource = recombineTask.getOutputs().front();
  auto recombineConsumers = consumersByResource.find(recombinedResource);
  if (recombineConsumers == consumersByResource.end() ||
      recombineConsumers->second.size() != 1)
    return std::nullopt;
  const TaskGraphNode *outputNode = recombineConsumers->second.front();
  sculptor::TaskCreateOp outputTask = outputNode->op;
  if ((outputTask.getTaskKind() != task_graph_names::kBiasAddTaskKind &&
       outputTask.getTaskKind() != kOutputRecombineTaskKind) ||
      outputTask.getInputs().size() != 1 ||
      outputTask.getInputs().front() != recombinedResource ||
      outputTask.getOutputs().size() != 1)
    return std::nullopt;

  auto outputCore = getI64Attr(outputTask, runtime_attrs::kTaskCoreIdAttrName);
  auto recombineCore =
      getI64Attr(recombineTask, runtime_attrs::kTaskCoreIdAttrName);
  auto homeIsland = getI64Attr(outputTask, schedule_attrs::kIslandIdAttrName);
  if (!outputCore || !recombineCore || !homeIsland || *outputCore != *coreId ||
      *recombineCore != *coreId)
    return std::nullopt;

  FailureOr<Type> outputType =
      getTaskResourceValueType(outputTask.getOutputs().front());
  if (failed(outputType) ||
      !hasExactStaticTensorType(
          *outputType, {descriptor->batch, descriptor->outputChannels,
                        descriptor->outputHeight, descriptor->outputWidth}))
    return std::nullopt;

  if (descriptor->hasBias !=
      (outputTask.getTaskKind() == task_graph_names::kBiasAddTaskKind))
    return std::nullopt;
  if (descriptor->hasBias &&
      !hasExactStaticTensorType(descriptor->bias.getType(),
                                {descriptor->outputChannels}))
    return std::nullopt;

  StreamingConvolutionMatch match;
  match.patchNode = &patchNode;
  match.recombineNode = recombineNode;
  match.outputNode = outputNode;
  match.arrays = std::move(arrays);
  match.activationResource = patchTask.getInputs().front();
  match.outputResource = outputTask.getOutputs().front();
  match.descriptor = *descriptor;
  match.coreId = *coreId;
  match.homeIslandId = *homeIsland;
  match.componentNodes.push_back(&patchNode);
  for (const ArrayStage &stage : match.arrays)
    match.componentNodes.push_back(stage.mvmNode);
  match.componentNodes.push_back(recombineNode);
  match.componentNodes.push_back(outputNode);
  llvm::sort(match.componentNodes,
             [](const TaskGraphNode *lhs, const TaskGraphNode *rhs) {
               return lhs->index < rhs->index;
             });

  if (!sameSourceLayer(match.componentNodes))
    return std::nullopt;
  collectExternalDependencies(match, dag);
  for (Value dependency : match.externalDependencies) {
    auto indexIt = dag.nodeIndexByTaskResult.find(dependency);
    if (indexIt == dag.nodeIndexByTaskResult.end() ||
        indexIt->second >= patchNode.index)
      return std::nullopt;
  }
  collectSourceIslands(match);
  return match;
}

static std::string sanitizeSymbolComponent(StringRef value) {
  std::string result;
  result.reserve(value.size());
  bool previousUnderscore = false;
  for (char character : value) {
    unsigned char unsignedCharacter = static_cast<unsigned char>(character);
    char next =
        (std::isalnum(unsignedCharacter) || character == '_') ? character : '_';
    if (next == '_' && previousUnderscore)
      continue;
    result.push_back(next);
    previousUnderscore = next == '_';
  }
  if (result.empty())
    return "streaming_convolution";
  return result;
}

static std::string buildUniqueFunctionName(ModuleOp module,
                                           StringRef sourceLayer) {
  std::string base = "task_";
  base += sanitizeSymbolComponent(sourceLayer);
  base += "_streaming_conv_mvm";
  std::string candidate = base;
  unsigned suffix = 0;
  while (module.lookupSymbol(candidate))
    candidate = base + "_" + std::to_string(suffix++);
  return candidate;
}

static Value createIndexConstant(OpBuilder &builder, Location loc,
                                 int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

static void attachArrayBinding(Operation *op, OpBuilder &builder,
                               unsigned bindingIndex, const ArrayStage &stage) {
  op->setAttr(runtime_attrs::kArrayBindingIndexAttrName,
              builder.getI64IntegerAttr(bindingIndex));
  op->setAttr(runtime_attrs::kTaskLocalArrayIdAttrName,
              builder.getI64IntegerAttr(stage.localArrayId));
  op->setAttr(runtime_attrs::kTaskPhysicalArrayIdAttrName,
              builder.getI64IntegerAttr(stage.physicalArrayId));
}

static void attachArrayStoreShape(Operation *op, OpBuilder &builder,
                                  const ArrayStage &stage) {
  op->setAttr(tiling_attrs::kTilePhysicalShapeAttrName,
              builder.getI64ArrayAttr({stage.arrayRows, stage.arrayCols}));
  op->setAttr(tiling_attrs::kTileValidShapeAttrName,
              builder.getI64ArrayAttr({stage.validRows, stage.validCols}));
}

static ArrayAttr buildArrayBindings(OpBuilder &builder,
                                    ArrayRef<ArrayStage> arrays) {
  SmallVector<Attribute, 4> bindings;
  bindings.reserve(arrays.size());
  for (auto indexedStage : llvm::enumerate(arrays)) {
    const ArrayStage &stage = indexedStage.value();
    bindings.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr(runtime_attrs::kArrayBindingInputIndexFieldName,
                             builder.getI64IntegerAttr(static_cast<int64_t>(
                                 indexedStage.index() + 1))),
        builder.getNamedAttr(runtime_attrs::kArrayBindingPhysicalIdFieldName,
                             builder.getI64IntegerAttr(stage.physicalArrayId)),
        builder.getNamedAttr(runtime_attrs::kArrayBindingLocalIdFieldName,
                             builder.getI64IntegerAttr(stage.localArrayId)),
    }));
  }
  return builder.getArrayAttr(bindings);
}

static FailureOr<int64_t> checkedMultiply(Operation *anchor,
                                          ArrayRef<int64_t> values,
                                          StringRef description) {
  int64_t product = 1;
  for (int64_t value : values) {
    if (value < 0 ||
        (value != 0 && product > std::numeric_limits<int64_t>::max() / value)) {
      anchor->emitError(description) << " overflow";
      return failure();
    }
    product *= value;
  }
  return product;
}

static FailureOr<func::FuncOp>
buildStreamingConvolutionCallee(ModuleOp module,
                                const StreamingConvolutionMatch &match) {
  sculptor::TaskCreateOp patchTask = match.patchNode->op;
  Location loc = patchTask.getLoc();
  OpBuilder builder(module.getContext());

  FailureOr<Type> activationType =
      getTaskResourceValueType(match.activationResource);
  FailureOr<Type> outputType = getTaskResourceValueType(match.outputResource);
  if (failed(activationType) || failed(outputType))
    return failure();

  SmallVector<Type, 4> inputTypes{*activationType};
  for (const ArrayStage &stage : match.arrays) {
    FailureOr<Type> arrayType =
        getTaskResourceValueType(stage.logicalArrayResource);
    if (failed(arrayType))
      return failure();
    inputTypes.push_back(*arrayType);
  }

  std::string functionName =
      buildUniqueFunctionName(module, patchTask.getSourceLayer());
  auto functionType = builder.getFunctionType(inputTypes, {*outputType});
  auto streamingFunc = func::FuncOp::create(loc, functionName, functionType);
  streamingFunc.setPrivate();
  streamingFunc->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  streamingFunc->setAttr(
      task_attrs::kTaskDomainAttrName,
      builder.getStringAttr(task_graph_names::kDigitalDomain));
  streamingFunc->setAttr(
      task_attrs::kTaskKindAttrName,
      builder.getStringAttr(task_graph_names::kStreamingConvolutionTaskKind));
  streamingFunc->setAttr(
      task_attrs::kTaskNameAttrName,
      builder.getStringAttr(
          (patchTask.getSourceLayer() + "_streaming_conv_mvm").str()));
  streamingFunc->setAttr(task_attrs::kSourceLayerAttrName,
                         patchTask.getSourceLayerAttr());
  streamingFunc->setAttr(task_attrs::kSourceTaskOrdinalAttrName,
                         patchTask.getSourceTaskOrdinalAttr());
  streamingFunc->setAttr(runtime_attrs::kTaskCoreIdAttrName,
                         builder.getI64IntegerAttr(match.coreId));
  streamingFunc->setAttr(runtime_attrs::kTaskArrayBindingsAttrName,
                         buildArrayBindings(builder, match.arrays));
  streamingFunc->setAttr(optimization_attrs::kSourceIslandIdsAttrName,
                         builder.getI64ArrayAttr(match.sourceIslandIds));

  int64_t outputPositions =
      match.descriptor.outputHeight * match.descriptor.outputWidth;
  SmallVector<int64_t, 4> executionCounts(match.arrays.size(), outputPositions);
  streamingFunc->setAttr(runtime_attrs::kTaskAnalogExecutionCountsAttrName,
                         builder.getI64ArrayAttr(executionCounts));

  int64_t totalPhysicalColumns = 0;
  int64_t totalPhysicalRows = 0;
  SmallVector<int64_t, 4> loadBytesPerArray;
  SmallVector<int64_t, 4> storeBytesPerArray;
  for (const ArrayStage &stage : match.arrays) {
    totalPhysicalColumns += stage.arrayCols;
    totalPhysicalRows += stage.arrayRows;
    auto arrayLoadBytes =
        checkedMultiply(patchTask, {outputPositions, stage.arrayCols, 4},
                        "per-array streaming convolution load bytes");
    auto arrayStoreBytes =
        checkedMultiply(patchTask, {outputPositions, stage.arrayRows, 4},
                        "per-array streaming convolution store bytes");
    if (failed(arrayLoadBytes) || failed(arrayStoreBytes))
      return failure();
    loadBytesPerArray.push_back(*arrayLoadBytes);
    storeBytesPerArray.push_back(*arrayStoreBytes);
  }
  auto loadBytes =
      checkedMultiply(patchTask, {outputPositions, totalPhysicalColumns, 4},
                      "streaming convolution analog load byte count");
  auto storeBytes =
      checkedMultiply(patchTask, {outputPositions, totalPhysicalRows, 4},
                      "streaming convolution analog store byte count");
  auto patchOps = checkedMultiply(
      patchTask,
      {outputPositions, match.descriptor.inputChannels,
       match.descriptor.kernelHeight, match.descriptor.kernelWidth},
      "streaming convolution digital patch operation count");
  auto reductionOps = checkedMultiply(
      patchTask,
      {outputPositions, match.descriptor.outputChannels,
       static_cast<int64_t>(match.arrays.size() - 1)},
      "streaming convolution digital reduction operation count");
  auto biasOps =
      checkedMultiply(patchTask,
                      {outputPositions, match.descriptor.outputChannels,
                       match.descriptor.hasBias ? 1 : 0},
                      "streaming convolution digital bias operation count");
  if (failed(loadBytes) || failed(storeBytes) || failed(patchOps) ||
      failed(reductionOps) || failed(biasOps) ||
      *patchOps > std::numeric_limits<int64_t>::max() - *reductionOps ||
      *patchOps + *reductionOps >
          std::numeric_limits<int64_t>::max() - *biasOps)
    return failure();
  int64_t digitalOps = *patchOps + *reductionOps + *biasOps;
  streamingFunc->setAttr(runtime_attrs::kTaskAnalogLoadBytesAttrName,
                         builder.getI64IntegerAttr(*loadBytes));
  streamingFunc->setAttr(runtime_attrs::kTaskAnalogStoreBytesAttrName,
                         builder.getI64IntegerAttr(*storeBytes));
  streamingFunc->setAttr(runtime_attrs::kTaskAnalogLoadBytesPerArrayAttrName,
                         builder.getI64ArrayAttr(loadBytesPerArray));
  streamingFunc->setAttr(runtime_attrs::kTaskAnalogStoreBytesPerArrayAttrName,
                         builder.getI64ArrayAttr(storeBytesPerArray));
  streamingFunc->setAttr(runtime_attrs::kTaskDigitalOpsAttrName,
                         builder.getI64IntegerAttr(digitalOps));

  builder.setInsertionPointToEnd(module.getBody());
  builder.insert(streamingFunc);
  Block *entry = streamingFunc.addEntryBlock();
  OpBuilder bodyBuilder(entry, entry->begin());

  const ConvolutionDescriptor &descriptor = match.descriptor;
  Type f32 = bodyBuilder.getF32Type();
  auto outputTensorType = cast<RankedTensorType>(*outputType);
  Value zero = createIndexConstant(bodyBuilder, loc, 0);
  Value one = createIndexConstant(bodyBuilder, loc, 1);
  Value outputPositionUpper =
      createIndexConstant(bodyBuilder, loc, outputPositions);
  Value outputWidth =
      createIndexConstant(bodyBuilder, loc, descriptor.outputWidth);
  Value outputChannelUpper =
      createIndexConstant(bodyBuilder, loc, descriptor.outputChannels);
  Value kernelPlane = createIndexConstant(
      bodyBuilder, loc, descriptor.kernelHeight * descriptor.kernelWidth);
  Value kernelWidth =
      createIndexConstant(bodyBuilder, loc, descriptor.kernelWidth);
  Value strideHeight =
      createIndexConstant(bodyBuilder, loc, descriptor.strideHeight);
  Value strideWidth =
      createIndexConstant(bodyBuilder, loc, descriptor.strideWidth);
  Value outputInit = bodyBuilder.create<tensor::EmptyOp>(
      loc, outputTensorType.getShape(), outputTensorType.getElementType());

  Value bias;
  if (descriptor.hasBias)
    bias = bodyBuilder
               .create<arith::ConstantOp>(loc, descriptor.bias.getType(),
                                          descriptor.bias)
               .getResult();

  auto positionLoop = bodyBuilder.create<scf::ForOp>(
      loc, zero, outputPositionUpper, one, ValueRange{outputInit},
      [&](OpBuilder &positionBuilder, Location positionLoc, Value position,
          ValueRange positionIterArgs) {
        Value outputHeightIndex = positionBuilder.create<arith::DivUIOp>(
            positionLoc, position, outputWidth);
        Value outputWidthIndex = positionBuilder.create<arith::RemUIOp>(
            positionLoc, position, outputWidth);

        SmallVector<Value, 4> executedResults;
        executedResults.reserve(match.arrays.size());
        for (auto indexedStage : llvm::enumerate(match.arrays)) {
          const ArrayStage &stage = indexedStage.value();
          auto vectorType = RankedTensorType::get({1, stage.arrayCols}, f32);
          auto zeroVectorAttr = DenseElementsAttr::get(
              vectorType, positionBuilder.getF32FloatAttr(0.0));
          Value zeroVector = positionBuilder
                                 .create<arith::ConstantOp>(
                                     positionLoc, vectorType, zeroVectorAttr)
                                 .getResult();
          Value validColumnUpper = createIndexConstant(
              positionBuilder, positionLoc, stage.validCols);
          Value globalColumnOffset = createIndexConstant(
              positionBuilder, positionLoc, stage.tileColumn * stage.arrayCols);

          auto featureLoop = positionBuilder.create<scf::ForOp>(
              positionLoc, zero, validColumnUpper, one, ValueRange{zeroVector},
              [&](OpBuilder &featureBuilder, Location featureLoc,
                  Value localColumn, ValueRange featureIterArgs) {
                Value flattenedIndex = featureBuilder.create<arith::AddIOp>(
                    featureLoc, globalColumnOffset, localColumn);
                Value channel = featureBuilder.create<arith::DivUIOp>(
                    featureLoc, flattenedIndex, kernelPlane);
                Value kernelOffset = featureBuilder.create<arith::RemUIOp>(
                    featureLoc, flattenedIndex, kernelPlane);
                Value kernelHeightIndex = featureBuilder.create<arith::DivUIOp>(
                    featureLoc, kernelOffset, kernelWidth);
                Value kernelWidthIndex = featureBuilder.create<arith::RemUIOp>(
                    featureLoc, kernelOffset, kernelWidth);
                Value inputHeightBase = featureBuilder.create<arith::MulIOp>(
                    featureLoc, outputHeightIndex, strideHeight);
                Value inputWidthBase = featureBuilder.create<arith::MulIOp>(
                    featureLoc, outputWidthIndex, strideWidth);
                Value inputHeightIndex = featureBuilder.create<arith::AddIOp>(
                    featureLoc, inputHeightBase, kernelHeightIndex);
                Value inputWidthIndex = featureBuilder.create<arith::AddIOp>(
                    featureLoc, inputWidthBase, kernelWidthIndex);
                Value inputValue = featureBuilder.create<tensor::ExtractOp>(
                    featureLoc, entry->getArgument(0),
                    ValueRange{zero, channel, inputHeightIndex,
                               inputWidthIndex});
                Value updated = featureBuilder.create<tensor::InsertOp>(
                    featureLoc, inputValue, featureIterArgs.front(),
                    ValueRange{zero, localColumn});
                featureBuilder.create<scf::YieldOp>(featureLoc, updated);
              });

          auto load = positionBuilder.create<sculptor::ArrayLoadOp>(
              positionLoc, featureLoop.getResult(0),
              entry->getArgument(indexedStage.index() + 1));
          attachArrayBinding(load, positionBuilder, indexedStage.index(),
                             stage);
          auto execute = positionBuilder.create<sculptor::ArrayExecuteOp>(
              positionLoc, sculptor::ArrayResultType::get(module.getContext()),
              entry->getArgument(indexedStage.index() + 1));
          attachArrayBinding(execute, positionBuilder, indexedStage.index(),
                             stage);
          executedResults.push_back(execute.getResult());
        }

        SmallVector<Value, 4> partialResults;
        partialResults.reserve(match.arrays.size());
        for (auto indexedStage : llvm::enumerate(match.arrays)) {
          const ArrayStage &stage = indexedStage.value();
          auto storedType = RankedTensorType::get({1, stage.validRows}, f32);
          auto store = positionBuilder.create<sculptor::ArrayStoreOp>(
              positionLoc, storedType, executedResults[indexedStage.index()]);
          attachArrayBinding(store, positionBuilder, indexedStage.index(),
                             stage);
          attachArrayStoreShape(store, positionBuilder, stage);
          partialResults.push_back(store.getOutput());
        }

        Value reduced = partialResults.front();
        auto reducedType = cast<RankedTensorType>(reduced.getType());
        for (Value partial : llvm::drop_begin(partialResults)) {
          Value init = positionBuilder.create<tensor::EmptyOp>(
              positionLoc, reducedType.getShape(),
              reducedType.getElementType());
          reduced = positionBuilder
                        .create<linalg::AddOp>(positionLoc,
                                               ValueRange{reduced, partial},
                                               ValueRange{init})
                        .getResult(0);
        }

        auto channelLoop = positionBuilder.create<scf::ForOp>(
            positionLoc, zero, outputChannelUpper, one, positionIterArgs,
            [&](OpBuilder &channelBuilder, Location channelLoc, Value channel,
                ValueRange channelIterArgs) {
              Value value = channelBuilder.create<tensor::ExtractOp>(
                  channelLoc, reduced, ValueRange{zero, channel});
              if (bias) {
                Value biasValue = channelBuilder.create<tensor::ExtractOp>(
                    channelLoc, bias, ValueRange{channel});
                value = channelBuilder.create<arith::AddFOp>(channelLoc, value,
                                                             biasValue);
              }
              Value updated = channelBuilder.create<tensor::InsertOp>(
                  channelLoc, value, channelIterArgs.front(),
                  ValueRange{zero, channel, outputHeightIndex,
                             outputWidthIndex});
              channelBuilder.create<scf::YieldOp>(channelLoc, updated);
            });
        positionBuilder.create<scf::YieldOp>(positionLoc,
                                             channelLoop.getResult(0));
      });

  bodyBuilder.setInsertionPointAfter(positionLoop);
  bodyBuilder.create<func::ReturnOp>(loc, positionLoop.getResult(0));
  return streamingFunc;
}

static void
replaceComponentDependencies(sculptor::TaskCreateOp replacement,
                             const SmallPtrSetImpl<Operation *> &componentOps,
                             func::FuncOp taskGraphFunc) {
  for (sculptor::TaskCreateOp task :
       taskGraphFunc.getOps<sculptor::TaskCreateOp>()) {
    if (task == replacement || componentOps.contains(task))
      continue;

    SmallVector<Value, 8> dependencies;
    bool changed = false;
    for (Value dependency : task.getDependencies()) {
      auto producer = dependency.getDefiningOp<sculptor::TaskCreateOp>();
      if (!producer || !componentOps.contains(producer)) {
        appendUniqueValue(dependencies, dependency);
        continue;
      }
      changed = true;
      appendUniqueValue(dependencies, replacement.getResult());
    }
    if (changed)
      task.getDependenciesMutable().assign(dependencies);
  }
}

static LogicalResult
rewriteStreamingConvolution(ModuleOp module, func::FuncOp taskGraphFunc,
                            const StreamingConvolutionMatch &match) {
  auto streamingFunc = buildStreamingConvolutionCallee(module, match);
  if (failed(streamingFunc))
    return failure();

  sculptor::TaskCreateOp patchTask = match.patchNode->op;
  OpBuilder builder(patchTask);
  SmallVector<Value, 4> inputs{match.activationResource};
  for (const ArrayStage &stage : match.arrays)
    inputs.push_back(stage.logicalArrayResource);

  std::string taskName =
      (patchTask.getSourceLayer() + "_streaming_conv_mvm").str();
  auto replacement = builder.create<sculptor::TaskCreateOp>(
      patchTask.getLoc(), patchTask.getResult().getType(), patchTask.getGraph(),
      FlatSymbolRefAttr::get(module.getContext(),
                             (*streamingFunc).getSymName()),
      builder.getStringAttr(task_graph_names::kDigitalDomain),
      builder.getStringAttr(task_graph_names::kStreamingConvolutionTaskKind),
      builder.getStringAttr(taskName),
      builder.getStringAttr(patchTask.getSourceLayer()),
      patchTask.getSourceTaskOrdinalAttr(), inputs,
      ValueRange{match.outputResource}, match.externalDependencies);
  replacement->setAttr(runtime_attrs::kTaskCoreIdAttrName,
                       builder.getI64IntegerAttr(match.coreId));
  replacement->setAttr(schedule_attrs::kIslandIdAttrName,
                       builder.getI64IntegerAttr(match.homeIslandId));
  replacement->setAttr(optimization_attrs::kSourceIslandIdsAttrName,
                       builder.getI64ArrayAttr(match.sourceIslandIds));
  replacement->setAttr(runtime_attrs::kTaskArrayBindingsAttrName,
                       buildArrayBindings(builder, match.arrays));
  for (StringRef attrName :
       {runtime_attrs::kTaskAnalogExecutionCountsAttrName,
        runtime_attrs::kTaskAnalogLoadBytesAttrName,
        runtime_attrs::kTaskAnalogStoreBytesAttrName,
        runtime_attrs::kTaskAnalogLoadBytesPerArrayAttrName,
        runtime_attrs::kTaskAnalogStoreBytesPerArrayAttrName,
        runtime_attrs::kTaskDigitalOpsAttrName}) {
    replacement->setAttr(attrName, (*streamingFunc)->getAttr(attrName));
  }
  replacement->setAttr(runtime_attrs::kTaskResultIndicesAttrName,
                       builder.getI64ArrayAttr({0}));

  SmallPtrSet<Operation *, 16> componentOps;
  for (const TaskGraphNode *node : match.componentNodes)
    componentOps.insert(node->op);
  replaceComponentDependencies(replacement, componentOps, taskGraphFunc);

  for (const TaskGraphNode *node : llvm::reverse(match.componentNodes)) {
    sculptor::TaskCreateOp task = node->op;
    if (!task.getResult().use_empty()) {
      task.emitError(
          "expected optimized convolution task to have no remaining users");
      return failure();
    }
    task.erase();
  }
  return success();
}

} // namespace

LogicalResult optimizeStreamingConvolution(ModuleOp module,
                                           func::FuncOp taskGraphFunc,
                                           const TaskGraphDAG &dag,
                                           bool &changed) {
  changed = false;
  DenseMap<Value, const TaskGraphNode *> producerByResource;
  DenseMap<Value, SmallVector<const TaskGraphNode *, 4>> consumersByResource;
  collectResourceProducers(dag, producerByResource);
  collectResourceConsumers(dag, consumersByResource);

  for (const TaskGraphNode &node : dag.nodes) {
    auto match = matchStreamingConvolution(
        module, dag, node, producerByResource, consumersByResource);
    if (!match)
      continue;
    if (failed(rewriteStreamingConvolution(module, taskGraphFunc, *match)))
      return failure();
    changed = true;
    return success();
  }
  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
