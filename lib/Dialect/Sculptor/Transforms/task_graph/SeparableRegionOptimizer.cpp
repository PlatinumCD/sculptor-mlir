#include "sculptor-mlir/Dialect/Sculptor/Transforms/task_graph/SeparableRegionOptimizer.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TaskGraphTaskAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace mlir {
namespace sculptor {
namespace task_graph {
namespace {

namespace task_attrs = mlir::sculptor::task_attrs;

constexpr llvm::StringLiteral
    kOutputRecombineTaskKind("digital.output_recombine");
constexpr llvm::StringLiteral kActivationTaskKind("digital.activation");
constexpr llvm::StringLiteral kDigitalDomain("digital");
constexpr llvm::StringLiteral
    kSeparableRegionAttr("sculptor.optimization.separable_region");
constexpr llvm::StringLiteral
    kSeparableSliceAttr("sculptor.optimization.separable_slice");
constexpr llvm::StringLiteral kSeparableParentOrdinalAttr(
    "sculptor.optimization.separable_parent_ordinal");

struct StaticSlice {
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
};

struct RecombineSlice {
  unsigned inputIndex = 0;
  StaticSlice slice;
};

struct SliceTaskMatch {
  const TaskGraphNode *node = nullptr;
  func::FuncOp callee;
  unsigned inputIndex = 0;
};

struct SeparableRegionMatch {
  const TaskGraphNode *recombineNode = nullptr;
  const TaskGraphNode *activationNode = nullptr;
  func::FuncOp activationCallee;
  linalg::GenericOp activationGeneric;
  SmallVector<RecombineSlice> recombineSlices;
  SmallVector<SliceTaskMatch> sliceTasks;
};

static bool isStaticF32Tensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.hasStaticShape() && tensor.getElementType().isF32();
}

static FailureOr<Type> getResourceValueType(Value resource) {
  auto type = dyn_cast<sculptor::TaskResourceType>(resource.getType());
  if (!type)
    return failure();
  return type.getValueType();
}

static FailureOr<func::FuncOp> lookupTaskCallee(ModuleOp module,
                                                sculptor::TaskCreateOp task) {
  auto callee =
      module.lookupSymbol<func::FuncOp>(task.getCalleeAttr().getValue());
  if (!callee)
    return failure();
  return callee;
}

static std::optional<StaticSlice> getStaticSlice(tensor::InsertSliceOp op) {
  if (llvm::any_of(op.getStaticOffsets(), ShapedType::isDynamic) ||
      llvm::any_of(op.getStaticSizes(), ShapedType::isDynamic) ||
      llvm::any_of(op.getStaticStrides(), ShapedType::isDynamic) ||
      llvm::any_of(op.getStaticStrides(),
                   [](int64_t stride) { return stride != 1; }))
    return std::nullopt;
  return StaticSlice{SmallVector<int64_t>(op.getStaticOffsets()),
                     SmallVector<int64_t>(op.getStaticSizes()),
                     SmallVector<int64_t>(op.getStaticStrides())};
}

static std::optional<StaticSlice> getStaticSlice(tensor::ExtractSliceOp op) {
  if (llvm::any_of(op.getStaticOffsets(), ShapedType::isDynamic) ||
      llvm::any_of(op.getStaticSizes(), ShapedType::isDynamic) ||
      llvm::any_of(op.getStaticStrides(), ShapedType::isDynamic) ||
      llvm::any_of(op.getStaticStrides(),
                   [](int64_t stride) { return stride != 1; }))
    return std::nullopt;
  return StaticSlice{SmallVector<int64_t>(op.getStaticOffsets()),
                     SmallVector<int64_t>(op.getStaticSizes()),
                     SmallVector<int64_t>(op.getStaticStrides())};
}

static bool sameSlice(const StaticSlice &lhs, const StaticSlice &rhs) {
  return lhs.offsets == rhs.offsets && lhs.sizes == rhs.sizes &&
         lhs.strides == rhs.strides;
}

static bool areDisjoint(const StaticSlice &lhs, const StaticSlice &rhs) {
  for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
       llvm::zip_equal(lhs.offsets, lhs.sizes, rhs.offsets, rhs.sizes)) {
    if (lhsOffset + lhsSize <= rhsOffset || rhsOffset + rhsSize <= lhsOffset)
      return true;
  }
  return false;
}

static int64_t getElementCount(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t extent : shape)
    count *= extent;
  return count;
}

static Value stripShapeOnlyOps(Value value,
                               SmallPtrSetImpl<Operation *> &allowed) {
  while (Operation *operation = value.getDefiningOp()) {
    if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(operation))
      break;
    allowed.insert(operation);
    value = operation->getOperand(0);
  }
  return value;
}

static FailureOr<SmallVector<RecombineSlice>>
matchRecombineFunction(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  if (type.getNumInputs() < 2 || type.getNumResults() != 1 ||
      !isStaticF32Tensor(type.getResult(0)) ||
      !function.getBody().hasOneBlock())
    return failure();
  for (Type input : type.getInputs()) {
    if (!isStaticF32Tensor(input))
      return failure();
  }

  Block &entry = function.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(entry.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1)
    return failure();

  SmallPtrSet<Operation *, 16> allowed;
  allowed.insert(returnOp);
  Value value = returnOp.getOperand(0);
  SmallVector<RecombineSlice> slices;
  while (auto insert = value.getDefiningOp<tensor::InsertSliceOp>()) {
    auto slice = getStaticSlice(insert);
    if (!slice)
      return failure();
    allowed.insert(insert);

    Value source = stripShapeOnlyOps(insert.getSource(), allowed);
    auto argument = dyn_cast<BlockArgument>(source);
    if (!argument || argument.getOwner() != &entry)
      return failure();
    unsigned inputIndex = argument.getArgNumber();
    if (inputIndex >= type.getNumInputs() ||
        getElementCount(slice->sizes) !=
            cast<RankedTensorType>(type.getInput(inputIndex)).getNumElements())
      return failure();
    slices.push_back(RecombineSlice{inputIndex, std::move(*slice)});
    value = insert.getDest();
  }

  auto empty = value.getDefiningOp<tensor::EmptyOp>();
  if (!empty || empty.getType() != type.getResult(0))
    return failure();
  allowed.insert(empty);
  if (slices.size() != type.getNumInputs())
    return failure();

  SmallVector<bool> seen(type.getNumInputs(), false);
  int64_t coveredElements = 0;
  for (const RecombineSlice &slice : slices) {
    if (seen[slice.inputIndex])
      return failure();
    seen[slice.inputIndex] = true;
    coveredElements += getElementCount(slice.slice.sizes);
  }
  auto outputType = cast<RankedTensorType>(type.getResult(0));
  if (coveredElements != outputType.getNumElements())
    return failure();
  for (auto [index, lhs] : llvm::enumerate(slices)) {
    for (const RecombineSlice &rhs : llvm::drop_begin(slices, index + 1)) {
      if (!areDisjoint(lhs.slice, rhs.slice))
        return failure();
    }
  }

  for (Operation &operation : entry) {
    if (!allowed.contains(&operation))
      return failure();
  }
  llvm::sort(slices, [](const RecombineSlice &lhs, const RecombineSlice &rhs) {
    return lhs.inputIndex < rhs.inputIndex;
  });
  return slices;
}

static FailureOr<StaticSlice> matchSliceFunction(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  if (type.getNumInputs() != 1 || type.getNumResults() != 1 ||
      !isStaticF32Tensor(type.getInput(0)) ||
      !isStaticF32Tensor(type.getResult(0)) ||
      !function.getBody().hasOneBlock())
    return failure();

  Block &entry = function.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(entry.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1)
    return failure();
  SmallPtrSet<Operation *, 8> allowed;
  allowed.insert(returnOp);
  Value value = stripShapeOnlyOps(returnOp.getOperand(0), allowed);
  auto extract = value.getDefiningOp<tensor::ExtractSliceOp>();
  if (!extract || extract.getSource() != entry.getArgument(0))
    return failure();
  auto slice = getStaticSlice(extract);
  if (!slice)
    return failure();
  allowed.insert(extract);
  if (getElementCount(slice->sizes) !=
      cast<RankedTensorType>(type.getResult(0)).getNumElements())
    return failure();
  for (Operation &operation : entry) {
    if (!allowed.contains(&operation))
      return failure();
  }
  return *slice;
}

static FailureOr<linalg::GenericOp>
matchPointwiseFunction(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  if (type.getNumInputs() != 1 || type.getNumResults() != 1 ||
      type.getInput(0) != type.getResult(0) ||
      !isStaticF32Tensor(type.getInput(0)) || !function.getBody().hasOneBlock())
    return failure();

  Block &entry = function.getBody().front();
  linalg::GenericOp generic;
  tensor::EmptyOp empty;
  func::ReturnOp returnOp;
  for (Operation &operation : entry) {
    if (auto candidate = dyn_cast<linalg::GenericOp>(operation)) {
      if (generic)
        return failure();
      generic = candidate;
    } else if (auto candidate = dyn_cast<tensor::EmptyOp>(operation)) {
      if (empty)
        return failure();
      empty = candidate;
    } else if (auto candidate = dyn_cast<func::ReturnOp>(operation)) {
      returnOp = candidate;
    } else {
      return failure();
    }
  }
  auto tensorType = cast<RankedTensorType>(type.getInput(0));
  if (!generic || !empty || !returnOp || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || generic->getNumResults() != 1 ||
      generic.getDpsInputs().front() != entry.getArgument(0) ||
      generic.getDpsInits().front() != empty.getResult() ||
      returnOp.getNumOperands() != 1 ||
      returnOp.getOperand(0) != generic.getResult(0) ||
      generic.getNumLoops() != tensorType.getRank())
    return failure();
  if (llvm::any_of(generic.getIteratorTypesArray(),
                   [](utils::IteratorType iterator) {
                     return iterator != utils::IteratorType::parallel;
                   }) ||
      llvm::any_of(generic.getIndexingMapsArray(), [&](AffineMap map) {
        return map.getNumDims() != tensorType.getRank() || !map.isIdentity();
      }))
    return failure();

  Block &body = generic.getRegion().front();
  if (!body.getArguments().back().use_empty() ||
      !isa<linalg::YieldOp>(body.getTerminator()))
    return failure();
  for (Operation &operation : body.without_terminator()) {
    if (isa<linalg::IndexOp>(operation) || !isMemoryEffectFree(&operation))
      return failure();
  }
  return generic;
}

static void
collectResourceEdges(const TaskGraphDAG &dag,
                     DenseMap<Value, const TaskGraphNode *> &producerByResource,
                     DenseMap<Value, SmallVector<const TaskGraphNode *, 4>>
                         &consumersByResource) {
  for (const TaskGraphNode &node : dag.nodes) {
    sculptor::TaskCreateOp task = node.op;
    for (Value output : task.getOutputs())
      producerByResource.try_emplace(output, &node);
    for (Value input : task.getInputs())
      consumersByResource[input].push_back(&node);
  }
}

static bool taskResultUsedOnlyBy(sculptor::TaskCreateOp task,
                                 ArrayRef<SliceTaskMatch> users) {
  for (OpOperand &use : task.getResult().getUses()) {
    auto user = dyn_cast<sculptor::TaskCreateOp>(use.getOwner());
    if (!user || llvm::none_of(users, [&](const SliceTaskMatch &match) {
          return match.node->op == user;
        }))
      return false;
  }
  return true;
}

static std::optional<SeparableRegionMatch> matchSeparableRegion(
    ModuleOp module, const TaskGraphDAG &dag,
    const TaskGraphNode &recombineNode,
    const DenseMap<Value, SmallVector<const TaskGraphNode *, 4>>
        &consumersByResource) {
  sculptor::TaskCreateOp recombineTask = recombineNode.op;
  if (recombineTask.getTaskKind() != kOutputRecombineTaskKind ||
      recombineTask.getInputs().size() < 2 ||
      recombineTask.getOutputs().size() != 1)
    return std::nullopt;
  auto recombineCallee = lookupTaskCallee(module, recombineTask);
  if (failed(recombineCallee))
    return std::nullopt;
  auto recombineSlices = matchRecombineFunction(*recombineCallee);
  if (failed(recombineSlices) ||
      recombineSlices->size() != recombineTask.getInputs().size())
    return std::nullopt;

  Value recombinedResource = recombineTask.getOutputs().front();
  auto activationIt = consumersByResource.find(recombinedResource);
  if (activationIt == consumersByResource.end() ||
      activationIt->second.size() != 1)
    return std::nullopt;
  const TaskGraphNode *activationNode = activationIt->second.front();
  sculptor::TaskCreateOp activationTask = activationNode->op;
  if (activationTask.getTaskKind() != kActivationTaskKind ||
      activationTask.getInputs().size() != 1 ||
      activationTask.getInputs().front() != recombinedResource ||
      activationTask.getOutputs().size() != 1 ||
      activationTask.getSourceLayer() != recombineTask.getSourceLayer())
    return std::nullopt;
  auto activationCallee = lookupTaskCallee(module, activationTask);
  if (failed(activationCallee))
    return std::nullopt;
  auto activationGeneric = matchPointwiseFunction(*activationCallee);
  if (failed(activationGeneric))
    return std::nullopt;

  FailureOr<Type> recombinedType = getResourceValueType(recombinedResource);
  FailureOr<Type> activatedType =
      getResourceValueType(activationTask.getOutputs().front());
  if (failed(recombinedType) || failed(activatedType) ||
      *recombinedType != activationCallee->getFunctionType().getInput(0) ||
      *activatedType != activationCallee->getFunctionType().getResult(0))
    return std::nullopt;

  Value activatedResource = activationTask.getOutputs().front();
  auto slicesIt = consumersByResource.find(activatedResource);
  if (slicesIt == consumersByResource.end() ||
      slicesIt->second.size() != recombineSlices->size())
    return std::nullopt;

  SmallVector<SliceTaskMatch> sliceTasks;
  SmallVector<bool> matchedInputs(recombineSlices->size(), false);
  for (const TaskGraphNode *sliceNode : slicesIt->second) {
    sculptor::TaskCreateOp sliceTask = sliceNode->op;
    if (sliceTask.getDomain() != kDigitalDomain ||
        sliceTask.getSourceLayer() != recombineTask.getSourceLayer() ||
        sliceTask.getInputs().size() != 1 ||
        sliceTask.getInputs().front() != activatedResource ||
        sliceTask.getOutputs().size() != 1)
      return std::nullopt;
    auto sliceCallee = lookupTaskCallee(module, sliceTask);
    if (failed(sliceCallee))
      return std::nullopt;
    auto extractedSlice = matchSliceFunction(*sliceCallee);
    if (failed(extractedSlice))
      return std::nullopt;

    std::optional<unsigned> inputIndex;
    for (const RecombineSlice &candidate : *recombineSlices) {
      if (!sameSlice(candidate.slice, *extractedSlice))
        continue;
      if (inputIndex || matchedInputs[candidate.inputIndex])
        return std::nullopt;
      inputIndex = candidate.inputIndex;
    }
    if (!inputIndex)
      return std::nullopt;

    FailureOr<Type> sliceOutputType =
        getResourceValueType(sliceTask.getOutputs().front());
    FailureOr<Type> recombineInputType =
        getResourceValueType(recombineTask.getInputs()[*inputIndex]);
    if (failed(sliceOutputType) || failed(recombineInputType) ||
        *sliceOutputType != *recombineInputType ||
        *sliceOutputType != sliceCallee->getFunctionType().getResult(0))
      return std::nullopt;
    matchedInputs[*inputIndex] = true;
    sliceTasks.push_back(SliceTaskMatch{sliceNode, *sliceCallee, *inputIndex});
  }
  if (llvm::any_of(matchedInputs, [](bool matched) { return !matched; }))
    return std::nullopt;
  llvm::sort(sliceTasks,
             [](const SliceTaskMatch &lhs, const SliceTaskMatch &rhs) {
               return lhs.inputIndex < rhs.inputIndex;
             });

  SmallVector<SliceTaskMatch> activationUsers = sliceTasks;
  if (!taskResultUsedOnlyBy(activationTask, activationUsers))
    return std::nullopt;
  for (OpOperand &use : recombineTask.getResult().getUses()) {
    if (use.getOwner() != activationTask)
      return std::nullopt;
  }

  return SeparableRegionMatch{&recombineNode,
                              activationNode,
                              *activationCallee,
                              *activationGeneric,
                              std::move(*recombineSlices),
                              std::move(sliceTasks)};
}

static std::string getUniqueSymbolName(ModuleOp module, StringRef base) {
  std::string candidate = base.str();
  unsigned suffix = 0;
  while (module.lookupSymbol(candidate))
    candidate = (base + "_" + Twine(suffix++)).str();
  return candidate;
}

static func::FuncOp buildSliceCallee(ModuleOp module,
                                     const SeparableRegionMatch &match,
                                     const SliceTaskMatch &sliceMatch) {
  sculptor::TaskCreateOp sliceTask = sliceMatch.node->op;
  sculptor::TaskCreateOp recombineTask = match.recombineNode->op;
  auto inputType = cast<RankedTensorType>(
      cast<sculptor::TaskResourceType>(
          recombineTask.getInputs()[sliceMatch.inputIndex].getType())
          .getValueType());
  OpBuilder builder(module.getContext());
  func::FuncOp sliceCallee = sliceMatch.callee;
  std::string symbol = getUniqueSymbolName(
      module, (sliceCallee.getSymName() + "_separable_activation").str());
  auto function = func::FuncOp::create(
      sliceTask.getLoc(), symbol,
      builder.getFunctionType(TypeRange{inputType}, TypeRange{inputType}));
  function.setPrivate();
  if (match.activationCallee->hasAttr("llvm.emit_c_interface"))
    function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  function->setAttr(task_attrs::kTaskDomainAttrName,
                    builder.getStringAttr(kDigitalDomain));
  function->setAttr(task_attrs::kTaskKindAttrName,
                    builder.getStringAttr(kActivationTaskKind));
  function->setAttr(
      task_attrs::kTaskNameAttrName,
      builder.getStringAttr((sliceTask.getTaskName() + ".separable").str()));
  function->setAttr(task_attrs::kSourceLayerAttrName,
                    sliceTask.getSourceLayerAttr());
  function->setAttr(task_attrs::kSourceTaskOrdinalAttrName,
                    sliceTask.getSourceTaskOrdinalAttr());
  function->setAttr(kSeparableRegionAttr, builder.getUnitAttr());
  function->setAttr(kSeparableSliceAttr,
                    builder.getI64IntegerAttr(sliceMatch.inputIndex));
  sculptor::TaskCreateOp activationTask = match.activationNode->op;
  function->setAttr(kSeparableParentOrdinalAttr,
                    activationTask.getSourceTaskOrdinalAttr());

  builder.setInsertionPointToEnd(module.getBody());
  builder.insert(function);
  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value empty = builder.create<tensor::EmptyOp>(
      sliceTask.getLoc(), inputType.getShape(), inputType.getElementType());
  AffineMap identity = AffineMap::getMultiDimIdentityMap(inputType.getRank(),
                                                         module.getContext());
  SmallVector<utils::IteratorType> iterators(inputType.getRank(),
                                             utils::IteratorType::parallel);
  linalg::GenericOp activationGeneric = match.activationGeneric;
  Block &sourceBody = activationGeneric.getRegion().front();
  auto sourceYield = cast<linalg::YieldOp>(sourceBody.getTerminator());
  auto generic = builder.create<linalg::GenericOp>(
      sliceTask.getLoc(), inputType, ValueRange{entry->getArgument(0)},
      ValueRange{empty}, ArrayRef<AffineMap>{identity, identity}, iterators,
      [&](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange arguments) {
        IRMapping mapping;
        for (auto [sourceArgument, targetArgument] :
             llvm::zip_equal(sourceBody.getArguments(), arguments))
          mapping.map(sourceArgument, targetArgument);
        for (Operation &operation : sourceBody.without_terminator())
          bodyBuilder.clone(operation, mapping);
        bodyBuilder.create<linalg::YieldOp>(
            bodyLoc, mapping.lookupOrDefault(sourceYield.getValues().front()));
      });
  builder.create<func::ReturnOp>(sliceTask.getLoc(), generic.getResult(0));
  return function;
}

static void appendUnique(SmallVectorImpl<Value> &values, Value value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

static SmallVector<Value> buildSliceDependencies(
    const TaskGraphDAG &dag, const SeparableRegionMatch &match,
    const SliceTaskMatch &sliceMatch,
    const DenseMap<Value, const TaskGraphNode *> &producerByResource) {
  sculptor::TaskCreateOp recombineTask = match.recombineNode->op;
  sculptor::TaskCreateOp activationTask = match.activationNode->op;
  sculptor::TaskCreateOp sliceTask = sliceMatch.node->op;
  SmallPtrSet<Operation *, 2> removed{recombineTask, activationTask};
  SmallVector<Value> dependencies;

  for (Value dependency : sliceTask.getDependencies()) {
    Operation *producer = dependency.getDefiningOp();
    if (!removed.contains(producer))
      appendUnique(dependencies, dependency);
  }
  for (Value dependency : activationTask.getDependencies()) {
    Operation *producer = dependency.getDefiningOp();
    if (!removed.contains(producer))
      appendUnique(dependencies, dependency);
  }

  Value selectedInput = recombineTask.getInputs()[sliceMatch.inputIndex];
  const TaskGraphNode *selectedProducer = nullptr;
  if (auto it = producerByResource.find(selectedInput);
      it != producerByResource.end())
    selectedProducer = it->second;

  SmallPtrSet<Operation *, 8> inputProducers;
  for (Value input : recombineTask.getInputs()) {
    if (auto it = producerByResource.find(input);
        it != producerByResource.end())
      inputProducers.insert(it->second->op);
  }
  for (Value dependency : recombineTask.getDependencies()) {
    auto producer = dependency.getDefiningOp<sculptor::TaskCreateOp>();
    if (producer && inputProducers.contains(producer) &&
        (!selectedProducer || producer != selectedProducer->op))
      continue;
    appendUnique(dependencies, dependency);
  }
  if (selectedProducer)
    appendUnique(dependencies,
                 sculptor::TaskCreateOp(selectedProducer->op).getResult());

  DenseMap<Value, unsigned> indexByTask;
  for (const TaskGraphNode &node : dag.nodes)
    indexByTask.try_emplace(sculptor::TaskCreateOp(node.op).getResult(),
                            node.index);
  llvm::sort(dependencies, [&](Value lhs, Value rhs) {
    return indexByTask.lookup(lhs) < indexByTask.lookup(rhs);
  });
  return dependencies;
}

static LogicalResult rewriteSeparableRegion(
    ModuleOp module, const TaskGraphDAG &dag, const SeparableRegionMatch &match,
    const DenseMap<Value, const TaskGraphNode *> &producerByResource) {
  sculptor::TaskCreateOp recombineTask = match.recombineNode->op;
  sculptor::TaskCreateOp activationTask = match.activationNode->op;
  OpBuilder builder(module.getContext());

  for (const SliceTaskMatch &sliceMatch : match.sliceTasks) {
    func::FuncOp callee = buildSliceCallee(module, match, sliceMatch);
    sculptor::TaskCreateOp task = sliceMatch.node->op;
    Value input = recombineTask.getInputs()[sliceMatch.inputIndex];
    SmallVector<Value> dependencies =
        buildSliceDependencies(dag, match, sliceMatch, producerByResource);
    task.setCalleeAttr(
        FlatSymbolRefAttr::get(module.getContext(), callee.getSymName()));
    task.setTaskKindAttr(builder.getStringAttr(kActivationTaskKind));
    task.setTaskNameAttr(
        builder.getStringAttr((task.getTaskName() + ".separable").str()));
    task.getInputsMutable().assign(ValueRange{input});
    task.getDependenciesMutable().assign(dependencies);
    task->setAttr(kSeparableRegionAttr, builder.getUnitAttr());
    task->setAttr(kSeparableSliceAttr,
                  builder.getI64IntegerAttr(sliceMatch.inputIndex));
    task->setAttr(kSeparableParentOrdinalAttr,
                  activationTask.getSourceTaskOrdinalAttr());
  }

  if (!activationTask.getResult().use_empty())
    return activationTask.emitError(
        "expected separable activation task to have no remaining users");
  activationTask.erase();
  if (!recombineTask.getResult().use_empty())
    return recombineTask.emitError(
        "expected separable recombine task to have no remaining users");
  recombineTask.erase();
  return success();
}

} // namespace

LogicalResult optimizeSeparableRegions(ModuleOp module,
                                       func::FuncOp taskGraphFunc,
                                       const TaskGraphDAG &dag, bool &changed) {
  changed = false;
  DenseMap<Value, const TaskGraphNode *> producerByResource;
  DenseMap<Value, SmallVector<const TaskGraphNode *, 4>> consumersByResource;
  collectResourceEdges(dag, producerByResource, consumersByResource);

  for (const TaskGraphNode &node : dag.nodes) {
    auto match = matchSeparableRegion(module, dag, node, consumersByResource);
    if (!match)
      continue;
    if (failed(rewriteSeparableRegion(module, dag, *match, producerByResource)))
      return failure();
    changed = true;
    return success();
  }
  return success();
}

} // namespace task_graph
} // namespace sculptor
} // namespace mlir
