#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ReductionTree.h"

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <optional>
#include <set>

namespace {

using namespace mlir;
using namespace mlir::sculptor::mapping;

template <typename T>
void appendUnique(llvm::SmallVectorImpl<T> &values, T value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

bool isDigitalBiasAdd(const ComputeOperation &operation) {
  if (operation.kind != ComputeOperationKind::Structured)
    return false;
  return llvm::any_of(operation.members, [](Operation *member) {
    auto section =
        member->getAttrOfType<StringAttr>("sculptor.semantic.section");
    return member->getName().getStringRef() == "linalg.add" && section &&
           section.getValue() == "digital.bias_add";
  });
}

bool isComputeResourceType(Type type) {
  return isa<ShapedType, sculptor::LogicalArrayType>(type);
}

int64_t getStaticTensorByteSize(Type type) {
  if (isa<sculptor::LogicalArrayType>(type))
    return 0;

  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType)
    return -1;
  if (!shapedType.hasStaticShape())
    return -1;

  Type elementType = shapedType.getElementType();
  if (!elementType.isIntOrFloat())
    return -1;

  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return -1;

  std::optional<int64_t> byteSize =
      llvm::checkedMul(shapedType.getNumElements(), int64_t{bitWidth / 8});
  return byteSize.value_or(-1);
}

bool isFunctionArgument(func::FuncOp func, Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner()->getParentOp() == func.getOperation();
}

void collectUpstreamComputeOperations(
    func::FuncOp func, Value value,
    const DenseMap<Operation *, int64_t> &operationIds,
    SmallVectorImpl<int64_t> &producerOperations, bool &isFunctionInput,
    llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (isFunctionArgument(func, value)) {
    isFunctionInput = true;
    return;
  }

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return;

  auto operationId = operationIds.find(definingOp);
  if (operationId != operationIds.end()) {
    appendUnique(producerOperations, operationId->second);
    return;
  }

  if (!visited.insert(definingOp).second)
    return;

  for (Value operand : definingOp->getOperands()) {
    if (!isComputeResourceType(operand.getType()))
      continue;
    collectUpstreamComputeOperations(func, operand, operationIds,
                                     producerOperations, isFunctionInput,
                                     visited);
  }
}

std::optional<ComputeOperationKind>
classifyComputeOperation(Operation *operation) {
  if (isa<sculptor::MVMOp>(operation))
    return ComputeOperationKind::LogicalMVM;
  if (isa<TilingInterface>(operation))
    return ComputeOperationKind::Structured;
  return std::nullopt;
}

FailureOr<ComputeOperationKind> classifyStage(Operation *operation,
                                              StringRef stageKind) {
  if (stageKind == kMatrixSetupStageKind)
    return ComputeOperationKind::MatrixSetup;
  if (stageKind == kDigitalStageKind)
    return ComputeOperationKind::DigitalStage;
  if (stageKind == kVectorTileStageKind)
    return ComputeOperationKind::VectorTile;
  if (stageKind == kPhysicalMVMStageKind)
    return ComputeOperationKind::PhysicalMVM;
  if (stageKind == kTileRecombineStageKind)
    return ComputeOperationKind::TileRecombine;
  operation->emitError("unknown mapping stage kind '") << stageKind << "'";
  return failure();
}

std::optional<std::pair<int64_t, int64_t>> getI64PairAttr(Operation *operation,
                                                          StringRef name) {
  auto values = operation->getAttrOfType<ArrayAttr>(name);
  if (!values || values.size() != 2)
    return std::nullopt;
  auto first = dyn_cast<IntegerAttr>(values[0]);
  auto second = dyn_cast<IntegerAttr>(values[1]);
  if (!first || !second)
    return std::nullopt;
  return std::make_pair(first.getInt(), second.getInt());
}

LogicalResult populateStageDomain(ComputeOperation &operation) {
  if (operation.kind == ComputeOperationKind::PhysicalMVM) {
    for (Operation *member : operation.members) {
      auto shape = getI64PairAttr(member, "sculptor.tile_physical_shape");
      if (!shape)
        continue;
      operation.analogMVM = AnalogMVMGeometry{shape->first, shape->second};
      return success();
    }
    return operation.operation->emitError(
        "physical MVM stage is missing sculptor.tile_physical_shape");
  }

  for (Operation *member : llvm::reverse(operation.members)) {
    for (Value result : llvm::reverse(member->getResults())) {
      auto type = dyn_cast<ShapedType>(result.getType());
      if (!type)
        continue;
      int64_t extent =
          type.hasStaticShape() ? type.getNumElements() : ShapedType::kDynamic;
      operation.iterationDomain.push_back(
          {/*loopIndex=*/0, ComputeIteratorKind::Parallel, extent});
      return success();
    }
  }
  return success();
}

LogicalResult populateMVMIterationDomain(ComputeOperation &computeOperation,
                                         sculptor::MVMOp mvm) {
  auto matrixType = dyn_cast<RankedTensorType>(mvm.getMatrix().getType());
  if (!matrixType || matrixType.getRank() != 2) {
    return mvm.emitOpError(
        "expected rank-2 matrix geometry for RA-tree construction");
  }

  int64_t outputRows = matrixType.getDimSize(0);
  int64_t inputColumns = matrixType.getDimSize(1);
  computeOperation.analogMVM = AnalogMVMGeometry{outputRows, inputColumns};
  computeOperation.iterationDomain.push_back(
      {/*loopIndex=*/0, ComputeIteratorKind::Parallel, outputRows});
  computeOperation.iterationDomain.push_back(
      {/*loopIndex=*/1, ComputeIteratorKind::Reduction, inputColumns});
  return success();
}

LogicalResult populateIterationDomain(ComputeOperation &computeOperation) {
  if (computeOperation.kind == ComputeOperationKind::LogicalMVM) {
    auto mvm = cast<sculptor::MVMOp>(computeOperation.operation);
    return populateMVMIterationDomain(computeOperation, mvm);
  }

  if (computeOperation.kind != ComputeOperationKind::Structured)
    return populateStageDomain(computeOperation);

  auto tiling = dyn_cast<TilingInterface>(computeOperation.operation);
  if (!tiling) {
    return computeOperation.operation->emitError(
        "supported RA-tree compute operation has no iteration-domain model");
  }
  SmallVector<utils::IteratorType> iteratorTypes =
      tiling.getLoopIteratorTypes();
  SmallVector<int64_t> staticExtents(iteratorTypes.size(),
                                     ShapedType::kDynamic);

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOperation.operation)) {
    SmallVector<int64_t> linalgExtents = linalgOp.getStaticLoopRanges();
    if (linalgExtents.size() == iteratorTypes.size())
      staticExtents = std::move(linalgExtents);
  }

  computeOperation.iterationDomain.reserve(iteratorTypes.size());
  for (auto [index, iteratorType] : llvm::enumerate(iteratorTypes)) {
    ComputeIteratorKind kind = iteratorType == utils::IteratorType::reduction
                                   ? ComputeIteratorKind::Reduction
                                   : ComputeIteratorKind::Parallel;
    computeOperation.iterationDomain.push_back(
        {static_cast<int64_t>(index), kind, staticExtents[index]});
  }
  return success();
}

LogicalResult populateSemanticTaskKind(ComputeOperation &operation) {
  std::optional<StringRef> semanticSection;
  for (Operation *member : operation.members) {
    auto section =
        member->getAttrOfType<StringAttr>("sculptor.semantic.section");
    if (!section)
      continue;
    if (semanticSection && *semanticSection != section.getValue()) {
      return operation.operation->emitError(
          "mapping operation contains conflicting sculptor.semantic.section "
          "values");
    }
    semanticSection = section.getValue();
  }
  if (semanticSection) {
    operation.semanticTaskKind = semanticSection->str();
    return success();
  }

  switch (operation.kind) {
  case ComputeOperationKind::LogicalMVM:
  case ComputeOperationKind::PhysicalMVM:
    operation.semanticTaskKind = "sculptor.mvm";
    break;
  case ComputeOperationKind::MatrixSetup:
    operation.semanticTaskKind = "sculptor.matrix_setup";
    break;
  case ComputeOperationKind::VectorTile:
    operation.semanticTaskKind = "digital.vector_tile";
    break;
  case ComputeOperationKind::TileRecombine:
    operation.semanticTaskKind = "digital.tile_recombine";
    break;
  case ComputeOperationKind::DigitalStage:
    operation.semanticTaskKind = "digital.compute";
    break;
  case ComputeOperationKind::Structured:
    operation.semanticTaskKind =
        operation.operation->getName().getStringRef().str();
    break;
  }
  return success();
}

LogicalResult populateReductionIdentity(ComputeOperation &operation) {
  struct ReductionIdentity {
    int64_t treeId;
    int64_t nodeId;
    int64_t level;
    int64_t ordinal;
    int64_t width;

    bool operator==(const ReductionIdentity &) const = default;
  };
  std::optional<ReductionIdentity> identity;
  for (Operation *member : operation.members) {
    auto treeId = member->getAttrOfType<IntegerAttr>(kReductionTreeIdAttrName);
    auto nodeId = member->getAttrOfType<IntegerAttr>(kReductionNodeIdAttrName);
    auto level = member->getAttrOfType<IntegerAttr>(kReductionLevelAttrName);
    auto ordinal =
        member->getAttrOfType<IntegerAttr>(kReductionOrdinalAttrName);
    auto width = member->getAttrOfType<IntegerAttr>(kReductionWidthAttrName);
    bool hasAny = treeId || nodeId || level || ordinal || width;
    if (!hasAny)
      continue;
    if (!treeId || !nodeId || !level || !ordinal || !width) {
      return member->emitError(
          "reduction mapping operation has incomplete node identity");
    }
    ReductionIdentity candidate{treeId.getInt(), nodeId.getInt(),
                                level.getInt(), ordinal.getInt(),
                                width.getInt()};
    if (candidate.treeId < 0 || candidate.nodeId < 0 || candidate.level < 0 ||
        candidate.ordinal < 0 || candidate.width < 2) {
      return member->emitError("reduction mapping identity is out of range");
    }
    if (identity && *identity != candidate) {
      return member->emitError(
          "one mapping stage contains conflicting reduction identities");
    }
    identity = candidate;
  }
  if (!identity)
    return success();
  operation.reductionTreeId = identity->treeId;
  operation.reductionNodeId = identity->nodeId;
  operation.reductionLevel = identity->level;
  operation.reductionOrdinal = identity->ordinal;
  operation.reductionWidth = identity->width;
  return success();
}

LogicalResult
buildLaneBindingGroups(ComputeGraph &graph,
                       const DenseMap<Operation *, int64_t> &operationIds) {
  DenseMap<Value, int64_t> groupByLogicalArray;

  for (ComputeOperation &operation : graph.operations) {
    if (operation.kind != ComputeOperationKind::MatrixSetup)
      continue;

    sculptor::ArraySetOp arraySet;
    for (Operation *member : operation.members) {
      auto candidate = dyn_cast<sculptor::ArraySetOp>(member);
      if (!candidate)
        continue;
      if (arraySet) {
        member->emitError(
            "matrix-setup mapping stage contains multiple array.set ops");
        return failure();
      }
      arraySet = candidate;
    }
    if (!arraySet) {
      operation.operation->emitError(
          "matrix-setup mapping stage has no array.set op");
      return failure();
    }

    int64_t groupId = static_cast<int64_t>(graph.laneBindingGroups.size());
    if (!groupByLogicalArray.try_emplace(arraySet.getArray(), groupId).second) {
      arraySet.emitOpError(
          "logical array appears in multiple lane-binding groups");
      return failure();
    }
    operation.laneBindingGroup = groupId;
    LaneBindingGroup group;
    group.id = groupId;
    group.setupOperationId = operation.id;
    group.operationIds.push_back(operation.id);
    graph.laneBindingGroups.push_back(std::move(group));
  }

  for (ComputeOperation &operation : graph.operations) {
    if (operation.kind != ComputeOperationKind::PhysicalMVM)
      continue;

    SmallVector<Value> referencedArrays;
    int64_t loadCount = 0;
    int64_t executeCount = 0;
    int64_t storeCount = 0;
    for (Operation *member : operation.members) {
      if (auto load = dyn_cast<sculptor::ArrayLoadOp>(member)) {
        ++loadCount;
        appendUnique(referencedArrays, Value(load.getArray()));
      } else if (auto execute = dyn_cast<sculptor::ArrayExecuteOp>(member)) {
        ++executeCount;
        appendUnique(referencedArrays, Value(execute.getArray()));
      } else if (auto store = dyn_cast<sculptor::ArrayStoreOp>(member)) {
        ++storeCount;
        appendUnique(referencedArrays, Value(store.getArray()));
      }
    }
    if (loadCount == 0 || executeCount == 0 || storeCount == 0) {
      operation.operation->emitError(
          "physical-MVM mapping stage must contain array.load, "
          "array.execute, and array.store operations");
      return failure();
    }
    if (referencedArrays.size() != 1) {
      operation.operation->emitError(
          "physical-MVM mapping stage must use one logical array");
      return failure();
    }

    auto group = groupByLogicalArray.find(referencedArrays.front());
    if (group == groupByLogicalArray.end()) {
      operation.operation->emitError(
          "physical-MVM logical array is not produced by a matrix-setup "
          "mapping stage");
      return failure();
    }
    auto definingSet =
        referencedArrays.front().getDefiningOp<sculptor::ArraySetOp>();
    auto setupOperation = definingSet
                              ? operationIds.find(definingSet.getOperation())
                              : operationIds.end();
    if (!definingSet || setupOperation == operationIds.end() ||
        setupOperation->second !=
            graph.laneBindingGroups[group->second].setupOperationId) {
      operation.operation->emitError(
          "physical-MVM lane binding does not resolve to its array.set "
          "producer");
      return failure();
    }

    operation.laneBindingGroup = group->second;
    graph.laneBindingGroups[group->second].operationIds.push_back(operation.id);
  }

  return success();
}

LogicalResult buildMVMWaves(ComputeGraph &graph) {
  for (ComputeOperation &seed : graph.operations) {
    if (seed.kind != ComputeOperationKind::PhysicalMVM || seed.mvmWaveId)
      continue;

    SmallVector<int64_t> recombineOperationIds;
    for (int64_t tensorId : seed.outputTensors) {
      for (int64_t consumerId : graph.tensors[tensorId].consumerOperations) {
        if (graph.operations[consumerId].kind ==
            ComputeOperationKind::TileRecombine)
          appendUnique(recombineOperationIds, consumerId);
      }
    }
    llvm::sort(recombineOperationIds);
    if (recombineOperationIds.size() > 1) {
      seed.operation->emitError(
          "physical MVM feeds multiple tile-recombine stages; cannot derive "
          "one MVM wave");
      return failure();
    }

    MVMWave wave;
    wave.id = static_cast<int64_t>(graph.mvmWaves.size());
    if (!recombineOperationIds.empty()) {
      wave.recombineOperationId = recombineOperationIds.front();
      const ComputeOperation &recombine =
          graph.operations[*wave.recombineOperationId];
      for (int64_t tensorId : recombine.inputTensors) {
        for (int64_t producerId : graph.tensors[tensorId].producerOperations) {
          if (graph.operations[producerId].kind ==
              ComputeOperationKind::PhysicalMVM)
            appendUnique(wave.physicalMVMOperationIds, producerId);
        }
      }
    } else {
      wave.physicalMVMOperationIds.push_back(seed.id);
    }

    llvm::sort(wave.physicalMVMOperationIds);
    if (wave.physicalMVMOperationIds.empty() ||
        !llvm::is_contained(wave.physicalMVMOperationIds, seed.id)) {
      seed.operation->emitError(
          "tile-recombine stage does not consume its physical MVM result");
      return failure();
    }

    std::set<int64_t> laneBindingGroups;
    for (int64_t physicalMVMId : wave.physicalMVMOperationIds) {
      ComputeOperation &physicalMVM = graph.operations[physicalMVMId];
      if (physicalMVM.mvmWaveId) {
        physicalMVM.operation->emitError(
            "physical MVM belongs to multiple derived MVM waves");
        return failure();
      }
      if (!physicalMVM.laneBindingGroup) {
        physicalMVM.operation->emitError(
            "physical MVM has no analog lane-binding group");
        return failure();
      }
      if (!laneBindingGroups.insert(*physicalMVM.laneBindingGroup).second) {
        physicalMVM.operation->emitError(
            "one MVM wave uses the same analog lane-binding group more than "
            "once");
        return failure();
      }

      for (int64_t tensorId : physicalMVM.inputTensors) {
        for (int64_t producerId : graph.tensors[tensorId].producerOperations) {
          if (graph.operations[producerId].kind ==
              ComputeOperationKind::VectorTile)
            appendUnique(wave.vectorTileOperationIds, producerId);
        }
      }
    }
    llvm::sort(wave.vectorTileOperationIds);

    int64_t waveSize =
        static_cast<int64_t>(wave.physicalMVMOperationIds.size());
    for (auto [memberIndex, physicalMVMId] :
         llvm::enumerate(wave.physicalMVMOperationIds)) {
      ComputeOperation &physicalMVM = graph.operations[physicalMVMId];
      physicalMVM.mvmWaveId = wave.id;
      physicalMVM.mvmWaveMember = static_cast<int64_t>(memberIndex);
      physicalMVM.mvmWaveSize = waveSize;
    }

    for (int64_t vectorTileId : wave.vectorTileOperationIds) {
      ComputeOperation &vectorTile = graph.operations[vectorTileId];
      if (vectorTile.mvmWaveId && *vectorTile.mvmWaveId != wave.id) {
        vectorTile.operation->emitError(
            "vector-tile stage feeds multiple derived MVM waves");
        return failure();
      }
      vectorTile.mvmWaveId = wave.id;
      vectorTile.mvmWaveSize = waveSize;

      SmallVector<int64_t> physicalConsumers;
      for (int64_t tensorId : vectorTile.outputTensors) {
        for (int64_t consumerId : graph.tensors[tensorId].consumerOperations) {
          if (llvm::is_contained(wave.physicalMVMOperationIds, consumerId))
            appendUnique(physicalConsumers, consumerId);
        }
      }
      if (physicalConsumers.size() == 1) {
        vectorTile.mvmWaveMember =
            graph.operations[physicalConsumers.front()].mvmWaveMember;
      } else {
        vectorTile.mvmWaveMember.reset();
      }
    }

    if (wave.recombineOperationId) {
      ComputeOperation &recombine =
          graph.operations[*wave.recombineOperationId];
      if (recombine.mvmWaveId && *recombine.mvmWaveId != wave.id) {
        recombine.operation->emitError(
            "tile-recombine stage belongs to multiple derived MVM waves");
        return failure();
      }
      recombine.mvmWaveId = wave.id;
      recombine.mvmWaveSize = waveSize;
    }

    SmallVector<int64_t> waveOutputOperationIds;
    if (wave.recombineOperationId)
      waveOutputOperationIds.push_back(*wave.recombineOperationId);
    else
      waveOutputOperationIds.append(wave.physicalMVMOperationIds.begin(),
                                    wave.physicalMVMOperationIds.end());

    SmallVector<int64_t> outputConsumers;
    bool outputIsFunctionResult = false;
    for (int64_t outputOperationId : waveOutputOperationIds) {
      for (int64_t tensorId :
           graph.operations[outputOperationId].outputTensors) {
        const ComputeTensor &tensor = graph.tensors[tensorId];
        outputIsFunctionResult |= tensor.isFunctionOutput;
        for (int64_t consumerId : tensor.consumerOperations)
          appendUnique(outputConsumers, consumerId);
      }
    }
    llvm::sort(outputConsumers);
    if (!outputIsFunctionResult && outputConsumers.size() == 1 &&
        isDigitalBiasAdd(graph.operations[outputConsumers.front()])) {
      ComputeOperation &biasAdd = graph.operations[outputConsumers.front()];
      if (biasAdd.mvmWaveId && *biasAdd.mvmWaveId != wave.id) {
        biasAdd.operation->emitError(
            "bias-add stage belongs to multiple derived MVM waves");
        return failure();
      }
      wave.biasAddOperationId = biasAdd.id;
      biasAdd.mvmWaveId = wave.id;
      biasAdd.mvmWaveSize = waveSize;
    }

    graph.mvmWaves.push_back(std::move(wave));
  }

  return success();
}

LogicalResult buildTopologicalOrder(ComputeGraph &graph, Operation *anchor) {
  SmallVector<SmallVector<int64_t>> successors(graph.operations.size());
  SmallVector<int64_t> indegree(graph.operations.size(), 0);

  for (const ComputeTensor &tensor : graph.tensors) {
    for (int64_t producer : tensor.producerOperations) {
      for (int64_t consumer : tensor.consumerOperations) {
        if (producer == consumer ||
            llvm::is_contained(successors[producer], consumer))
          continue;
        successors[producer].push_back(consumer);
        ++indegree[consumer];
      }
    }
  }

  std::set<int64_t> ready;
  for (auto [operationId, degree] : llvm::enumerate(indegree)) {
    if (degree == 0)
      ready.insert(static_cast<int64_t>(operationId));
  }

  while (!ready.empty()) {
    int64_t operationId = *ready.begin();
    ready.erase(ready.begin());
    graph.topologicalOrder.push_back(operationId);
    for (int64_t successor : successors[operationId]) {
      if (--indegree[successor] == 0)
        ready.insert(successor);
    }
  }

  if (graph.topologicalOrder.size() != graph.operations.size()) {
    anchor->emitError(
        "cannot build an RA Tree from a cyclic structured-operation graph");
    return failure();
  }
  return success();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

StringRef stringifyComputeOperationKind(ComputeOperationKind kind) {
  switch (kind) {
  case ComputeOperationKind::Structured:
    return "structured";
  case ComputeOperationKind::LogicalMVM:
    return "analog_mvm";
  case ComputeOperationKind::MatrixSetup:
    return kMatrixSetupStageKind;
  case ComputeOperationKind::DigitalStage:
    return kDigitalStageKind;
  case ComputeOperationKind::VectorTile:
    return kVectorTileStageKind;
  case ComputeOperationKind::PhysicalMVM:
    return kPhysicalMVMStageKind;
  case ComputeOperationKind::TileRecombine:
    return kTileRecombineStageKind;
  }
  llvm_unreachable("unknown compute operation kind");
}

StringRef stringifyComputeIteratorKind(ComputeIteratorKind kind) {
  switch (kind) {
  case ComputeIteratorKind::Parallel:
    return "parallel";
  case ComputeIteratorKind::Reduction:
    return "reduction";
  }
  llvm_unreachable("unknown compute iterator kind");
}

FailureOr<ComputeGraph> buildComputeGraph(func::FuncOp func) {
  ComputeGraph graph;
  graph.functionSymbol = func.getSymName().str();

  DenseMap<Operation *, int64_t> operationIds;
  DenseMap<int64_t, int64_t> stageOperationIds;
  auto appendStageMembers = [&](ComputeOperation &computeOperation,
                                Operation *root, int64_t operationId) {
    root->walk([&](Operation *member) {
      computeOperation.members.push_back(member);
      operationIds[member] = operationId;
    });
  };
  for (Block &block : func.getBody()) {
    for (Operation &operation : block) {
      if (auto stageId =
              operation.getAttrOfType<IntegerAttr>(kStageIdAttrName)) {
        auto stageKind =
            operation.getAttrOfType<StringAttr>(kStageKindAttrName);
        if (!stageKind) {
          operation.emitError("mapping stage is missing its kind");
          return failure();
        }

        FailureOr<ComputeOperationKind> kind =
            classifyStage(&operation, stageKind.getValue());
        if (failed(kind))
          return failure();

        auto existing = stageOperationIds.find(stageId.getInt());
        if (existing != stageOperationIds.end()) {
          ComputeOperation &computeOperation =
              graph.operations[existing->second];
          if (*kind != computeOperation.kind) {
            operation.emitError("one mapping stage ID has inconsistent kinds");
            return failure();
          }
          appendStageMembers(computeOperation, &operation, existing->second);
          continue;
        }

        int64_t operationId = static_cast<int64_t>(graph.operations.size());
        stageOperationIds[stageId.getInt()] = operationId;
        ComputeOperation computeOperation;
        computeOperation.id = operationId;
        computeOperation.operation = &operation;
        appendStageMembers(computeOperation, &operation, operationId);
        computeOperation.kind = *kind;
        if (auto name = operation.getAttrOfType<StringAttr>(kStageNameAttrName))
          computeOperation.stageName = name.getValue().str();
        graph.operations.push_back(std::move(computeOperation));
        continue;
      }

      std::optional<ComputeOperationKind> kind =
          classifyComputeOperation(&operation);
      if (!kind)
        continue;
      int64_t operationId = static_cast<int64_t>(graph.operations.size());
      operationIds[&operation] = operationId;
      ComputeOperation computeOperation;
      computeOperation.id = operationId;
      computeOperation.operation = &operation;
      computeOperation.members.push_back(&operation);
      computeOperation.kind = *kind;
      graph.operations.push_back(std::move(computeOperation));
    }
  }

  for (ComputeOperation &operation : graph.operations) {
    operation.requiredLane = classifyLogicalLaneRequirement(operation.kind);
    if (failed(populateSemanticTaskKind(operation)))
      return failure();
    if (failed(populateReductionIdentity(operation)))
      return failure();
    if (failed(populateIterationDomain(operation)))
      return failure();
  }

  if (failed(buildLaneBindingGroups(graph, operationIds)))
    return failure();

  if (graph.operations.empty())
    return graph;

  DenseMap<Value, int64_t> tensorIds;
  auto getOrCreateTensor = [&](Value value) -> int64_t {
    auto existing = tensorIds.find(value);
    if (existing != tensorIds.end())
      return existing->second;

    ComputeTensor tensor;
    tensor.id = static_cast<int64_t>(graph.tensors.size());
    tensor.value = value;
    tensor.type = value.getType();
    tensor.byteSize = getStaticTensorByteSize(tensor.type);
    tensor.isLogicalArray = isa<sculptor::LogicalArrayType>(tensor.type);
    llvm::SmallPtrSet<Operation *, 16> visited;
    collectUpstreamComputeOperations(func, value, operationIds,
                                     tensor.producerOperations,
                                     tensor.isFunctionInput, visited);
    llvm::sort(tensor.producerOperations);
    graph.tensors.push_back(std::move(tensor));
    tensorIds[value] = graph.tensors.back().id;
    return graph.tensors.back().id;
  };

  for (ComputeOperation &computeOperation : graph.operations) {
    for (Operation *member : computeOperation.members) {
      for (Value operand : member->getOperands()) {
        if (!isComputeResourceType(operand.getType()))
          continue;
        Operation *definingOperation = operand.getDefiningOp();
        auto producer = operationIds.find(definingOperation);
        if (producer != operationIds.end() &&
            producer->second == computeOperation.id)
          continue;
        int64_t tensorId = getOrCreateTensor(operand);
        appendUnique(computeOperation.inputTensors, tensorId);
        appendUnique(graph.tensors[tensorId].consumerOperations,
                     computeOperation.id);
      }
    }
  }

  for (Block &block : func.getBody()) {
    auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator());
    if (!returnOp)
      continue;
    for (Value operand : returnOp.getOperands()) {
      if (!isComputeResourceType(operand.getType()))
        continue;
      graph.tensors[getOrCreateTensor(operand)].isFunctionOutput = true;
    }
  }

  for (ComputeTensor &tensor : graph.tensors) {
    for (int64_t producer : tensor.producerOperations)
      appendUnique(graph.operations[producer].outputTensors, tensor.id);
  }

  if (failed(buildMVMWaves(graph)))
    return failure();

  if (failed(buildTopologicalOrder(graph, func)))
    return failure();
  return graph;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
