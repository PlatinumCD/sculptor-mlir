#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/Support/IR/SemanticLayerIdentity.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ReductionTree.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <limits>
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

bool valueCarriesRequiredData(
    func::FuncOp func, Value value,
    const DenseMap<Operation *, int64_t> &operationIds,
    llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (isFunctionArgument(func, value))
    return true;

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return true;
  if (operationIds.contains(definingOp))
    return true;
  if (isa<tensor::EmptyOp>(definingOp))
    return false;
  if (definingOp->hasTrait<OpTrait::ConstantLike>())
    return true;
  if (!visited.insert(definingOp).second)
    return true;

  bool hasResourceOperand = false;
  for (Value operand : definingOp->getOperands()) {
    if (!isComputeResourceType(operand.getType()))
      continue;
    hasResourceOperand = true;
    if (valueCarriesRequiredData(func, operand, operationIds, visited))
      return true;
  }
  // Unknown tensor-producing sources are conservatively data-bearing. A
  // support chain is destination-only only when every shaped source traces to
  // tensor.empty.
  return !hasResourceOperand;
}

bool operandCarriesRequiredData(
    func::FuncOp func, OpOperand &operand,
    const DenseMap<Operation *, int64_t> &operationIds) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(operand.getOwner())) {
    if (!linalgOp.payloadUsesValueFromOperand(&operand) &&
        !linalgOp.isDpsInit(&operand))
      return false;
  }
  // A non-empty DPS destination still carries a conservative storage/order
  // dependency even when the payload overwrites every element. Empty-backed
  // destinations fall through to valueCarriesRequiredData and are excluded.
  llvm::SmallPtrSet<Operation *, 16> visited;
  return valueCarriesRequiredData(func, operand.get(), operationIds, visited);
}

void collectUpstreamComputeOperations(
    func::FuncOp func, Value value,
    const DenseMap<Operation *, int64_t> &operationIds,
    SmallVectorImpl<int64_t> &producerOperations, bool &isFunctionInput,
    DenseMap<int64_t, SmallVector<Value>> &producerValues,
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
    appendUnique(producerValues[operationId->second], value);
    return;
  }

  if (!visited.insert(definingOp).second)
    return;

  for (Value operand : definingOp->getOperands()) {
    if (!isComputeResourceType(operand.getType()))
      continue;
    collectUpstreamComputeOperations(func, operand, operationIds,
                                     producerOperations, isFunctionInput,
                                     producerValues, visited);
  }
}

std::optional<ComputeOperationKind>
classifyComputeOperation(Operation *operation) {
  if (isa<sculptor::MVMOp>(operation))
    return ComputeOperationKind::LogicalMVM;
  // Do not classify every operation that happens to implement
  // TilingInterface as independently scheduled work. Interface models can be
  // attached to shape/view-like operations over time, and those operations
  // belong in the closure of their consuming routine. Linalg operations and
  // tensor.pad both materialize payload data and therefore own mapping work.
  if (isa<linalg::LinalgOp, tensor::PadOp>(operation))
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
  } else if (auto padOp = dyn_cast<tensor::PadOp>(computeOperation.operation)) {
    // PadOp's external TilingInterface reifies its iteration domain from the
    // result shape. Avoid manufacturing IR solely to recover information that
    // is already present in a static result type.
    RankedTensorType resultType = padOp.getResultType();
    if (resultType.hasStaticShape() &&
        static_cast<size_t>(resultType.getRank()) == iteratorTypes.size())
      staticExtents.assign(resultType.getShape().begin(),
                           resultType.getShape().end());
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

LogicalResult populateSemanticLayerIdentity(ComputeOperation &operation) {
  std::optional<int64_t> layerId;
  std::optional<StringRef> layerKind;
  SmallVector<Operation *> untaggedMembers;

  for (Operation *member : operation.members) {
    auto id = member->getAttrOfType<IntegerAttr>(
        mlir::sculptor::kSemanticLayerIdAttrName);
    auto kind = member->getAttrOfType<StringAttr>(
        mlir::sculptor::kSemanticLayerKindAttrName);
    if (!id && !kind) {
      untaggedMembers.push_back(member);
      continue;
    }
    if (!id || !kind) {
      return member->emitError(
          "semantic layer identity requires both layer_id and layer_kind");
    }
    if (id.getInt() < 0)
      return member->emitError("semantic layer ID must be non-negative");
    if (kind.getValue().empty())
      return member->emitError("semantic layer kind must not be empty");
    if (layerId && *layerId != id.getInt()) {
      return member->emitError(
          "one mapping operation contains conflicting semantic layer IDs");
    }
    if (layerKind && *layerKind != kind.getValue()) {
      return member->emitError(
          "one mapping operation contains conflicting semantic layer kinds");
    }
    layerId = id.getInt();
    layerKind = kind.getValue();
  }

  if (layerId) {
    operation.semanticLayerId = *layerId;
    operation.semanticLayerKind = layerKind->str();
    // Region-producing rewrites may synthesize a support operation after the
    // containing stage was annotated (for example affine.apply while slicing
    // a linalg body). The tagged stage is authoritative, so complete that
    // identity here before outlining can detach the support operation.
    for (Operation *member : untaggedMembers) {
      member->setAttr(mlir::sculptor::kSemanticLayerIdAttrName,
                      IntegerAttr::get(IntegerType::get(member->getContext(),
                                                        /*width=*/64),
                                       *layerId));
      member->setAttr(mlir::sculptor::kSemanticLayerKindAttrName,
                      StringAttr::get(member->getContext(), *layerKind));
    }
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
      if (!physicalConsumers.empty()) {
        // A vector tile may feed several physical arrays (for example one
        // input-column tile broadcast across multiple output-row tiles). Give
        // it a deterministic execution owner instead of leaving it unowned,
        // which would pin every shared vector tile in the wave to the same
        // home core and destroy otherwise valid nested fork/join parallelism.
        int64_t owner = *std::min_element(
            physicalConsumers.begin(), physicalConsumers.end(),
            [&](int64_t left, int64_t right) {
              return std::pair(
                         *graph.operations[left].mvmWaveMember, left) <
                     std::pair(
                         *graph.operations[right].mvmWaveMember, right);
            });
        vectorTile.mvmWaveMember = graph.operations[owner].mvmWaveMember;
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

// Keep cheap post-layer tensor work (ReLU, clamp, pointwise affine transforms,
// and layout-preserving copies) with the one semantic layer that produces it.
// Reductions and joins remain independent: absorbing either would hide a real
// scheduling boundary or arbitrarily choose among multiple parent layers.
bool isProducerAffineLayerEpilogue(const ComputeOperation &operation) {
  return operation.kind == ComputeOperationKind::Structured &&
         !operation.iterationDomain.empty() &&
         llvm::all_of(operation.iterationDomain,
                      [](const ComputeIterationDimension &dimension) {
                        return dimension.kind == ComputeIteratorKind::Parallel;
                      });
}

std::optional<int64_t>
findUniqueSemanticProducerRegion(const ComputeGraph &graph,
                                 const ComputeOperation &operation) {
  std::optional<int64_t> candidateRegion;
  for (int64_t tensorId : operation.inputTensors) {
    for (int64_t producerId : graph.tensors[tensorId].producerOperations) {
      int64_t producerRegion = graph.operations[producerId].layerRegionId;
      if (producerRegion < 0)
        continue;
      const LayerRegion &region = graph.layerRegions[producerRegion];
      if (!region.semanticLayerId)
        return std::nullopt;
      if (candidateRegion && *candidateRegion != producerRegion)
        return std::nullopt;
      candidateRegion = producerRegion;
    }
  }
  return candidateRegion;
}

void inheritSemanticRegion(ComputeOperation &operation, LayerRegion &region) {
  operation.semanticLayerId = region.semanticLayerId;
  operation.semanticLayerKind = region.semanticLayerKind;
  for (Operation *member : operation.members) {
    member->setAttr(
        mlir::sculptor::kSemanticLayerIdAttrName,
        IntegerAttr::get(IntegerType::get(member->getContext(), /*width=*/64),
                         *region.semanticLayerId));
    member->setAttr(mlir::sculptor::kSemanticLayerKindAttrName,
                    StringAttr::get(member->getContext(),
                                    region.semanticLayerKind));
  }
}

std::optional<int64_t> findDestinationInitializerConsumer(
    const ComputeOperation &operation,
    const DenseMap<Operation *, int64_t> &operationIds) {
  auto fill = dyn_cast<linalg::FillOp>(operation.operation);
  if (!fill || fill->getNumResults() != 1)
    return std::nullopt;

  Value result = fill->getResult(0);
  if (!result.hasOneUse())
    return std::nullopt;

  OpOperand &use = *result.getUses().begin();
  auto consumer = dyn_cast<linalg::LinalgOp>(use.getOwner());
  if (!consumer || isa<linalg::FillOp>(consumer.getOperation()) ||
      !consumer.isDpsInit(&use))
    return std::nullopt;

  auto found = operationIds.find(consumer.getOperation());
  if (found == operationIds.end() || found->second == operation.id)
    return std::nullopt;
  return found->second;
}

LogicalResult buildLayerRegions(ComputeGraph &graph, Operation *anchor) {
  DenseMap<int64_t, int64_t> regionBySemanticLayerId;
  DenseMap<Operation *, int64_t> operationIds;
  for (const ComputeOperation &operation : graph.operations) {
    for (Operation *member : operation.members)
      operationIds.try_emplace(member, operation.id);
  }

  // A one-use linalg.fill that feeds a consumer's DPS initialization is the
  // consumer's prologue, not an independently schedulable layer. Defer these
  // fills until their consumers have received a layer region.
  DenseMap<int64_t, int64_t> initializerConsumers;
  for (const ComputeOperation &operation : graph.operations) {
    if (std::optional<int64_t> consumer =
            findDestinationInitializerConsumer(operation, operationIds))
      initializerConsumers.try_emplace(operation.id, *consumer);
  }

  for (int64_t operationId : graph.topologicalOrder) {
    if (initializerConsumers.contains(operationId))
      continue;

    ComputeOperation &operation = graph.operations[operationId];
    int64_t regionId = -1;
    if (operation.semanticLayerId) {
      auto [entry, inserted] = regionBySemanticLayerId.try_emplace(
          *operation.semanticLayerId,
          static_cast<int64_t>(graph.layerRegions.size()));
      regionId = entry->second;
      if (inserted) {
        LayerRegion region;
        region.id = regionId;
        region.semanticLayerId = operation.semanticLayerId;
        region.semanticLayerKind = operation.semanticLayerKind;
        graph.layerRegions.push_back(std::move(region));
      } else if (graph.layerRegions[regionId].semanticLayerKind !=
                 operation.semanticLayerKind) {
        operation.operation->emitError(
            "one semantic layer ID is associated with multiple layer kinds");
        return failure();
      }
    } else if (isProducerAffineLayerEpilogue(operation)) {
      std::optional<int64_t> producerRegion =
          findUniqueSemanticProducerRegion(graph, operation);
      if (producerRegion) {
        regionId = *producerRegion;
        inheritSemanticRegion(operation, graph.layerRegions[regionId]);
      }
    }

    if (regionId < 0) {
      regionId = static_cast<int64_t>(graph.layerRegions.size());
      LayerRegion region;
      region.id = regionId;
      region.isSingletonFallback = true;
      graph.layerRegions.push_back(std::move(region));
    }

    operation.layerRegionId = regionId;
    graph.layerRegions[regionId].operationIds.push_back(operationId);
  }

  for (int64_t operationId : graph.topologicalOrder) {
    auto found = initializerConsumers.find(operationId);
    if (found == initializerConsumers.end())
      continue;

    ComputeOperation &operation = graph.operations[operationId];
    int64_t consumerId = found->second;
    int64_t regionId = graph.operations[consumerId].layerRegionId;
    if (regionId < 0 ||
        static_cast<size_t>(regionId) >= graph.layerRegions.size()) {
      operation.operation->emitError(
          "destination initializer consumer has no layer region");
      return failure();
    }

    operation.layerRegionId = regionId;
    LayerRegion &region = graph.layerRegions[regionId];
    region.isSingletonFallback = false;
    auto consumerPosition = llvm::find(region.operationIds, consumerId);
    if (consumerPosition == region.operationIds.end()) {
      operation.operation->emitError(
          "destination initializer consumer is missing from its layer region");
      return failure();
    }
    region.operationIds.insert(consumerPosition, operationId);
  }

  for (const ComputeTensor &tensor : graph.tensors) {
    SmallVector<int64_t> producerRegions;
    SmallVector<int64_t> consumerRegions;
    for (int64_t operationId : tensor.producerOperations)
      appendUnique(producerRegions,
                   graph.operations[operationId].layerRegionId);
    for (int64_t operationId : tensor.consumerOperations)
      appendUnique(consumerRegions,
                   graph.operations[operationId].layerRegionId);

    for (int64_t regionId : consumerRegions) {
      bool hasProducerInRegion = llvm::is_contained(producerRegions, regionId);
      bool hasProducerOutsideRegion = llvm::any_of(
          producerRegions, [=](int64_t producerRegion) {
            return producerRegion != regionId;
          });
      if (tensor.isFunctionInput || producerRegions.empty() ||
          hasProducerOutsideRegion)
        appendUnique(graph.layerRegions[regionId].inputTensors, tensor.id);
      if (hasProducerInRegion)
        appendUnique(graph.layerRegions[regionId].internalTensors, tensor.id);
    }

    for (int64_t regionId : producerRegions) {
      bool hasConsumerInRegion = llvm::is_contained(consumerRegions, regionId);
      bool hasConsumerOutsideRegion = llvm::any_of(
          consumerRegions, [=](int64_t consumerRegion) {
            return consumerRegion != regionId;
          });
      if (tensor.isFunctionOutput || consumerRegions.empty() ||
          hasConsumerOutsideRegion)
        appendUnique(graph.layerRegions[regionId].outputTensors, tensor.id);
      if (hasConsumerInRegion)
        appendUnique(graph.layerRegions[regionId].internalTensors, tensor.id);
    }
  }

  SmallVector<int64_t> membershipCount(graph.operations.size(), 0);
  for (const LayerRegion &region : graph.layerRegions) {
    if (region.id < 0 || static_cast<size_t>(region.id) >=
                             graph.layerRegions.size()) {
      anchor->emitError("layer region has an invalid dense ID");
      return failure();
    }
    if (region.operationIds.empty()) {
      anchor->emitError("layer region contains no compute operations");
      return failure();
    }
    if (region.isSingletonFallback && region.operationIds.size() != 1) {
      anchor->emitError("fallback layer region is not a singleton");
      return failure();
    }
    for (int64_t operationId : region.operationIds) {
      if (operationId < 0 || static_cast<size_t>(operationId) >=
                                 graph.operations.size() ||
          graph.operations[operationId].layerRegionId != region.id) {
        anchor->emitError("layer region operation membership is invalid");
        return failure();
      }
      ++membershipCount[operationId];
    }
  }
  if (llvm::any_of(membershipCount,
                   [](int64_t count) { return count != 1; })) {
    anchor->emitError(
        "every compute operation must belong to exactly one layer region");
    return failure();
  }

  SmallVector<SmallVector<int64_t>> regionSuccessors(
      graph.layerRegions.size());
  SmallVector<int64_t> regionIndegree(graph.layerRegions.size(), 0);
  for (const ComputeTensor &tensor : graph.tensors) {
    for (int64_t producerId : tensor.producerOperations) {
      int64_t producerRegion = graph.operations[producerId].layerRegionId;
      for (int64_t consumerId : tensor.consumerOperations) {
        int64_t consumerRegion = graph.operations[consumerId].layerRegionId;
        if (producerRegion == consumerRegion ||
            llvm::is_contained(regionSuccessors[producerRegion],
                               consumerRegion))
          continue;
        regionSuccessors[producerRegion].push_back(consumerRegion);
        ++regionIndegree[consumerRegion];
      }
    }
  }
  std::set<int64_t> readyRegions;
  for (auto [regionId, indegree] : llvm::enumerate(regionIndegree)) {
    if (indegree == 0)
      readyRegions.insert(static_cast<int64_t>(regionId));
  }
  while (!readyRegions.empty()) {
    int64_t regionId = *readyRegions.begin();
    readyRegions.erase(readyRegions.begin());
    graph.topologicalLayerRegionOrder.push_back(regionId);
    for (int64_t successor : regionSuccessors[regionId]) {
      if (--regionIndegree[successor] == 0)
        readyRegions.insert(successor);
    }
  }
  if (graph.topologicalLayerRegionOrder.size() != graph.layerRegions.size()) {
    anchor->emitError(
        "semantic layer regions form a cyclic parent-level dependency graph");
    return failure();
  }
  return success();
}

int64_t addStaticEstimate(int64_t total, int64_t value) {
  if (total < 0 || value < 0 ||
      total > std::numeric_limits<int64_t>::max() - value)
    return -1;
  return total + value;
}

int64_t estimateOperationWorkItems(const ComputeOperation &operation) {
  int64_t workItems = 1;
  for (const ComputeIterationDimension &dimension : operation.iterationDomain) {
    if (ShapedType::isDynamic(dimension.staticExtent) ||
        dimension.staticExtent <= 0 ||
        workItems >
            std::numeric_limits<int64_t>::max() / dimension.staticExtent)
      return -1;
    workItems *= dimension.staticExtent;
  }
  return workItems;
}

void estimateLayerRegionResources(ComputeGraph &graph) {
  for (LayerRegion &region : graph.layerRegions) {
    std::set<int64_t> analogBindingGroups;
    for (int64_t operationId : region.operationIds) {
      const ComputeOperation &operation = graph.operations[operationId];
      if (operation.requiredLane == LogicalLaneKind::Analog &&
          operation.laneBindingGroup)
        analogBindingGroups.insert(*operation.laneBindingGroup);
      if (operation.requiredLane == LogicalLaneKind::Digital) {
        region.estimatedDigitalWorkItems = addStaticEstimate(
            region.estimatedDigitalWorkItems,
            estimateOperationWorkItems(operation));
      }
    }
    region.analogLaneDemand =
        static_cast<int64_t>(analogBindingGroups.size());

    std::set<int64_t> residentTensorIds;
    residentTensorIds.insert(region.inputTensors.begin(),
                             region.inputTensors.end());
    residentTensorIds.insert(region.outputTensors.begin(),
                             region.outputTensors.end());
    residentTensorIds.insert(region.internalTensors.begin(),
                             region.internalTensors.end());
    for (int64_t tensorId : residentTensorIds) {
      region.estimatedStaticMemoryBytes = addStaticEstimate(
          region.estimatedStaticMemoryBytes, graph.tensors[tensorId].byteSize);
    }
    for (int64_t tensorId : region.inputTensors) {
      region.estimatedInputBytes = addStaticEstimate(
          region.estimatedInputBytes, graph.tensors[tensorId].byteSize);
    }
    for (int64_t tensorId : region.outputTensors) {
      region.estimatedOutputBytes = addStaticEstimate(
          region.estimatedOutputBytes, graph.tensors[tensorId].byteSize);
    }
  }
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
    computeOperation.members.push_back(root);
    operationIds[root] = operationId;
    root->walk([&](Operation *member) {
      if (member == root)
        return;
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
    if (failed(populateSemanticLayerIdentity(operation)))
      return failure();
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
    DenseMap<int64_t, SmallVector<Value>> producerValues;
    llvm::SmallPtrSet<Operation *, 16> visited;
    collectUpstreamComputeOperations(func, value, operationIds,
                                     tensor.producerOperations,
                                     tensor.isFunctionInput, producerValues,
                                     visited);
    llvm::sort(tensor.producerOperations);
    tensor.producerByteSizes.reserve(tensor.producerOperations.size());
    for (int64_t producerId : tensor.producerOperations) {
      int64_t contributionBytes = 0;
      for (Value boundary : producerValues.lookup(producerId)) {
        int64_t boundaryBytes = getStaticTensorByteSize(boundary.getType());
        if (boundaryBytes < 0 ||
            contributionBytes >
                std::numeric_limits<int64_t>::max() - boundaryBytes) {
          contributionBytes = -1;
          break;
        }
        contributionBytes += boundaryBytes;
      }
      tensor.producerByteSizes.push_back(contributionBytes);
    }
    graph.tensors.push_back(std::move(tensor));
    tensorIds[value] = graph.tensors.back().id;
    return graph.tensors.back().id;
  };

  for (ComputeOperation &computeOperation : graph.operations) {
    for (Operation *member : computeOperation.members) {
      for (OpOperand &operand : member->getOpOperands()) {
        if (!isComputeResourceType(operand.get().getType()))
          continue;
        Operation *definingOperation = operand.get().getDefiningOp();
        auto producer = operationIds.find(definingOperation);
        if (producer != operationIds.end() &&
            producer->second == computeOperation.id)
          continue;
        if (!operandCarriesRequiredData(func, operand, operationIds))
          continue;
        int64_t tensorId = getOrCreateTensor(operand.get());
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
  if (failed(buildLayerRegions(graph, func)))
    return failure();
  estimateLayerRegionResources(graph);
  return graph;
}

int64_t getProducerContributionByteSize(const ComputeTensor &tensor,
                                        int64_t producerOperationId) {
  auto producer = llvm::find(tensor.producerOperations, producerOperationId);
  if (producer == tensor.producerOperations.end())
    return -1;
  size_t index = std::distance(tensor.producerOperations.begin(), producer);
  if (index >= tensor.producerByteSizes.size())
    return -1;
  return tensor.producerByteSizes[index];
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
