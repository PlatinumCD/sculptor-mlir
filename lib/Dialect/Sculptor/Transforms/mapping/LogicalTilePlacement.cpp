#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingConfig.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingCostProfile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

FailureOr<int64_t> getPhysicalTileCapacity(const PhysicalMeshGeometry &mesh,
                                           Operation *anchor) {
  if (mesh.rows <= 0 || mesh.columns <= 0 || mesh.arraysPerCore <= 0) {
    anchor->emitError("physical logical-tile placement requires positive mesh "
                      "dimensions and arrays per core");
    return failure();
  }
  std::optional<int64_t> capacity = llvm::checkedMul(mesh.rows, mesh.columns);
  if (!capacity) {
    anchor->emitError("physical mesh capacity overflow");
    return failure();
  }
  return *capacity;
}

PhysicalTileLocation getLocation(int64_t physicalTileId,
                                 const PhysicalMeshGeometry &mesh) {
  return {physicalTileId, physicalTileId / mesh.columns,
          physicalTileId % mesh.columns};
}

int64_t getManhattanDistance(const PhysicalTileLocation &source,
                             const PhysicalTileLocation &target) {
  return std::abs(source.row - target.row) +
         std::abs(source.column - target.column);
}

FailureOr<int64_t> checkedTransferCost(int64_t byteSize, int64_t hops,
                                       Operation *anchor) {
  if (byteSize < 0 || hops < 0) {
    anchor->emitError("logical-tile transfer cost requires nonnegative bytes "
                      "and Manhattan hops");
    return failure();
  }
  std::optional<int64_t> result = llvm::checkedMul(byteSize, hops);
  if (!result) {
    anchor->emitError("logical-tile transfer cost overflow");
    return failure();
  }
  return *result;
}

FailureOr<int64_t> checkedAddCost(int64_t current, int64_t increment,
                                  Operation *anchor) {
  std::optional<int64_t> result = llvm::checkedAdd(current, increment);
  if (!result) {
    anchor->emitError("logical-tile placement score overflow");
    return failure();
  }
  return *result;
}

FailureOr<std::optional<int64_t>>
getStaticByteSize(Type type, std::optional<ArrayRef<int64_t>> shapeOverride,
                  Operation *anchor) {
  if (isa<sculptor::LogicalArrayType>(type))
    return std::optional<int64_t>{0};
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType || !shapedType.getElementType().isIntOrFloat())
    return std::optional<int64_t>{};
  unsigned bitWidth = shapedType.getElementType().getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return std::optional<int64_t>{};

  ArrayRef<int64_t> shape =
      shapeOverride ? *shapeOverride : shapedType.getShape();
  int64_t elements = 1;
  for (int64_t dimension : shape) {
    if (ShapedType::isDynamic(dimension) || dimension < 0)
      return std::optional<int64_t>{};
    std::optional<int64_t> updated = llvm::checkedMul(elements, dimension);
    if (!updated) {
      anchor->emitError("logical-tile memory element-count overflow");
      return failure();
    }
    elements = *updated;
  }
  std::optional<int64_t> bytes =
      llvm::checkedMul(elements, static_cast<int64_t>(bitWidth / 8));
  if (!bytes) {
    anchor->emitError("logical-tile memory byte-size overflow");
    return failure();
  }
  return std::optional<int64_t>{*bytes};
}

LogicalResult addEstimatedBytes(int64_t &target, std::optional<int64_t> bytes,
                                bool &complete, Operation *anchor) {
  if (!bytes) {
    complete = false;
    return success();
  }
  std::optional<int64_t> updated = llvm::checkedAdd(target, *bytes);
  if (!updated)
    return anchor->emitError("logical-tile memory estimate overflow");
  target = *updated;
  return success();
}

enum class EstimateTransientKind { Produced, Incoming };

struct EstimateLiveRange {
  int64_t bytes = 0;
  double beginNs = 0.0;
  double endNs = 0.0;
  EstimateTransientKind kind = EstimateTransientKind::Produced;
  int64_t sourceTileId = -1;
  int64_t sourceOperationId = -1;
  int64_t sourceWorkUnitId = -1;
  int64_t targetOperationId = -1;
  int64_t targetWorkUnitId = -1;
  int64_t tensorId = -1;
};

struct EstimateEventBucket {
  int64_t producedStarts = 0;
  int64_t producedEnds = 0;
  int64_t producedInstant = 0;
  int64_t incomingStarts = 0;
  int64_t incomingEnds = 0;
  int64_t incomingInstant = 0;
};

struct EndpointAssignmentLocation {
  int64_t tileId = -1;
  const LogicalTileAssignment *assignment = nullptr;
};

using EstimateEndpoint = std::pair<int64_t, int64_t>;

LogicalResult checkedAccumulateEstimate(int64_t &target, int64_t bytes,
                                        Operation *anchor) {
  if (bytes < 0)
    return anchor->emitError("logical-tile memory estimate has negative bytes");
  std::optional<int64_t> updated = llvm::checkedAdd(target, bytes);
  if (!updated)
    return anchor->emitError("logical-tile memory estimate overflow");
  target = *updated;
  return success();
}

LogicalResult checkedReleaseEstimate(int64_t &target, int64_t bytes,
                                     Operation *anchor) {
  if (bytes < 0 || bytes > target)
    return anchor->emitError(
        "logical-tile memory lifetime event is unbalanced");
  target -= bytes;
  return success();
}

LogicalResult
buildLogicalTileMemoryEstimates(const ComputeGraph &graph,
                                const ResourceAllocationTree &tree,
                                LogicalTilePlacementProblem &problem) {
  DenseMap<int64_t, const MappingWorkUnit *> workUnitsById;
  std::map<EstimateEndpoint, SmallVector<int64_t>>
      sourceTensorIdsByWorkUnitEndpoint;
  for (const MappingWorkUnit &workUnit : tree.workUnits) {
    if (!workUnitsById.try_emplace(workUnit.id, &workUnit).second)
      return problem.anchor->emitError(
          "duplicate work-unit ID while estimating tile memory");
  }
  for (const MappingWorkUnitEdge &edge : tree.workUnitEdges) {
    SmallVector<int64_t> &tensorIds = sourceTensorIdsByWorkUnitEndpoint[
        {edge.sourceOperationId, edge.sourceWorkUnitId}];
    if (!llvm::is_contained(tensorIds, edge.tensorId))
      tensorIds.push_back(edge.tensorId);
  }

  llvm::DenseSet<Value> modelOutputValues;
  llvm::DenseSet<Value> internallyConsumedModelOutputValues;
  for (const ComputeTensor &tensor : graph.tensors) {
    if (!tensor.isFunctionOutput)
      continue;
    modelOutputValues.insert(tensor.value);
    for (Operation *user : tensor.value.getUsers()) {
      if (!isa<func::ReturnOp>(user)) {
        internallyConsumedModelOutputValues.insert(tensor.value);
        break;
      }
    }
  }
  auto isOutputOnlyValue = [&](Value value) {
    return modelOutputValues.contains(value) &&
           !internallyConsumedModelOutputValues.contains(value);
  };

  // Endpoint schedules are global even though memory is estimated one logical
  // tile at a time. Cross-tile payload lifetimes therefore use the source and
  // target assignment times without requiring a physical placement.
  std::map<EstimateEndpoint, SmallVector<EndpointAssignmentLocation>>
      endpointAssignments;
  std::map<int64_t, SmallVector<EndpointAssignmentLocation>>
      workUnitAssignmentsByOperation;
  auto indexAssignment = [&](int64_t tileId,
                             const LogicalTileAssignment &assignment) {
    EndpointAssignmentLocation location{tileId, &assignment};
    endpointAssignments[{assignment.operationId, assignment.workUnitId}]
        .push_back(location);
    if (assignment.workUnitId >= 0)
      workUnitAssignmentsByOperation[assignment.operationId].push_back(
          location);
  };
  for (const LogicalTile &tile : problem.tileGraph.tiles) {
    for (const LogicalTileAssignment &assignment : tile.digitalAssignments)
      indexAssignment(tile.id, assignment);
    for (const LogicalTileAnalogLane &lane : tile.analogLanes)
      for (const LogicalTileAssignment &assignment : lane.assignments)
        indexAssignment(tile.id, assignment);
  }
  auto findAssignments = [&](int64_t operationId, int64_t workUnitId,
                             int64_t tileId) {
    SmallVector<const LogicalTileAssignment *> result;
    auto found = endpointAssignments.find({operationId, workUnitId});
    if (found != endpointAssignments.end()) {
      for (const EndpointAssignmentLocation &location : found->second)
        if (location.tileId == tileId)
          result.push_back(location.assignment);
    }
    // An operation-wide dependency can target an operation that was expanded
    // into work-unit endpoints after the dependency was discovered.
    if (!result.empty() || workUnitId >= 0)
      return result;
    auto expanded = workUnitAssignmentsByOperation.find(operationId);
    if (expanded != workUnitAssignmentsByOperation.end())
      for (const EndpointAssignmentLocation &location : expanded->second)
        if (location.tileId == tileId)
          result.push_back(location.assignment);
    return result;
  };

  problem.memoryEstimates.clear();
  problem.memoryEstimateIndexByTileId.clear();
  problem.memoryEstimates.reserve(problem.tileGraph.tiles.size());
  bool emittedCapacityDiagnostic = false;
  for (const LogicalTile &tile : problem.tileGraph.tiles) {
    LogicalTileMemoryEstimate estimate;
    estimate.logicalTileId = tile.id;
    estimate.complete = true;
    std::set<int64_t> persistentTensorIds;
    std::set<std::pair<int64_t, int64_t>> endpoints;
    llvm::DenseSet<Value> persistentValues;
    llvm::DenseSet<Value> producedValues;
    SmallVector<EstimateLiveRange> liveRanges;
    std::map<std::tuple<int64_t, int64_t, int64_t>, SmallVector<unsigned>>
        producedByEndpointAndTensor;
    std::map<std::pair<int64_t, int64_t>, SmallVector<unsigned>>
        producedByOperationAndTensor;
    auto markIncomplete = [&](StringRef reason) {
      estimate.complete = false;
      if (estimate.incompleteReason.empty())
        estimate.incompleteReason = reason.str();
    };

    auto validAssignmentTimes = [&](const LogicalTileAssignment &assignment) {
      return std::isfinite(assignment.startNs) &&
             std::isfinite(assignment.finishNs) &&
             assignment.startNs >= 0.0 &&
             assignment.finishNs >= assignment.startNs;
    };

    auto addInternalProducedRange =
        [&](const LogicalTileAssignment &assignment,
            std::optional<int64_t> bytes) -> LogicalResult {
      if (!bytes) {
        markIncomplete("a produced resource has unknown static size");
        return success();
      }
      if (*bytes == 0)
        return success();
      if (*bytes < 0)
        return problem.anchor->emitError(
            "logical-tile produced resource has negative bytes");
      if (!validAssignmentTimes(assignment)) {
        markIncomplete("an assignment has invalid schedule times");
        return success();
      }
      liveRanges.push_back(
          {*bytes, assignment.startNs, assignment.finishNs,
           EstimateTransientKind::Produced, tile.id, assignment.operationId,
           assignment.workUnitId});
      return success();
    };

    auto addBoundaryProducedRange =
        [&](const LogicalTileAssignment &assignment,
            std::optional<int64_t> bytes, ArrayRef<int64_t> tensorIds,
            bool includeAssignmentExecution) -> LogicalResult {
      if (!bytes) {
        markIncomplete("a produced boundary has unknown static size");
        return success();
      }
      if (*bytes == 0)
        return success();
      if (*bytes < 0)
        return problem.anchor->emitError(
            "logical-tile produced boundary has negative bytes");
      if (!validAssignmentTimes(assignment)) {
        markIncomplete("an assignment has invalid schedule times");
        return success();
      }
      unsigned index = liveRanges.size();
      double beginNs = includeAssignmentExecution ? assignment.startNs
                                                  : assignment.finishNs;
      liveRanges.push_back(
          {*bytes, beginNs, assignment.finishNs,
           EstimateTransientKind::Produced, tile.id, assignment.operationId,
           assignment.workUnitId, -1, -1,
           tensorIds.empty() ? -1 : tensorIds.front()});
      for (int64_t tensorId : tensorIds) {
        producedByEndpointAndTensor[
            {assignment.operationId, assignment.workUnitId, tensorId}]
            .push_back(index);
        producedByOperationAndTensor[{assignment.operationId, tensorId}]
            .push_back(index);
      }
      return success();
    };

    auto accountAssignment =
        [&](const LogicalTileAssignment &assignment) -> LogicalResult {
      if (assignment.operationId < 0 ||
          assignment.operationId >=
              static_cast<int64_t>(graph.operations.size()))
        return problem.anchor->emitError(
            "logical tile references an unknown operation during memory "
            "estimation");
      if (!endpoints.insert({assignment.operationId, assignment.workUnitId})
               .second)
        return success();
      const ComputeOperation &operation =
          graph.operations[assignment.operationId];

      if (operation.kind == ComputeOperationKind::MatrixSetup) {
        for (int64_t tensorId : operation.inputTensors) {
          if (tensorId < 0 ||
              tensorId >= static_cast<int64_t>(graph.tensors.size()))
            return problem.anchor->emitError(
                "matrix setup references an unknown tensor during memory "
                "estimation");
          const ComputeTensor &tensor = graph.tensors[tensorId];
          if (tensor.isLogicalArray || tensor.isFunctionInput ||
              !persistentTensorIds.insert(tensorId).second ||
              !persistentValues.insert(tensor.value).second)
            continue;
          FailureOr<std::optional<int64_t>> bytes =
              getStaticByteSize(tensor.type, std::nullopt, problem.anchor);
          if (failed(bytes) ||
              failed(addEstimatedBytes(estimate.persistentBytes, *bytes,
                                       estimate.complete, problem.anchor)))
            return failure();
        }
        for (Operation *member : operation.members) {
          for (Value result : member->getResults()) {
            if (!isa<ShapedType>(result.getType()) ||
                !persistentValues.insert(result).second)
              continue;
            FailureOr<std::optional<int64_t>> bytes = getStaticByteSize(
                result.getType(), std::nullopt, problem.anchor);
            if (failed(bytes) ||
                failed(addEstimatedBytes(estimate.persistentBytes, *bytes,
                                         estimate.complete, problem.anchor)))
              return failure();
          }
        }
        return success();
      }

      if (assignment.workUnitId >= 0) {
        const MappingWorkUnit *workUnit =
            workUnitsById.lookup(assignment.workUnitId);
        if (!workUnit || workUnit->operationId != assignment.operationId ||
            workUnit->resultNumber < 0 ||
            workUnit->resultNumber >=
                static_cast<int64_t>(operation.operation->getNumResults())) {
          markIncomplete("an assignment references an invalid work unit");
          return success();
        }
        Value result = operation.operation->getResult(workUnit->resultNumber);
        if (isOutputOnlyValue(result))
          return success();
        FailureOr<std::optional<int64_t>> bytes = getStaticByteSize(
            result.getType(), ArrayRef<int64_t>(workUnit->resultSizes),
            problem.anchor);
        if (failed(bytes))
          return failure();
        SmallVector<int64_t> tensorIds = sourceTensorIdsByWorkUnitEndpoint[
            {assignment.operationId, assignment.workUnitId}];
        if (tensorIds.empty())
          tensorIds.push_back(-1);
        if (failed(addBoundaryProducedRange(
                assignment, *bytes, tensorIds,
                /*includeAssignmentExecution=*/true)))
          return failure();
        return success();
      }

      // The compute graph's output-tensor list intentionally follows graph
      // boundaries and can omit a value consumed by a non-compute support op
      // such as tensor.concat.  Account the endpoint root's shaped results
      // directly so mixed expanded/unexpanded plans still have a complete
      // transient estimate.
      for (Value result : operation.operation->getResults()) {
        if (!isa<ShapedType>(result.getType()) ||
            isa<sculptor::LogicalArrayType>(result.getType()) ||
            isOutputOnlyValue(result) ||
            !producedValues.insert(result).second)
          continue;
        FailureOr<std::optional<int64_t>> bytes =
            getStaticByteSize(result.getType(), std::nullopt, problem.anchor);
        if (failed(bytes) ||
            failed(addInternalProducedRange(assignment, *bytes)))
          return failure();
      }

      for (Operation *member : operation.members) {
        bool destinationStyle = isa<DestinationStyleOpInterface>(member);
        for (Value result : member->getResults()) {
          if (destinationStyle || isa<tensor::ExtractSliceOp>(member) ||
              !isa<ShapedType, sculptor::LogicalArrayType>(result.getType()) ||
              isa<sculptor::LogicalArrayType>(result.getType()) ||
              isOutputOnlyValue(result) ||
              !producedValues.insert(result).second)
            continue;
          FailureOr<std::optional<int64_t>> bytes =
              getStaticByteSize(result.getType(), std::nullopt, problem.anchor);
          if (failed(bytes) ||
              failed(addInternalProducedRange(assignment, *bytes)))
            return failure();
        }
      }
      for (int64_t tensorId : operation.outputTensors) {
        if (tensorId < 0 ||
            tensorId >= static_cast<int64_t>(graph.tensors.size()))
          return problem.anchor->emitError(
              "compute output references an unknown tensor during memory "
              "estimation");
        const ComputeTensor &tensor = graph.tensors[tensorId];
        if (tensor.isLogicalArray || isOutputOnlyValue(tensor.value))
          continue;
        int64_t contributionBytes = getProducerContributionByteSize(
            tensor, assignment.operationId);
        if (contributionBytes < 0) {
          markIncomplete(
              "a tensor producer contribution has unknown static size");
          continue;
        }
        if (failed(addBoundaryProducedRange(
                assignment, std::optional<int64_t>{contributionBytes},
                ArrayRef<int64_t>{tensorId},
                /*includeAssignmentExecution=*/false)))
          return failure();
      }
      return success();
    };

    for (const LogicalTileAssignment &assignment : tile.digitalAssignments)
      if (failed(accountAssignment(assignment)))
        return failure();
    for (const LogicalTileAnalogLane &lane : tile.analogLanes)
      for (const LogicalTileAssignment &assignment : lane.assignments)
        if (failed(accountAssignment(assignment)))
          return failure();
    if (!estimate.complete && estimate.incompleteReason.empty())
      estimate.incompleteReason =
          "a persistent resource has unknown static size";

    auto extendProducedLifetime = [&](const LogicalTileDependency &dependency,
                                      double finalUseNs) -> LogicalResult {
      auto ranges = producedByEndpointAndTensor.find(
          {dependency.sourceOperationId, dependency.sourceWorkUnitId,
           dependency.tensorId});
      if (ranges == producedByEndpointAndTensor.end())
        ranges = producedByEndpointAndTensor.find(
            {dependency.sourceOperationId, dependency.sourceWorkUnitId, -1});
      if (ranges == producedByEndpointAndTensor.end()) {
        bool extendedExpandedSource = false;
        if (dependency.sourceWorkUnitId < 0) {
          auto expanded = producedByOperationAndTensor.find(
              {dependency.sourceOperationId, dependency.tensorId});
          if (expanded == producedByOperationAndTensor.end())
            expanded = producedByOperationAndTensor.find(
                {dependency.sourceOperationId, -1});
          if (expanded != producedByOperationAndTensor.end()) {
            for (unsigned rangeIndex : expanded->second)
              if (finalUseNs >= liveRanges[rangeIndex].endNs) {
                liveRanges[rangeIndex].endNs = finalUseNs;
                liveRanges[rangeIndex].targetOperationId =
                    dependency.targetOperationId;
                liveRanges[rangeIndex].targetWorkUnitId =
                    dependency.targetWorkUnitId;
              }
            extendedExpandedSource = !expanded->second.empty();
          }
        }
        if (!extendedExpandedSource && dependency.byteSize != 0) {
          std::string reason =
              "dependency source operation " +
              std::to_string(dependency.sourceOperationId) + " work unit " +
              std::to_string(dependency.sourceWorkUnitId) +
              " has no estimated transient resource";
          markIncomplete(reason);
        }
        return success();
      }
      if (!std::isfinite(finalUseNs) || finalUseNs < 0.0) {
        markIncomplete("a produced resource has an invalid final-use time");
        return success();
      }
      for (unsigned rangeIndex : ranges->second) {
        if (finalUseNs < liveRanges[rangeIndex].endNs)
          continue;
        liveRanges[rangeIndex].endNs = finalUseNs;
        liveRanges[rangeIndex].targetOperationId =
            dependency.targetOperationId;
        liveRanges[rangeIndex].targetWorkUnitId =
            dependency.targetWorkUnitId;
      }
      return success();
    };

    for (const LogicalTileDependency &dependency : tile.internalDependencies) {
      SmallVector<const LogicalTileAssignment *> targets = findAssignments(
          dependency.targetOperationId, dependency.targetWorkUnitId, tile.id);
      if (targets.empty() || llvm::any_of(targets, [&](const auto *target) {
            return !validAssignmentTimes(*target);
          })) {
        markIncomplete("an internal dependency has no scheduled target");
        continue;
      }
      double finalUseNs =
          (*llvm::max_element(targets, [](const auto *left,
                                         const auto *right) {
            return left->finishNs < right->finishNs;
          }))->finishNs;
      if (failed(extendProducedLifetime(dependency, finalUseNs)))
        return failure();
    }

    // One received payload can feed several consumers on the same tile. Keep
    // one range and extend it through the final consuming assignment.
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, unsigned>
        incomingPayloads;
    for (const LogicalTileEdge &edge : problem.tileGraph.edges) {
      if (edge.sourceTileId == tile.id) {
        for (const LogicalTileDependency &dependency : edge.dependencies) {
          SmallVector<const LogicalTileAssignment *> targets = findAssignments(
              dependency.targetOperationId, dependency.targetWorkUnitId,
              edge.targetTileId);
          if (targets.empty() || llvm::any_of(targets, [&](const auto *target) {
                return !validAssignmentTimes(*target);
              })) {
            markIncomplete("an outgoing dependency has no scheduled target");
            continue;
          }
          // The source buffer must survive until the route can hand the value
          // to its destination. Target start is a conservative pre-placement
          // proxy for route completion.
          double finalUseNs =
              (*llvm::max_element(targets, [](const auto *left,
                                             const auto *right) {
                return left->startNs < right->startNs;
              }))->startNs;
          if (failed(extendProducedLifetime(dependency, finalUseNs)))
            return failure();
        }
      }
      if (edge.targetTileId != tile.id)
        continue;
      for (const LogicalTileDependency &dependency : edge.dependencies) {
        if (dependency.byteSize < 0) {
          markIncomplete("an incoming dependency has unknown byte size");
          continue;
        }
        if (dependency.byteSize == 0)
          continue;
        SmallVector<const LogicalTileAssignment *> sources = findAssignments(
            dependency.sourceOperationId, dependency.sourceWorkUnitId,
            edge.sourceTileId);
        SmallVector<const LogicalTileAssignment *> targets = findAssignments(
            dependency.targetOperationId, dependency.targetWorkUnitId, tile.id);
        if (sources.empty() || targets.empty() ||
            llvm::any_of(sources, [&](const auto *source) {
              return !validAssignmentTimes(*source);
            }) ||
            llvm::any_of(targets, [&](const auto *target) {
              return !validAssignmentTimes(*target);
            })) {
          markIncomplete("an incoming dependency has an unscheduled endpoint");
          continue;
        }
        double sourceFinishNs =
            (*llvm::max_element(sources, [](const auto *left,
                                           const auto *right) {
              return left->finishNs < right->finishNs;
            }))->finishNs;
        double targetFinishNs =
            (*llvm::max_element(targets, [](const auto *left,
                                           const auto *right) {
              return left->finishNs < right->finishNs;
            }))->finishNs;
        auto key = std::make_tuple(
            dependency.sourceOperationId, dependency.sourceWorkUnitId,
            dependency.tensorId, dependency.byteSize);
        auto existing = incomingPayloads.find(key);
        if (existing == incomingPayloads.end()) {
          unsigned index = liveRanges.size();
          liveRanges.push_back(
              {dependency.byteSize,
               sourceFinishNs,
               targetFinishNs,
               EstimateTransientKind::Incoming,
               edge.sourceTileId,
               dependency.sourceOperationId,
               dependency.sourceWorkUnitId,
               dependency.targetOperationId,
               dependency.targetWorkUnitId,
               dependency.tensorId});
          incomingPayloads.emplace(key, index);
        } else {
          EstimateLiveRange &range = liveRanges[existing->second];
          range.beginNs = std::min(range.beginNs, sourceFinishNs);
          range.endNs = std::max(range.endNs, targetFinishNs);
        }
      }
    }

    std::map<double, EstimateEventBucket> events;
    for (EstimateLiveRange &range : liveRanges) {
      if (!std::isfinite(range.beginNs) || !std::isfinite(range.endNs) ||
          range.beginNs < 0.0) {
        markIncomplete("a resource has invalid lifetime bounds");
        continue;
      }
      if (range.endNs < range.beginNs) {
        markIncomplete("a resource dies before it becomes live");
        range.endNs = range.beginNs;
      }
      EstimateEventBucket &begin = events[range.beginNs];
      if (range.beginNs == range.endNs) {
        int64_t &instant = range.kind == EstimateTransientKind::Produced
                               ? begin.producedInstant
                               : begin.incomingInstant;
        if (failed(checkedAccumulateEstimate(instant, range.bytes,
                                             problem.anchor)))
          return failure();
        continue;
      }
      EstimateEventBucket &end = events[range.endNs];
      int64_t &starts = range.kind == EstimateTransientKind::Produced
                            ? begin.producedStarts
                            : begin.incomingStarts;
      int64_t &ends = range.kind == EstimateTransientKind::Produced
                          ? end.producedEnds
                          : end.incomingEnds;
      if (failed(checkedAccumulateEstimate(starts, range.bytes,
                                           problem.anchor)) ||
          failed(checkedAccumulateEstimate(ends, range.bytes,
                                           problem.anchor)))
        return failure();
    }

    int64_t liveProduced = 0;
    int64_t liveIncoming = 0;
    double peakTimeNs = 0.0;
    bool hasPeakTime = false;
    estimate.requiredBytes = estimate.persistentBytes;
    for (const auto &[time, event] : events) {
      // A resource whose final consumer has completed can be reused by work
      // beginning at the same scheduled instant.
      if (failed(checkedReleaseEstimate(liveProduced, event.producedEnds,
                                        problem.anchor)) ||
          failed(checkedReleaseEstimate(liveIncoming, event.incomingEnds,
                                        problem.anchor)) ||
          failed(checkedAccumulateEstimate(liveProduced, event.producedStarts,
                                           problem.anchor)) ||
          failed(checkedAccumulateEstimate(liveIncoming, event.incomingStarts,
                                           problem.anchor)) ||
          failed(checkedAccumulateEstimate(liveProduced,
                                           event.producedInstant,
                                           problem.anchor)) ||
          failed(checkedAccumulateEstimate(liveIncoming,
                                           event.incomingInstant,
                                           problem.anchor)))
        return failure();

      std::optional<int64_t> transient =
          llvm::checkedAdd(liveProduced, liveIncoming);
      std::optional<int64_t> required =
          transient ? llvm::checkedAdd(estimate.persistentBytes, *transient)
                    : std::nullopt;
      if (!required)
        return problem.anchor->emitError(
            "logical-tile peak-memory estimate overflow");
      if (*required > estimate.requiredBytes) {
        estimate.requiredBytes = *required;
        estimate.producedBytes = liveProduced;
        estimate.incomingBytes = liveIncoming;
        peakTimeNs = time;
        hasPeakTime = true;
      }

      if (failed(checkedReleaseEstimate(liveProduced,
                                        event.producedInstant,
                                        problem.anchor)) ||
          failed(checkedReleaseEstimate(liveIncoming,
                                        event.incomingInstant,
                                        problem.anchor)))
        return failure();
    }
    if (liveProduced != 0 || liveIncoming != 0)
      return problem.anchor->emitError(
          "logical-tile memory lifetimes do not close at schedule end");

    // Keep capacity failures actionable without attaching diagnostics to the
    // potentially enormous one-line module operation. Report the first
    // offending tile and its largest payloads at the exact peak as a compact,
    // grep-friendly record on stderr.
    if (!emittedCapacityDiagnostic && problem.tileMemoryCapacityBytes > 0 &&
        estimate.requiredBytes > problem.tileMemoryCapacityBytes) {
      emittedCapacityDiagnostic = true;
      SmallVector<const EstimateLiveRange *> activeRanges;
      size_t activeProducedCount = 0;
      size_t activeIncomingCount = 0;
      if (hasPeakTime) {
        for (const EstimateLiveRange &range : liveRanges) {
          bool live = range.beginNs == range.endNs
                          ? range.beginNs == peakTimeNs
                          : range.beginNs <= peakTimeNs &&
                                peakTimeNs < range.endNs;
          if (!live)
            continue;
          activeRanges.push_back(&range);
          if (range.kind == EstimateTransientKind::Produced)
            ++activeProducedCount;
          else
            ++activeIncomingCount;
        }
      }
      llvm::sort(activeRanges, [](const EstimateLiveRange *left,
                                  const EstimateLiveRange *right) {
        return std::tie(left->bytes, left->beginNs, left->endNs) >
               std::tie(right->bytes, right->beginNs, right->endNs);
      });
      constexpr size_t maxReportedPayloads = 12;
      llvm::errs() << "SCULPTOR_LOGICAL_TILE_MEMORY_PEAK tile=" << tile.id
                   << " capacity_bytes=" << problem.tileMemoryCapacityBytes
                   << " required_bytes=" << estimate.requiredBytes
                   << " persistent_bytes=" << estimate.persistentBytes
                   << " produced_bytes=" << estimate.producedBytes
                   << " incoming_bytes=" << estimate.incomingBytes
                   << " time_ns=";
      if (hasPeakTime)
        llvm::errs() << peakTimeNs;
      else
        llvm::errs() << "none";
      llvm::errs() << " active_produced_count=" << activeProducedCount
                   << " active_incoming_count=" << activeIncomingCount
                   << '\n';
      for (const EstimateLiveRange *range :
           llvm::ArrayRef(activeRanges).take_front(
               std::min(maxReportedPayloads, activeRanges.size()))) {
        StringRef sourceName = "unknown";
        StringRef targetName = "unknown";
        StringRef tensorDefiningName = "block_argument";
        if (range->sourceOperationId >= 0 &&
            range->sourceOperationId <
                static_cast<int64_t>(graph.operations.size()))
          sourceName = graph.operations[range->sourceOperationId]
                           .operation->getName()
                           .getStringRef();
        if (range->targetOperationId >= 0 &&
            range->targetOperationId <
                static_cast<int64_t>(graph.operations.size()))
          targetName = graph.operations[range->targetOperationId]
                           .operation->getName()
                           .getStringRef();
        if (range->tensorId >= 0 &&
            range->tensorId < static_cast<int64_t>(graph.tensors.size()))
          if (Operation *defining =
                  graph.tensors[range->tensorId].value.getDefiningOp())
            tensorDefiningName = defining->getName().getStringRef();
        llvm::errs() << "SCULPTOR_LOGICAL_TILE_MEMORY_PAYLOAD"
                     << " kind="
                     << (range->kind == EstimateTransientKind::Produced
                             ? "produced"
                             : "incoming")
                     << " bytes=" << range->bytes
                     << " source_tile=" << range->sourceTileId
                     << " source_operation=" << range->sourceOperationId
                     << " source_name=" << sourceName
                     << " source_work_unit=" << range->sourceWorkUnitId
                     << " target_operation=" << range->targetOperationId
                     << " target_name=" << targetName
                     << " target_work_unit=" << range->targetWorkUnitId
                     << " tensor=" << range->tensorId
                     << " tensor_defining_name=" << tensorDefiningName
                     << " begin_ns=" << range->beginNs
                     << " end_ns=" << range->endNs << '\n';
      }
    }
    problem.memoryEstimateIndexByTileId[tile.id] =
        problem.memoryEstimates.size();
    problem.memoryEstimates.push_back(estimate);
  }
  return success();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

LogicalResult initializeLogicalTilePlacementProblem(
    const ComputeGraph &computeGraph,
    const ResourceAllocationTree &resourceAllocationTree,
    MappingCostProfile &costProfileStorage,
    LogicalTilePlacementProblem &problem) {
  if (!problem.anchor)
    return failure();

  MappingHardwareModel hardware;
  hardware.meshRows = problem.mesh.rows;
  hardware.meshCols = problem.mesh.columns;
  hardware.arraysPerCore = problem.mesh.arraysPerCore;
  auto profileAttr = problem.anchor->getAttrOfType<DictionaryAttr>(
      kMappingCostProfileAttrName);
  if (profileAttr) {
    if (auto clock = profileAttr.getAs<IntegerAttr>("clock_frequency_hz"))
      hardware.clockFrequencyHz = clock.getInt();
    if (auto network = profileAttr.getAs<DictionaryAttr>("network")) {
      if (auto wordBits = network.getAs<IntegerAttr>("word_bits"))
        hardware.networkWordBits = wordBits.getInt();
    }
  }

  FailureOr<MappingCostProfile> resolvedProfile =
      profileAttr
          ? deserializeMappingCostProfile(profileAttr, hardware, problem.anchor)
          : FailureOr<MappingCostProfile>(
                getLegacyMappingCostProfile(hardware));
  if (failed(resolvedProfile))
    return failure();

  costProfileStorage = std::move(*resolvedProfile);
  problem.computeGraph = &computeGraph;
  problem.raTree = &resourceAllocationTree;
  problem.costProfile = &costProfileStorage;
  return buildLogicalTileMemoryEstimates(computeGraph, resourceAllocationTree,
                                         problem);
}

LogicalResult initializeLogicalTilePlacementProblemFromPlan(
    LogicalTilePlacementAttr placementAttr, const ComputeGraph &computeGraph,
    const ResourceAllocationTree &resourceAllocationTree,
    MappingCostProfile &costProfileStorage,
    LogicalTilePlacementProblem &problem) {
  problem.mesh = {placementAttr.getMeshRows().getInt(),
                  placementAttr.getMeshCols().getInt(),
                  placementAttr.getArraysPerCore().getInt()};
  problem.tileMemoryCapacityBytes =
      placementAttr.getTileMemoryCapacityBytes().getInt();
  FailureOr<PlacementObjectiveKind> objective = parsePlacementObjective(
      placementAttr.getObjective().getValue(), problem.anchor);
  FailureOr<TemporalNetworkMode> networkMode = parseTemporalNetworkMode(
      placementAttr.getNetworkMode().getValue(), problem.anchor);
  FailureOr<TemporalTimingScope> timingScope = parseTemporalTimingScope(
      placementAttr.getTimingScope().getValue(), problem.anchor);
  if (failed(objective) || failed(networkMode) || failed(timingScope) ||
      failed(initializeLogicalTilePlacementProblem(
          computeGraph, resourceAllocationTree, costProfileStorage, problem)))
    return failure();

  problem.objective = *objective;
  problem.networkMode = *networkMode;
  problem.timingScope = *timingScope;
  return success();
}

FailureOr<LogicalTileScheduleKind>
parseLogicalTileSchedule(StringRef value, Operation *anchor,
                         bool allowAnnealing) {
  if (value == "random")
    return LogicalTileScheduleKind::Random;
  if (value == "snake")
    return LogicalTileScheduleKind::Snake;
  if (value == "greedy")
    return LogicalTileScheduleKind::Greedy;
  if (value == "annealing" && allowAnnealing)
    return LogicalTileScheduleKind::Annealing;
  anchor->emitError("unknown logical-tile placement schedule '")
      << value << "'";
  return failure();
}

StringRef stringifyLogicalTileSchedule(LogicalTileScheduleKind schedule) {
  switch (schedule) {
  case LogicalTileScheduleKind::Random:
    return "random";
  case LogicalTileScheduleKind::Snake:
    return "snake";
  case LogicalTileScheduleKind::Greedy:
    return "greedy";
  case LogicalTileScheduleKind::Annealing:
    return "annealing";
  }
  llvm_unreachable("unknown logical-tile placement schedule");
}

FailureOr<GreedyTileOrder> parseGreedyTileOrder(StringRef value,
                                                Operation *anchor) {
  if (value == "sequential")
    return GreedyTileOrder::Sequential;
  if (value == "priority")
    return GreedyTileOrder::Priority;
  anchor->emitError("unknown greedy logical-tile order '") << value << "'";
  return failure();
}

StringRef stringifyGreedyTileOrder(GreedyTileOrder order) {
  switch (order) {
  case GreedyTileOrder::Sequential:
    return "sequential";
  case GreedyTileOrder::Priority:
    return "priority";
  }
  llvm_unreachable("unknown greedy logical-tile order");
}

FailureOr<GreedyPriorityMode> parseGreedyPriorityMode(StringRef value,
                                                      Operation *anchor) {
  if (value == "sum")
    return GreedyPriorityMode::Sum;
  if (value == "max")
    return GreedyPriorityMode::Max;
  anchor->emitError("unknown greedy priority mode '") << value << "'";
  return failure();
}

StringRef stringifyGreedyPriorityMode(GreedyPriorityMode mode) {
  switch (mode) {
  case GreedyPriorityMode::Sum:
    return "sum";
  case GreedyPriorityMode::Max:
    return "max";
  }
  llvm_unreachable("unknown greedy priority mode");
}

FailureOr<GreedyCandidateScope> parseGreedyCandidateScope(StringRef value,
                                                          Operation *anchor) {
  if (value == "cardinal")
    return GreedyCandidateScope::Cardinal;
  if (value == "diagonal")
    return GreedyCandidateScope::Diagonal;
  if (value == "frontier")
    return GreedyCandidateScope::Frontier;
  anchor->emitError("unknown greedy candidate scope '") << value << "'";
  return failure();
}

StringRef stringifyGreedyCandidateScope(GreedyCandidateScope scope) {
  switch (scope) {
  case GreedyCandidateScope::Cardinal:
    return "cardinal";
  case GreedyCandidateScope::Diagonal:
    return "diagonal";
  case GreedyCandidateScope::Frontier:
    return "frontier";
  }
  llvm_unreachable("unknown greedy candidate scope");
}

LogicalResult validateLogicalTilePlacementProblem(
    const LogicalTilePlacementProblem &problem) {
  FailureOr<int64_t> capacity =
      getPhysicalTileCapacity(problem.mesh, problem.anchor);
  if (failed(capacity))
    return failure();
  if (problem.tileGraph.tiles.empty()) {
    problem.anchor->emitError("logical-tile placement requires an active tile");
    return failure();
  }
  if (problem.mesh.rows != problem.tileGraph.plannedMeshRows ||
      problem.mesh.columns != problem.tileGraph.plannedMeshCols) {
    problem.anchor->emitError("physical placement mesh ")
        << problem.mesh.rows << "x" << problem.mesh.columns
        << " does not match logical planning mesh "
        << problem.tileGraph.plannedMeshRows << "x"
        << problem.tileGraph.plannedMeshCols;
    return failure();
  }
  if (problem.mesh.arraysPerCore != problem.tileGraph.analogLanesPerTile) {
    problem.anchor->emitError("physical placement arrays per core ")
        << problem.mesh.arraysPerCore
        << " does not match logical planning arrays per core "
        << problem.tileGraph.analogLanesPerTile;
    return failure();
  }
  if (problem.tileGraph.tiles.size() > static_cast<size_t>(*capacity)) {
    problem.anchor->emitError("physical mesh has ")
        << *capacity << " tiles but the mapping requires "
        << problem.tileGraph.tiles.size() << " active logical tiles";
    return failure();
  }
  for (const LogicalTile &tile : problem.tileGraph.tiles) {
    if (tile.analogLanes.size() >
        static_cast<size_t>(problem.mesh.arraysPerCore)) {
      problem.anchor->emitError("logical tile ")
          << tile.id << " exceeds physical analog-lane capacity";
      return failure();
    }
  }
  if (problem.tileMemoryCapacityBytes < 0) {
    problem.anchor->emitError(
        "tile memory capacity must be nonnegative; zero disables it");
    return failure();
  }
  if (problem.memoryEstimates.size() != problem.tileGraph.tiles.size()) {
    problem.anchor->emitError(
        "logical-tile memory estimate count does not match active tiles");
    return failure();
  }
  llvm::DenseSet<int64_t> estimatedTileIds;
  for (const LogicalTileMemoryEstimate &estimate : problem.memoryEstimates) {
    if (!problem.tileGraph.tileIndexById.contains(estimate.logicalTileId) ||
        !estimatedTileIds.insert(estimate.logicalTileId).second ||
        estimate.persistentBytes < 0 || estimate.producedBytes < 0 ||
        estimate.incomingBytes < 0 || estimate.requiredBytes < 0) {
      problem.anchor->emitError(
          "logical-tile memory estimate has invalid identity or bytes");
      return failure();
    }
    std::optional<int64_t> partial =
        llvm::checkedAdd(estimate.persistentBytes, estimate.producedBytes);
    std::optional<int64_t> required =
        partial ? llvm::checkedAdd(*partial, estimate.incomingBytes)
                : std::nullopt;
    if (!required || *required != estimate.requiredBytes) {
      problem.anchor->emitError(
          "logical-tile memory estimate categories do not add up");
      return failure();
    }
    if (problem.tileMemoryCapacityBytes == 0)
      continue;
    if (!estimate.complete) {
      problem.anchor->emitError("cannot enforce tile memory capacity for "
                                "logical tile ")
          << estimate.logicalTileId
          << ": its pre-outlining memory estimate is incomplete"
          << (estimate.incompleteReason.empty()
                  ? ""
                  : (" (" + estimate.incompleteReason + ")"));
      return failure();
    }
    if (estimate.requiredBytes > problem.tileMemoryCapacityBytes) {
      problem.anchor->emitError("logical tile ")
          << estimate.logicalTileId << " requires " << estimate.requiredBytes
          << " conservative local-memory bytes (persistent "
          << estimate.persistentBytes << ", peak produced "
          << estimate.producedBytes << ", peak incoming "
          << estimate.incomingBytes << "), exceeding the per-tile "
             "capacity of "
          << problem.tileMemoryCapacityBytes << " bytes";
      return failure();
    }
  }
  return success();
}

FailureOr<int64_t>
scoreLogicalTilePlacement(const LogicalTilePlacementProblem &problem,
                          ArrayRef<int64_t> physicalTileByLogicalTileIndex) {
  if (failed(validateLogicalTilePlacementProblem(problem)))
    return failure();
  if (physicalTileByLogicalTileIndex.size() != problem.tileGraph.tiles.size()) {
    problem.anchor->emitError(
        "logical-tile placement score requires one location per active tile");
    return failure();
  }
  FailureOr<int64_t> capacity =
      getPhysicalTileCapacity(problem.mesh, problem.anchor);
  if (failed(capacity))
    return failure();
  std::set<int64_t> usedPhysicalTiles;
  for (int64_t physicalTileId : physicalTileByLogicalTileIndex) {
    if (physicalTileId < 0 || physicalTileId >= *capacity ||
        !usedPhysicalTiles.insert(physicalTileId).second) {
      problem.anchor->emitError(
          "logical-tile placement contains an invalid or duplicate physical "
          "tile");
      return failure();
    }
  }

  int64_t total = 0;
  for (const LogicalTileEdge &edge : problem.tileGraph.edges) {
    auto source = problem.tileGraph.tileIndexById.find(edge.sourceTileId);
    auto target = problem.tileGraph.tileIndexById.find(edge.targetTileId);
    if (source == problem.tileGraph.tileIndexById.end() ||
        target == problem.tileGraph.tileIndexById.end()) {
      problem.anchor->emitError(
          "logical-tile placement edge references an inactive tile");
      return failure();
    }
    PhysicalTileLocation sourceLocation = getLocation(
        physicalTileByLogicalTileIndex[source->second], problem.mesh);
    PhysicalTileLocation targetLocation = getLocation(
        physicalTileByLogicalTileIndex[target->second], problem.mesh);
    FailureOr<int64_t> cost = checkedTransferCost(
        edge.byteSize, getManhattanDistance(sourceLocation, targetLocation),
        problem.anchor);
    if (failed(cost))
      return failure();
    FailureOr<int64_t> updated = checkedAddCost(total, *cost, problem.anchor);
    if (failed(updated))
      return failure();
    total = *updated;
  }
  return total;
}

FailureOr<LogicalTilePlacementPlan>
buildLogicalTilePlacementPlan(const LogicalTilePlacementProblem &problem,
                              ArrayRef<int64_t> physicalTileByLogicalTileIndex,
                              StringRef schedule, int64_t initialScore,
                              int64_t evaluations) {
  FailureOr<PlacementObjectiveEvaluation> objective =
      evaluatePlacementObjective(problem, physicalTileByLogicalTileIndex);
  if (failed(objective))
    return failure();

  LogicalTilePlacementPlan plan;
  plan.schedule = schedule.str();
  plan.objective = problem.objective;
  plan.networkMode = problem.networkMode;
  plan.timingScope = problem.timingScope;
  if (problem.costProfile) {
    plan.costProfileName = problem.costProfile->name;
    plan.costProfileHash = problem.costProfile->contentHash;
  }
  plan.mesh = problem.mesh;
  plan.tileMemoryCapacityBytes = problem.tileMemoryCapacityBytes;
  plan.memoryEstimates = problem.memoryEstimates;
  for (auto indexedEstimate : llvm::enumerate(plan.memoryEstimates))
    plan.memoryEstimateIndexByTileId[indexedEstimate.value().logicalTileId] =
        indexedEstimate.index();
  plan.initialScore = initialScore;
  plan.objectiveScore = objective->score;
  plan.totalTransferCost = objective->transferCost;
  plan.predictedMakespanNs = objective->makespanNs;
  plan.evaluations = evaluations;
  plan.assignments.reserve(problem.tileGraph.tiles.size());
  for (auto indexedTile : llvm::enumerate(problem.tileGraph.tiles)) {
    LogicalTilePhysicalAssignment assignment;
    assignment.logicalTileId = indexedTile.value().id;
    assignment.location = getLocation(
        physicalTileByLogicalTileIndex[indexedTile.index()], problem.mesh);
    plan.assignmentIndexByTileId[assignment.logicalTileId] =
        plan.assignments.size();
    plan.assignments.push_back(assignment);
  }

  plan.edges.reserve(problem.tileGraph.edges.size());
  for (const LogicalTileEdge &edge : problem.tileGraph.edges) {
    const LogicalTilePhysicalAssignment &source =
        plan.assignments[plan.assignmentIndexByTileId.lookup(
            edge.sourceTileId)];
    const LogicalTilePhysicalAssignment &target =
        plan.assignments[plan.assignmentIndexByTileId.lookup(
            edge.targetTileId)];
    int64_t hops = getManhattanDistance(source.location, target.location);
    FailureOr<int64_t> cost =
        checkedTransferCost(edge.byteSize, hops, problem.anchor);
    if (failed(cost))
      return failure();
    plan.edges.push_back({edge.id, edge.sourceTileId, edge.targetTileId,
                          edge.byteSize, hops, *cost});
  }
  if (failed(verifyLogicalTilePlacementPlan(problem, plan)))
    return failure();
  return plan;
}

LogicalResult
verifyLogicalTilePlacementPlan(const LogicalTilePlacementProblem &problem,
                               const LogicalTilePlacementPlan &plan) {
  if (failed(validateLogicalTilePlacementProblem(problem)))
    return failure();
  if (plan.version != 3)
    return problem.anchor->emitError(
               "logical-tile placement version mismatch: ")
           << plan.version << " versus 3";
  if (plan.schedule.empty())
    return problem.anchor->emitError(
        "logical-tile placement schedule must not be empty");
  if (plan.objective != problem.objective)
    return problem.anchor->emitError(
               "logical-tile placement objective mismatch: '")
           << stringifyPlacementObjective(plan.objective) << "' versus '"
           << stringifyPlacementObjective(problem.objective) << "'";
  if (plan.networkMode != problem.networkMode)
    return problem.anchor->emitError(
               "logical-tile placement network mode mismatch: '")
           << stringifyTemporalNetworkMode(plan.networkMode) << "' versus '"
           << stringifyTemporalNetworkMode(problem.networkMode) << "'";
  if (plan.timingScope != problem.timingScope)
    return problem.anchor->emitError(
               "logical-tile placement timing scope mismatch: '")
           << stringifyTemporalTimingScope(plan.timingScope) << "' versus '"
           << stringifyTemporalTimingScope(problem.timingScope) << "'";
  if (plan.routePolicy != "xy")
    return problem.anchor->emitError(
               "logical-tile placement route policy mismatch: '")
           << plan.routePolicy << "' versus 'xy'";
  if (problem.costProfile && plan.costProfileName != problem.costProfile->name)
    return problem.anchor->emitError(
               "logical-tile placement cost profile name mismatch: '")
           << plan.costProfileName << "' versus '" << problem.costProfile->name
           << "'";
  if (problem.costProfile &&
      plan.costProfileHash != problem.costProfile->contentHash)
    return problem.anchor->emitError(
        "logical-tile placement cost profile hash mismatch");
  if (plan.mesh.rows != problem.mesh.rows)
    return problem.anchor->emitError(
               "logical-tile placement mesh-row mismatch: ")
           << plan.mesh.rows << " versus " << problem.mesh.rows;
  if (plan.mesh.columns != problem.mesh.columns)
    return problem.anchor->emitError(
               "logical-tile placement mesh-column mismatch: ")
           << plan.mesh.columns << " versus " << problem.mesh.columns;
  if (plan.mesh.arraysPerCore != problem.mesh.arraysPerCore)
    return problem.anchor->emitError(
               "logical-tile placement arrays-per-core mismatch: ")
           << plan.mesh.arraysPerCore << " versus "
           << problem.mesh.arraysPerCore;
  if (plan.tileMemoryCapacityBytes != problem.tileMemoryCapacityBytes)
    return problem.anchor->emitError(
               "logical-tile placement memory-capacity mismatch: ")
           << plan.tileMemoryCapacityBytes << " versus "
           << problem.tileMemoryCapacityBytes;
  if (plan.memoryEstimates.size() != problem.memoryEstimates.size())
    return problem.anchor->emitError(
        "logical-tile placement memory-estimate count mismatch");
  for (const LogicalTileMemoryEstimate &expected : problem.memoryEstimates) {
    auto indexed =
        plan.memoryEstimateIndexByTileId.find(expected.logicalTileId);
    if (indexed == plan.memoryEstimateIndexByTileId.end())
      return problem.anchor->emitError(
          "logical-tile placement is missing a memory estimate");
    const LogicalTileMemoryEstimate &actual =
        plan.memoryEstimates[indexed->second];
    if (actual.logicalTileId != expected.logicalTileId ||
        actual.persistentBytes != expected.persistentBytes ||
        actual.producedBytes != expected.producedBytes ||
        actual.incomingBytes != expected.incomingBytes ||
        actual.requiredBytes != expected.requiredBytes ||
        actual.complete != expected.complete)
      return problem.anchor->emitError(
          "logical-tile placement memory estimate does not match the "
          "reconstructed placement problem");
  }
  if (plan.initialScore < 0)
    return problem.anchor->emitError(
        "logical-tile placement initial score must be nonnegative");
  if (plan.objectiveScore < 0)
    return problem.anchor->emitError(
        "logical-tile placement objective score must be nonnegative");
  if (plan.totalTransferCost < 0)
    return problem.anchor->emitError(
        "logical-tile placement transfer cost must be nonnegative");
  if (!std::isfinite(plan.predictedMakespanNs) ||
      plan.predictedMakespanNs < 0.0)
    return problem.anchor->emitError(
        "logical-tile placement predicted makespan must be finite and "
        "nonnegative");
  if (plan.evaluations < 0)
    return problem.anchor->emitError(
        "logical-tile placement evaluation count must be nonnegative");
  if (plan.assignments.size() != problem.tileGraph.tiles.size())
    return problem.anchor->emitError(
               "logical-tile placement assignment-count mismatch: ")
           << plan.assignments.size() << " versus "
           << problem.tileGraph.tiles.size();
  if (plan.edges.size() != problem.tileGraph.edges.size())
    return problem.anchor->emitError(
               "logical-tile placement edge-count mismatch: ")
           << plan.edges.size() << " versus " << problem.tileGraph.edges.size();

  FailureOr<int64_t> capacity =
      getPhysicalTileCapacity(problem.mesh, problem.anchor);
  if (failed(capacity))
    return failure();
  std::set<int64_t> logicalTiles;
  std::set<int64_t> physicalTiles;
  DenseMap<int64_t, PhysicalTileLocation> locations;
  for (const LogicalTilePhysicalAssignment &assignment : plan.assignments) {
    if (!problem.tileGraph.tileIndexById.contains(assignment.logicalTileId) ||
        !logicalTiles.insert(assignment.logicalTileId).second ||
        assignment.location.physicalTileId < 0 ||
        assignment.location.physicalTileId >= *capacity ||
        !physicalTiles.insert(assignment.location.physicalTileId).second ||
        assignment.location.row !=
            assignment.location.physicalTileId / problem.mesh.columns ||
        assignment.location.column !=
            assignment.location.physicalTileId % problem.mesh.columns) {
      problem.anchor->emitError(
          "invalid or duplicate logical-to-physical tile assignment");
      return failure();
    }
    locations[assignment.logicalTileId] = assignment.location;
  }

  int64_t total = 0;
  for (auto indexedEdge : llvm::enumerate(plan.edges)) {
    const PlacedLogicalTileEdge &placed = indexedEdge.value();
    const LogicalTileEdge &logical =
        problem.tileGraph.edges[indexedEdge.index()];
    if (placed.edgeId != logical.id ||
        placed.sourceTileId != logical.sourceTileId ||
        placed.targetTileId != logical.targetTileId ||
        placed.byteSize != logical.byteSize) {
      problem.anchor->emitError(
          "placed edge does not match its logical-tile edge");
      return failure();
    }
    int64_t expectedHops =
        getManhattanDistance(locations.lookup(placed.sourceTileId),
                             locations.lookup(placed.targetTileId));
    FailureOr<int64_t> expectedCost =
        checkedTransferCost(placed.byteSize, expectedHops, problem.anchor);
    if (failed(expectedCost) || placed.manhattanHops != expectedHops ||
        placed.transferCost != *expectedCost) {
      problem.anchor->emitError("placed edge has an invalid distance or cost");
      return failure();
    }
    FailureOr<int64_t> updated =
        checkedAddCost(total, placed.transferCost, problem.anchor);
    if (failed(updated))
      return failure();
    total = *updated;
  }
  if (total != plan.totalTransferCost) {
    problem.anchor->emitError(
        "logical-tile placement total does not match its edges");
    return failure();
  }
  if (plan.objective == PlacementObjectiveKind::TransferCost &&
      (plan.objectiveScore != plan.totalTransferCost ||
       plan.predictedMakespanNs != 0.0)) {
    problem.anchor->emitError(
        "transfer-cost placement has inconsistent objective metadata");
    return failure();
  }
  if (plan.objective == PlacementObjectiveKind::Makespan) {
    FailureOr<PlacementObjectiveEvaluation> objective =
        evaluatePlacementObjective(problem, [&]() {
          SmallVector<int64_t> physical;
          physical.reserve(problem.tileGraph.tiles.size());
          for (const LogicalTile &tile : problem.tileGraph.tiles)
            physical.push_back(locations.lookup(tile.id).physicalTileId);
          return physical;
        }());
    if (failed(objective) || objective->score != plan.objectiveScore ||
        objective->makespanNs != plan.predictedMakespanNs)
      return problem.anchor->emitError(
          "makespan placement has inconsistent objective metadata");
  }
  return success();
}

LogicalTilePlacementAttr
serializeLogicalTilePlacement(MLIRContext *context,
                              const LogicalTilePlacementPlan &plan) {
  Builder builder(context);
  SmallVector<Attribute> assignments;
  assignments.reserve(plan.assignments.size());
  for (const LogicalTilePhysicalAssignment &assignment : plan.assignments) {
    assignments.push_back(PhysicalTileAssignmentAttr::get(
        context, builder.getI64IntegerAttr(assignment.logicalTileId),
        builder.getI64IntegerAttr(assignment.location.physicalTileId),
        builder.getI64IntegerAttr(assignment.location.row),
        builder.getI64IntegerAttr(assignment.location.column)));
  }
  SmallVector<Attribute> edges;
  edges.reserve(plan.edges.size());
  for (const PlacedLogicalTileEdge &edge : plan.edges) {
    edges.push_back(PlacedLogicalTileEdgeAttr::get(
        context, builder.getI64IntegerAttr(edge.edgeId),
        builder.getI64IntegerAttr(edge.sourceTileId),
        builder.getI64IntegerAttr(edge.targetTileId),
        builder.getI64IntegerAttr(edge.byteSize),
        builder.getI64IntegerAttr(edge.manhattanHops),
        builder.getI64IntegerAttr(edge.transferCost)));
  }
  SmallVector<Attribute> memoryEstimates;
  memoryEstimates.reserve(plan.memoryEstimates.size());
  for (const LogicalTileMemoryEstimate &estimate : plan.memoryEstimates) {
    memoryEstimates.push_back(LogicalTileMemoryEstimateAttr::get(
        context, builder.getI64IntegerAttr(estimate.logicalTileId),
        builder.getI64IntegerAttr(estimate.persistentBytes),
        builder.getI64IntegerAttr(estimate.producedBytes),
        builder.getI64IntegerAttr(estimate.incomingBytes),
        builder.getI64IntegerAttr(estimate.requiredBytes),
        builder.getBoolAttr(estimate.complete)));
  }
  return LogicalTilePlacementAttr::get(
      context, builder.getI64IntegerAttr(plan.version),
      builder.getStringAttr(plan.schedule),
      builder.getStringAttr(stringifyPlacementObjective(plan.objective)),
      builder.getStringAttr(stringifyTemporalNetworkMode(plan.networkMode)),
      builder.getStringAttr(stringifyTemporalTimingScope(plan.timingScope)),
      builder.getStringAttr(plan.routePolicy),
      builder.getStringAttr(plan.costProfileName),
      builder.getStringAttr(plan.costProfileHash),
      builder.getI64IntegerAttr(plan.mesh.rows),
      builder.getI64IntegerAttr(plan.mesh.columns),
      builder.getI64IntegerAttr(plan.mesh.arraysPerCore),
      builder.getI64IntegerAttr(plan.tileMemoryCapacityBytes),
      builder.getArrayAttr(memoryEstimates),
      builder.getI64IntegerAttr(plan.initialScore),
      builder.getI64IntegerAttr(plan.objectiveScore),
      builder.getI64IntegerAttr(plan.totalTransferCost),
      builder.getF64FloatAttr(plan.predictedMakespanNs),
      builder.getI64IntegerAttr(plan.evaluations),
      builder.getArrayAttr(assignments), builder.getArrayAttr(edges));
}

FailureOr<LogicalTilePlacementPlan>
deserializeLogicalTilePlacement(LogicalTilePlacementAttr attr,
                                const LogicalTilePlacementProblem &problem) {
  LogicalTilePlacementPlan plan;
  plan.version = attr.getVersion().getInt();
  plan.schedule = attr.getSchedule().getValue().str();
  FailureOr<PlacementObjectiveKind> objective =
      parsePlacementObjective(attr.getObjective().getValue(), problem.anchor);
  if (failed(objective))
    return failure();
  plan.objective = *objective;
  FailureOr<TemporalNetworkMode> networkMode = parseTemporalNetworkMode(
      attr.getNetworkMode().getValue(), problem.anchor);
  FailureOr<TemporalTimingScope> timingScope = parseTemporalTimingScope(
      attr.getTimingScope().getValue(), problem.anchor);
  if (failed(networkMode) || failed(timingScope))
    return failure();
  plan.networkMode = *networkMode;
  plan.timingScope = *timingScope;
  plan.routePolicy = attr.getRoutePolicy().getValue().str();
  plan.costProfileName = attr.getCostProfileName().getValue().str();
  plan.costProfileHash = attr.getCostProfileHash().getValue().str();
  plan.mesh = {attr.getMeshRows().getInt(), attr.getMeshCols().getInt(),
               attr.getArraysPerCore().getInt()};
  plan.tileMemoryCapacityBytes = attr.getTileMemoryCapacityBytes().getInt();
  for (Attribute value : attr.getMemoryEstimates()) {
    auto estimateAttr = dyn_cast<LogicalTileMemoryEstimateAttr>(value);
    if (!estimateAttr) {
      problem.anchor->emitError("logical-tile memory estimates must be typed");
      return failure();
    }
    LogicalTileMemoryEstimate estimate{
        estimateAttr.getLogicalTileId().getInt(),
        estimateAttr.getPersistentBytes().getInt(),
        estimateAttr.getProducedBytes().getInt(),
        estimateAttr.getIncomingBytes().getInt(),
        estimateAttr.getRequiredBytes().getInt(),
        estimateAttr.getComplete().getValue()};
    if (plan.memoryEstimateIndexByTileId.contains(estimate.logicalTileId)) {
      problem.anchor->emitError(
          "logical-tile placement contains a duplicate memory estimate");
      return failure();
    }
    plan.memoryEstimateIndexByTileId[estimate.logicalTileId] =
        plan.memoryEstimates.size();
    plan.memoryEstimates.push_back(estimate);
  }
  plan.initialScore = attr.getInitialScore().getInt();
  plan.objectiveScore = attr.getObjectiveScore().getInt();
  plan.totalTransferCost = attr.getTotalTransferCost().getInt();
  plan.predictedMakespanNs = attr.getPredictedMakespanNs().getValueAsDouble();
  plan.evaluations = attr.getEvaluations().getInt();
  for (Attribute value : attr.getAssignments()) {
    auto assignmentAttr = dyn_cast<PhysicalTileAssignmentAttr>(value);
    if (!assignmentAttr) {
      problem.anchor->emitError(
          "logical-tile placement assignments must be typed");
      return failure();
    }
    LogicalTilePhysicalAssignment assignment{
        assignmentAttr.getLogicalTileId().getInt(),
        {assignmentAttr.getPhysicalTileId().getInt(),
         assignmentAttr.getRow().getInt(),
         assignmentAttr.getColumn().getInt()}};
    if (plan.assignmentIndexByTileId.contains(assignment.logicalTileId)) {
      problem.anchor->emitError(
          "logical-tile placement contains a duplicate logical tile");
      return failure();
    }
    plan.assignmentIndexByTileId[assignment.logicalTileId] =
        plan.assignments.size();
    plan.assignments.push_back(assignment);
  }
  for (Attribute value : attr.getEdges()) {
    auto edgeAttr = dyn_cast<PlacedLogicalTileEdgeAttr>(value);
    if (!edgeAttr) {
      problem.anchor->emitError("logical-tile placed edges must be typed");
      return failure();
    }
    plan.edges.push_back(
        {edgeAttr.getEdgeId().getInt(), edgeAttr.getSourceTileId().getInt(),
         edgeAttr.getTargetTileId().getInt(), edgeAttr.getByteSize().getInt(),
         edgeAttr.getManhattanHops().getInt(),
         edgeAttr.getTransferCost().getInt()});
  }
  if (failed(verifyLogicalTilePlacementPlan(problem, plan)))
    return failure();
  return plan;
}

LogicalTileAnnealingTraceAttr
serializeLogicalTileAnnealingTrace(MLIRContext *context,
                                   const LogicalTileAnnealingTrace &trace) {
  Builder builder(context);
  SmallVector<Attribute> samples;
  samples.reserve(trace.samples.size());
  for (const LogicalTileAnnealingSample &sample : trace.samples) {
    samples.push_back(AnnealingScoreSampleAttr::get(
        context, builder.getI64IntegerAttr(sample.iteration),
        builder.getI64IntegerAttr(sample.candidateScore),
        builder.getI64IntegerAttr(sample.currentScore),
        builder.getI64IntegerAttr(sample.bestScore),
        builder.getBoolAttr(sample.accepted)));
  }
  return LogicalTileAnnealingTraceAttr::get(
      context, builder.getI64IntegerAttr(trace.version),
      builder.getI64IntegerAttr(trace.initialScore),
      builder.getI64IntegerAttr(trace.finalScore),
      builder.getI64IntegerAttr(trace.evaluations),
      builder.getArrayAttr(samples));
}

FailureOr<LogicalTileAnnealingTrace>
deserializeLogicalTileAnnealingTrace(LogicalTileAnnealingTraceAttr attr,
                                     const LogicalTilePlacementPlan &plan,
                                     Operation *anchor) {
  LogicalTileAnnealingTrace trace;
  trace.version = attr.getVersion().getInt();
  trace.initialScore = attr.getInitialScore().getInt();
  trace.finalScore = attr.getFinalScore().getInt();
  trace.evaluations = attr.getEvaluations().getInt();
  for (Attribute value : attr.getSamples()) {
    auto sampleAttr = dyn_cast<AnnealingScoreSampleAttr>(value);
    if (!sampleAttr) {
      anchor->emitError("logical-tile annealing samples must be typed");
      return failure();
    }
    trace.samples.push_back({sampleAttr.getIteration().getInt(),
                             sampleAttr.getCandidateScore().getInt(),
                             sampleAttr.getCurrentScore().getInt(),
                             sampleAttr.getBestScore().getInt(),
                             sampleAttr.getAccepted().getValue()});
  }

  if (trace.version != 1 || plan.schedule != "annealing" ||
      trace.initialScore != plan.initialScore ||
      trace.finalScore != plan.objectiveScore ||
      trace.evaluations != plan.evaluations || trace.evaluations < 0 ||
      trace.samples.empty() || trace.samples.front().iteration != 0 ||
      trace.samples.back().iteration != trace.evaluations) {
    anchor->emitError("annealing trace does not match its placement plan");
    return failure();
  }

  int64_t previousCurrent = trace.initialScore;
  int64_t previousBest = trace.initialScore;
  for (auto indexedSample : llvm::enumerate(trace.samples)) {
    const LogicalTileAnnealingSample &sample = indexedSample.value();
    if ((indexedSample.index() > 0 &&
         sample.iteration <=
             trace.samples[indexedSample.index() - 1].iteration) ||
        sample.iteration < 0 || sample.iteration > trace.evaluations ||
        sample.candidateScore < 0 || sample.currentScore < 0 ||
        sample.bestScore < 0) {
      anchor->emitError("invalid annealing score sample metadata");
      return failure();
    }
    if (indexedSample.index() == 0) {
      if (!sample.accepted || sample.candidateScore != trace.initialScore ||
          sample.currentScore != trace.initialScore ||
          sample.bestScore != trace.initialScore) {
        anchor->emitError("invalid initial annealing score sample");
        return failure();
      }
      continue;
    }
    bool consecutive = sample.iteration ==
                       trace.samples[indexedSample.index() - 1].iteration + 1;
    int64_t expectedCurrent =
        sample.accepted ? sample.candidateScore : previousCurrent;
    int64_t expectedBest = std::min(previousBest, expectedCurrent);
    if ((consecutive && (sample.currentScore != expectedCurrent ||
                         sample.bestScore != expectedBest)) ||
        (!consecutive && (sample.bestScore > previousBest ||
                          sample.bestScore > sample.currentScore))) {
      anchor->emitError("annealing score trajectory is inconsistent");
      return failure();
    }
    previousCurrent = sample.currentScore;
    previousBest = sample.bestScore;
  }
  if (previousBest != trace.finalScore) {
    anchor->emitError("annealing trace final score is inconsistent");
    return failure();
  }
  return trace;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
