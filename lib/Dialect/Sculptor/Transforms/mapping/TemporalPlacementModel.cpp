#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/TemporalPlacementModel.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingCostProfile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::sculptor::mapping;

struct EventState {
  TemporalTaskEvent event;
  unsigned logicalTileIndex = 0;
  int64_t physicalTile = -1;
  SmallVector<int64_t> predecessors;
  SmallVector<int64_t> successors;
  double startNs = 0.0;
  double finishNs = 0.0;
  double criticalRemainingNs = 0.0;
  int64_t criticalPredecessor = -1;
};

struct AnalogPhases {
  double loadNs = 0.0;
  double executeNs = 0.0;
  double storeNs = 0.0;
};

int64_t manhattanDistance(int64_t source, int64_t target,
                          const PhysicalMeshGeometry &mesh) {
  return std::abs(source / mesh.columns - target / mesh.columns) +
         std::abs(source % mesh.columns - target % mesh.columns);
}

SmallVector<std::pair<int64_t, int64_t>>
buildXYPath(int64_t source, int64_t target, const PhysicalMeshGeometry &mesh) {
  SmallVector<std::pair<int64_t, int64_t>> path;
  int64_t current = source;
  int64_t row = current / mesh.columns;
  int64_t column = current % mesh.columns;
  int64_t targetRow = target / mesh.columns;
  int64_t targetColumn = target % mesh.columns;
  while (column != targetColumn) {
    int64_t nextColumn = column + (targetColumn > column ? 1 : -1);
    int64_t next = row * mesh.columns + nextColumn;
    path.push_back({current, next});
    current = next;
    column = nextColumn;
  }
  while (row != targetRow) {
    int64_t nextRow = row + (targetRow > row ? 1 : -1);
    int64_t next = nextRow * mesh.columns + column;
    path.push_back({current, next});
    current = next;
    row = nextRow;
  }
  return path;
}

FailureOr<int64_t> addBytes(int64_t current, int64_t increment,
                            Operation *anchor) {
  std::optional<int64_t> total = llvm::checkedAdd(current, increment);
  if (!total) {
    anchor->emitError("temporal placement dependency byte count overflow");
    return failure();
  }
  return *total;
}

FailureOr<int64_t> getWords(int64_t bytes, const MappingCostProfile &profile,
                            Operation *anchor) {
  if (bytes < 0) {
    anchor->emitError("temporal route byte count must be nonnegative");
    return failure();
  }
  std::optional<int64_t> bits = llvm::checkedMul(bytes, int64_t{8});
  if (!bits) {
    anchor->emitError("temporal route bit count overflow");
    return failure();
  }
  return llvm::divideCeil(*bits, profile.network.wordBits);
}

FailureOr<double> finiteRouteDelay(int64_t bytes, int64_t source,
                                   int64_t target,
                                   const LogicalTilePlacementProblem &problem) {
  if (bytes < 0) {
    problem.anchor->emitError("temporal route byte count must be nonnegative");
    return failure();
  }
  if (bytes == 0 || source == target ||
      problem.networkMode == TemporalNetworkMode::Ideal)
    return 0.0;
  const MappingCostProfile &profile = *problem.costProfile;
  FailureOr<int64_t> words = getWords(bytes, profile, problem.anchor);
  if (failed(words))
    return failure();
  double wordTimeNs = 1.0e9 / static_cast<double>(profile.clockFrequencyHz);
  return profile.runtime.routeSetupNs + profile.network.injectFixedNs +
         static_cast<double>(*words) * wordTimeNs +
         static_cast<double>(manhattanDistance(source, target, problem.mesh)) *
             profile.network.hopPipelineNs +
         profile.network.ejectFixedNs +
         profile.network.dmaNsPerByte * static_cast<double>(bytes);
}

AnalogPhases getAnalogPhases(const EventState &state,
                             const LogicalTilePlacementProblem &problem) {
  const ComputeOperation &operation =
      problem.computeGraph->operations[state.event.operationId];
  if (operation.kind != ComputeOperationKind::PhysicalMVM ||
      !operation.analogMVM)
    return {0.0, state.event.durationNs, 0.0};

  int64_t executionCount = 1;
  for (const StructuralRATreeNode &node : problem.raTree->nodes) {
    if (node.id == state.event.leafId) {
      executionCount = node.workGroupCount;
      break;
    }
  }
  const MappingCostProfile &profile = *problem.costProfile;
  int64_t loadBytes = operation.analogMVM->inputColumns * int64_t{4};
  int64_t storeBytes = operation.analogMVM->outputRows * int64_t{4};
  return {profile.runtime.taskDispatchNs +
              (profile.analog.loadFixedNs +
               profile.analog.loadNsPerByte * static_cast<double>(loadBytes)) *
                  executionCount,
          profile.analog.executeNs * executionCount,
          (profile.analog.storeFixedNs +
           profile.analog.storeNsPerByte * static_cast<double>(storeBytes)) *
              executionCount};
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<TemporalPlacementEvaluation>
evaluateTemporalPlacement(const LogicalTilePlacementProblem &problem,
                          ArrayRef<int64_t> physicalTileByLogicalTileIndex) {
  if (!problem.computeGraph || !problem.raTree || !problem.costProfile) {
    problem.anchor->emitError(
        "makespan placement requires the compute graph, RA tree, and cost "
        "profile");
    return failure();
  }
  if (physicalTileByLogicalTileIndex.size() != problem.tileGraph.tiles.size()) {
    problem.anchor->emitError(
        "temporal placement requires one physical tile per logical tile");
    return failure();
  }
  std::optional<int64_t> capacityValue =
      llvm::checkedMul(problem.mesh.rows, problem.mesh.columns);
  if (!capacityValue || *capacityValue <= 0) {
    problem.anchor->emitError("temporal placement mesh capacity is invalid");
    return failure();
  }
  for (int64_t physicalTile : physicalTileByLogicalTileIndex) {
    if (physicalTile < 0 || physicalTile >= *capacityValue) {
      problem.anchor->emitError(
          "temporal placement contains an invalid physical tile");
      return failure();
    }
  }
  if (problem.temporalCandidateLimit <= 0) {
    problem.anchor->emitError("temporal candidate limit must be positive");
    return failure();
  }

  SmallVector<EventState> events;
  std::map<std::pair<int64_t, int64_t>, SmallVector<int64_t>> eventsByEndpoint;
  DenseMap<int64_t, int64_t> eventByLeaf;
  for (auto indexedTile : llvm::enumerate(problem.tileGraph.tiles)) {
    const LogicalTile &tile = indexedTile.value();
    auto appendAssignment = [&](const LogicalTileAssignment &assignment) {
      EventState state;
      state.event.eventId = events.size();
      state.event.leafId = assignment.leafId;
      state.event.logicalTileId = tile.id;
      state.event.operationId = assignment.operationId;
      state.event.workUnitId = assignment.workUnitId;
      state.event.laneKind = assignment.laneKind;
      state.event.laneIndex = assignment.laneIndex;
      state.event.durationNs =
          std::max(0.0, assignment.finishNs - assignment.startNs);
      const ComputeOperation &operation =
          problem.computeGraph->operations[assignment.operationId];
      if (problem.timingScope == TemporalTimingScope::Warm &&
          operation.kind == ComputeOperationKind::MatrixSetup)
        state.event.durationNs = 0.0;
      state.logicalTileIndex = indexedTile.index();
      state.physicalTile = physicalTileByLogicalTileIndex[indexedTile.index()];
      eventsByEndpoint[{assignment.operationId, assignment.workUnitId}]
          .push_back(state.event.eventId);
      if (assignment.workUnitId != -1)
        eventsByEndpoint[{assignment.operationId, -1}].push_back(
            state.event.eventId);
      if (!eventByLeaf.try_emplace(assignment.leafId, state.event.eventId)
               .second) {
        problem.anchor->emitError("temporal placement has duplicate RA leaf ")
            << assignment.leafId;
        return failure();
      }
      events.push_back(std::move(state));
      return success();
    };
    for (const LogicalTileAssignment &assignment : tile.digitalAssignments) {
      if (failed(appendAssignment(assignment)))
        return failure();
    }
    for (const LogicalTileAnalogLane &lane : tile.analogLanes) {
      for (const LogicalTileAssignment &assignment : lane.assignments) {
        if (failed(appendAssignment(assignment)))
          return failure();
      }
    }
  }
  if (events.empty()) {
    problem.anchor->emitError("temporal placement requires task events");
    return failure();
  }

  std::map<std::pair<int64_t, int64_t>, int64_t> dependencyBytes;
  auto addEventEdge = [&](int64_t source, int64_t target,
                          int64_t bytes) -> LogicalResult {
    if (source == target)
      return success();
    if (!llvm::is_contained(events[source].successors, target)) {
      events[source].successors.push_back(target);
      events[target].predecessors.push_back(source);
    }
    FailureOr<int64_t> total =
        addBytes(dependencyBytes[{source, target}], bytes, problem.anchor);
    if (failed(total))
      return failure();
    dependencyBytes[{source, target}] = *total;
    return success();
  };
  auto addDependency = [&](const LogicalTileDependency &dependency) {
    auto sources = eventsByEndpoint.find(
        {dependency.sourceOperationId, dependency.sourceWorkUnitId});
    auto targets = eventsByEndpoint.find(
        {dependency.targetOperationId, dependency.targetWorkUnitId});
    if (sources == eventsByEndpoint.end() ||
        targets == eventsByEndpoint.end()) {
      problem.anchor->emitError(
          "temporal placement cannot resolve a logical-tile dependency");
      return failure();
    }
    for (int64_t source : sources->second) {
      for (int64_t target : targets->second) {
        if (failed(addEventEdge(source, target, dependency.byteSize)))
          return failure();
      }
    }
    return success();
  };
  for (const LogicalTile &tile : problem.tileGraph.tiles) {
    for (const LogicalTileDependency &dependency : tile.internalDependencies) {
      if (failed(addDependency(dependency)))
        return failure();
    }
  }
  for (const LogicalTileEdge &edge : problem.tileGraph.edges) {
    for (const LogicalTileDependency &dependency : edge.dependencies) {
      if (failed(addDependency(dependency)))
        return failure();
    }
  }

  DenseMap<int64_t, const StructuralRATreeNode *> nodesById;
  for (const StructuralRATreeNode &node : problem.raTree->nodes)
    nodesById[node.id] = &node;
  DenseMap<int64_t, SmallVector<int64_t>> subtreeEvents;
  std::function<FailureOr<SmallVector<int64_t>>(int64_t)> collectEvents =
      [&](int64_t nodeId) -> FailureOr<SmallVector<int64_t>> {
    auto cached = subtreeEvents.find(nodeId);
    if (cached != subtreeEvents.end())
      return cached->second;
    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    if (!node) {
      problem.anchor->emitError("temporal placement cannot resolve RA node ")
          << nodeId;
      return failure();
    }
    SmallVector<int64_t> values;
    if (node->kind == RATreeNodeKind::Leaf) {
      auto event = eventByLeaf.find(node->id);
      if (event == eventByLeaf.end()) {
        problem.anchor->emitError("temporal placement cannot resolve RA leaf ")
            << node->id;
        return failure();
      }
      values.push_back(event->second);
    } else {
      for (int64_t child : node->childIds) {
        FailureOr<SmallVector<int64_t>> childEvents = collectEvents(child);
        if (failed(childEvents))
          return failure();
        values.append(childEvents->begin(), childEvents->end());
      }
    }
    subtreeEvents[nodeId] = values;
    return values;
  };
  if (failed(collectEvents(problem.raTree->rootId)))
    return failure();

  for (const StructuralRATreeNode &node : problem.raTree->nodes) {
    if (node.kind != RATreeNodeKind::TemporalCut)
      continue;
    for (size_t childIndex = 1; childIndex < node.childIds.size();
         ++childIndex) {
      const SmallVector<int64_t> &left =
          subtreeEvents.lookup(node.childIds[childIndex - 1]);
      const SmallVector<int64_t> &right =
          subtreeEvents.lookup(node.childIds[childIndex]);
      llvm::DenseSet<int64_t> leftSet(left.begin(), left.end());
      llvm::DenseSet<int64_t> rightSet(right.begin(), right.end());
      SmallVector<int64_t> sinks;
      SmallVector<int64_t> sources;
      for (int64_t event : left) {
        if (llvm::none_of(events[event].successors, [&](int64_t successor) {
              return leftSet.contains(successor);
            }))
          sinks.push_back(event);
      }
      for (int64_t event : right) {
        if (llvm::none_of(events[event].predecessors, [&](int64_t predecessor) {
              return rightSet.contains(predecessor);
            }))
          sources.push_back(event);
      }
      for (int64_t sink : sinks) {
        for (int64_t source : sources) {
          if (failed(addEventEdge(sink, source, /*bytes=*/0)))
            return failure();
        }
      }
    }
  }

  SmallVector<int64_t> indegree(events.size(), 0);
  std::set<int64_t> topoReady;
  for (EventState &event : events) {
    llvm::sort(event.predecessors);
    llvm::sort(event.successors);
    indegree[event.event.eventId] = event.predecessors.size();
    if (event.predecessors.empty())
      topoReady.insert(event.event.eventId);
  }
  SmallVector<int64_t> topologicalOrder;
  SmallVector<int64_t> topoIndegree = indegree;
  while (!topoReady.empty()) {
    int64_t event = *topoReady.begin();
    topoReady.erase(topoReady.begin());
    topologicalOrder.push_back(event);
    for (int64_t successor : events[event].successors) {
      if (--topoIndegree[successor] == 0)
        topoReady.insert(successor);
    }
  }
  if (topologicalOrder.size() != events.size()) {
    problem.anchor->emitError(
        "temporal placement event graph contains a dependency cycle");
    return failure();
  }
  for (int64_t eventId : llvm::reverse(topologicalOrder)) {
    EventState &event = events[eventId];
    double successorCost = 0.0;
    for (int64_t successor : event.successors)
      successorCost =
          std::max(successorCost, events[successor].criticalRemainingNs);
    event.criticalRemainingNs = event.event.durationNs + successorCost;
  }

  int64_t capacity = *capacityValue;
  SmallVector<double> digitalAvailable(capacity, 0.0);
  SmallVector<SmallVector<double>> analogAvailable(
      capacity, SmallVector<double>(problem.mesh.arraysPerCore, 0.0));
  SmallVector<double> analogIOAvailable(capacity, 0.0);
  SmallVector<double> sourceNICAvailable(capacity, 0.0);
  SmallVector<double> receiveDMAAvailable(capacity, 0.0);
  SmallVector<double> tileLoad(capacity, 0.0);
  std::map<std::pair<int64_t, int64_t>, double> linkAvailable;
  std::map<std::pair<int64_t, int64_t>, int64_t> linkWords;
  std::set<int64_t> ready;
  for (auto [eventId, degree] : llvm::enumerate(indegree)) {
    if (degree == 0)
      ready.insert(eventId);
  }

  TemporalPlacementEvaluation result;
  auto estimateDependencyReady = [&](int64_t eventId) -> FailureOr<double> {
    double readyNs = 0.0;
    for (int64_t predecessor : events[eventId].predecessors) {
      int64_t bytes = dependencyBytes[{predecessor, eventId}];
      FailureOr<double> delay =
          finiteRouteDelay(bytes, events[predecessor].physicalTile,
                           events[eventId].physicalTile, problem);
      if (failed(delay))
        return failure();
      readyNs = std::max(readyNs, events[predecessor].finishNs + *delay);
    }
    return readyNs;
  };
  auto estimateStart = [&](int64_t eventId) -> FailureOr<double> {
    const EventState &event = events[eventId];
    FailureOr<double> dependencyReady = estimateDependencyReady(eventId);
    if (failed(dependencyReady))
      return failure();
    if (event.event.laneKind == LogicalLaneKind::Digital)
      return std::max(*dependencyReady, digitalAvailable[event.physicalTile]);
    if (event.event.laneIndex < 0 ||
        event.event.laneIndex >= problem.mesh.arraysPerCore)
      return std::numeric_limits<double>::infinity();
    double loadStart =
        std::max(*dependencyReady, analogIOAvailable[event.physicalTile]);
    return loadStart;
  };

  auto scheduleRoute = [&](int64_t sourceEvent,
                           int64_t targetEvent) -> FailureOr<double> {
    EventState &source = events[sourceEvent];
    EventState &target = events[targetEvent];
    int64_t bytes = dependencyBytes[{sourceEvent, targetEvent}];
    if (bytes == 0 || source.physicalTile == target.physicalTile ||
        problem.networkMode == TemporalNetworkMode::Ideal)
      return source.finishNs;
    const MappingCostProfile &profile = *problem.costProfile;
    FailureOr<int64_t> words = getWords(bytes, profile, problem.anchor);
    if (failed(words))
      return failure();
    double wordTimeNs = 1.0e9 / static_cast<double>(profile.clockFrequencyHz);
    double serializationNs = *words * wordTimeNs;
    FailureOr<double> baseline = finiteRouteDelay(bytes, source.physicalTile,
                                                  target.physicalTile, problem);
    if (failed(baseline))
      return failure();
    if (problem.networkMode == TemporalNetworkMode::Finite) {
      result.exposedTransportNs += *baseline;
      return source.finishNs + *baseline;
    }

    double cursor =
        std::max(source.finishNs, sourceNICAvailable[source.physicalTile]);
    cursor += profile.runtime.routeSetupNs + profile.network.injectFixedNs;
    sourceNICAvailable[source.physicalTile] = cursor + serializationNs;
    SmallVector<std::pair<int64_t, int64_t>> path =
        buildXYPath(source.physicalTile, target.physicalTile, problem.mesh);
    for (const auto &link : path) {
      cursor = std::max(cursor, linkAvailable[link]);
      linkAvailable[link] = cursor + serializationNs;
      std::optional<int64_t> updatedWords =
          llvm::checkedAdd(linkWords[link], *words);
      if (!updatedWords) {
        problem.anchor->emitError("temporal directed-link word count overflow");
        return failure();
      }
      linkWords[link] = *updatedWords;
      result.maximumDirectedLinkWords =
          std::max(result.maximumDirectedLinkWords, linkWords[link]);
      cursor += profile.network.hopPipelineNs;
    }
    cursor += serializationNs + profile.network.ejectFixedNs;
    cursor = std::max(cursor, receiveDMAAvailable[target.physicalTile]);
    double dmaNs = profile.network.dmaNsPerByte * bytes;
    receiveDMAAvailable[target.physicalTile] = cursor + dmaNs;
    cursor += dmaNs;
    double actualDelay = cursor - source.finishNs;
    result.exposedTransportNs += actualDelay;
    result.exposedContentionNs += std::max(0.0, actualDelay - *baseline);
    return cursor;
  };

  SmallVector<int64_t> remainingIndegree = indegree;
  int64_t scheduled = 0;
  while (!ready.empty()) {
    int64_t selected = -1;
    double selectedStart = 0.0;
    for (int64_t candidate : ready) {
      FailureOr<double> start = estimateStart(candidate);
      if (failed(start))
        return failure();
      bool better = selected < 0 || *start < selectedStart;
      if (!better && selected >= 0 && *start == selectedStart) {
        better = events[candidate].criticalRemainingNs >
                 events[selected].criticalRemainingNs;
        if (events[candidate].criticalRemainingNs ==
            events[selected].criticalRemainingNs) {
          better = std::tie(events[candidate].event.operationId,
                            events[candidate].event.workUnitId,
                            events[candidate].event.eventId) <
                   std::tie(events[selected].event.operationId,
                            events[selected].event.workUnitId,
                            events[selected].event.eventId);
        }
      }
      if (better) {
        selected = candidate;
        selectedStart = *start;
      }
    }
    ready.erase(selected);
    EventState &event = events[selected];
    double dependencyReady = 0.0;
    int64_t criticalPredecessor = -1;
    for (int64_t predecessor : event.predecessors) {
      FailureOr<double> arrival = scheduleRoute(predecessor, selected);
      if (failed(arrival))
        return failure();
      if (*arrival >= dependencyReady) {
        dependencyReady = *arrival;
        criticalPredecessor = predecessor;
      }
    }

    if (event.event.laneKind == LogicalLaneKind::Digital) {
      event.startNs =
          std::max(dependencyReady, digitalAvailable[event.physicalTile]);
      event.finishNs = event.startNs + event.event.durationNs;
      digitalAvailable[event.physicalTile] = event.finishNs;
    } else {
      if (event.event.laneIndex < 0 ||
          event.event.laneIndex >= problem.mesh.arraysPerCore) {
        problem.anchor->emitError("temporal analog event has invalid lane ")
            << event.event.laneIndex;
        return failure();
      }
      AnalogPhases phases = getAnalogPhases(event, problem);
      double loadStart =
          std::max(dependencyReady, analogIOAvailable[event.physicalTile]);
      double loadFinish = loadStart + phases.loadNs;
      analogIOAvailable[event.physicalTile] = loadFinish;
      double executeStart =
          std::max(loadFinish,
                   analogAvailable[event.physicalTile][event.event.laneIndex]);
      double executeFinish = executeStart + phases.executeNs;
      analogAvailable[event.physicalTile][event.event.laneIndex] =
          executeFinish;
      double storeStart =
          std::max(executeFinish, analogIOAvailable[event.physicalTile]);
      event.startNs = loadStart;
      event.finishNs = storeStart + phases.storeNs;
      analogIOAvailable[event.physicalTile] = event.finishNs;
    }
    event.criticalPredecessor = criticalPredecessor;
    tileLoad[event.physicalTile] += event.event.durationNs;
    result.maximumTileLoadNs =
        std::max(result.maximumTileLoadNs, tileLoad[event.physicalTile]);
    result.makespanNs = std::max(result.makespanNs, event.finishNs);
    ++scheduled;
    for (int64_t successor : event.successors) {
      if (--remainingIndegree[successor] == 0)
        ready.insert(successor);
    }
  }
  if (scheduled != static_cast<int64_t>(events.size())) {
    problem.anchor->emitError(
        "temporal list scheduler did not visit all tasks");
    return failure();
  }

  int64_t critical = 0;
  for (const EventState &event : events) {
    if (event.finishNs > events[critical].finishNs)
      critical = event.event.eventId;
  }
  while (critical >= 0) {
    result.criticalEventIds.push_back(critical);
    result.taskTimeOnCriticalChainNs += events[critical].event.durationNs;
    critical = events[critical].criticalPredecessor;
  }
  std::reverse(result.criticalEventIds.begin(), result.criticalEventIds.end());
  return result;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
