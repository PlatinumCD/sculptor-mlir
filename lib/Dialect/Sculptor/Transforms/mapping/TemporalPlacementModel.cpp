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
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

struct EventState {
  TemporalTaskEvent event;
  unsigned logicalTileIndex = 0;
  SmallVector<int64_t> predecessors;
  SmallVector<int64_t> predecessorBytes;
  SmallVector<int64_t> predecessorWords;
  SmallVector<double> predecessorBaseDelayNs;
  SmallVector<int64_t> successors;
  double criticalRemainingNs = 0.0;
};

struct AnalogPhases {
  double loadNs = 0.0;
  double executeNs = 0.0;
  double storeNs = 0.0;
};

struct PreparedTemporalModel {
  int64_t capacity = 0;
  SmallVector<EventState, 0> events;
  SmallVector<int64_t> indegree;
  SmallVector<SmallVector<int64_t>> eventsByLogicalTile;
  SmallVector<AnalogPhases> analogPhases;
};

struct ScheduledTemporalPlacement {
  TemporalPlacementEvaluation evaluation;
  SmallVector<int64_t, 128> placement;
  SmallVector<int64_t, 256> scheduleOrder;
  SmallVector<int64_t, 256> schedulePosition;
};

int64_t manhattanDistance(int64_t source, int64_t target,
                          const PhysicalMeshGeometry &mesh) {
  return std::abs(source / mesh.columns - target / mesh.columns) +
         std::abs(source % mesh.columns - target % mesh.columns);
}

int64_t getDirectedLinkIndex(int64_t source, int64_t target,
                             const PhysicalMeshGeometry &mesh) {
  int64_t sourceRow = source / mesh.columns;
  int64_t sourceColumn = source % mesh.columns;
  int64_t targetRow = target / mesh.columns;
  int64_t targetColumn = target % mesh.columns;
  if (sourceRow == targetRow) {
    int64_t edge =
        sourceRow * (mesh.columns - 1) + std::min(sourceColumn, targetColumn);
    return edge * 2 + (targetColumn > sourceColumn ? 0 : 1);
  }
  int64_t horizontalLinks = mesh.rows * (mesh.columns - 1) * 2;
  int64_t edge = std::min(sourceRow, targetRow) * mesh.columns + sourceColumn;
  return horizontalLinks + edge * 2 + (targetRow > sourceRow ? 0 : 1);
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

FailureOr<PreparedTemporalModel>
prepareTemporalModel(const LogicalTilePlacementProblem &problem) {
  if (!problem.computeGraph || !problem.raTree || !problem.costProfile) {
    problem.anchor->emitError(
        "makespan placement requires the compute graph, RA tree, and cost "
        "profile");
    return failure();
  }
  std::optional<int64_t> capacity =
      llvm::checkedMul(problem.mesh.rows, problem.mesh.columns);
  if (!capacity || *capacity <= 0) {
    problem.anchor->emitError("temporal placement mesh capacity is invalid");
    return failure();
  }
  if (problem.temporalCandidateLimit <= 0) {
    problem.anchor->emitError("temporal candidate limit must be positive");
    return failure();
  }

  PreparedTemporalModel model;
  model.capacity = *capacity;
  model.eventsByLogicalTile.resize(problem.tileGraph.tiles.size());
  std::map<std::pair<int64_t, int64_t>, SmallVector<int64_t>> eventsByEndpoint;
  DenseMap<int64_t, int64_t> eventByLeaf;
  for (auto indexedTile : llvm::enumerate(problem.tileGraph.tiles)) {
    const LogicalTile &tile = indexedTile.value();
    auto appendAssignment = [&](const LogicalTileAssignment &assignment) {
      EventState state;
      state.event.eventId = model.events.size();
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
      model.eventsByLogicalTile[indexedTile.index()].push_back(
          state.event.eventId);
      model.events.push_back(std::move(state));
      return success();
    };
    for (const LogicalTileAssignment &assignment : tile.digitalAssignments)
      if (failed(appendAssignment(assignment)))
        return failure();
    for (const LogicalTileAnalogLane &lane : tile.analogLanes)
      for (const LogicalTileAssignment &assignment : lane.assignments)
        if (failed(appendAssignment(assignment)))
          return failure();
  }
  if (model.events.empty()) {
    problem.anchor->emitError("temporal placement requires task events");
    return failure();
  }

  std::map<std::pair<int64_t, int64_t>, int64_t> dependencyBytes;
  auto addEventEdge = [&](int64_t source, int64_t target,
                          int64_t bytes) -> LogicalResult {
    if (source == target)
      return success();
    if (!llvm::is_contained(model.events[source].successors, target)) {
      model.events[source].successors.push_back(target);
      model.events[target].predecessors.push_back(source);
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
    for (int64_t source : sources->second)
      for (int64_t target : targets->second)
        if (failed(addEventEdge(source, target, dependency.byteSize)))
          return failure();
    return success();
  };
  for (const LogicalTile &tile : problem.tileGraph.tiles)
    for (const LogicalTileDependency &dependency : tile.internalDependencies)
      if (failed(addDependency(dependency)))
        return failure();
  for (const LogicalTileEdge &edge : problem.tileGraph.edges)
    for (const LogicalTileDependency &dependency : edge.dependencies)
      if (failed(addDependency(dependency)))
        return failure();

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
      for (int64_t event : left)
        if (llvm::none_of(
                model.events[event].successors,
                [&](int64_t successor) { return leftSet.contains(successor); }))
          sinks.push_back(event);
      for (int64_t event : right)
        if (llvm::none_of(model.events[event].predecessors,
                          [&](int64_t predecessor) {
                            return rightSet.contains(predecessor);
                          }))
          sources.push_back(event);
      for (int64_t sink : sinks)
        for (int64_t source : sources)
          if (failed(addEventEdge(sink, source, /*bytes=*/0)))
            return failure();
    }
  }

  model.indegree.assign(model.events.size(), 0);
  std::set<int64_t> topoReady;
  for (EventState &event : model.events) {
    llvm::sort(event.predecessors);
    llvm::sort(event.successors);
    event.predecessorBytes.reserve(event.predecessors.size());
    event.predecessorWords.reserve(event.predecessors.size());
    event.predecessorBaseDelayNs.reserve(event.predecessors.size());
    const MappingCostProfile &profile = *problem.costProfile;
    double wordTimeNs = 1.0e9 / static_cast<double>(profile.clockFrequencyHz);
    for (int64_t predecessor : event.predecessors) {
      int64_t bytes = dependencyBytes.at({predecessor, event.event.eventId});
      FailureOr<int64_t> words = getWords(bytes, profile, problem.anchor);
      if (failed(words))
        return failure();
      event.predecessorBytes.push_back(bytes);
      event.predecessorWords.push_back(*words);
      event.predecessorBaseDelayNs.push_back(
          profile.runtime.routeSetupNs + profile.network.injectFixedNs +
          static_cast<double>(*words) * wordTimeNs +
          profile.network.ejectFixedNs +
          profile.network.dmaNsPerByte * static_cast<double>(bytes));
    }
    model.indegree[event.event.eventId] = event.predecessors.size();
    if (event.predecessors.empty())
      topoReady.insert(event.event.eventId);
  }
  SmallVector<int64_t> topologicalOrder;
  SmallVector<int64_t> topoIndegree = model.indegree;
  while (!topoReady.empty()) {
    int64_t event = *topoReady.begin();
    topoReady.erase(topoReady.begin());
    topologicalOrder.push_back(event);
    for (int64_t successor : model.events[event].successors)
      if (--topoIndegree[successor] == 0)
        topoReady.insert(successor);
  }
  if (topologicalOrder.size() != model.events.size()) {
    problem.anchor->emitError(
        "temporal placement event graph contains a dependency cycle");
    return failure();
  }
  for (int64_t eventId : llvm::reverse(topologicalOrder)) {
    EventState &event = model.events[eventId];
    double successorCost = 0.0;
    for (int64_t successor : event.successors)
      successorCost =
          std::max(successorCost, model.events[successor].criticalRemainingNs);
    event.criticalRemainingNs = event.event.durationNs + successorCost;
  }
  model.analogPhases.reserve(model.events.size());
  for (const EventState &event : model.events)
    model.analogPhases.push_back(getAnalogPhases(event, problem));
  return model;
}

LogicalResult validatePlacement(const LogicalTilePlacementProblem &problem,
                                const PreparedTemporalModel &model,
                                ArrayRef<int64_t> placement) {
  if (placement.size() != problem.tileGraph.tiles.size())
    return problem.anchor->emitError(
        "temporal placement requires one physical tile per logical tile");
  for (int64_t physicalTile : placement)
    if (physicalTile < 0 || physicalTile >= model.capacity)
      return problem.anchor->emitError(
          "temporal placement contains an invalid physical tile");
  return success();
}

FailureOr<ScheduledTemporalPlacement>
runTemporalSchedule(const LogicalTilePlacementProblem &problem,
                    const PreparedTemporalModel &model,
                    ArrayRef<int64_t> placement,
                    ArrayRef<int64_t> unchangedPrefix = {}) {
  if (failed(validatePlacement(problem, model, placement)))
    return failure();

  const SmallVector<EventState, 0> &events = model.events;
  SmallVector<int64_t, 256> physicalTileByEvent(events.size());
  SmallVector<double, 256> startNs(events.size(), 0.0);
  SmallVector<double, 256> finishNs(events.size(), 0.0);
  SmallVector<int64_t, 256> criticalPredecessor(events.size(), -1);
  for (const EventState &event : events)
    physicalTileByEvent[event.event.eventId] =
        placement[event.logicalTileIndex];
  SmallVector<double, 128> digitalAvailable(model.capacity, 0.0);
  SmallVector<SmallVector<double>, 128> analogAvailable(
      model.capacity, SmallVector<double>(problem.mesh.arraysPerCore, 0.0));
  SmallVector<double, 128> analogIOAvailable(model.capacity, 0.0);
  SmallVector<double, 128> sourceNICAvailable(model.capacity, 0.0);
  SmallVector<double, 128> receiveDMAAvailable(model.capacity, 0.0);
  SmallVector<double, 128> tileLoad(model.capacity, 0.0);
  SmallVector<double, 256> estimatedDependencyReady(events.size(), 0.0);
  int64_t directedLinkCount = (problem.mesh.rows * (problem.mesh.columns - 1) +
                               (problem.mesh.rows - 1) * problem.mesh.columns) *
                              2;
  SmallVector<double, 256> linkAvailable(directedLinkCount, 0.0);
  SmallVector<int64_t, 256> linkWords(directedLinkCount, 0);
  SmallVector<int64_t, 256> ready;
  ready.reserve(events.size());
  for (auto [eventId, degree] : llvm::enumerate(model.indegree))
    if (degree == 0)
      ready.push_back(eventId);

  ScheduledTemporalPlacement scheduledPlacement;
  scheduledPlacement.placement.assign(placement.begin(), placement.end());
  scheduledPlacement.scheduleOrder.reserve(events.size());
  TemporalPlacementEvaluation &result = scheduledPlacement.evaluation;
  auto getBaselineDelay = [&](int64_t targetEvent, size_t predecessorIndex) {
    const EventState &target = events[targetEvent];
    int64_t predecessor = target.predecessors[predecessorIndex];
    if (target.predecessorBytes[predecessorIndex] == 0 ||
        physicalTileByEvent[predecessor] == physicalTileByEvent[targetEvent] ||
        problem.networkMode == TemporalNetworkMode::Ideal)
      return 0.0;
    return target.predecessorBaseDelayNs[predecessorIndex] +
           static_cast<double>(manhattanDistance(
               physicalTileByEvent[predecessor],
               physicalTileByEvent[targetEvent], problem.mesh)) *
               problem.costProfile->network.hopPipelineNs;
  };
  auto estimateDependencyReady = [&](int64_t eventId) -> FailureOr<double> {
    double readyNs = 0.0;
    for (auto [predecessorIndex, predecessor] :
         llvm::enumerate(events[eventId].predecessors)) {
      double delay = getBaselineDelay(eventId, predecessorIndex);
      readyNs = std::max(readyNs, finishNs[predecessor] + delay);
    }
    return readyNs;
  };
  auto estimateStart = [&](int64_t eventId) -> FailureOr<double> {
    const EventState &event = events[eventId];
    double dependencyReady = estimatedDependencyReady[eventId];
    if (event.event.laneKind == LogicalLaneKind::Digital)
      return std::max(dependencyReady,
                      digitalAvailable[physicalTileByEvent[eventId]]);
    if (event.event.laneIndex < 0 ||
        event.event.laneIndex >= problem.mesh.arraysPerCore)
      return std::numeric_limits<double>::infinity();
    return std::max(dependencyReady,
                    analogIOAvailable[physicalTileByEvent[eventId]]);
  };
  auto scheduleRoute = [&](int64_t sourceEvent, int64_t targetEvent,
                           int64_t bytes, int64_t words,
                           double baseline) -> FailureOr<double> {
    int64_t sourcePhysicalTile = physicalTileByEvent[sourceEvent];
    int64_t targetPhysicalTile = physicalTileByEvent[targetEvent];
    if (bytes == 0 || sourcePhysicalTile == targetPhysicalTile ||
        problem.networkMode == TemporalNetworkMode::Ideal)
      return finishNs[sourceEvent];
    const MappingCostProfile &profile = *problem.costProfile;
    double wordTimeNs = 1.0e9 / static_cast<double>(profile.clockFrequencyHz);
    double serializationNs = words * wordTimeNs;
    if (problem.networkMode == TemporalNetworkMode::Finite) {
      result.exposedTransportNs += baseline;
      return finishNs[sourceEvent] + baseline;
    }

    double cursor =
        std::max(finishNs[sourceEvent], sourceNICAvailable[sourcePhysicalTile]);
    cursor += profile.runtime.routeSetupNs + profile.network.injectFixedNs;
    sourceNICAvailable[sourcePhysicalTile] = cursor + serializationNs;
    auto scheduleLink = [&](int64_t source, int64_t target) -> LogicalResult {
      int64_t link = getDirectedLinkIndex(source, target, problem.mesh);
      cursor = std::max(cursor, linkAvailable[link]);
      linkAvailable[link] = cursor + serializationNs;
      std::optional<int64_t> updatedWords =
          llvm::checkedAdd(linkWords[link], words);
      if (!updatedWords) {
        problem.anchor->emitError("temporal directed-link word count overflow");
        return failure();
      }
      linkWords[link] = *updatedWords;
      result.maximumDirectedLinkWords =
          std::max(result.maximumDirectedLinkWords, linkWords[link]);
      cursor += profile.network.hopPipelineNs;
      return success();
    };
    int64_t current = sourcePhysicalTile;
    int64_t row = current / problem.mesh.columns;
    int64_t column = current % problem.mesh.columns;
    int64_t targetRow = targetPhysicalTile / problem.mesh.columns;
    int64_t targetColumn = targetPhysicalTile % problem.mesh.columns;
    while (column != targetColumn) {
      int64_t nextColumn = column + (targetColumn > column ? 1 : -1);
      int64_t next = row * problem.mesh.columns + nextColumn;
      if (failed(scheduleLink(current, next)))
        return failure();
      current = next;
      column = nextColumn;
    }
    while (row != targetRow) {
      int64_t nextRow = row + (targetRow > row ? 1 : -1);
      int64_t next = nextRow * problem.mesh.columns + column;
      if (failed(scheduleLink(current, next)))
        return failure();
      current = next;
      row = nextRow;
    }
    cursor += serializationNs + profile.network.ejectFixedNs;
    cursor = std::max(cursor, receiveDMAAvailable[targetPhysicalTile]);
    double dmaNs = profile.network.dmaNsPerByte * bytes;
    receiveDMAAvailable[targetPhysicalTile] = cursor + dmaNs;
    cursor += dmaNs;
    double actualDelay = cursor - finishNs[sourceEvent];
    result.exposedTransportNs += actualDelay;
    result.exposedContentionNs += std::max(0.0, actualDelay - baseline);
    return cursor;
  };

  SmallVector<int64_t, 256> remainingIndegree(model.indegree.begin(),
                                              model.indegree.end());
  auto scheduleEvent = [&](int64_t selected) -> LogicalResult {
    auto selectedPosition = llvm::find(ready, selected);
    if (selectedPosition == ready.end())
      return problem.anchor->emitError(
          "incremental temporal prefix is not schedulable");
    ready.erase(selectedPosition);
    const EventState &event = events[selected];
    int64_t eventPhysicalTile = physicalTileByEvent[selected];
    double dependencyReady = 0.0;
    int64_t eventCriticalPredecessor = -1;
    for (auto [predecessorIndex, predecessor] :
         llvm::enumerate(event.predecessors)) {
      FailureOr<double> arrival = scheduleRoute(
          predecessor, selected, event.predecessorBytes[predecessorIndex],
          event.predecessorWords[predecessorIndex],
          getBaselineDelay(selected, predecessorIndex));
      if (failed(arrival))
        return failure();
      if (*arrival >= dependencyReady) {
        dependencyReady = *arrival;
        eventCriticalPredecessor = predecessor;
      }
    }

    if (event.event.laneKind == LogicalLaneKind::Digital) {
      startNs[selected] =
          std::max(dependencyReady, digitalAvailable[eventPhysicalTile]);
      finishNs[selected] = startNs[selected] + event.event.durationNs;
      digitalAvailable[eventPhysicalTile] = finishNs[selected];
    } else {
      if (event.event.laneIndex < 0 ||
          event.event.laneIndex >= problem.mesh.arraysPerCore)
        return problem.anchor->emitError(
                   "temporal analog event has invalid lane ")
               << event.event.laneIndex;
      const AnalogPhases &phases = model.analogPhases[selected];
      double loadStart =
          std::max(dependencyReady, analogIOAvailable[eventPhysicalTile]);
      double loadFinish = loadStart + phases.loadNs;
      analogIOAvailable[eventPhysicalTile] = loadFinish;
      double executeStart =
          std::max(loadFinish,
                   analogAvailable[eventPhysicalTile][event.event.laneIndex]);
      double executeFinish = executeStart + phases.executeNs;
      analogAvailable[eventPhysicalTile][event.event.laneIndex] = executeFinish;
      double storeStart =
          std::max(executeFinish, analogIOAvailable[eventPhysicalTile]);
      startNs[selected] = loadStart;
      finishNs[selected] = storeStart + phases.storeNs;
      analogIOAvailable[eventPhysicalTile] = finishNs[selected];
    }
    criticalPredecessor[selected] = eventCriticalPredecessor;
    tileLoad[eventPhysicalTile] += event.event.durationNs;
    result.maximumTileLoadNs =
        std::max(result.maximumTileLoadNs, tileLoad[eventPhysicalTile]);
    result.makespanNs = std::max(result.makespanNs, finishNs[selected]);
    scheduledPlacement.scheduleOrder.push_back(selected);
    for (int64_t successor : event.successors) {
      if (--remainingIndegree[successor] == 0) {
        FailureOr<double> estimate = estimateDependencyReady(successor);
        if (failed(estimate))
          return failure();
        estimatedDependencyReady[successor] = *estimate;
        ready.push_back(successor);
      }
    }
    return success();
  };

  for (int64_t selected : unchangedPrefix)
    if (failed(scheduleEvent(selected)))
      return failure();

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
            events[selected].criticalRemainingNs)
          better = std::tie(events[candidate].event.operationId,
                            events[candidate].event.workUnitId,
                            events[candidate].event.eventId) <
                   std::tie(events[selected].event.operationId,
                            events[selected].event.workUnitId,
                            events[selected].event.eventId);
      }
      if (better) {
        selected = candidate;
        selectedStart = *start;
      }
    }
    if (failed(scheduleEvent(selected)))
      return failure();
  }
  if (scheduledPlacement.scheduleOrder.size() != events.size()) {
    problem.anchor->emitError(
        "temporal list scheduler did not visit all tasks");
    return failure();
  }

  int64_t critical = 0;
  for (const EventState &event : events)
    if (finishNs[event.event.eventId] > finishNs[critical])
      critical = event.event.eventId;
  while (critical >= 0) {
    result.criticalEventIds.push_back(critical);
    result.taskTimeOnCriticalChainNs += events[critical].event.durationNs;
    critical = criticalPredecessor[critical];
  }
  std::reverse(result.criticalEventIds.begin(), result.criticalEventIds.end());

  scheduledPlacement.schedulePosition.assign(events.size(), -1);
  for (auto [position, event] :
       llvm::enumerate(scheduledPlacement.scheduleOrder))
    scheduledPlacement.schedulePosition[event] = position;
  return scheduledPlacement;
}

size_t findRollbackStep(const PreparedTemporalModel &model,
                        const ScheduledTemporalPlacement &current,
                        ArrayRef<int64_t> candidate,
                        ArrayRef<unsigned> changedLogicalTileIndices) {
  llvm::DenseSet<unsigned> changed(changedLogicalTileIndices.begin(),
                                   changedLogicalTileIndices.end());
  for (auto [index, physicalTile] : llvm::enumerate(candidate))
    if (physicalTile != current.placement[index])
      changed.insert(index);

  size_t rollback = current.scheduleOrder.size();
  for (unsigned logicalTileIndex : changed) {
    if (logicalTileIndex >= model.eventsByLogicalTile.size())
      return 0;
    for (int64_t eventId : model.eventsByLogicalTile[logicalTileIndex]) {
      size_t readyStep = 0;
      for (int64_t predecessor : model.events[eventId].predecessors)
        readyStep = std::max(
            readyStep,
            static_cast<size_t>(current.schedulePosition[predecessor] + 1));
      rollback = std::min(rollback, readyStep);
    }
  }
  return rollback;
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

struct IncrementalTemporalPlacementEvaluator::Impl {
  const LogicalTilePlacementProblem *problem = nullptr;
  PreparedTemporalModel model;
  ScheduledTemporalPlacement current;
  std::optional<ScheduledTemporalPlacement> candidate;
};

IncrementalTemporalPlacementEvaluator::IncrementalTemporalPlacementEvaluator(
    std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

IncrementalTemporalPlacementEvaluator::
    ~IncrementalTemporalPlacementEvaluator() = default;

IncrementalTemporalPlacementEvaluator::IncrementalTemporalPlacementEvaluator(
    IncrementalTemporalPlacementEvaluator &&) noexcept = default;

IncrementalTemporalPlacementEvaluator &
IncrementalTemporalPlacementEvaluator::operator=(
    IncrementalTemporalPlacementEvaluator &&) noexcept = default;

FailureOr<std::unique_ptr<IncrementalTemporalPlacementEvaluator>>
IncrementalTemporalPlacementEvaluator::create(
    const LogicalTilePlacementProblem &problem,
    ArrayRef<int64_t> initialPlacement) {
  FailureOr<PreparedTemporalModel> model = prepareTemporalModel(problem);
  if (failed(model))
    return failure();
  FailureOr<ScheduledTemporalPlacement> initial =
      runTemporalSchedule(problem, *model, initialPlacement);
  if (failed(initial))
    return failure();
  auto impl = std::make_unique<Impl>();
  impl->problem = &problem;
  impl->model = std::move(*model);
  impl->current = std::move(*initial);
  return std::unique_ptr<IncrementalTemporalPlacementEvaluator>(
      new IncrementalTemporalPlacementEvaluator(std::move(impl)));
}

const TemporalPlacementEvaluation &
IncrementalTemporalPlacementEvaluator::getCurrentEvaluation() const {
  return impl->current.evaluation;
}

FailureOr<TemporalPlacementEvaluation>
IncrementalTemporalPlacementEvaluator::evaluateCandidate(
    ArrayRef<int64_t> placement, ArrayRef<unsigned> changedLogicalTileIndices) {
  size_t rollback = findRollbackStep(impl->model, impl->current, placement,
                                     changedLogicalTileIndices);
  ArrayRef<int64_t> prefix(impl->current.scheduleOrder);
  prefix = prefix.take_front(rollback);
  FailureOr<ScheduledTemporalPlacement> candidate =
      runTemporalSchedule(*impl->problem, impl->model, placement, prefix);
  if (failed(candidate)) {
    impl->candidate.reset();
    return failure();
  }
  impl->candidate = std::move(*candidate);
  return impl->candidate->evaluation;
}

void IncrementalTemporalPlacementEvaluator::commitCandidate() {
  assert(impl->candidate && "no incremental temporal candidate to commit");
  impl->current = std::move(*impl->candidate);
  impl->candidate.reset();
}

void IncrementalTemporalPlacementEvaluator::discardCandidate() {
  impl->candidate.reset();
}

FailureOr<TemporalPlacementEvaluation>
IncrementalTemporalPlacementEvaluator::evaluateFromScratch(
    ArrayRef<int64_t> placement) const {
  FailureOr<ScheduledTemporalPlacement> result =
      runTemporalSchedule(*impl->problem, impl->model, placement);
  if (failed(result))
    return failure();
  return result->evaluation;
}

FailureOr<TemporalPlacementEvaluation>
evaluateTemporalPlacement(const LogicalTilePlacementProblem &problem,
                          ArrayRef<int64_t> physicalTileByLogicalTileIndex) {
  FailureOr<PreparedTemporalModel> model = prepareTemporalModel(problem);
  if (failed(model))
    return failure();
  FailureOr<ScheduledTemporalPlacement> result =
      runTemporalSchedule(problem, *model, physicalTileByLogicalTileIndex);
  if (failed(result))
    return failure();
  return result->evaluation;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
