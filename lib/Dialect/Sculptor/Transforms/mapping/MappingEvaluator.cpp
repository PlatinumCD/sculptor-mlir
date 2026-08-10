#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingEvaluator.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/GolemMVMPlanning.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingRealization.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <utility>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

using ChildEdge = std::pair<int64_t, int64_t>;
using LeafEndpoint = std::pair<int64_t, int64_t>;

struct ReferenceEvaluationContext {
  ReferenceEvaluationContext(const MappingProblem &problem,
                             const ResourceAllocationTree &tree)
      : problem(problem), tree(tree) {
    for (const MappingWorkUnitEdge &edge : tree.workUnitEdges) {
      workUnitEdgesBySourceOperation[edge.sourceOperationId].push_back(&edge);
      refinedOperationEdges.insert(
          {edge.sourceOperationId, edge.targetOperationId});
    }
  }

  const MappingProblem &problem;
  const ResourceAllocationTree &tree;
  DenseMap<int64_t, const StructuralRATreeNode *> nodesById;
  DenseMap<int64_t, SmallVector<int64_t>> subtreeOperations;
  DenseMap<int64_t, SmallVector<LeafEndpoint>> subtreeEndpoints;
  DenseMap<int64_t, MappingNodeEvaluation> evaluations;
  DenseMap<int64_t, SmallVector<double>> childStartOffsets;
  DenseMap<int64_t, SmallVector<const MappingWorkUnitEdge *>>
      workUnitEdgesBySourceOperation;
  std::set<std::pair<int64_t, int64_t>> refinedOperationEdges;
};

FailureOr<int64_t> checkedAddI64(int64_t lhs, int64_t rhs, Operation *anchor,
                                 StringRef description) {
  std::optional<int64_t> result = llvm::checkedAdd(lhs, rhs);
  if (!result) {
    anchor->emitError(description) << " overflows a signed 64-bit integer";
    return failure();
  }
  return *result;
}

FailureOr<SmallVector<LeafEndpoint>>
collectSubtreeEndpoints(ReferenceEvaluationContext &context, int64_t nodeId) {
  auto cached = context.subtreeEndpoints.find(nodeId);
  if (cached != context.subtreeEndpoints.end())
    return cached->second;

  const StructuralRATreeNode *node = context.nodesById.lookup(nodeId);
  if (!node) {
    context.problem.anchor->emitError(
        "mapping evaluator cannot resolve RA-tree node ")
        << nodeId;
    return failure();
  }

  SmallVector<LeafEndpoint> endpoints;
  if (node->kind == RATreeNodeKind::Leaf) {
    endpoints.push_back({node->operationId, node->workUnitId});
  } else {
    for (int64_t childId : node->childIds) {
      FailureOr<SmallVector<LeafEndpoint>> childEndpoints =
          collectSubtreeEndpoints(context, childId);
      if (failed(childEndpoints))
        return failure();
      endpoints.append(childEndpoints->begin(), childEndpoints->end());
    }
  }
  context.subtreeEndpoints[nodeId] = endpoints;
  return endpoints;
}

FailureOr<int64_t> checkedMulI64(int64_t lhs, int64_t rhs, Operation *anchor,
                                 StringRef description) {
  std::optional<int64_t> result = llvm::checkedMul(lhs, rhs);
  if (!result) {
    anchor->emitError(description) << " overflows a signed 64-bit integer";
    return failure();
  }
  return *result;
}

FailureOr<SmallVector<int64_t>>
collectSubtreeOperations(ReferenceEvaluationContext &context, int64_t nodeId) {
  auto cached = context.subtreeOperations.find(nodeId);
  if (cached != context.subtreeOperations.end())
    return cached->second;

  const StructuralRATreeNode *node = context.nodesById.lookup(nodeId);
  if (!node) {
    context.problem.anchor->emitError(
        "mapping evaluator cannot resolve RA-tree "
        "node ")
        << nodeId;
    return failure();
  }

  SmallVector<int64_t> operations;
  if (node->kind == RATreeNodeKind::Leaf) {
    operations.push_back(node->operationId);
  } else {
    for (int64_t childId : node->childIds) {
      FailureOr<SmallVector<int64_t>> childOperations =
          collectSubtreeOperations(context, childId);
      if (failed(childOperations))
        return failure();
      operations.append(childOperations->begin(), childOperations->end());
    }
  }
  context.subtreeOperations[nodeId] = operations;
  return operations;
}

FailureOr<int64_t> estimateLeafWork(const MappingProblem &problem,
                                    const ResourceAllocationTree &tree,
                                    const StructuralRATreeNode &node) {
  const ComputeOperation &operation =
      problem.graph.operations[node.operationId];
  const MappingWorkUnit *workUnit = nullptr;
  if (node.workUnitId >= 0) {
    for (const MappingWorkUnit &candidate : tree.workUnits) {
      if (candidate.id == node.workUnitId) {
        workUnit = &candidate;
        break;
      }
    }
    if (!workUnit) {
      problem.anchor->emitError("mapping evaluator cannot resolve work unit ")
          << node.workUnitId;
      return failure();
    }
  }

  int64_t workItems = 1;
  for (auto [index, dimension] : llvm::enumerate(operation.iterationDomain)) {
    int64_t extent =
        workUnit ? workUnit->iterationSizes[index] : dimension.staticExtent;
    if (ShapedType::isDynamic(extent)) {
      operation.operation->emitError(
          "reference mapping evaluation requires static iteration extents");
      return failure();
    }
    if (extent <= 0) {
      operation.operation->emitError(
          "reference mapping evaluation requires positive iteration extents");
      return failure();
    }
    FailureOr<int64_t> product = checkedMulI64(
        workItems, extent, problem.anchor, "mapping leaf work estimate");
    if (failed(product))
      return failure();
    workItems = *product;
  }
  return workItems;
}

double cyclesToNanoseconds(int64_t cycles,
                           const MappingHardwareModel &hardware) {
  return static_cast<double>(cycles) * 1.0e9 /
         static_cast<double>(hardware.clockFrequencyHz);
}

FailureOr<double>
estimateTransferNanoseconds(int64_t bytes, const MappingHardwareModel &hardware,
                            Operation *anchor) {
  FailureOr<int64_t> bits =
      checkedMulI64(bytes, int64_t{8}, anchor, "mapping transfer bit count");
  if (failed(bits))
    return failure();
  int64_t words = llvm::divideCeil(*bits, hardware.networkWordBits);
  FailureOr<int64_t> cycles = checkedMulI64(
      words, hardware.networkHopCycles, anchor, "mapping transfer cycle count");
  if (failed(cycles))
    return failure();
  return cyclesToNanoseconds(*cycles, hardware);
}

FailureOr<std::map<ChildEdge, int64_t>>
collectChildEdges(ReferenceEvaluationContext &context,
                  const StructuralRATreeNode &node) {
  std::map<LeafEndpoint, int64_t> endpointToChild;
  std::map<int64_t, std::set<int64_t>> operationToChildren;
  for (auto [childIndex, childId] : llvm::enumerate(node.childIds)) {
    FailureOr<SmallVector<LeafEndpoint>> endpoints =
        collectSubtreeEndpoints(context, childId);
    if (failed(endpoints))
      return failure();
    for (LeafEndpoint endpoint : *endpoints) {
      endpointToChild[endpoint] = static_cast<int64_t>(childIndex);
      operationToChildren[endpoint.first].insert(
          static_cast<int64_t>(childIndex));
    }
  }

  std::map<ChildEdge, int64_t> edgeBytes;
  for (const auto &[sourceOperationId, sourceOperationChildren] :
       operationToChildren) {
    auto indexedEdges =
        context.workUnitEdgesBySourceOperation.find(sourceOperationId);
    if (indexedEdges == context.workUnitEdgesBySourceOperation.end())
      continue;
    for (const MappingWorkUnitEdge *edge : indexedEdges->second) {
      auto targetOperationChildren =
          operationToChildren.find(edge->targetOperationId);
      if (targetOperationChildren == operationToChildren.end())
        continue;

      std::set<int64_t> sourceChildren;
      std::set<int64_t> targetChildren;
      if (edge->sourceWorkUnitId >= 0) {
        auto child = endpointToChild.find(
            {edge->sourceOperationId, edge->sourceWorkUnitId});
        if (child != endpointToChild.end())
          sourceChildren.insert(child->second);
      } else {
        sourceChildren = sourceOperationChildren;
      }
      if (edge->targetWorkUnitId >= 0) {
        auto child = endpointToChild.find(
            {edge->targetOperationId, edge->targetWorkUnitId});
        if (child != endpointToChild.end())
          targetChildren.insert(child->second);
      } else {
        targetChildren = targetOperationChildren->second;
      }

      for (int64_t source : sourceChildren) {
        for (int64_t target : targetChildren) {
          if (source == target)
            continue;
          FailureOr<int64_t> bytes = checkedAddI64(
              edgeBytes[{source, target}], edge->byteSize,
              context.problem.anchor, "mapping work-unit edge byte count");
          if (failed(bytes))
            return failure();
          edgeBytes[{source, target}] = *bytes;
        }
      }
    }
  }

  std::set<int64_t> relevantTensorIds;
  for (const auto &[operationId, children] : operationToChildren) {
    (void)children;
    for (int64_t tensorId :
         context.problem.graph.operations[operationId].outputTensors)
      relevantTensorIds.insert(tensorId);
  }
  for (int64_t tensorId : relevantTensorIds) {
    const ComputeTensor &tensor = context.problem.graph.tensors[tensorId];
    std::set<ChildEdge> tensorEdges;
    for (int64_t producerId : tensor.producerOperations) {
      for (int64_t consumerId : tensor.consumerOperations) {
        if (context.refinedOperationEdges.contains({producerId, consumerId}))
          continue;
        auto producerChildren = operationToChildren.find(producerId);
        auto consumerChildren = operationToChildren.find(consumerId);
        if (producerChildren == operationToChildren.end() ||
            consumerChildren == operationToChildren.end())
          continue;
        for (int64_t producerChild : producerChildren->second) {
          for (int64_t consumerChild : consumerChildren->second) {
            if (producerChild != consumerChild)
              tensorEdges.insert({producerChild, consumerChild});
          }
        }
      }
    }
    if (tensorEdges.empty())
      continue;
    if (tensor.byteSize < 0) {
      context.problem.anchor->emitError(
          "reference mapping evaluation requires static byte sizes for "
          "crossing tensors");
      return failure();
    }
    for (ChildEdge edge : tensorEdges) {
      FailureOr<int64_t> bytes = checkedAddI64(edgeBytes[edge], tensor.byteSize,
                                               context.problem.anchor,
                                               "mapping child-edge byte count");
      if (failed(bytes))
        return failure();
      edgeBytes[edge] = *bytes;
    }
  }
  return edgeBytes;
}

MappingNodeEvaluation makeInfeasible(int64_t nodeId, StringRef reason) {
  MappingNodeEvaluation evaluation;
  evaluation.nodeId = nodeId;
  evaluation.feasible = false;
  evaluation.infeasibilityReason = reason.str();
  return evaluation;
}

FailureOr<MappingNodeEvaluation>
evaluateAnalogMVMLeaf(const MappingProblem &problem,
                      const StructuralRATreeNode &node,
                      const ComputeOperation &operation) {
  if (!operation.analogMVM) {
    operation.operation->emitError(
        "analog MVM compute record is missing logical matrix geometry");
    return failure();
  }

  FailureOr<GolemMVMPlan> plan =
      planGolemMVM(operation.operation, operation.analogMVM->outputRows,
                   operation.analogMVM->inputColumns,
                   problem.hardware.arrayRows, problem.hardware.arrayCols);
  if (failed(plan))
    return failure();

  int64_t requiredCores =
      llvm::divideCeil(plan->arrayCount, problem.hardware.arraysPerCore);
  FailureOr<int64_t> coreCount = problem.hardware.getCoreCount(problem.anchor);
  if (failed(coreCount))
    return failure();
  if (requiredCores > *coreCount) {
    return makeInfeasible(
        node.id,
        (Twine("logical MVM requires ") + Twine(plan->arrayCount) +
         " arrays across " + Twine(requiredCores) +
         " cores, but the mesh provides " + Twine(*coreCount) + " cores")
            .str());
  }

  // Cores transfer concurrently. The shared I/O limit applies to the arrays
  // resident on one core, so the busiest core determines leaf I/O latency.
  int64_t arraysOnBusiestCore =
      std::min(plan->arrayCount, problem.hardware.arraysPerCore);
  FailureOr<int64_t> elementsPerArray =
      checkedAddI64(problem.hardware.arrayCols, problem.hardware.arrayRows,
                    problem.anchor, "analog MVM per-array I/O element count");
  if (failed(elementsPerArray))
    return failure();
  FailureOr<int64_t> ioElements =
      checkedMulI64(arraysOnBusiestCore, *elementsPerArray, problem.anchor,
                    "analog MVM shared I/O element count");
  if (failed(ioElements))
    return failure();
  FailureOr<int64_t> ioBits = checkedMulI64(
      *ioElements, int64_t{32}, problem.anchor, "analog MVM shared I/O bits");
  if (failed(ioBits))
    return failure();
  int64_t ioCycles =
      llvm::divideCeil(*ioBits, problem.hardware.analogIOBitsPerCycle);

  int64_t vectorWidth = problem.hardware.digitalVectorBitsPerCycle / 32;
  int64_t effectiveOpsPerCycle =
      std::max(problem.hardware.digitalIssueWidth, vectorWidth);
  int64_t recombinationCycles =
      llvm::divideCeil(plan->recombinationAddOps, effectiveOpsPerCycle);
  FailureOr<int64_t> supportCycles =
      checkedAddI64(ioCycles, recombinationCycles, problem.anchor,
                    "analog MVM support cycle count");
  if (failed(supportCycles))
    return failure();

  MappingNodeEvaluation result;
  result.nodeId = node.id;
  result.requiredResourceUnits = requiredCores;
  result.pipelineStages = 1;
  result.estimatedLatencyNs =
      (static_cast<double>(problem.hardware.analogMVMLatencyNs) +
       cyclesToNanoseconds(*supportCycles, problem.hardware)) *
      static_cast<double>(node.workGroupCount);
  return result;
}

FailureOr<MappingNodeEvaluation>
evaluatePhysicalMVMLeaf(const MappingProblem &problem,
                        const StructuralRATreeNode &node,
                        const ComputeOperation &operation) {
  if (!operation.analogMVM) {
    operation.operation->emitError(
        "physical MVM compute record is missing array geometry");
    return failure();
  }

  FailureOr<int64_t> ioElements = checkedAddI64(
      operation.analogMVM->inputColumns, operation.analogMVM->outputRows,
      problem.anchor, "physical MVM I/O element count");
  if (failed(ioElements))
    return failure();
  FailureOr<int64_t> ioBits = checkedMulI64(
      *ioElements, int64_t{32}, problem.anchor, "physical MVM I/O bit count");
  if (failed(ioBits))
    return failure();
  int64_t ioCycles =
      llvm::divideCeil(*ioBits, problem.hardware.analogIOBitsPerCycle);

  MappingNodeEvaluation result;
  result.nodeId = node.id;
  result.requiredResourceUnits = 1;
  result.pipelineStages = 1;
  result.estimatedLatencyNs =
      (static_cast<double>(problem.hardware.analogMVMLatencyNs) +
       cyclesToNanoseconds(ioCycles, problem.hardware)) *
      static_cast<double>(node.workGroupCount);
  return result;
}

FailureOr<MappingNodeEvaluation>
evaluateNode(ReferenceEvaluationContext &context, int64_t nodeId) {
  auto cached = context.evaluations.find(nodeId);
  if (cached != context.evaluations.end())
    return cached->second;

  const StructuralRATreeNode *node = context.nodesById.lookup(nodeId);
  if (!node) {
    context.problem.anchor->emitError(
        "mapping evaluator cannot resolve RA-tree "
        "node ")
        << nodeId;
    return failure();
  }
  MappingNodeEvaluation result;
  result.nodeId = nodeId;
  result.pipelineStages = 1;

  if (node->kind == RATreeNodeKind::Leaf) {
    const ComputeOperation &operation =
        context.problem.graph.operations[node->operationId];
    if (operation.kind == ComputeOperationKind::LogicalMVM) {
      FailureOr<MappingNodeEvaluation> analog =
          evaluateAnalogMVMLeaf(context.problem, *node, operation);
      if (failed(analog))
        return failure();
      context.evaluations[nodeId] = *analog;
      return *analog;
    }
    if (operation.kind == ComputeOperationKind::PhysicalMVM) {
      FailureOr<MappingNodeEvaluation> physical =
          evaluatePhysicalMVMLeaf(context.problem, *node, operation);
      if (failed(physical))
        return failure();
      context.evaluations[nodeId] = *physical;
      return *physical;
    }

    FailureOr<int64_t> workItems =
        estimateLeafWork(context.problem, context.tree, *node);
    if (failed(workItems))
      return failure();
    int64_t vectorWidth =
        context.problem.hardware.digitalVectorBitsPerCycle / 32;
    int64_t effectiveOpsPerCycle =
        std::max(context.problem.hardware.digitalIssueWidth, vectorWidth);
    int64_t cycles = llvm::divideCeil(*workItems, effectiveOpsPerCycle);
    FailureOr<int64_t> groupedCycles =
        checkedMulI64(cycles, node->workGroupCount, context.problem.anchor,
                      "mapping leaf grouped cycle count");
    if (failed(groupedCycles))
      return failure();
    result.estimatedLatencyNs =
        cyclesToNanoseconds(*groupedCycles, context.problem.hardware);
    result.requiredResourceUnits = 1;
    context.evaluations[nodeId] = result;
    return result;
  }

  SmallVector<MappingNodeEvaluation> children;
  children.reserve(node->childIds.size());
  for (int64_t childId : node->childIds) {
    FailureOr<MappingNodeEvaluation> child = evaluateNode(context, childId);
    if (failed(child))
      return failure();
    if (!child->feasible) {
      result = makeInfeasible(nodeId, child->infeasibilityReason);
      context.evaluations[nodeId] = result;
      return result;
    }
    children.push_back(*child);
  }

  FailureOr<std::map<ChildEdge, int64_t>> edgeBytes =
      collectChildEdges(context, *node);
  if (failed(edgeBytes))
    return failure();

  for (const MappingNodeEvaluation &child : children) {
    FailureOr<int64_t> bytes = checkedAddI64(
        result.crossingBytes, child.crossingBytes, context.problem.anchor,
        "mapping subtree crossing-byte count");
    if (failed(bytes))
      return failure();
    result.crossingBytes = *bytes;
    result.estimatedCommunicationNs += child.estimatedCommunicationNs;
  }
  for (const auto &[edge, bytes] : *edgeBytes) {
    (void)edge;
    FailureOr<int64_t> total =
        checkedAddI64(result.crossingBytes, bytes, context.problem.anchor,
                      "mapping cut crossing-byte count");
    if (failed(total))
      return failure();
    result.crossingBytes = *total;
  }

  if (node->kind == RATreeNodeKind::TemporalCut) {
    result.requiredResourceUnits = 0;
    result.pipelineStages = 0;
    SmallVector<double> childOffsets;
    childOffsets.reserve(children.size());
    for (const MappingNodeEvaluation &child : children) {
      childOffsets.push_back(result.estimatedLatencyNs);
      result.estimatedLatencyNs += child.estimatedLatencyNs;
      result.requiredResourceUnits =
          std::max(result.requiredResourceUnits, child.requiredResourceUnits);
      FailureOr<int64_t> stages = checkedAddI64(
          result.pipelineStages, child.pipelineStages, context.problem.anchor,
          "mapping temporal pipeline-stage count");
      if (failed(stages))
        return failure();
      result.pipelineStages = *stages;
    }
    context.childStartOffsets[nodeId] = std::move(childOffsets);
  } else {
    DenseMap<int64_t, int64_t> childByLaneBindingGroup;
    for (auto [childIndex, childId] : llvm::enumerate(node->childIds)) {
      FailureOr<SmallVector<int64_t>> childOperations =
          collectSubtreeOperations(context, childId);
      if (failed(childOperations))
        return failure();
      for (int64_t operationId : *childOperations) {
        const std::optional<int64_t> &bindingGroup =
            context.problem.graph.operations[operationId].laneBindingGroup;
        if (!bindingGroup)
          continue;
        auto [binding, inserted] = childByLaneBindingGroup.try_emplace(
            *bindingGroup, static_cast<int64_t>(childIndex));
        if (!inserted && binding->second != static_cast<int64_t>(childIndex)) {
          result = makeInfeasible(
              nodeId, (Twine("spatial cut splits analog lane-binding group ") +
                       Twine(*bindingGroup) + " across multiple resource lanes")
                          .str());
          context.evaluations[nodeId] = result;
          return result;
        }
      }
    }

    FailureOr<SmallVector<int64_t>> subtreeOperations =
        collectSubtreeOperations(context, nodeId);
    if (failed(subtreeOperations))
      return failure();
    bool isSetupFrontier =
        llvm::all_of(*subtreeOperations, [&](int64_t operationId) {
          return context.problem.graph.operations[operationId].kind ==
                 ComputeOperationKind::MatrixSetup;
        });

    std::optional<int64_t> mvmWaveId;
    bool isMVMWaveFrontier = !subtreeOperations->empty();
    int64_t mvmWaveAnalogLaneCount = 0;
    for (int64_t operationId : *subtreeOperations) {
      const ComputeOperation &operation =
          context.problem.graph.operations[operationId];
      if (!operation.mvmWaveId ||
          (operation.kind != ComputeOperationKind::VectorTile &&
           operation.kind != ComputeOperationKind::PhysicalMVM)) {
        isMVMWaveFrontier = false;
        break;
      }
      if (operation.kind == ComputeOperationKind::PhysicalMVM)
        ++mvmWaveAnalogLaneCount;
      if (!mvmWaveId)
        mvmWaveId = operation.mvmWaveId;
      else if (*mvmWaveId != *operation.mvmWaveId) {
        isMVMWaveFrontier = false;
        break;
      }
    }
    isMVMWaveFrontier &= mvmWaveAnalogLaneCount > 0;

    if (isSetupFrontier || isMVMWaveFrontier) {
      int64_t analogLaneCount = static_cast<int64_t>(
          context.problem.logicalTileShape.analogLanes.size());
      if (analogLaneCount <= 0) {
        context.problem.anchor->emitError(
            "analog frontier requires at least one logical analog lane");
        return failure();
      }
      int64_t requiredAnalogLanes =
          isSetupFrontier ? static_cast<int64_t>(subtreeOperations->size())
                          : mvmWaveAnalogLaneCount;
      result.requiredResourceUnits =
          isMVMWaveFrontier &&
                  context.problem.mvmBodyPolicy == MVMBodyPolicy::Spread
              ? requiredAnalogLanes
              : llvm::divideCeil(requiredAnalogLanes, analogLaneCount);
    } else {
      result.requiredResourceUnits = 0;
      for (const MappingNodeEvaluation &child : children) {
        FailureOr<int64_t> resources = checkedAddI64(
            result.requiredResourceUnits, child.requiredResourceUnits,
            context.problem.anchor, "mapping spatial resource count");
        if (failed(resources))
          return failure();
        result.requiredResourceUnits = *resources;
      }
    }

    FailureOr<int64_t> coreCount =
        context.problem.hardware.getCoreCount(context.problem.anchor);
    if (failed(coreCount))
      return failure();
    if (result.requiredResourceUnits > *coreCount) {
      result = makeInfeasible(
          nodeId, (Twine("spatial cut requires ") +
                   Twine(result.requiredResourceUnits) +
                   " resource units but the mesh provides " + Twine(*coreCount))
                      .str());
      context.evaluations[nodeId] = result;
      return result;
    }

    SmallVector<SmallVector<int64_t>> successors(children.size());
    SmallVector<SmallVector<int64_t>> predecessors(children.size());
    SmallVector<int64_t> indegree(children.size(), 0);
    std::map<ChildEdge, double> transferNs;
    for (const auto &[edge, bytes] : *edgeBytes) {
      int64_t source = edge.first;
      int64_t destination = edge.second;
      successors[source].push_back(destination);
      predecessors[destination].push_back(source);
      ++indegree[destination];
      FailureOr<double> transfer = estimateTransferNanoseconds(
          bytes, context.problem.hardware, context.problem.anchor);
      if (failed(transfer))
        return failure();
      transferNs[edge] = *transfer;
      result.estimatedCommunicationNs += *transfer;
    }

    std::set<int64_t> ready;
    for (auto [childIndex, degree] : llvm::enumerate(indegree)) {
      if (degree == 0)
        ready.insert(static_cast<int64_t>(childIndex));
    }
    SmallVector<double> finishNs(children.size(), 0.0);
    SmallVector<double> startOffsets(children.size(), 0.0);
    SmallVector<int64_t> stage(children.size(), 0);
    int64_t visited = 0;
    while (!ready.empty()) {
      int64_t childIndex = *ready.begin();
      ready.erase(ready.begin());
      ++visited;

      double startNs = 0.0;
      for (int64_t predecessor : predecessors[childIndex]) {
        startNs = std::max(startNs, finishNs[predecessor] +
                                        transferNs[{predecessor, childIndex}]);
        stage[childIndex] = std::max(stage[childIndex], stage[predecessor] + 1);
      }
      startOffsets[childIndex] = startNs;
      finishNs[childIndex] = startNs + children[childIndex].estimatedLatencyNs;
      result.estimatedLatencyNs =
          std::max(result.estimatedLatencyNs, finishNs[childIndex]);
      result.pipelineStages =
          std::max(result.pipelineStages, stage[childIndex] + 1);

      for (int64_t successor : successors[childIndex]) {
        if (--indegree[successor] == 0)
          ready.insert(successor);
      }
    }
    if (visited != static_cast<int64_t>(children.size())) {
      result = makeInfeasible(
          nodeId, "spatial-cut child dependency graph contains a cycle");
      context.evaluations[nodeId] = result;
      return result;
    }
    context.childStartOffsets[nodeId] = std::move(startOffsets);
  }

  result.estimatedLatencyNs *= static_cast<double>(node->workGroupCount);
  if (!std::isfinite(result.estimatedLatencyNs) ||
      !std::isfinite(result.estimatedCommunicationNs)) {
    context.problem.anchor->emitError(
        "mapping evaluation produced a non-finite cost");
    return failure();
  }
  context.evaluations[nodeId] = result;
  return result;
}

LogicalResult assignRealizationTimes(ReferenceEvaluationContext &context,
                                     MappingRealization &realization) {
  DenseMap<int64_t, MappingLeafAssignment *> assignments;
  for (MappingLeafAssignment &assignment : realization.leafAssignments)
    assignments[assignment.leafId] = &assignment;

  std::function<LogicalResult(int64_t, double)> assignNode =
      [&](int64_t nodeId, double startNs) -> LogicalResult {
    const StructuralRATreeNode *node = context.nodesById.lookup(nodeId);
    if (!node) {
      context.problem.anchor->emitError(
          "mapping timing cannot resolve RA-tree node ")
          << nodeId;
      return failure();
    }
    const MappingNodeEvaluation &evaluation =
        context.evaluations.lookup(nodeId);
    if (node->kind == RATreeNodeKind::Leaf) {
      MappingLeafAssignment *assignment = assignments.lookup(nodeId);
      if (!assignment) {
        context.problem.anchor->emitError(
            "mapping timing cannot resolve assignment for RA-tree leaf ")
            << nodeId;
        return failure();
      }
      assignment->startNs = startNs;
      assignment->finishNs = startNs + evaluation.estimatedLatencyNs;
      return success();
    }

    auto offsets = context.childStartOffsets.find(nodeId);
    if (offsets == context.childStartOffsets.end() ||
        offsets->second.size() != node->childIds.size()) {
      context.problem.anchor->emitError(
          "mapping timing has no child schedule for RA-tree node ")
          << nodeId;
      return failure();
    }
    for (auto [childIndex, childId] : llvm::enumerate(node->childIds)) {
      if (failed(assignNode(childId, startNs + offsets->second[childIndex])))
        return failure();
    }
    return success();
  };

  return assignNode(context.tree.rootId, 0.0);
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<MappingEvaluation>
ReferenceMappingEvaluator::evaluate(const MappingProblem &problem,
                                    const ResourceAllocationTree &tree) const {
  if (!problem.anchor)
    return failure();
  if (failed(verifyResourceAllocationTree(tree, problem.graph, problem.anchor)))
    return failure();

  ReferenceEvaluationContext context{problem, tree};
  for (const StructuralRATreeNode &node : tree.nodes)
    context.nodesById[node.id] = &node;

  FailureOr<MappingNodeEvaluation> root = evaluateNode(context, tree.rootId);
  if (failed(root))
    return failure();

  MappingEvaluation evaluation;
  evaluation.feasible = root->feasible;
  evaluation.estimatedLatencyNs = root->estimatedLatencyNs;
  evaluation.crossingBytes = root->crossingBytes;
  evaluation.estimatedCommunicationNs = root->estimatedCommunicationNs;
  evaluation.requiredResourceUnits = root->requiredResourceUnits;
  evaluation.pipelineStages = root->pipelineStages;
  evaluation.infeasibilityReason = root->infeasibilityReason;
  evaluation.nodes.reserve(tree.nodes.size());
  for (const StructuralRATreeNode &node : tree.nodes)
    evaluation.nodes.push_back(context.evaluations.lookup(node.id));
  if (evaluation.feasible && problem.requireRealization) {
    FailureOr<MappingRealization> realization =
        realizeResourceAllocationTree(problem, tree);
    if (failed(realization))
      return failure();
    if (!realization->feasible) {
      evaluation.feasible = false;
      evaluation.infeasibilityReason = realization->infeasibilityReason;
    } else {
      if (failed(assignRealizationTimes(context, *realization)))
        return failure();
      evaluation.realization = std::move(*realization);
    }
  }
  return evaluation;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
