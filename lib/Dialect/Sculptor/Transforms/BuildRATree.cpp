#include "sculptor-mlir/Dialect/Sculptor/Transforms/BuildRATree.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CheckedArithmetic.h"

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

FailureOr<SmallVector<int64_t>>
parseI64Array(ArrayAttr values, Operation *anchor, StringRef description) {
  SmallVector<int64_t> result;
  result.reserve(values.size());
  for (Attribute value : values) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer) {
      anchor->emitError(description) << " must contain integer attributes";
      return failure();
    }
    result.push_back(integer.getInt());
  }
  return result;
}

class ExpandedDigitalWorkTreeBuilder {
public:
  ExpandedDigitalWorkTreeBuilder(const ComputeGraph &graph,
                                 const ResourceAllocationTree &source,
                                 Operation *anchor)
      : graph(graph), anchor(anchor) {
    for (const StructuralRATreeNode &node : source.nodes)
      sourceNodes[node.id] = &node;
    for (const ComputeOperation &operation : graph.operations) {
      for (Operation *member : operation.members)
        operationIds[member] = operation.id;
    }
  }

  FailureOr<ResourceAllocationTree> build(int64_t sourceRootId) {
    FailureOr<int64_t> root = clone(sourceRootId);
    if (failed(root))
      return failure();
    tree.rootId = *root;
    tree.nodes[tree.rootId].parentId = -1;
    if (failed(buildDestinationStyleEdges()))
      return failure();
    return std::move(tree);
  }

private:
  int64_t addLeaf(int64_t operationId, int64_t workUnitId,
                  int64_t workGroupCount) {
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    tree.nodes.push_back({id,
                          RATreeNodeKind::Leaf,
                          -1,
                          {},
                          operationId,
                          workUnitId,
                          workGroupCount});
    return id;
  }

  int64_t addCut(RATreeNodeKind kind, ArrayRef<int64_t> children,
                 int64_t workGroupCount = 1) {
    assert(children.size() >= 2 && "cuts require at least two children");
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    StructuralRATreeNode node;
    node.id = id;
    node.kind = kind;
    node.childIds.assign(children.begin(), children.end());
    node.workGroupCount = workGroupCount;
    tree.nodes.push_back(std::move(node));
    for (int64_t childId : children)
      tree.nodes[childId].parentId = id;
    return id;
  }

  FailureOr<MappingWorkUnit> parseWorkUnit(MappingWorkUnitAttr attribute,
                                           int64_t operationId) {
    if (attribute.getOperationId().getInt() != operationId) {
      graph.operations[operationId].operation->emitError(
          "expanded digital work references a stale operation ID");
      return failure();
    }

    FailureOr<SmallVector<int64_t>> resultOffsets =
        parseI64Array(attribute.getResultOffsets(), anchor,
                      "expanded digital work result offsets");
    FailureOr<SmallVector<int64_t>> resultSizes =
        parseI64Array(attribute.getResultSizes(), anchor,
                      "expanded digital work result sizes");
    FailureOr<SmallVector<int64_t>> iterationOffsets =
        parseI64Array(attribute.getIterationOffsets(), anchor,
                      "expanded digital work iteration offsets");
    FailureOr<SmallVector<int64_t>> iterationSizes =
        parseI64Array(attribute.getIterationSizes(), anchor,
                      "expanded digital work iteration sizes");
    if (failed(resultOffsets) || failed(resultSizes) ||
        failed(iterationOffsets) || failed(iterationSizes))
      return failure();

    MappingWorkUnit workUnit;
    workUnit.id = attribute.getId().getInt();
    workUnit.operationId = operationId;
    workUnit.resultNumber = attribute.getResultNumber().getInt();
    workUnit.resultOffsets = std::move(*resultOffsets);
    workUnit.resultSizes = std::move(*resultSizes);
    workUnit.iterationOffsets = std::move(*iterationOffsets);
    workUnit.iterationSizes = std::move(*iterationSizes);
    return workUnit;
  }

  FailureOr<int64_t> clone(int64_t sourceNodeId) {
    const StructuralRATreeNode *source = sourceNodes.lookup(sourceNodeId);
    if (!source) {
      anchor->emitError("cannot resolve baseline RA-tree node ")
          << sourceNodeId;
      return failure();
    }
    if (source->kind == RATreeNodeKind::Leaf) {
      Operation *operation = graph.operations[source->operationId].operation;
      auto expanded =
          operation->getAttrOfType<ArrayAttr>(kExpandedDigitalWorkAttrName);
      if (!expanded)
        return addLeaf(source->operationId, source->workUnitId,
                       source->workGroupCount);

      SmallVector<int64_t> workerLeaves;
      workerLeaves.reserve(expanded.size());
      for (Attribute value : expanded) {
        auto workUnitAttribute = dyn_cast<MappingWorkUnitAttr>(value);
        if (!workUnitAttribute) {
          operation->emitError(
              "expanded digital work must contain mapping work-unit "
              "attributes");
          return failure();
        }
        FailureOr<MappingWorkUnit> workUnit =
            parseWorkUnit(workUnitAttribute, source->operationId);
        if (failed(workUnit))
          return failure();
        int64_t workUnitId = workUnit->id;
        tree.workUnits.push_back(std::move(*workUnit));
        workerLeaves.push_back(
            addLeaf(source->operationId, workUnitId, source->workGroupCount));
      }
      if (workerLeaves.size() < 2) {
        operation->emitError(
            "expanded digital work requires at least two work units");
        return failure();
      }
      return addCut(RATreeNodeKind::SpatialCut, workerLeaves);
    }

    SmallVector<int64_t> children;
    children.reserve(source->childIds.size());
    for (int64_t childId : source->childIds) {
      FailureOr<int64_t> child = clone(childId);
      if (failed(child))
        return failure();
      children.push_back(*child);
    }
    return addCut(source->kind, children, source->workGroupCount);
  }

  FailureOr<int64_t> getWorkUnitByteSize(const MappingWorkUnit &workUnit) {
    auto resultType = dyn_cast<RankedTensorType>(
        graph.operations[workUnit.operationId]
            .operation->getResult(workUnit.resultNumber)
            .getType());
    if (!resultType || !resultType.getElementType().isIntOrFloat())
      return failure();
    unsigned bitWidth = resultType.getElementType().getIntOrFloatBitWidth();
    if (bitWidth == 0 || bitWidth % 8 != 0)
      return failure();

    int64_t bytes = static_cast<int64_t>(bitWidth / 8);
    for (int64_t size : workUnit.resultSizes) {
      std::optional<int64_t> next = llvm::checkedMul(bytes, size);
      if (!next)
        return failure();
      bytes = *next;
    }
    return bytes;
  }

  LogicalResult buildDestinationStyleEdges() {
    DenseMap<int64_t, SmallVector<const MappingWorkUnit *>> unitsByOperation;
    for (const MappingWorkUnit &workUnit : tree.workUnits)
      unitsByOperation[workUnit.operationId].push_back(&workUnit);

    for (const ComputeOperation &producer : graph.operations) {
      auto producerUnits = unitsByOperation.find(producer.id);
      if (producerUnits == unitsByOperation.end() ||
          producer.operation->getNumResults() != 1)
        continue;

      Value result = producer.operation->getResult(0);
      for (OpOperand &use : result.getUses()) {
        auto consumer = dyn_cast<linalg::LinalgOp>(use.getOwner());
        if (!consumer || !llvm::is_contained(consumer.getDpsInits(), result))
          continue;
        auto operationId = operationIds.find(consumer.getOperation());
        if (operationId == operationIds.end())
          continue;
        auto consumerUnits = unitsByOperation.find(operationId->second);
        if (consumerUnits == unitsByOperation.end())
          continue;

        for (const MappingWorkUnit *source : producerUnits->second) {
          const MappingWorkUnit *target = nullptr;
          for (const MappingWorkUnit *candidate : consumerUnits->second) {
            if (source->resultOffsets == candidate->resultOffsets &&
                source->resultSizes == candidate->resultSizes) {
              target = candidate;
              break;
            }
          }
          if (!target)
            continue;
          FailureOr<int64_t> byteSize = getWorkUnitByteSize(*source);
          if (failed(byteSize)) {
            producer.operation->emitError(
                "cannot compute a static byte size for an expanded digital "
                "work-unit dependency");
            return failure();
          }
          tree.workUnitEdges.push_back({producer.id, source->id,
                                        operationId->second, target->id,
                                        *byteSize});
        }
      }
    }
    return success();
  }

  const ComputeGraph &graph;
  Operation *anchor;
  ResourceAllocationTree tree;
  DenseMap<int64_t, const StructuralRATreeNode *> sourceNodes;
  DenseMap<Operation *, int64_t> operationIds;
};

FailureOr<ResourceAllocationTree>
expandDigitalWork(const ComputeGraph &graph,
                  const ResourceAllocationTree &baseline, Operation *anchor) {
  ExpandedDigitalWorkTreeBuilder builder(graph, baseline, anchor);
  return builder.build(baseline.rootId);
}

} // namespace

namespace mlir {
namespace sculptor {

void BuildRATreePass::runOnOperation() {
  bool builtTree = false;
  for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;

    FailureOr<mapping::ComputeGraph> graph = mapping::buildComputeGraph(func);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    if (graph->operations.empty())
      continue;

    for (const mapping::ComputeOperation &operation : graph->operations) {
      if (operation.kind != mapping::ComputeOperationKind::LogicalMVM)
        continue;
      operation.operation->emitError(
          "logical sculptor.mvm must be expanded before RA-tree "
          "construction; run --sculptor-expand-mvm-to-golem first");
      signalPassFailure();
      return;
    }

    FailureOr<mapping::ResourceAllocationTree> tree =
        mapping::buildTemporalBaselineRATree(*graph, func);
    if (failed(tree)) {
      signalPassFailure();
      return;
    }
    FailureOr<mapping::ResourceAllocationTree> expanded =
        expandDigitalWork(*graph, *tree, func);
    if (failed(expanded) || failed(mapping::verifyResourceAllocationTree(
                                *expanded, *graph, func))) {
      signalPassFailure();
      return;
    }
    tree = std::move(expanded);

    Builder builder(&getContext());
    for (const mapping::ComputeOperation &operation : graph->operations) {
      for (Operation *member : operation.members) {
        auto setOptionalI64 = [&](StringRef name,
                                  std::optional<int64_t> value) {
          if (value)
            member->setAttr(name, builder.getI64IntegerAttr(*value));
          else
            member->removeAttr(name);
        };
        member->setAttr(mapping::kMappingOperationIdAttrName,
                        builder.getI64IntegerAttr(operation.id));
        setOptionalI64(mapping::kLaneBindingGroupAttrName,
                       operation.laneBindingGroup);
        setOptionalI64(mapping::kMVMWaveIdAttrName, operation.mvmWaveId);
        setOptionalI64(mapping::kMVMWaveMemberAttrName,
                       operation.mvmWaveMember);
        setOptionalI64(mapping::kMVMWaveSizeAttrName, operation.mvmWaveSize);
      }
    }

    std::string fingerprint = mapping::computeGraphFingerprint(*graph);
    func->removeAttr(mapping::kMappingPlanAttrName);
    func->removeAttr(mapping::kLogicalTileGraphAttrName);
    func->removeAttr(mapping::kLogicalTilePlacementAttrName);
    func->removeAttr(mapping::kLogicalTileAnnealingTraceAttrName);
    func->setAttr(mapping::kRATreeAttrName,
                  mapping::serializeResourceAllocationTree(
                      &getContext(), *tree, *graph, fingerprint));
    builtTree = true;
  }

  if (!builtTree) {
    getOperation().emitError(
        "expected at least one supported top-level compute operation "
        "(TilingInterface, sculptor.mvm, or expanded realization stage)");
    signalPassFailure();
  }
}

void registerBuildRATreePass() { PassRegistration<BuildRATreePass>(); }

} // namespace sculptor
} // namespace mlir
