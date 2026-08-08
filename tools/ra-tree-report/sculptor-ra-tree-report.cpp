#include "RATreeReportHTML.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorOps.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

llvm::cl::OptionCategory reportCategory("Sculptor RA-tree report options");

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input MLIR>"),
                                         llvm::cl::Required,
                                         llvm::cl::cat(reportCategory));

llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output self-contained HTML file"),
                   llvm::cl::value_desc("filename"), llvm::cl::init(""),
                   llvm::cl::cat(reportCategory));

llvm::cl::opt<std::string>
    jsonOutputFilename("json-output",
                       llvm::cl::desc("Optional standalone JSON report"),
                       llvm::cl::value_desc("filename"), llvm::cl::init(""),
                       llvm::cl::cat(reportCategory));

llvm::cl::opt<std::string> expandedIRFilename(
    "expanded-ir",
    llvm::cl::desc("Optional Golem-expanded MLIR to overlay on RA leaves"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""),
    llvm::cl::cat(reportCategory));

llvm::cl::opt<std::string> reportTitle("title", llvm::cl::desc("Report title"),
                                       llvm::cl::value_desc("text"),
                                       llvm::cl::init(""),
                                       llvm::cl::cat(reportCategory));

llvm::cl::opt<std::string>
    functionFilter("function",
                   llvm::cl::desc("Export only this function symbol"),
                   llvm::cl::value_desc("symbol"), llvm::cl::init(""),
                   llvm::cl::cat(reportCategory));

struct ReportEdge {
  int64_t source = -1;
  int64_t target = -1;
  SmallVector<int64_t> tensorIds;
  int64_t byteSize = 0;
  bool hasUnknownByteSize = false;
};

struct RealizationStage {
  int64_t id = -1;
  int64_t operationId = -1;
  int64_t raLeafId = -1;
  int64_t stageIndex = -1;
  std::string kind;
  std::string name;
  std::string location;
  std::string mlir;
  int64_t inputCount = 0;
  int64_t outputCount = 0;
  int64_t arraySetCount = 0;
  int64_t arrayLoadCount = 0;
  int64_t arrayExecuteCount = 0;
  int64_t arrayStoreCount = 0;
  SmallVector<int64_t> predecessorStageIds;
  SmallVector<std::pair<std::string, std::string>> attributes;
};

struct ReportFunction {
  func::FuncOp function;
  RATreeAttr treeAttr;
  MappingPlanAttr planAttr;
  ResourceAllocationTree tree;
  ComputeGraph graph;
  std::optional<LogicalTileGraph> logicalTileGraph;
  std::optional<LogicalTilePlacementPlan> logicalTilePlacement;
  SmallVector<ReportEdge> edges;
  SmallVector<RealizationStage, 0> realizationStages;
};

std::string printType(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

std::string printAttribute(Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return text;
}

std::string printLocation(Location location) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  location.print(stream);
  return text;
}

std::string printOperationGroup(ArrayRef<Operation *> operations) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  OpPrintingFlags flags;
  flags.elideLargeElementsAttrs().elideLargeResourceString();
  for (Operation *operation : operations) {
    operation->print(stream, flags);
    stream << '\n';
  }
  return text;
}

std::string printComputeOperation(const ComputeOperation &operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  OpPrintingFlags flags;
  flags.elideLargeElementsAttrs().elideLargeResourceString();
  for (Operation *member : operation.members) {
    member->print(stream, flags);
    stream << '\n';
  }
  return text;
}

SmallVector<ReportEdge> buildEdges(const ComputeGraph &graph) {
  std::map<std::pair<int64_t, int64_t>, ReportEdge> edges;
  for (const ComputeTensor &tensor : graph.tensors) {
    for (int64_t producer : tensor.producerOperations) {
      for (int64_t consumer : tensor.consumerOperations) {
        if (producer == consumer)
          continue;
        auto key = std::make_pair(producer, consumer);
        ReportEdge &edge = edges[key];
        edge.source = producer;
        edge.target = consumer;
        edge.tensorIds.push_back(tensor.id);
        if (tensor.byteSize < 0) {
          edge.hasUnknownByteSize = true;
        } else {
          edge.byteSize += tensor.byteSize;
        }
      }
    }
  }

  SmallVector<ReportEdge> result;
  result.reserve(edges.size());
  for (auto &entry : edges)
    result.push_back(std::move(entry.second));
  return result;
}

FailureOr<ReportFunction> buildReportFunction(func::FuncOp function,
                                              RATreeAttr treeAttr) {
  FailureOr<ComputeGraph> graph = buildComputeGraph(function);
  if (failed(graph))
    return failure();

  FailureOr<ResourceAllocationTree> tree =
      deserializeResourceAllocationTree(treeAttr, *graph, function);
  if (failed(tree))
    return failure();

  auto planAttr =
      function->getAttrOfType<MappingPlanAttr>(kMappingPlanAttrName);
  if (planAttr && planAttr.getRaTreeFingerprint().getValue() !=
                      computeRATreeFingerprint(*tree)) {
    function.emitError("mapping-plan RA-tree fingerprint does not match the "
                       "selected structural tree");
    return failure();
  }

  std::optional<LogicalTileGraph> logicalTileGraph;
  auto logicalTileGraphAttr =
      function->getAttrOfType<LogicalTileGraphAttr>(kLogicalTileGraphAttrName);
  if (logicalTileGraphAttr) {
    FailureOr<LogicalTileGraph> deserialized = deserializeLogicalTileGraph(
        logicalTileGraphAttr, *graph, *tree, function);
    if (failed(deserialized))
      return failure();
    logicalTileGraph = std::move(*deserialized);
  }

  std::optional<LogicalTilePlacementPlan> logicalTilePlacement;
  auto placementAttr = function->getAttrOfType<LogicalTilePlacementAttr>(
      kLogicalTilePlacementAttrName);
  auto annealingTraceAttr =
      function->getAttrOfType<LogicalTileAnnealingTraceAttr>(
          kLogicalTileAnnealingTraceAttrName);
  if (annealingTraceAttr && !placementAttr) {
    function.emitError("annealing trace has no physical placement");
    return failure();
  }
  if (placementAttr) {
    if (!logicalTileGraph) {
      function.emitError(
          "physical logical-tile placement has no logical-tile graph");
      return failure();
    }
    LogicalTilePlacementProblem problem{
        *logicalTileGraph,
        {placementAttr.getMeshRows().getInt(),
         placementAttr.getMeshCols().getInt(),
         placementAttr.getArraysPerCore().getInt()},
        function};
    FailureOr<LogicalTilePlacementPlan> deserialized =
        deserializeLogicalTilePlacement(placementAttr, problem);
    if (failed(deserialized))
      return failure();
    if (annealingTraceAttr) {
      FailureOr<LogicalTileAnnealingTrace> trace =
          deserializeLogicalTileAnnealingTrace(annealingTraceAttr,
                                               *deserialized, function);
      if (failed(trace))
        return failure();
      deserialized->annealingTrace = std::move(*trace);
    }
    logicalTilePlacement = std::move(*deserialized);
  }

  ReportFunction report{function,
                        treeAttr,
                        planAttr,
                        std::move(*tree),
                        std::move(*graph),
                        std::move(logicalTileGraph),
                        std::move(logicalTilePlacement),
                        {},
                        {}};
  report.edges = buildEdges(report.graph);
  return report;
}

bool isRealizationMetadata(StringRef name) {
  return name == "sculptor.source_resource" ||
         name.starts_with("sculptor.tile") ||
         name.starts_with("sculptor.vector_tile") ||
         name.starts_with("sculptor.semantic.");
}

LogicalResult
collectExpandedRealization(ModuleOp expandedModule,
                           MutableArrayRef<ReportFunction> reports) {
  for (ReportFunction &report : reports) {
    auto expandedFunction =
        expandedModule.lookupSymbol<func::FuncOp>(report.graph.functionSymbol);
    if (!expandedFunction) {
      expandedModule.emitError("expanded IR has no function @")
          << report.graph.functionSymbol;
      return failure();
    }

    struct ExpandedStageGroup {
      int64_t stageId = -1;
      int64_t operationId = -1;
      SmallVector<Operation *> members;
    };

    SmallVector<ExpandedStageGroup> groups;
    DenseSet<int64_t> seenStageIds;
    for (Operation &operation : expandedFunction.front().without_terminator()) {
      auto stageId =
          operation.getAttrOfType<IntegerAttr>(mapping::kStageIdAttrName);
      auto operationId = operation.getAttrOfType<IntegerAttr>(
          mapping::kMappingOperationIdAttrName);
      if (!stageId || !operationId)
        continue;

      if (groups.empty() || groups.back().stageId != stageId.getInt()) {
        if (!seenStageIds.insert(stageId.getInt()).second) {
          operation.emitError("expanded realization stage is not contiguous: ")
              << stageId.getInt();
          return failure();
        }
        groups.push_back(
            ExpandedStageGroup{stageId.getInt(), operationId.getInt(), {}});
      } else if (groups.back().operationId != operationId.getInt()) {
        operation.emitError("one expanded realization stage references "
                            "multiple logical operations");
        return failure();
      }
      groups.back().members.push_back(&operation);
    }

    DenseMap<int64_t, int64_t> nextStageIndex;
    DenseSet<int64_t> realizedMVMs;
    DenseMap<Operation *, int64_t> reportStageByOperation;
    SmallVector<int64_t> reportStageByGroup;
    DenseMap<int64_t, int64_t> raLeafByOperation;
    for (const StructuralRATreeNode &node : report.tree.nodes) {
      if (node.operationId >= 0)
        raLeafByOperation[node.operationId] = node.id;
    }

    for (const ExpandedStageGroup &group : groups) {
      Operation *anchor = group.members.front();
      const int64_t id = group.operationId;
      if (id < 0 ||
          id >= static_cast<int64_t>(report.graph.operations.size()) ||
          report.graph.operations[id].id != id) {
        anchor->emitError("expanded stage references unknown logical operation "
                          "ID ")
            << id;
        return failure();
      }

      const ComputeOperation &logicalOperation = report.graph.operations[id];
      if (logicalOperation.kind != ComputeOperationKind::LogicalMVM) {
        anchor->emitError("expanded stage references operation ")
            << id << ", which is not a logical sculptor.mvm";
        return failure();
      }

      auto expectedLeafId =
          logicalOperation.operation->getAttrOfType<IntegerAttr>(
              mapping::kRALeafIdAttrName);
      auto structuralLeaf = raLeafByOperation.find(id);
      if (structuralLeaf == raLeafByOperation.end()) {
        anchor->emitError("logical operation has no RA-tree leaf: ") << id;
        return failure();
      }

      const int64_t expectedLeaf = structuralLeaf->second;
      if (expectedLeafId && expectedLeafId.getInt() != expectedLeaf) {
        anchor->emitError("logical operation RA-leaf identity does not match "
                          "its structural leaf");
        return failure();
      }
      for (Operation *member : group.members) {
        auto memberOperationId = member->getAttrOfType<IntegerAttr>(
            mapping::kMappingOperationIdAttrName);
        auto memberLeafId = member->getAttrOfType<IntegerAttr>(
            mapping::kRALeafIdAttrName);
        if (!memberOperationId || memberOperationId.getInt() != id ||
            !memberLeafId || memberLeafId.getInt() != expectedLeaf) {
          member->emitError("expanded stage identity does not match logical "
                            "operation ")
              << id;
          return failure();
        }
      }

      RealizationStage stage;
      stage.id = report.realizationStages.size();
      stage.operationId = id;
      stage.raLeafId = expectedLeaf;
      stage.stageIndex = nextStageIndex[id]++;
      if (auto kind = anchor->getAttrOfType<StringAttr>(
              mapping::kStageKindAttrName))
        stage.kind = kind.getValue().str();
      if (auto name = anchor->getAttrOfType<StringAttr>(
              mapping::kStageNameAttrName))
        stage.name = name.getValue().str();
      stage.location = printLocation(anchor->getLoc());
      stage.mlir = printOperationGroup(group.members);

      DenseSet<std::pair<StringRef, Attribute>> seenAttributes;
      std::function<void(Operation *)> recordOperation = [&](Operation *op) {
        reportStageByOperation[op] = stage.id;
        if (isa<ArraySetOp>(op))
          ++stage.arraySetCount;
        if (isa<ArrayLoadOp>(op))
          ++stage.arrayLoadCount;
        if (isa<ArrayExecuteOp>(op))
          ++stage.arrayExecuteCount;
        if (isa<ArrayStoreOp>(op))
          ++stage.arrayStoreCount;
        for (Region &region : op->getRegions()) {
          for (Block &block : region) {
            for (Operation &nested : block)
              recordOperation(&nested);
          }
        }
      };
      for (Operation *member : group.members) {
        recordOperation(member);
        for (NamedAttribute namedAttribute : member->getAttrs()) {
          StringRef attrName = namedAttribute.getName().strref();
          auto key = std::make_pair(attrName, namedAttribute.getValue());
          if (!isRealizationMetadata(attrName) ||
              !seenAttributes.insert(key).second)
            continue;
          stage.attributes.emplace_back(
              attrName.str(), printAttribute(namedAttribute.getValue()));
        }
      }

      realizedMVMs.insert(id);
      reportStageByGroup.push_back(stage.id);
      report.realizationStages.push_back(std::move(stage));
    }

    for (auto [group, reportStageId] :
         llvm::zip_equal(groups, reportStageByGroup)) {
      RealizationStage &stage = report.realizationStages[reportStageId];
      DenseSet<Value> boundaryInputs;
      DenseSet<Value> boundaryOutputs;
      DenseSet<int64_t> predecessorIds;
      std::function<void(Operation *)> analyzeOperation = [&](Operation *op) {
        for (Value operand : op->getOperands()) {
          if (auto argument = dyn_cast<BlockArgument>(operand)) {
            Operation *parent = argument.getOwner()->getParentOp();
            auto parentStage = reportStageByOperation.find(parent);
            if (parentStage != reportStageByOperation.end() &&
                parentStage->second == reportStageId)
              continue;
          }

          Operation *producer = operand.getDefiningOp();
          auto producerStage = reportStageByOperation.find(producer);
          if (producerStage != reportStageByOperation.end() &&
              producerStage->second == reportStageId)
            continue;

          boundaryInputs.insert(operand);
          if (producerStage != reportStageByOperation.end())
            predecessorIds.insert(producerStage->second);
        }

        for (Value result : op->getResults()) {
          for (OpOperand &use : result.getUses()) {
            auto consumerStage = reportStageByOperation.find(use.getOwner());
            if (consumerStage == reportStageByOperation.end() ||
                consumerStage->second != reportStageId) {
              boundaryOutputs.insert(result);
              break;
            }
          }
        }

        for (Region &region : op->getRegions()) {
          for (Block &block : region) {
            for (Operation &nested : block)
              analyzeOperation(&nested);
          }
        }
      };
      for (Operation *member : group.members)
        analyzeOperation(member);

      stage.inputCount = boundaryInputs.size();
      stage.outputCount = boundaryOutputs.size();
      stage.predecessorStageIds.assign(predecessorIds.begin(),
                                       predecessorIds.end());
      llvm::sort(stage.predecessorStageIds);
    }

    for (const ComputeOperation &operation : report.graph.operations) {
      if (operation.kind != ComputeOperationKind::LogicalMVM ||
          realizedMVMs.contains(operation.id))
        continue;
      operation.operation->emitError(
          "logical sculptor.mvm has no physical realization in expanded IR");
      return failure();
    }
  }
  return success();
}

void emitI64Array(llvm::json::OStream &json, llvm::StringRef name,
                  ArrayRef<int64_t> values) {
  json.attributeArray(name, [&] {
    for (int64_t value : values)
      json.value(value);
  });
}

void emitSemanticAttributes(llvm::json::OStream &json, Operation *operation) {
  json.attributeObject("semantic", [&] {
    for (NamedAttribute namedAttribute : operation->getAttrs()) {
      StringRef name = namedAttribute.getName().strref();
      if (!name.consume_front("sculptor.semantic."))
        continue;

      Attribute attribute = namedAttribute.getValue();
      if (auto stringAttr = dyn_cast<StringAttr>(attribute)) {
        json.attribute(name, stringAttr.getValue());
      } else if (auto integerAttr = dyn_cast<IntegerAttr>(attribute)) {
        json.attribute(name, integerAttr.getInt());
      } else if (auto floatAttr = dyn_cast<FloatAttr>(attribute)) {
        json.attribute(name, floatAttr.getValueAsDouble());
      } else if (auto boolAttr = dyn_cast<BoolAttr>(attribute)) {
        json.attribute(name, boolAttr.getValue());
      } else {
        json.attribute(name, printAttribute(attribute));
      }
    }
  });
}

void emitTree(llvm::json::OStream &json, const ReportFunction &report) {
  DenseMap<int64_t, const StructuralRATreeNode *> nodesById;
  for (const StructuralRATreeNode &node : report.tree.nodes)
    nodesById[node.id] = &node;
  DenseMap<int64_t, std::optional<int64_t>> inferredWaveIds;
  std::function<std::optional<int64_t>(int64_t)> inferWaveId =
      [&](int64_t nodeId) -> std::optional<int64_t> {
    auto cached = inferredWaveIds.find(nodeId);
    if (cached != inferredWaveIds.end())
      return cached->second;

    const StructuralRATreeNode *node = nodesById.lookup(nodeId);
    if (!node)
      return std::nullopt;
    if (node->kind == RATreeNodeKind::Leaf) {
      std::optional<int64_t> waveId =
          report.graph.operations[node->operationId].mvmWaveId;
      inferredWaveIds[nodeId] = waveId;
      return waveId;
    }

    std::optional<int64_t> waveId;
    for (int64_t childId : node->childIds) {
      std::optional<int64_t> childWaveId = inferWaveId(childId);
      if (!childWaveId || (waveId && *waveId != *childWaveId)) {
        inferredWaveIds[nodeId] = std::nullopt;
        return std::nullopt;
      }
      waveId = childWaveId;
    }
    inferredWaveIds[nodeId] = waveId;
    return waveId;
  };

  json.attributeObject("tree", [&] {
    json.attribute("version", report.treeAttr.getVersion().getInt());
    json.attribute("root_id", report.tree.rootId);
    json.attribute("operation_count",
                   report.treeAttr.getOperationCount().getInt());
    json.attribute("tensor_count", report.treeAttr.getTensorCount().getInt());
    json.attribute("fingerprint",
                   report.treeAttr.getGraphFingerprint().getValue());
    json.attributeArray("work_units", [&] {
      for (const MappingWorkUnit &workUnit : report.tree.workUnits) {
        json.object([&] {
          json.attribute("id", workUnit.id);
          json.attribute("operation_id", workUnit.operationId);
          json.attribute("result_number", workUnit.resultNumber);
          emitI64Array(json, "result_offsets", workUnit.resultOffsets);
          emitI64Array(json, "result_sizes", workUnit.resultSizes);
          emitI64Array(json, "iteration_offsets", workUnit.iterationOffsets);
          emitI64Array(json, "iteration_sizes", workUnit.iterationSizes);
        });
      }
    });
    json.attributeArray("work_unit_edges", [&] {
      for (const MappingWorkUnitEdge &edge : report.tree.workUnitEdges) {
        json.object([&] {
          json.attribute("source_operation_id", edge.sourceOperationId);
          json.attribute("source_work_unit_id", edge.sourceWorkUnitId);
          json.attribute("target_operation_id", edge.targetOperationId);
          json.attribute("target_work_unit_id", edge.targetWorkUnitId);
          json.attribute("byte_size", edge.byteSize);
        });
      }
    });
    json.attributeArray("nodes", [&] {
      for (const StructuralRATreeNode &node : report.tree.nodes) {
        json.object([&] {
          json.attribute("id", node.id);
          json.attribute("kind", stringifyRATreeNodeKind(node.kind));
          json.attribute("parent_id", node.parentId);
          emitI64Array(json, "child_ids", node.childIds);
          json.attribute("operation_id", node.operationId);
          json.attribute("work_unit_id", node.workUnitId);
          json.attribute("work_group_count", node.workGroupCount);
          json.attribute("mvm_wave_id", inferWaveId(node.id).value_or(-1));
        });
      }
    });
  });
}

void emitPlan(llvm::json::OStream &json, const ReportFunction &report) {
  if (!report.planAttr)
    return;

  MappingPlanAttr plan = report.planAttr;
  json.attributeObject("plan", [&] {
    json.attribute("version", plan.getVersion().getInt());
    json.attribute("planner", plan.getPlanner().getValue());
    json.attribute("objective",
                   stringifyMappingPlanObjective(plan.getObjective()));
    json.attribute("mvm_body_policy", plan.getMvmBodyPolicy().getValue());
    json.attribute("setup_binding_policy",
                   plan.getSetupBindingPolicy().getValue());
    json.attribute("ra_tree_fingerprint",
                   plan.getRaTreeFingerprint().getValue());
    json.attribute("feasible", plan.getFeasible().getValue());
    json.attribute("estimated_latency_ns",
                   plan.getEstimatedLatencyNs().getValueAsDouble());
    json.attribute("crossing_bytes", plan.getCrossingBytes().getInt());
    json.attribute("estimated_communication_ns",
                   plan.getEstimatedCommunicationNs().getValueAsDouble());
    json.attribute("required_resource_units",
                   plan.getRequiredResourceUnits().getInt());
    json.attribute("pipeline_stages", plan.getPipelineStages().getInt());
    json.attributeArray("candidates", [&] {
      for (Attribute attribute : plan.getCandidates()) {
        auto candidate = cast<MappingCandidateEvaluationAttr>(attribute);
        json.object([&] {
          json.attribute("name", candidate.getName().getValue());
          json.attribute("selected", candidate.getSelected().getValue());
          json.attribute("feasible", candidate.getFeasible().getValue());
          json.attribute("estimated_latency_ns",
                         candidate.getEstimatedLatencyNs().getValueAsDouble());
          json.attribute("crossing_bytes",
                         candidate.getCrossingBytes().getInt());
          json.attribute(
              "estimated_communication_ns",
              candidate.getEstimatedCommunicationNs().getValueAsDouble());
          json.attribute("required_resource_units",
                         candidate.getRequiredResourceUnits().getInt());
          json.attribute("pipeline_stages",
                         candidate.getPipelineStages().getInt());
          json.attribute("infeasibility_reason",
                         candidate.getInfeasibilityReason().getValue());
        });
      }
    });
    json.attributeArray("node_evaluations", [&] {
      for (Attribute attribute : plan.getNodeEvaluations()) {
        auto node = cast<RATreeNodeEvaluationAttr>(attribute);
        json.object([&] {
          json.attribute("node_id", node.getNodeId().getInt());
          json.attribute("feasible", node.getFeasible().getValue());
          json.attribute("estimated_latency_ns",
                         node.getEstimatedLatencyNs().getValueAsDouble());
          json.attribute("crossing_bytes", node.getCrossingBytes().getInt());
          json.attribute("estimated_communication_ns",
                         node.getEstimatedCommunicationNs().getValueAsDouble());
          json.attribute("required_resource_units",
                         node.getRequiredResourceUnits().getInt());
          json.attribute("pipeline_stages", node.getPipelineStages().getInt());
          json.attribute("infeasibility_reason",
                         node.getInfeasibilityReason().getValue());
        });
      }
    });
    MappingRealizationAttr realization = plan.getRealization();
    json.attributeObject("realization", [&] {
      json.attribute("version", realization.getVersion().getInt());
      json.attribute("logical_tile_count",
                     realization.getLogicalTileCount().getInt());
      json.attribute("analog_lanes_per_tile",
                     realization.getAnalogLanesPerTile().getInt());
      json.attributeArray("digital_work_per_tile", [&] {
        for (Attribute work : realization.getDigitalWorkPerTile())
          json.value(cast<IntegerAttr>(work).getInt());
      });
      json.attributeArray("node_allocations", [&] {
        for (Attribute attribute : realization.getNodeAllocations()) {
          auto allocation = cast<MappingNodeAllocationAttr>(attribute);
          json.object([&] {
            json.attribute("node_id", allocation.getNodeId().getInt());
            json.attributeArray("digital_tile_ids", [&] {
              for (Attribute tileId : allocation.getDigitalTileIds())
                json.value(cast<IntegerAttr>(tileId).getInt());
            });
            json.attributeArray("analog_lanes", [&] {
              for (auto [tileId, laneIndex] :
                   llvm::zip_equal(allocation.getAnalogTileIds(),
                                   allocation.getAnalogLaneIndices())) {
                json.object([&] {
                  json.attribute("tile_id", cast<IntegerAttr>(tileId).getInt());
                  json.attribute("lane_index",
                                 cast<IntegerAttr>(laneIndex).getInt());
                });
              }
            });
          });
        }
      });
      json.attributeArray("leaf_assignments", [&] {
        for (Attribute attribute : realization.getLeafAssignments()) {
          auto assignment = cast<MappingLeafAssignmentAttr>(attribute);
          json.object([&] {
            json.attribute("leaf_id", assignment.getLeafId().getInt());
            json.attribute("operation_id",
                           assignment.getOperationId().getInt());
            json.attribute("tile_id", assignment.getTileId().getInt());
            json.attribute("lane_kind",
                           stringifyMappingLaneKind(assignment.getLaneKind()));
            json.attribute("lane_index", assignment.getLaneIndex().getInt());
            json.attribute("start_ns",
                           assignment.getStartNs().getValueAsDouble());
            json.attribute("finish_ns",
                           assignment.getFinishNs().getValueAsDouble());
          });
        }
      });
    });
  });
}

void emitOperations(llvm::json::OStream &json, const ReportFunction &report) {
  json.attributeArray("operations", [&] {
    for (const ComputeOperation &operation : report.graph.operations) {
      json.object([&] {
        json.attribute("id", operation.id);
        json.attribute("name",
                       operation.stageName.empty()
                           ? operation.operation->getName().getStringRef()
                           : StringRef(operation.stageName));
        json.attribute("kind", stringifyComputeOperationKind(operation.kind));
        json.attribute("required_lane",
                       operation.requiredLane
                           ? stringifyLogicalLaneKind(*operation.requiredLane)
                           : StringRef("unassigned"));
        json.attribute("lane_binding_group", operation.laneBindingGroup
                                                 ? *operation.laneBindingGroup
                                                 : int64_t{-1});
        json.attribute("mvm_wave_id", operation.mvmWaveId ? *operation.mvmWaveId
                                                          : int64_t{-1});
        json.attribute("mvm_wave_member", operation.mvmWaveMember
                                              ? *operation.mvmWaveMember
                                              : int64_t{-1});
        json.attribute("mvm_wave_size", operation.mvmWaveSize
                                            ? *operation.mvmWaveSize
                                            : int64_t{-1});
        json.attribute("member_count", operation.members.size());
        json.attribute("location",
                       printLocation(operation.operation->getLoc()));
        json.attribute("mlir", printComputeOperation(operation));
        emitI64Array(json, "input_tensors", operation.inputTensors);
        emitI64Array(json, "output_tensors", operation.outputTensors);
        emitSemanticAttributes(json, operation.operation);
        if (operation.analogMVM) {
          json.attributeObject("analog_mvm", [&] {
            json.attribute("output_rows", operation.analogMVM->outputRows);
            json.attribute("input_columns", operation.analogMVM->inputColumns);
          });
        }
        json.attributeArray("iteration_domain", [&] {
          for (const ComputeIterationDimension &dimension :
               operation.iterationDomain) {
            json.object([&] {
              json.attribute("loop", dimension.loopIndex);
              json.attribute("kind",
                             stringifyComputeIteratorKind(dimension.kind));
              json.attribute("extent", dimension.staticExtent);
            });
          }
        });
      });
    }
  });
}

void emitLogicalTileAssignment(llvm::json::OStream &json,
                               const LogicalTileAssignment &assignment) {
  json.object([&] {
    json.attribute("leaf_id", assignment.leafId);
    json.attribute("operation_id", assignment.operationId);
    json.attribute("work_unit_id", assignment.workUnitId);
    json.attribute("lane_kind", stringifyLogicalLaneKind(assignment.laneKind));
    json.attribute("lane_index", assignment.laneIndex);
    emitI64Array(json, "ra_node_path", assignment.raNodePath);
    json.attribute("start_ns", assignment.startNs);
    json.attribute("finish_ns", assignment.finishNs);
  });
}

void emitLogicalTileDependency(llvm::json::OStream &json,
                               const LogicalTileDependency &dependency) {
  json.object([&] {
    json.attribute("source_operation_id", dependency.sourceOperationId);
    json.attribute("source_work_unit_id", dependency.sourceWorkUnitId);
    json.attribute("target_operation_id", dependency.targetOperationId);
    json.attribute("target_work_unit_id", dependency.targetWorkUnitId);
    json.attribute("tensor_id", dependency.tensorId);
    json.attribute("byte_size", dependency.byteSize);
  });
}

void emitLogicalTileGraph(llvm::json::OStream &json,
                          const ReportFunction &report) {
  if (!report.logicalTileGraph)
    return;
  const LogicalTileGraph &tileGraph = *report.logicalTileGraph;
  json.attributeObject("logical_tile_graph", [&] {
    json.attribute("version", tileGraph.version);
    json.attribute("planned_mesh_rows", tileGraph.plannedMeshRows);
    json.attribute("planned_mesh_cols", tileGraph.plannedMeshCols);
    json.attribute("logical_tile_capacity", tileGraph.logicalTileCapacity);
    json.attribute("analog_lanes_per_tile", tileGraph.analogLanesPerTile);
    json.attributeArray("tiles", [&] {
      for (const LogicalTile &tile : tileGraph.tiles) {
        json.object([&] {
          json.attribute("id", tile.id);
          json.attribute("digital_work", tile.digitalWork);
          emitI64Array(json, "model_input_tensor_ids",
                       tile.modelInputTensorIds);
          emitI64Array(json, "model_output_tensor_ids",
                       tile.modelOutputTensorIds);
          json.attributeArray("digital_assignments", [&] {
            for (const LogicalTileAssignment &assignment :
                 tile.digitalAssignments)
              emitLogicalTileAssignment(json, assignment);
          });
          json.attributeArray("analog_lanes", [&] {
            for (const LogicalTileAnalogLane &lane : tile.analogLanes) {
              json.object([&] {
                json.attribute("lane_index", lane.laneIndex);
                json.attribute("lane_binding_group",
                               lane.laneBindingGroup.value_or(-1));
                json.attributeArray("assignments", [&] {
                  for (const LogicalTileAssignment &assignment :
                       lane.assignments)
                    emitLogicalTileAssignment(json, assignment);
                });
              });
            }
          });
          json.attributeArray("internal_dependencies", [&] {
            for (const LogicalTileDependency &dependency :
                 tile.internalDependencies)
              emitLogicalTileDependency(json, dependency);
          });
        });
      }
    });
    json.attributeArray("edges", [&] {
      for (const LogicalTileEdge &edge : tileGraph.edges) {
        json.object([&] {
          json.attribute("id", edge.id);
          json.attribute("source_tile_id", edge.sourceTileId);
          json.attribute("target_tile_id", edge.targetTileId);
          json.attribute("byte_size", edge.byteSize);
          json.attributeArray("dependencies", [&] {
            for (const LogicalTileDependency &dependency : edge.dependencies)
              emitLogicalTileDependency(json, dependency);
          });
        });
      }
    });
  });
}

void emitLogicalTilePlacement(llvm::json::OStream &json,
                              const ReportFunction &report) {
  if (!report.logicalTilePlacement)
    return;
  const LogicalTilePlacementPlan &placement = *report.logicalTilePlacement;
  json.attributeObject("physical_placement", [&] {
    json.attribute("version", placement.version);
    json.attribute("schedule", placement.schedule);
    json.attribute("mesh_rows", placement.mesh.rows);
    json.attribute("mesh_cols", placement.mesh.columns);
    json.attribute("arrays_per_core", placement.mesh.arraysPerCore);
    json.attribute("initial_score", placement.initialScore);
    json.attribute("total_transfer_cost", placement.totalTransferCost);
    json.attribute("evaluations", placement.evaluations);
    json.attributeArray("assignments", [&] {
      for (const LogicalTilePhysicalAssignment &assignment :
           placement.assignments) {
        json.object([&] {
          json.attribute("logical_tile_id", assignment.logicalTileId);
          json.attribute("physical_tile_id",
                         assignment.location.physicalTileId);
          json.attribute("row", assignment.location.row);
          json.attribute("column", assignment.location.column);
        });
      }
    });
    json.attributeArray("edges", [&] {
      for (const PlacedLogicalTileEdge &edge : placement.edges) {
        json.object([&] {
          json.attribute("edge_id", edge.edgeId);
          json.attribute("source_tile_id", edge.sourceTileId);
          json.attribute("target_tile_id", edge.targetTileId);
          json.attribute("byte_size", edge.byteSize);
          json.attribute("manhattan_hops", edge.manhattanHops);
          json.attribute("transfer_cost", edge.transferCost);
        });
      }
    });
    if (placement.annealingTrace) {
      const LogicalTileAnnealingTrace &trace = *placement.annealingTrace;
      json.attributeObject("annealing_trace", [&] {
        json.attribute("version", trace.version);
        json.attribute("initial_score", trace.initialScore);
        json.attribute("final_score", trace.finalScore);
        json.attribute("evaluations", trace.evaluations);
        json.attributeArray("samples", [&] {
          for (const LogicalTileAnnealingSample &sample : trace.samples) {
            json.object([&] {
              json.attribute("iteration", sample.iteration);
              json.attribute("candidate_score", sample.candidateScore);
              json.attribute("current_score", sample.currentScore);
              json.attribute("best_score", sample.bestScore);
              json.attribute("accepted", sample.accepted);
            });
          }
        });
      });
    }
  });
}

void emitLaneBindingGroups(llvm::json::OStream &json,
                           const ReportFunction &report) {
  json.attributeArray("lane_binding_groups", [&] {
    for (const LaneBindingGroup &group : report.graph.laneBindingGroups) {
      json.object([&] {
        json.attribute("id", group.id);
        json.attribute("setup_operation_id", group.setupOperationId);
        emitI64Array(json, "operation_ids", group.operationIds);
      });
    }
  });
}

void emitMVMWaves(llvm::json::OStream &json, const ReportFunction &report) {
  json.attributeArray("mvm_waves", [&] {
    for (const MVMWave &wave : report.graph.mvmWaves) {
      json.object([&] {
        json.attribute("id", wave.id);
        emitI64Array(json, "vector_tile_operation_ids",
                     wave.vectorTileOperationIds);
        emitI64Array(json, "physical_mvm_operation_ids",
                     wave.physicalMVMOperationIds);
        json.attribute("recombine_operation_id",
                       wave.recombineOperationId ? *wave.recombineOperationId
                                                 : int64_t{-1});
        json.attribute("bias_add_operation_id", wave.biasAddOperationId
                                                    ? *wave.biasAddOperationId
                                                    : int64_t{-1});
      });
    }
  });
}

void emitTensors(llvm::json::OStream &json, const ReportFunction &report) {
  json.attributeArray("tensors", [&] {
    for (const ComputeTensor &tensor : report.graph.tensors) {
      json.object([&] {
        json.attribute("id", tensor.id);
        json.attribute("type", printType(tensor.type));
        json.attribute("byte_size", tensor.byteSize);
        json.attribute("is_logical_array", tensor.isLogicalArray);
        json.attribute("is_function_input", tensor.isFunctionInput);
        json.attribute("is_function_output", tensor.isFunctionOutput);
        emitI64Array(json, "producers", tensor.producerOperations);
        emitI64Array(json, "consumers", tensor.consumerOperations);
      });
    }
  });
}

void emitEdges(llvm::json::OStream &json, const ReportFunction &report) {
  json.attributeArray("edges", [&] {
    for (const ReportEdge &edge : report.edges) {
      json.object([&] {
        json.attribute("source", edge.source);
        json.attribute("target", edge.target);
        json.attribute("byte_size", edge.byteSize);
        json.attribute("has_unknown_byte_size", edge.hasUnknownByteSize);
        emitI64Array(json, "tensor_ids", edge.tensorIds);
      });
    }
  });
}

void emitRealization(llvm::json::OStream &json, const ReportFunction &report) {
  if (report.realizationStages.empty())
    return;

  json.attributeObject("realization", [&] {
    json.attribute("stage_count", report.realizationStages.size());
    json.attributeArray("stages", [&] {
      for (const RealizationStage &stage : report.realizationStages) {
        json.object([&] {
          json.attribute("operation_id", stage.operationId);
          json.attribute("id", stage.id);
          json.attribute("ra_leaf_id", stage.raLeafId);
          json.attribute("stage_index", stage.stageIndex);
          json.attribute("kind", stage.kind);
          json.attribute("name", stage.name);
          json.attribute("location", stage.location);
          json.attribute("mlir", stage.mlir);
          json.attribute("input_count", stage.inputCount);
          json.attribute("output_count", stage.outputCount);
          json.attribute("array_set_count", stage.arraySetCount);
          json.attribute("array_load_count", stage.arrayLoadCount);
          json.attribute("array_execute_count", stage.arrayExecuteCount);
          json.attribute("array_store_count", stage.arrayStoreCount);
          emitI64Array(json, "predecessor_stage_ids",
                       stage.predecessorStageIds);
          json.attributeObject("attributes", [&] {
            for (const auto &[name, value] : stage.attributes)
              json.attribute(name, value);
          });
        });
      }
    });
  });
}

void emitFunction(llvm::json::OStream &json, const ReportFunction &report) {
  json.object([&] {
    json.attribute("symbol", report.graph.functionSymbol);
    emitTree(json, report);
    emitPlan(json, report);
    emitLogicalTileGraph(json, report);
    emitLogicalTilePlacement(json, report);
    emitI64Array(json, "topological_order", report.graph.topologicalOrder);
    emitOperations(json, report);
    emitLaneBindingGroups(json, report);
    emitMVMWaves(json, report);
    emitTensors(json, report);
    emitEdges(json, report);
    emitRealization(json, report);
  });
}

std::string buildJSON(ArrayRef<ReportFunction> functions, StringRef title) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  llvm::json::OStream json(stream, /*IndentSize=*/0);
  json.object([&] {
    json.attribute("schema_version", 9);
    json.attribute("format", "sculptor.ra_tree.report");
    json.attribute("title", title);
    json.attribute("source", inputFilename);
    if (!expandedIRFilename.empty())
      json.attribute("expanded_source", expandedIRFilename);
    json.attributeArray("functions", [&] {
      for (const ReportFunction &function : functions)
        emitFunction(json, function);
    });
  });
  return text;
}

LogicalResult writeFile(StringRef path, StringRef contents) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "error: cannot open '" << path << "': " << error.message()
                 << '\n';
    return failure();
  }
  output << contents;
  return success();
}

std::string escapeJSONForHTML(StringRef json) {
  std::string escaped;
  escaped.reserve(json.size());
  for (char character : json) {
    switch (character) {
    case '<':
      escaped += "\\u003c";
      break;
    case '>':
      escaped += "\\u003e";
      break;
    case '&':
      escaped += "\\u0026";
      break;
    default:
      escaped.push_back(character);
      break;
    }
  }
  return escaped;
}

std::string defaultOutputPath() {
  SmallString<256> path(inputFilename);
  StringRef stem = llvm::sys::path::stem(path);
  SmallString<256> result(llvm::sys::path::parent_path(path));
  llvm::sys::path::append(result, (stem + "-ra-tree.html").str());
  return std::string(result);
}

std::string defaultTitle() {
  StringRef stem = llvm::sys::path::stem(inputFilename);
  return (stem + " RA Tree").str();
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::HideUnrelatedOptions(reportCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Sculptor RA-tree visual report\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<SculptorDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module)
    return 1;

  SmallVector<ReportFunction, 0> functions;
  bool foundRequestedFunction = functionFilter.empty();
  for (func::FuncOp function : module->getOps<func::FuncOp>()) {
    if (!functionFilter.empty() && function.getSymName() != functionFilter)
      continue;
    foundRequestedFunction = true;

    auto treeAttr = function->getAttrOfType<RATreeAttr>(kRATreeAttrName);
    if (!treeAttr) {
      if (!functionFilter.empty()) {
        function.emitError("requested function has no RA tree; run "
                           "--sculptor-build-ra-tree first");
        return 1;
      }
      continue;
    }

    FailureOr<ReportFunction> report = buildReportFunction(function, treeAttr);
    if (failed(report))
      return 1;
    functions.push_back(std::move(*report));
  }

  if (!foundRequestedFunction) {
    module->emitError("requested function '")
        << functionFilter << "' does not exist";
    return 1;
  }
  if (functions.empty()) {
    module->emitError("no function contains sculptor.mapping.ra_tree; run "
                      "--sculptor-build-ra-tree first");
    return 1;
  }

  OwningOpRef<ModuleOp> expandedModule;
  if (!expandedIRFilename.empty()) {
    expandedModule = parseSourceFile<ModuleOp>(expandedIRFilename, &context);
    if (!expandedModule ||
        failed(collectExpandedRealization(*expandedModule, functions)))
      return 1;
  }

  std::string title = reportTitle.empty() ? defaultTitle() : reportTitle;
  std::string json = buildJSON(functions, title);
  if (!jsonOutputFilename.empty() &&
      failed(writeFile(jsonOutputFilename, json + "\n")))
    return 1;

  std::string html = kRATreeReportHTML.str();
  constexpr StringLiteral marker = "__SCULPTOR_RA_TREE_REPORT_DATA__";
  size_t markerPosition = html.find(marker.str());
  if (markerPosition == std::string::npos) {
    llvm::errs() << "error: embedded report template has no data marker\n";
    return 1;
  }
  html.replace(markerPosition, marker.size(), escapeJSONForHTML(json));

  std::string output =
      outputFilename.empty() ? defaultOutputPath() : outputFilename;
  if (failed(writeFile(output, html)))
    return 1;

  llvm::outs() << "Wrote " << output << " (" << functions.size() << " function"
               << (functions.size() == 1 ? "" : "s") << ")\n";
  return 0;
}
