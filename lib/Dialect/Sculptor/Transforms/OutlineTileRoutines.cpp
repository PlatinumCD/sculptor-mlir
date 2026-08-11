#include "sculptor-mlir/Dialect/Sculptor/Transforms/OutlineTileRoutines.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingCostProfile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ReductionTree.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

namespace tile_runtime_attrs = mlir::sculptor::tile_runtime_attrs;

using EndpointKey = std::pair<int64_t, int64_t>;

constexpr StringLiteral kDeploymentKindAttr = "sculptor.deployment.kind";
constexpr StringLiteral kDeploymentVersionAttr = "sculptor.deployment.version";
constexpr StringLiteral kDeploymentMeshRowsAttr =
    "sculptor.deployment.mesh_rows";
constexpr StringLiteral kDeploymentMeshColsAttr =
    "sculptor.deployment.mesh_cols";
constexpr StringLiteral kDeploymentArraysPerTileAttr =
    "sculptor.deployment.arrays_per_tile";
constexpr StringLiteral kDeploymentActiveTileIdsAttr =
    "sculptor.deployment.active_tile_ids";
constexpr StringLiteral kDeploymentRoutesAttr = "sculptor.deployment.routes";
constexpr StringLiteral kDeploymentLocalBindingsAttr =
    "sculptor.deployment.local_bindings";
constexpr StringLiteral kDeploymentModelInputsAttr =
    "sculptor.deployment.model_inputs";
constexpr StringLiteral kDeploymentModelOutputsAttr =
    "sculptor.deployment.model_outputs";
constexpr StringLiteral kDeploymentIncomingRoutesAttr =
    "sculptor.deployment.incoming_routes";
constexpr StringLiteral kDeploymentOutgoingRoutesAttr =
    "sculptor.deployment.outgoing_routes";
constexpr StringLiteral kDeploymentPhysicalTileIdAttr =
    "sculptor.deployment.physical_tile_id";
constexpr StringLiteral kDeploymentTileRowAttr = "sculptor.deployment.tile_row";
constexpr StringLiteral kDeploymentTileColAttr = "sculptor.deployment.tile_col";
constexpr StringLiteral kDeploymentGlobalRoutineIdAttr =
    "sculptor.deployment.global_routine_id";
constexpr StringLiteral kDeploymentLocalRoutineIndexAttr =
    "sculptor.deployment.local_routine_index";
constexpr StringLiteral kDeploymentRoutineKindAttr =
    "sculptor.deployment.routine_kind";
constexpr StringLiteral kDeploymentLogicalTileIdsAttr =
    "sculptor.deployment.logical_tile_ids";
constexpr StringLiteral kDeploymentSourceLeafIdsAttr =
    "sculptor.deployment.source_leaf_ids";
constexpr StringLiteral kDeploymentInputResourceIdsAttr =
    "sculptor.deployment.input_resource_ids";
constexpr StringLiteral kDeploymentOutputResourceIdsAttr =
    "sculptor.deployment.output_resource_ids";

struct EndpointPlan {
  EndpointKey key{-1, -1};
  int64_t leafId = -1;
  int64_t logicalTileId = -1;
  PhysicalTileLocation location;
  LogicalLaneKind laneKind = LogicalLaneKind::Digital;
  int64_t laneIndex = -1;
  ComputeOperationKind operationKind = ComputeOperationKind::Structured;
  SmallVector<Operation *> mappedOperations;
  SmallVector<Value> producedValues;
};

struct TilePart {
  int64_t workUnitId = -1;
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  Value value;
};

struct RoutinePlan {
  int64_t globalId = -1;
  int64_t localIndex = -1;
  int64_t physicalTileId = -1;
  int64_t tileRow = -1;
  int64_t tileCol = -1;
  bool boot = false;
  bool syntheticOutput = false;
  SmallVector<unsigned> endpointIndices;
  SmallVector<int64_t> logicalTileIds;
  SmallVector<int64_t> sourceLeafIds;
  llvm::SetVector<Operation *> selectedOperations;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  func::FuncOp function;
};

struct RouteRecord {
  int64_t id = -1;
  unsigned sourceRoutine = 0;
  unsigned sourceOutput = 0;
  unsigned destinationRoutine = 0;
  unsigned destinationInput = 0;
  int64_t resourceId = -1;
  int64_t tensorId = -1;
  int64_t byteSize = -1;
};

struct LocalBindingRecord {
  unsigned sourceRoutine = 0;
  unsigned sourceOutput = 0;
  unsigned destinationRoutine = 0;
  unsigned destinationInput = 0;
  int64_t resourceId = -1;
  int64_t byteSize = -1;
};

enum class RoutineDependencyKind { Route, LocalBinding };

struct RoutineDependencyEdge {
  unsigned sourceRoutine = 0;
  unsigned destinationRoutine = 0;
  RoutineDependencyKind kind = RoutineDependencyKind::LocalBinding;
  int64_t id = -1;
  int64_t byteSize = -1;
};

SmallVector<Attribute> getI64Attrs(OpBuilder &builder,
                                   ArrayRef<int64_t> values) {
  SmallVector<Attribute> result;
  result.reserve(values.size());
  for (int64_t value : values)
    result.push_back(builder.getI64IntegerAttr(value));
  return result;
}

ArrayAttr getI64Array(OpBuilder &builder, ArrayRef<int64_t> values) {
  return builder.getArrayAttr(getI64Attrs(builder, values));
}

void appendUnique(SmallVectorImpl<int64_t> &values, int64_t value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

void appendUnique(SmallVectorImpl<Value> &values, Value value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

FailureOr<int64_t> getStaticByteSize(Type type, Operation *anchor) {
  if (isa<LogicalArrayType>(type))
    return int64_t{0};
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape()) {
    anchor->emitError("tile-routine boundary requires a statically shaped "
                      "tensor, but found ")
        << type;
    return failure();
  }
  Type elementType = shaped.getElementType();
  if (!elementType.isIntOrFloat()) {
    anchor->emitError("tile-routine boundary requires an integer or floating "
                      "point element type, but found ")
        << elementType;
    return failure();
  }
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0) {
    anchor->emitError("tile-routine boundary has an unsupported element "
                      "bit width ")
        << bitWidth;
    return failure();
  }
  std::optional<int64_t> bytes =
      llvm::checkedMul(shaped.getNumElements(), int64_t{bitWidth / 8});
  if (!bytes) {
    anchor->emitError("tile-routine boundary byte size overflow");
    return failure();
  }
  return *bytes;
}

bool matchesStaticSlice(tensor::ExtractSliceOp slice, ArrayRef<int64_t> offsets,
                        ArrayRef<int64_t> sizes) {
  return slice.getStaticOffsets() == offsets &&
         slice.getStaticSizes() == sizes &&
         llvm::all_of(slice.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; });
}

bool regionsOverlap(ArrayRef<int64_t> lhsOffsets, ArrayRef<int64_t> lhsSizes,
                    ArrayRef<int64_t> rhsOffsets, ArrayRef<int64_t> rhsSizes) {
  for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
       llvm::zip_equal(lhsOffsets, lhsSizes, rhsOffsets, rhsSizes)) {
    if (lhsOffset + lhsSize <= rhsOffset || rhsOffset + rhsSize <= lhsOffset)
      return false;
  }
  return true;
}

FailureOr<int64_t> getStaticRegionElementCount(ArrayRef<int64_t> sizes,
                                               Operation *anchor) {
  int64_t elements = 1;
  for (int64_t size : sizes) {
    if (size <= 0) {
      anchor->emitError("expected positive static tile extent");
      return failure();
    }
    std::optional<int64_t> next = llvm::checkedMul(elements, size);
    if (!next) {
      anchor->emitError("static tile element count overflow");
      return failure();
    }
    elements = *next;
  }
  return elements;
}

LogicalResult validateTilePartition(ArrayRef<TilePart> parts,
                                    RankedTensorType resultType,
                                    Operation *anchor) {
  int64_t rank = resultType.getRank();
  int64_t coveredElements = 0;
  for (const TilePart &part : parts) {
    auto partType = dyn_cast<RankedTensorType>(part.value.getType());
    if (part.offsets.size() != static_cast<size_t>(rank) ||
        part.sizes.size() != static_cast<size_t>(rank) || !partType ||
        !partType.hasStaticShape() ||
        partType.getElementType() != resultType.getElementType() ||
        partType.getShape() != ArrayRef<int64_t>(part.sizes)) {
      return anchor->emitError(
          "work-unit tile metadata does not match its tensor value type");
    }
    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      int64_t offset = part.offsets[dimension];
      int64_t size = part.sizes[dimension];
      int64_t extent = resultType.getDimSize(dimension);
      if (offset < 0 || size <= 0 || offset > extent - size) {
        return anchor->emitError(
            "work-unit tile lies outside its full tensor result");
      }
    }
    FailureOr<int64_t> partElements =
        getStaticRegionElementCount(part.sizes, anchor);
    if (failed(partElements) ||
        coveredElements > std::numeric_limits<int64_t>::max() - *partElements) {
      if (succeeded(partElements))
        anchor->emitError("work-unit tile coverage overflow");
      return failure();
    }
    coveredElements += *partElements;
  }

  for (unsigned lhs = 0; lhs < parts.size(); ++lhs) {
    for (unsigned rhs = lhs + 1; rhs < parts.size(); ++rhs) {
      if (regionsOverlap(parts[lhs].offsets, parts[lhs].sizes,
                         parts[rhs].offsets, parts[rhs].sizes)) {
        return anchor->emitError(
            "work-unit tiles overlap in one tensor result");
      }
    }
  }

  if (coveredElements != resultType.getNumElements()) {
    return anchor->emitError(
        "work-unit tiles do not completely cover their tensor result");
  }
  return success();
}

struct DemandPiece {
  const TilePart *part = nullptr;
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
};

LogicalResult rewriteStaticSubsetConsumers(
    IRRewriter &rewriter, Value fullValue, RankedTensorType fullType,
    ArrayRef<TilePart> parts,
    DenseMap<Value, SmallVector<TilePart>> &partsByFullValue) {
  SmallVector<tensor::ExtractSliceOp> consumers;
  for (OpOperand &use : fullValue.getUses()) {
    auto slice = dyn_cast<tensor::ExtractSliceOp>(use.getOwner());
    if (slice && slice.getSource() == fullValue)
      consumers.push_back(slice);
  }

  for (tensor::ExtractSliceOp slice : consumers) {
    ArrayRef<int64_t> demandOffsets = slice.getStaticOffsets();
    ArrayRef<int64_t> demandSizes = slice.getStaticSizes();
    ArrayRef<int64_t> demandStrides = slice.getStaticStrides();
    auto demandType = dyn_cast<RankedTensorType>(slice.getType());
    if (!demandType || demandType.getRank() != fullType.getRank() ||
        demandType.getShape() != demandSizes ||
        llvm::any_of(demandOffsets, ShapedType::isDynamic) ||
        llvm::any_of(demandSizes, ShapedType::isDynamic) ||
        llvm::any_of(demandStrides, ShapedType::isDynamic) ||
        !llvm::all_of(demandStrides,
                      [](int64_t stride) { return stride == 1; }))
      continue;

    bool isFullDemand =
        llvm::all_of(demandOffsets,
                     [](int64_t offset) { return offset == 0; }) &&
        demandSizes == fullType.getShape();
    if (isFullDemand)
      continue;

    SmallVector<DemandPiece> pieces;
    int64_t coveredElements = 0;
    for (const TilePart &part : parts) {
      DemandPiece piece;
      piece.part = &part;
      bool overlaps = true;
      for (int64_t dimension = 0; dimension < fullType.getRank(); ++dimension) {
        int64_t begin =
            std::max(part.offsets[dimension], demandOffsets[dimension]);
        int64_t end =
            std::min(part.offsets[dimension] + part.sizes[dimension],
                     demandOffsets[dimension] + demandSizes[dimension]);
        if (begin >= end) {
          overlaps = false;
          break;
        }
        piece.offsets.push_back(begin);
        piece.sizes.push_back(end - begin);
      }
      if (!overlaps)
        continue;
      FailureOr<int64_t> pieceElements =
          getStaticRegionElementCount(piece.sizes, slice);
      if (failed(pieceElements) ||
          coveredElements >
              std::numeric_limits<int64_t>::max() - *pieceElements) {
        if (succeeded(pieceElements))
          slice.emitOpError("subset-demand coverage overflow");
        return failure();
      }
      coveredElements += *pieceElements;
      pieces.push_back(std::move(piece));
    }

    FailureOr<int64_t> demandElements =
        getStaticRegionElementCount(demandSizes, slice);
    if (failed(demandElements))
      return failure();
    if (pieces.empty() || coveredElements != *demandElements) {
      return slice.emitOpError(
          "static subset demand is not completely covered by producer tiles");
    }
    llvm::sort(pieces, [](const DemandPiece &lhs, const DemandPiece &rhs) {
      return std::lexicographical_compare(
          lhs.offsets.begin(), lhs.offsets.end(), rhs.offsets.begin(),
          rhs.offsets.end());
    });

    rewriter.setInsertionPoint(slice);
    SmallVector<OpFoldResult> unitStrides(fullType.getRank(),
                                          rewriter.getIndexAttr(1));
    SmallVector<TilePart> rebasedParts;
    Value replacement;
    if (pieces.size() > 1) {
      replacement = rewriter.create<tensor::EmptyOp>(
          slice.getLoc(), demandType.getShape(), demandType.getElementType());
    }

    for (const DemandPiece &piece : pieces) {
      SmallVector<int64_t> sourceOffsets;
      SmallVector<int64_t> destinationOffsets;
      SmallVector<OpFoldResult> sourceOffsetAttrs;
      SmallVector<OpFoldResult> destinationOffsetAttrs;
      SmallVector<OpFoldResult> sizeAttrs;
      for (int64_t dimension = 0; dimension < fullType.getRank(); ++dimension) {
        sourceOffsets.push_back(piece.offsets[dimension] -
                                piece.part->offsets[dimension]);
        destinationOffsets.push_back(piece.offsets[dimension] -
                                     demandOffsets[dimension]);
        sourceOffsetAttrs.push_back(
            rewriter.getIndexAttr(sourceOffsets.back()));
        destinationOffsetAttrs.push_back(
            rewriter.getIndexAttr(destinationOffsets.back()));
        sizeAttrs.push_back(rewriter.getIndexAttr(piece.sizes[dimension]));
      }

      Value pieceValue = piece.part->value;
      if (sourceOffsets != SmallVector<int64_t>(fullType.getRank(), 0) ||
          piece.sizes != piece.part->sizes) {
        auto pieceType = RankedTensorType::get(
            piece.sizes, demandType.getElementType(), demandType.getEncoding());
        pieceValue = rewriter.create<tensor::ExtractSliceOp>(
            slice.getLoc(), pieceType, pieceValue, sourceOffsetAttrs, sizeAttrs,
            unitStrides);
      }

      if (pieces.size() == 1) {
        if (pieceValue.getType() != demandType) {
          return slice.emitOpError(
              "subset-demand result type disagrees with producer tile type");
        }
        replacement = pieceValue;
      } else {
        replacement = rewriter.create<tensor::InsertSliceOp>(
            slice.getLoc(), pieceValue, replacement, destinationOffsetAttrs,
            sizeAttrs, unitStrides);
      }
      rebasedParts.push_back({piece.part->workUnitId, destinationOffsets,
                              piece.sizes, pieceValue});
    }

    rewriter.replaceOp(slice, replacement);
    partsByFullValue[replacement] = std::move(rebasedParts);
  }
  return success();
}

FailureOr<PhysicalTileLocation>
getPhysicalLocation(const LogicalTilePlacementPlan &placement,
                    int64_t logicalTileId, Operation *anchor) {
  auto found = placement.assignmentIndexByTileId.find(logicalTileId);
  if (found == placement.assignmentIndexByTileId.end()) {
    anchor->emitError("logical tile ")
        << logicalTileId << " has no locked physical placement";
    return failure();
  }
  return placement.assignments[found->second].location;
}

LogicalResult validateLockedMapping(const LogicalTileGraph &tileGraph,
                                    const LogicalTilePlacementPlan &placement,
                                    Operation *anchor) {
  if (tileGraph.plannedMeshRows != placement.mesh.rows ||
      tileGraph.plannedMeshCols != placement.mesh.columns ||
      tileGraph.analogLanesPerTile != placement.mesh.arraysPerCore) {
    return anchor->emitError(
        "tile-routine outlining requires a physical placement locked to the "
        "logical plan geometry");
  }
  if (placement.assignments.size() != tileGraph.tiles.size()) {
    return anchor->emitError(
        "tile-routine outlining requires one placement assignment per "
        "logical tile");
  }
  for (const LogicalTile &tile : tileGraph.tiles) {
    FailureOr<PhysicalTileLocation> location =
        getPhysicalLocation(placement, tile.id, anchor);
    if (failed(location))
      return failure();
    if (location->physicalTileId < 0 || location->row < 0 ||
        location->column < 0 || location->row >= placement.mesh.rows ||
        location->column >= placement.mesh.columns ||
        location->physicalTileId !=
            location->row * placement.mesh.columns + location->column) {
      return anchor->emitError("logical tile ")
             << tile.id << " has an invalid physical tile location";
    }
  }
  return success();
}

LogicalResult addEndpoint(std::map<EndpointKey, EndpointPlan> &endpoints,
                          DenseMap<int64_t, unsigned> &operationCoverage,
                          const LogicalTileAssignment &assignment,
                          int64_t logicalTileId,
                          const PhysicalTileLocation &location,
                          const ComputeGraph &graph, Operation *anchor) {
  if (assignment.operationId < 0 ||
      assignment.operationId >= static_cast<int64_t>(graph.operations.size())) {
    return anchor->emitError("logical-tile assignment references unknown "
                             "operation ")
           << assignment.operationId;
  }
  EndpointKey key{assignment.operationId, assignment.workUnitId};
  if (endpoints.count(key)) {
    return anchor->emitError("mapping endpoint (")
           << key.first << ", " << key.second << ") is assigned more than once";
  }
  const ComputeOperation &operation = graph.operations[assignment.operationId];
  EndpointPlan endpoint;
  endpoint.key = key;
  endpoint.leafId = assignment.leafId;
  endpoint.logicalTileId = logicalTileId;
  endpoint.location = location;
  endpoint.laneKind = assignment.laneKind;
  endpoint.laneIndex = assignment.laneIndex;
  endpoint.operationKind = operation.kind;
  if (assignment.workUnitId < 0) {
    // A stage may contain nested implementation operations, such as the
    // arith body of a linalg op. Only operations carrying the stage's mapping
    // identity are independently outlined; nested operations remain inside
    // their owning operation and are cloned with it.
    for (Operation *member : operation.members)
      if (member->hasAttr(kStageIdAttrName))
        endpoint.mappedOperations.push_back(member);

    // Digital expansion removes stage identities before exposing independent
    // structured operations. An operation that cannot be split still has a
    // valid mapping endpoint and must be outlined as its original root.
    if (endpoint.mappedOperations.empty())
      endpoint.mappedOperations.push_back(operation.operation);
  }
  endpoints.emplace(key, std::move(endpoint));
  ++operationCoverage[assignment.operationId];
  return success();
}

FailureOr<std::map<EndpointKey, EndpointPlan>>
buildEndpoints(const LogicalTileGraph &tileGraph,
               const LogicalTilePlacementPlan &placement,
               const ComputeGraph &graph, Operation *anchor) {
  std::map<EndpointKey, EndpointPlan> endpoints;
  DenseMap<int64_t, unsigned> operationCoverage;
  for (const LogicalTile &tile : tileGraph.tiles) {
    FailureOr<PhysicalTileLocation> location =
        getPhysicalLocation(placement, tile.id, anchor);
    if (failed(location))
      return failure();
    for (const LogicalTileAssignment &assignment : tile.digitalAssignments) {
      if (failed(addEndpoint(endpoints, operationCoverage, assignment, tile.id,
                             *location, graph, anchor)))
        return failure();
    }
    for (const LogicalTileAnalogLane &lane : tile.analogLanes) {
      for (const LogicalTileAssignment &assignment : lane.assignments) {
        if (failed(addEndpoint(endpoints, operationCoverage, assignment,
                               tile.id, *location, graph, anchor)))
          return failure();
      }
    }
  }
  for (const ComputeOperation &operation : graph.operations) {
    if (!operationCoverage.contains(operation.id)) {
      operation.operation->emitError(
          "compute operation is absent from the locked logical-tile plan");
      return failure();
    }
  }
  return endpoints;
}

LogicalResult replaceGeneratedSlices(
    IRRewriter &rewriter, ArrayRef<Operation *> generatedSlices,
    const DenseMap<Value, SmallVector<TilePart>> &partsByFullValue) {
  for (Operation *generated : generatedSlices) {
    auto slice = dyn_cast_or_null<tensor::ExtractSliceOp>(generated);
    if (!slice || !slice->getBlock())
      continue;
    auto found = partsByFullValue.find(slice.getSource());
    if (found != partsByFullValue.end()) {
      const TilePart *match = nullptr;
      for (const TilePart &part : found->second) {
        if (!matchesStaticSlice(slice, part.offsets, part.sizes))
          continue;
        if (match)
          return slice.emitOpError(
              "generated tile slice matches multiple producer work units");
        match = &part;
      }
      if (match) {
        if (slice.getType() != match->value.getType())
          return slice.emitOpError(
              "generated tile slice type disagrees with producer work unit");
        rewriter.replaceOp(slice, match->value);
        continue;
      }
    }

    if (!slice.getSource().getDefiningOp<tensor::EmptyOp>())
      continue;
    auto tileType = dyn_cast<RankedTensorType>(slice.getType());
    if (!tileType || !tileType.hasStaticShape())
      return slice.emitOpError(
          "generated destination slice must have a static ranked type");
    rewriter.setInsertionPoint(slice);
    Value empty = rewriter.create<tensor::EmptyOp>(
        slice.getLoc(), tileType.getShape(), tileType.getElementType());
    rewriter.replaceOp(slice, empty);
  }
  return success();
}

LogicalResult
materializeWorkUnits(func::FuncOp function, const ComputeGraph &graph,
                     const ResourceAllocationTree &tree,
                     std::map<EndpointKey, EndpointPlan> &endpoints) {
  DenseMap<int64_t, SmallVector<const MappingWorkUnit *>> byOperation;
  for (const MappingWorkUnit &workUnit : tree.workUnits)
    byOperation[workUnit.operationId].push_back(&workUnit);
  if (byOperation.empty())
    return success();

  DenseMap<int64_t, unsigned> topologicalPosition;
  for (auto [position, operationId] : llvm::enumerate(graph.topologicalOrder))
    topologicalPosition[operationId] = position;
  SmallVector<int64_t> operationIds;
  operationIds.reserve(byOperation.size());
  for (const auto &entry : byOperation)
    operationIds.push_back(entry.first);
  llvm::sort(operationIds, [&](int64_t lhs, int64_t rhs) {
    return topologicalPosition.lookup(lhs) < topologicalPosition.lookup(rhs);
  });

  IRRewriter rewriter(function.getContext());
  DenseMap<Value, SmallVector<TilePart>> partsByFullValue;
  for (int64_t operationId : operationIds) {
    if (operationId < 0 ||
        operationId >= static_cast<int64_t>(graph.operations.size()))
      return function.emitError("work unit references unknown operation ")
             << operationId;
    const ComputeOperation &computeOperation = graph.operations[operationId];
    Operation *operation = computeOperation.operation;
    auto tiling = dyn_cast<TilingInterface>(operation);
    if (!tiling)
      return operation->emitError(
          "expanded mapping work requires TilingInterface materialization");

    SmallVector<const MappingWorkUnit *> workUnits = byOperation[operationId];
    llvm::sort(workUnits,
               [](const MappingWorkUnit *lhs, const MappingWorkUnit *rhs) {
                 return lhs->id < rhs->id;
               });
    DenseMap<int64_t, SmallVector<TilePart>> partsByResult;
    for (const MappingWorkUnit *workUnit : workUnits) {
      auto endpoint = endpoints.find({operationId, workUnit->id});
      if (endpoint == endpoints.end()) {
        return operation->emitError("work unit ")
               << workUnit->id << " has no logical-tile assignment";
      }
      SmallVector<OpFoldResult> iterationOffsets;
      SmallVector<OpFoldResult> iterationSizes;
      for (int64_t value : workUnit->iterationOffsets)
        iterationOffsets.push_back(rewriter.getIndexAttr(value));
      for (int64_t value : workUnit->iterationSizes)
        iterationSizes.push_back(rewriter.getIndexAttr(value));

      rewriter.setInsertionPoint(operation);
      FailureOr<TilingResult> tiled = tiling.getTiledImplementation(
          rewriter, iterationOffsets, iterationSizes);
      if (failed(tiled) || workUnit->resultNumber < 0 ||
          workUnit->resultNumber >=
              static_cast<int64_t>(tiled->tiledValues.size()) ||
          tiled->tiledOps.empty()) {
        return operation->emitError("failed to materialize mapping work unit ")
               << workUnit->id;
      }
      if (failed(replaceGeneratedSlices(rewriter, tiled->generatedSlices,
                                        partsByFullValue)))
        return failure();

      EndpointPlan &plan = endpoint->second;
      plan.mappedOperations.append(tiled->tiledOps.begin(),
                                   tiled->tiledOps.end());
      Value tiledValue = tiled->tiledValues[workUnit->resultNumber];
      plan.producedValues.push_back(tiledValue);
      for (Operation *tiledOperation : tiled->tiledOps) {
        tiledOperation->setAttr(kMappingOperationIdAttrName,
                                rewriter.getI64IntegerAttr(operationId));
        tiledOperation->setAttr(kMappingWorkUnitIdAttrName,
                                rewriter.getI64IntegerAttr(workUnit->id));
        tiledOperation->setAttr(kRALeafIdAttrName,
                                rewriter.getI64IntegerAttr(plan.leafId));
      }
      partsByResult[workUnit->resultNumber].push_back(
          {workUnit->id, workUnit->resultOffsets, workUnit->resultSizes,
           tiledValue});
    }

    for (auto &entry : partsByResult) {
      int64_t resultNumber = entry.first;
      if (resultNumber < 0 ||
          resultNumber >= static_cast<int64_t>(operation->getNumResults()))
        return operation->emitError("work unit has invalid result number ")
               << resultNumber;
      Value originalResult = operation->getResult(resultNumber);
      auto resultType = dyn_cast<RankedTensorType>(originalResult.getType());
      if (!resultType || !resultType.hasStaticShape())
        return operation->emitError(
            "work-unit reassembly requires a static ranked tensor result");

      SmallVector<TilePart> &parts = entry.second;
      llvm::sort(parts, [](const TilePart &lhs, const TilePart &rhs) {
        return lhs.workUnitId < rhs.workUnitId;
      });
      if (failed(validateTilePartition(parts, resultType, operation)) ||
          failed(rewriteStaticSubsetConsumers(
              rewriter, originalResult, resultType, parts, partsByFullValue)))
        return failure();

      // A static subset consumer can be satisfied directly from producer
      // tiles. Reconstruct the full result only for remaining whole-value or
      // unsupported consumers.
      if (originalResult.use_empty())
        continue;

      rewriter.setInsertionPoint(operation);
      Value assembled = rewriter.create<tensor::EmptyOp>(
          operation->getLoc(), resultType.getShape(),
          resultType.getElementType());
      SmallVector<OpFoldResult> strides(resultType.getRank(),
                                        rewriter.getIndexAttr(1));
      for (const TilePart &part : parts) {
        SmallVector<OpFoldResult> partOffsets;
        SmallVector<OpFoldResult> partSizes;
        for (int64_t value : part.offsets)
          partOffsets.push_back(rewriter.getIndexAttr(value));
        for (int64_t value : part.sizes)
          partSizes.push_back(rewriter.getIndexAttr(value));
        assembled = rewriter
                        .create<tensor::InsertSliceOp>(
                            operation->getLoc(), part.value, assembled,
                            partOffsets, partSizes, strides)
                        .getResult();
      }
      originalResult.replaceAllUsesWith(assembled);
      partsByFullValue[assembled] = parts;
    }
  }
  return success();
}

FailureOr<SmallVector<EndpointPlan>>
flattenEndpoints(std::map<EndpointKey, EndpointPlan> &&orderedEndpoints,
                 Operation *anchor) {
  SmallVector<EndpointPlan> endpoints;
  endpoints.reserve(orderedEndpoints.size());
  for (auto &entry : orderedEndpoints) {
    if (entry.second.mappedOperations.empty()) {
      anchor->emitError("mapping endpoint (")
          << entry.first.first << ", " << entry.first.second
          << ") has no materialized operations";
      return failure();
    }
    endpoints.push_back(std::move(entry.second));
  }
  return endpoints;
}

FailureOr<SmallVector<RoutinePlan, 0>>
buildRoutineRegions(ArrayRef<EndpointPlan> endpoints) {
  SmallVector<RoutinePlan, 0> routines;
  routines.reserve(endpoints.size());
  for (unsigned index = 0; index < endpoints.size(); ++index) {
    const EndpointPlan &endpoint = endpoints[index];
    RoutinePlan routine;
    routine.physicalTileId = endpoint.location.physicalTileId;
    routine.tileRow = endpoint.location.row;
    routine.tileCol = endpoint.location.column;
    routine.endpointIndices.push_back(index);
    appendUnique(routine.logicalTileIds, endpoint.logicalTileId);
    appendUnique(routine.sourceLeafIds, endpoint.leafId);
    routine.boot = endpoint.operationKind == ComputeOperationKind::MatrixSetup;
    for (Operation *operation : endpoint.mappedOperations)
      routine.selectedOperations.insert(operation);
    routines.push_back(std::move(routine));
  }
  return routines;
}

LogicalResult copyReductionMetadata(func::FuncOp function,
                                    const RoutinePlan &routine) {
  static constexpr StringLiteral mappingNames[] = {
      kReductionTreeIdAttrName, kReductionNodeIdAttrName,
      kReductionLevelAttrName, kReductionOrdinalAttrName,
      kReductionWidthAttrName};
  static constexpr StringLiteral taskNames[] = {
      task_graph_attrs::kTaskReductionTreeIdAttrName,
      task_graph_attrs::kTaskReductionLevelAttrName,
      task_graph_attrs::kTaskReductionWidthAttrName};

  DenseMap<StringRef, Attribute> values;
  for (Operation *operation : routine.selectedOperations) {
    for (StringRef name : mappingNames) {
      Attribute value = operation->getAttr(name);
      if (!value)
        continue;
      auto [found, inserted] = values.try_emplace(name, value);
      if (!inserted && found->second != value) {
        return operation->emitError(
                   "outlined routine contains conflicting reduction metadata '")
               << name << "'";
      }
    }
    for (StringRef name : taskNames) {
      Attribute value = operation->getAttr(name);
      if (!value)
        continue;
      auto [found, inserted] = values.try_emplace(name, value);
      if (!inserted && found->second != value) {
        return operation->emitError(
                   "outlined routine contains conflicting reduction metadata '")
               << name << "'";
      }
    }
    Attribute reduction =
        operation->getAttr(task_graph_attrs::kTaskReductionAttrName);
    if (reduction) {
      auto [found, inserted] = values.try_emplace(
          task_graph_attrs::kTaskReductionAttrName, reduction);
      if (!inserted && found->second != reduction) {
        return operation->emitError(
            "outlined routine contains conflicting reduction kinds");
      }
    }
  }

  bool hasReduction = values.contains(kReductionTreeIdAttrName);
  if (!hasReduction)
    return success();
  for (StringRef name : mappingNames) {
    if (!values.contains(name))
      return function.emitError("outlined reduction routine is missing '")
             << name << "'";
  }
  for (StringRef name : taskNames) {
    if (!values.contains(name))
      return function.emitError("outlined reduction routine is missing '")
             << name << "'";
  }
  if (!values.contains(task_graph_attrs::kTaskReductionAttrName)) {
    return function.emitError(
        "outlined reduction routine is missing reduction semantics");
  }
  for (const auto &entry : values)
    function->setAttr(entry.first, entry.second);
  return success();
}

SmallVector<Value> getExternalValuesUsedBy(Operation *root) {
  SmallVector<Value> values;
  auto isDefinedInside = [&](Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation *parent = argument.getOwner()->getParentOp();
      return parent == root || (parent && root->isProperAncestor(parent));
    }
    Operation *defining = value.getDefiningOp();
    return defining == root || (defining && root->isProperAncestor(defining));
  };
  root->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands()) {
      if (!isDefinedInside(operand) && !llvm::is_contained(values, operand))
        values.push_back(operand);
    }
  });
  return values;
}

LogicalResult collectRoutineValue(
    unsigned routineIndex, Value value, func::FuncOp source,
    SmallVectorImpl<RoutinePlan> &routines,
    const DenseMap<Operation *, unsigned> &mappedOwner,
    llvm::SmallDenseSet<std::pair<unsigned, Value>, 32> &active) {
  RoutinePlan &routine = routines[routineIndex];
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (argument.getOwner()->getParentOp() != source.getOperation())
      return source.emitError(
          "routine closure encountered a non-entry block argument");
    appendUnique(routine.inputs, value);
    return success();
  }

  Operation *defining = value.getDefiningOp();
  if (!defining)
    return source.emitError("routine closure encountered a value without a "
                            "definition");
  auto owner = mappedOwner.find(defining);
  if (owner != mappedOwner.end()) {
    if (owner->second != routineIndex) {
      appendUnique(routine.inputs, value);
      appendUnique(routines[owner->second].outputs, value);
    }
    return success();
  }
  if (defining->getBlock() != &source.getBody().front()) {
    return defining->emitError(
        "cannot outline a support operation outside the source entry block");
  }

  std::pair<unsigned, Value> activeKey{routineIndex, value};
  if (!active.insert(activeKey).second)
    return defining->emitError("cycle while constructing routine closure");
  routine.selectedOperations.insert(defining);
  for (Value operand : getExternalValuesUsedBy(defining)) {
    if (failed(collectRoutineValue(routineIndex, operand, source, routines,
                                   mappedOwner, active)))
      return failure();
  }
  active.erase(activeKey);
  return success();
}

LogicalResult
buildRoutineClosures(func::FuncOp source,
                     SmallVectorImpl<RoutinePlan> &routines,
                     DenseMap<Operation *, unsigned> &mappedOwner) {
  for (auto [routineIndex, routine] : llvm::enumerate(routines)) {
    for (Operation *operation : routine.selectedOperations) {
      if (!mappedOwner.try_emplace(operation, routineIndex).second)
        return operation->emitError(
            "one mapped operation belongs to multiple routine regions");
    }
  }

  for (unsigned routineIndex = 0; routineIndex < routines.size();
       ++routineIndex) {
    SmallVector<Operation *> seeds(
        routines[routineIndex].selectedOperations.begin(),
        routines[routineIndex].selectedOperations.end());
    llvm::SmallDenseSet<std::pair<unsigned, Value>, 32> active;
    for (Operation *operation : seeds) {
      for (Value operand : getExternalValuesUsedBy(operation)) {
        if (failed(collectRoutineValue(routineIndex, operand, source, routines,
                                       mappedOwner, active)))
          return failure();
      }
    }
  }
  return success();
}

void collectMappedOwners(Value value,
                         const DenseMap<Operation *, unsigned> &mappedOwner,
                         llvm::SmallDenseSet<unsigned> &owners,
                         llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (isa<BlockArgument>(value))
    return;
  Operation *defining = value.getDefiningOp();
  if (!defining || !visited.insert(defining).second)
    return;
  auto owner = mappedOwner.find(defining);
  if (owner != mappedOwner.end()) {
    owners.insert(owner->second);
    return;
  }
  for (Value operand : defining->getOperands())
    collectMappedOwners(operand, mappedOwner, owners, visited);
}

LogicalResult
attachModelOutputs(func::FuncOp source, SmallVectorImpl<RoutinePlan> &routines,
                   const DenseMap<Operation *, unsigned> &mappedOwner,
                   SmallVectorImpl<std::pair<unsigned, Value>> &modelOutputs) {
  auto returnOp =
      dyn_cast<func::ReturnOp>(source.getBody().front().getTerminator());
  if (!returnOp)
    return source.emitError("expected a func.return terminator");
  for (Value value : returnOp.getOperands()) {
    llvm::SmallDenseSet<unsigned> owners;
    llvm::SmallPtrSet<Operation *, 32> visited;
    collectMappedOwners(value, mappedOwner, owners, visited);
    if (owners.empty()) {
      return source.emitError(
          "model output is not derived from a mapped compute operation");
    }
    unsigned owner = *std::min_element(owners.begin(), owners.end());
    llvm::SmallDenseSet<std::pair<unsigned, Value>, 32> active;
    if (failed(collectRoutineValue(owner, value, source, routines, mappedOwner,
                                   active)))
      return failure();
    appendUnique(routines[owner].outputs, value);
    modelOutputs.emplace_back(owner, value);
  }
  return success();
}

SmallVector<unsigned> getRoutineOrder(ArrayRef<RoutinePlan> routines,
                                      ArrayRef<EndpointPlan> endpoints) {
  SmallVector<unsigned> order(routines.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::sort(order, [&](unsigned lhs, unsigned rhs) {
    const RoutinePlan &left = routines[lhs];
    const RoutinePlan &right = routines[rhs];
    auto key = [&](const RoutinePlan &routine) {
      int64_t operationId = std::numeric_limits<int64_t>::max();
      int64_t workUnitId = std::numeric_limits<int64_t>::max();
      for (unsigned endpointIndex : routine.endpointIndices) {
        operationId = std::min(operationId, endpoints[endpointIndex].key.first);
        workUnitId = std::min(workUnitId, endpoints[endpointIndex].key.second);
      }
      return std::tuple<int64_t, int64_t, int64_t, bool>{
          operationId, workUnitId, routine.physicalTileId, !routine.boot};
    };
    return key(left) < key(right);
  });
  return order;
}

LogicalResult assignRoutineIds(SmallVectorImpl<RoutinePlan> &routines,
                               ArrayRef<unsigned> order) {
  DenseMap<int64_t, int64_t> nextLocalIndex;
  for (auto [globalId, routineIndex] : llvm::enumerate(order)) {
    RoutinePlan &routine = routines[routineIndex];
    routine.globalId = globalId;
    routine.localIndex = nextLocalIndex[routine.physicalTileId]++;
  }
  return success();
}

FailureOr<unsigned> findOutputPort(const RoutinePlan &routine, Value value,
                                   Operation *anchor) {
  auto found = llvm::find(routine.outputs, value);
  if (found == routine.outputs.end()) {
    anchor->emitError("routine output value was not registered");
    return failure();
  }
  return static_cast<unsigned>(std::distance(routine.outputs.begin(), found));
}

FailureOr<unsigned>
findSourceRoutine(Value value,
                  const DenseMap<Operation *, unsigned> &mappedOwner,
                  Operation *anchor) {
  Operation *defining = value.getDefiningOp();
  auto found = mappedOwner.find(defining);
  if (!defining || found == mappedOwner.end()) {
    anchor->emitError(
        "routine boundary is not produced by another mapped routine");
    return failure();
  }
  return found->second;
}

int64_t findDependencyTensorId(const RoutinePlan &source,
                               const RoutinePlan &target,
                               ArrayRef<EndpointPlan> endpoints,
                               const LogicalTileGraph &tileGraph,
                               int64_t byteSize) {
  std::set<EndpointKey> sourceEndpoints;
  std::set<EndpointKey> targetEndpoints;
  for (unsigned endpoint : source.endpointIndices)
    sourceEndpoints.insert(endpoints[endpoint].key);
  for (unsigned endpoint : target.endpointIndices)
    targetEndpoints.insert(endpoints[endpoint].key);
  auto containsEndpoint = [](const std::set<EndpointKey> &endpointSet,
                             int64_t operationId, int64_t workUnitId) {
    if (endpointSet.count({operationId, workUnitId}))
      return true;
    if (workUnitId >= 0)
      return false;
    return llvm::any_of(endpointSet, [&](const EndpointKey &endpoint) {
      return endpoint.first == operationId;
    });
  };
  auto matches = [&](const LogicalTileDependency &dependency) {
    return containsEndpoint(sourceEndpoints, dependency.sourceOperationId,
                            dependency.sourceWorkUnitId) &&
           containsEndpoint(targetEndpoints, dependency.targetOperationId,
                            dependency.targetWorkUnitId) &&
           (dependency.byteSize == byteSize || dependency.byteSize == 0 ||
            byteSize == 0);
  };
  int64_t best = -1;
  for (const LogicalTile &tile : tileGraph.tiles) {
    for (const LogicalTileDependency &dependency : tile.internalDependencies) {
      if (matches(dependency) && (best < 0 || dependency.tensorId < best))
        best = dependency.tensorId;
    }
  }
  for (const LogicalTileEdge &edge : tileGraph.edges) {
    for (const LogicalTileDependency &dependency : edge.dependencies) {
      if (matches(dependency) && (best < 0 || dependency.tensorId < best))
        best = dependency.tensorId;
    }
  }
  return best;
}

LogicalResult assignResourcesAndConnections(
    func::FuncOp source, SmallVectorImpl<RoutinePlan> &routines,
    ArrayRef<unsigned> routineOrder, ArrayRef<EndpointPlan> endpoints,
    const LogicalTileGraph &tileGraph,
    const DenseMap<Operation *, unsigned> &mappedOwner,
    DenseMap<Value, int64_t> &resourceIdByValue,
    SmallVectorImpl<RouteRecord> &routes,
    SmallVectorImpl<LocalBindingRecord> &localBindings) {
  int64_t nextResourceId = 0;
  for (BlockArgument argument : source.getArguments()) {
    bool used = llvm::any_of(routines, [&](const RoutinePlan &routine) {
      return llvm::is_contained(routine.inputs, Value(argument));
    });
    if (used)
      resourceIdByValue[argument] = nextResourceId++;
  }
  for (unsigned routineIndex : routineOrder) {
    for (Value output : routines[routineIndex].outputs) {
      if (!resourceIdByValue.contains(output))
        resourceIdByValue[output] = nextResourceId++;
    }
  }

  for (unsigned targetIndex : routineOrder) {
    const RoutinePlan &target = routines[targetIndex];
    for (auto [inputIndex, input] : llvm::enumerate(target.inputs)) {
      if (isa<BlockArgument>(input))
        continue;
      FailureOr<unsigned> sourceIndex =
          findSourceRoutine(input, mappedOwner, source);
      if (failed(sourceIndex))
        return failure();
      const RoutinePlan &producer = routines[*sourceIndex];
      FailureOr<unsigned> sourceOutput =
          findOutputPort(producer, input, source);
      FailureOr<int64_t> byteSize = getStaticByteSize(input.getType(), source);
      if (failed(sourceOutput) || failed(byteSize))
        return failure();
      int64_t resourceId = resourceIdByValue.lookup(input);
      if (producer.physicalTileId == target.physicalTileId) {
        localBindings.push_back({*sourceIndex, *sourceOutput, targetIndex,
                                 static_cast<unsigned>(inputIndex), resourceId,
                                 *byteSize});
        continue;
      }
      if (isa<LogicalArrayType>(input.getType())) {
        return source.emitError(
            "logical-array state cannot cross a physical tile boundary");
      }
      routes.push_back({-1, *sourceIndex, *sourceOutput, targetIndex,
                        static_cast<unsigned>(inputIndex), resourceId,
                        findDependencyTensorId(producer, target, endpoints,
                                               tileGraph, *byteSize),
                        *byteSize});
    }
  }

  llvm::sort(routes, [&](const RouteRecord &lhs, const RouteRecord &rhs) {
    const RoutinePlan &lhsSource = routines[lhs.sourceRoutine];
    const RoutinePlan &rhsSource = routines[rhs.sourceRoutine];
    const RoutinePlan &lhsTarget = routines[lhs.destinationRoutine];
    const RoutinePlan &rhsTarget = routines[rhs.destinationRoutine];
    return std::tuple<int64_t, unsigned, int64_t, int64_t, unsigned>{
               lhsSource.globalId, lhs.sourceOutput, lhsTarget.physicalTileId,
               lhsTarget.globalId, lhs.destinationInput} <
           std::tuple<int64_t, unsigned, int64_t, int64_t, unsigned>{
               rhsSource.globalId, rhs.sourceOutput, rhsTarget.physicalTileId,
               rhsTarget.globalId, rhs.destinationInput};
  });
  for (auto [routeId, route] : llvm::enumerate(routes))
    route.id = routeId;

  llvm::sort(localBindings, [&](const LocalBindingRecord &lhs,
                                const LocalBindingRecord &rhs) {
    return std::tuple<int64_t, unsigned, int64_t, unsigned>{
               routines[lhs.sourceRoutine].globalId, lhs.sourceOutput,
               routines[lhs.destinationRoutine].globalId,
               lhs.destinationInput} <
           std::tuple<int64_t, unsigned, int64_t, unsigned>{
               routines[rhs.sourceRoutine].globalId, rhs.sourceOutput,
               routines[rhs.destinationRoutine].globalId, rhs.destinationInput};
  });
  return success();
}

LogicalResult
verifyRoutineDependencyDAG(Operation *anchor, ArrayRef<RoutinePlan> routines,
                           ArrayRef<RouteRecord> routes,
                           ArrayRef<LocalBindingRecord> localBindings) {
  SmallVector<RoutineDependencyEdge> edges;
  edges.reserve(routes.size() + localBindings.size());
  for (const RouteRecord &route : routes) {
    edges.push_back({route.sourceRoutine, route.destinationRoutine,
                     RoutineDependencyKind::Route, route.id, route.byteSize});
  }
  for (auto [bindingId, binding] : llvm::enumerate(localBindings)) {
    edges.push_back({binding.sourceRoutine, binding.destinationRoutine,
                     RoutineDependencyKind::LocalBinding,
                     static_cast<int64_t>(bindingId), binding.byteSize});
  }

  for (const RoutineDependencyEdge &edge : edges) {
    if (edge.sourceRoutine >= routines.size() ||
        edge.destinationRoutine >= routines.size()) {
      return anchor->emitError(
          "outlined runtime dependency references an unknown routine");
    }
  }
  llvm::sort(edges, [&](const RoutineDependencyEdge &lhs,
                        const RoutineDependencyEdge &rhs) {
    return std::tuple<int64_t, int64_t, unsigned, int64_t>{
               routines[lhs.sourceRoutine].globalId,
               routines[lhs.destinationRoutine].globalId,
               static_cast<unsigned>(lhs.kind),
               lhs.id} < std::tuple<int64_t, int64_t, unsigned, int64_t>{
                             routines[rhs.sourceRoutine].globalId,
                             routines[rhs.destinationRoutine].globalId,
                             static_cast<unsigned>(rhs.kind), rhs.id};
  });

  SmallVector<SmallVector<unsigned>> outgoing(routines.size());
  for (auto [edgeIndex, edge] : llvm::enumerate(edges))
    outgoing[edge.sourceRoutine].push_back(edgeIndex);

  SmallVector<unsigned> traversalOrder(routines.size());
  std::iota(traversalOrder.begin(), traversalOrder.end(), 0);
  llvm::sort(traversalOrder, [&](unsigned lhs, unsigned rhs) {
    return routines[lhs].globalId < routines[rhs].globalId;
  });

  SmallVector<uint8_t> state(routines.size(), 0);
  SmallVector<std::optional<unsigned>> parentEdge(routines.size());
  SmallVector<unsigned> cycle;
  std::function<bool(unsigned)> visit = [&](unsigned routineIndex) {
    state[routineIndex] = 1;
    for (unsigned edgeIndex : outgoing[routineIndex]) {
      const RoutineDependencyEdge &edge = edges[edgeIndex];
      unsigned destination = edge.destinationRoutine;
      if (state[destination] == 0) {
        parentEdge[destination] = edgeIndex;
        if (visit(destination))
          return true;
        continue;
      }
      if (state[destination] != 1)
        continue;

      SmallVector<unsigned> reversedPath;
      unsigned cursor = routineIndex;
      while (cursor != destination) {
        if (!parentEdge[cursor])
          return false;
        unsigned pathEdge = *parentEdge[cursor];
        reversedPath.push_back(pathEdge);
        cursor = edges[pathEdge].sourceRoutine;
      }
      cycle.assign(reversedPath.rbegin(), reversedPath.rend());
      cycle.push_back(edgeIndex);
      return true;
    }
    state[routineIndex] = 2;
    return false;
  };

  for (unsigned routineIndex : traversalOrder) {
    if (state[routineIndex] == 0 && visit(routineIndex))
      break;
  }
  if (cycle.empty())
    return success();

  InFlightDiagnostic diagnostic =
      anchor->emitError("outlined runtime dependency cycle:");
  for (unsigned edgeIndex : cycle) {
    const RoutineDependencyEdge &edge = edges[edgeIndex];
    const RoutinePlan &source = routines[edge.sourceRoutine];
    const RoutinePlan &destination = routines[edge.destinationRoutine];
    diagnostic << "\n  routine " << source.globalId << " on tile "
               << source.physicalTileId << " --";
    if (edge.kind == RoutineDependencyKind::Route) {
      diagnostic << "route " << edge.id;
    } else if (source.boot) {
      diagnostic << "boot dependency " << edge.id;
    } else if (edge.byteSize == 0) {
      diagnostic << "zero-byte synchronization " << edge.id;
    } else {
      diagnostic << "local binding " << edge.id;
    }
    diagnostic << "--> routine " << destination.globalId << " on tile "
               << destination.physicalTileId;
  }
  return failure();
}

ArrayAttr buildRouteAttrs(OpBuilder &builder, ArrayRef<RouteRecord> routes,
                          ArrayRef<RoutinePlan> routines,
                          function_ref<bool(const RouteRecord &)> predicate) {
  SmallVector<Attribute> attributes;
  for (const RouteRecord &route : routes) {
    if (!predicate(route))
      continue;
    const RoutinePlan &source = routines[route.sourceRoutine];
    const RoutinePlan &target = routines[route.destinationRoutine];
    attributes.push_back(TileRoutineRouteAttr::get(
        builder.getContext(), builder.getI64IntegerAttr(route.id),
        builder.getI64IntegerAttr(source.physicalTileId),
        builder.getI64IntegerAttr(source.globalId),
        builder.getI64IntegerAttr(route.sourceOutput),
        builder.getI64IntegerAttr(target.physicalTileId),
        builder.getI64IntegerAttr(target.globalId),
        builder.getI64IntegerAttr(route.destinationInput),
        builder.getI64IntegerAttr(route.resourceId),
        builder.getI64IntegerAttr(route.tensorId),
        builder.getI64IntegerAttr(route.byteSize)));
  }
  return builder.getArrayAttr(attributes);
}

ArrayAttr
buildLocalBindingAttrs(OpBuilder &builder,
                       ArrayRef<LocalBindingRecord> bindings,
                       ArrayRef<RoutinePlan> routines,
                       std::optional<int64_t> physicalTileId = std::nullopt) {
  SmallVector<Attribute> attributes;
  for (const LocalBindingRecord &binding : bindings) {
    const RoutinePlan &source = routines[binding.sourceRoutine];
    const RoutinePlan &target = routines[binding.destinationRoutine];
    if (physicalTileId && source.physicalTileId != *physicalTileId)
      continue;
    attributes.push_back(TileRoutineBindingAttr::get(
        builder.getContext(), builder.getI64IntegerAttr(source.globalId),
        builder.getI64IntegerAttr(binding.sourceOutput),
        builder.getI64IntegerAttr(target.globalId),
        builder.getI64IntegerAttr(binding.destinationInput),
        builder.getI64IntegerAttr(binding.resourceId),
        builder.getI64IntegerAttr(binding.byteSize)));
  }
  return builder.getArrayAttr(attributes);
}

LogicalResult createRoutineFunctions(
    ModuleOp outer, func::FuncOp source, SmallVectorImpl<RoutinePlan> &routines,
    ArrayRef<unsigned> routineOrder, ArrayRef<EndpointPlan> endpoints,
    int64_t arraysPerTile, const DenseMap<Value, int64_t> &resourceIdByValue,
    DenseMap<int64_t, ModuleOp> &tileModules) {
  OpBuilder outerBuilder(outer.getContext());
  std::set<int64_t> physicalTiles;
  for (const RoutinePlan &routine : routines)
    physicalTiles.insert(routine.physicalTileId);
  for (int64_t physicalTileId : physicalTiles) {
    const RoutinePlan *representative = nullptr;
    for (const RoutinePlan &routine : routines) {
      if (routine.physicalTileId == physicalTileId) {
        representative = &routine;
        break;
      }
    }
    outerBuilder.setInsertionPointToEnd(outer.getBody());
    ModuleOp tile = outerBuilder.create<ModuleOp>(
        source.getLoc(), "tile_" + std::to_string(physicalTileId));
    tile->setAttr(kDeploymentPhysicalTileIdAttr,
                  outerBuilder.getI64IntegerAttr(physicalTileId));
    tile->setAttr(kDeploymentTileRowAttr,
                  outerBuilder.getI64IntegerAttr(representative->tileRow));
    tile->setAttr(kDeploymentTileColAttr,
                  outerBuilder.getI64IntegerAttr(representative->tileCol));
    tileModules[physicalTileId] = tile;
  }

  Block &sourceBlock = source.getBody().front();
  for (unsigned routineIndex : routineOrder) {
    RoutinePlan &routine = routines[routineIndex];
    DenseMap<Operation *, int64_t> analogLaneByOperation;
    for (unsigned endpointIndex : routine.endpointIndices) {
      const EndpointPlan &endpoint = endpoints[endpointIndex];
      if (endpoint.laneKind != LogicalLaneKind::Analog)
        continue;
      if (endpoint.laneIndex < 0 || endpoint.laneIndex >= arraysPerTile) {
        return source.emitError("analog routine endpoint has invalid lane ")
               << endpoint.laneIndex;
      }
      for (Operation *operation : endpoint.mappedOperations) {
        auto [found, inserted] =
            analogLaneByOperation.try_emplace(operation, endpoint.laneIndex);
        if (!inserted && found->second != endpoint.laneIndex) {
          return operation->emitError(
              "one operation is assigned to multiple analog lanes");
        }
      }
    }
    SmallVector<Type> inputTypes;
    SmallVector<Type> outputTypes;
    for (Value input : routine.inputs)
      inputTypes.push_back(input.getType());
    for (Value output : routine.outputs)
      outputTypes.push_back(output.getType());

    ModuleOp tile = tileModules.lookup(routine.physicalTileId);
    OpBuilder builder(tile.getContext());
    builder.setInsertionPointToEnd(tile.getBody());
    std::string name = routine.boot ? "boot_setup_" : "routine_";
    name += std::to_string(routine.globalId);
    func::FuncOp function = builder.create<func::FuncOp>(
        source.getLoc(), name,
        builder.getFunctionType(inputTypes, outputTypes));
    function.setPrivate();
    function->setAttr(kDeploymentGlobalRoutineIdAttr,
                      builder.getI64IntegerAttr(routine.globalId));
    function->setAttr(kDeploymentLocalRoutineIndexAttr,
                      builder.getI64IntegerAttr(routine.localIndex));
    function->setAttr(kDeploymentRoutineKindAttr,
                      builder.getStringAttr(routine.boot ? "boot" : "compute"));
    function->setAttr(kDeploymentPhysicalTileIdAttr,
                      builder.getI64IntegerAttr(routine.physicalTileId));
    function->setAttr(kDeploymentLogicalTileIdsAttr,
                      getI64Array(builder, routine.logicalTileIds));
    function->setAttr(kDeploymentSourceLeafIdsAttr,
                      getI64Array(builder, routine.sourceLeafIds));
    if (failed(copyReductionMetadata(function, routine)))
      return failure();
    if (routine.boot) {
      const EndpointPlan &endpoint = endpoints[routine.endpointIndices.front()];
      int64_t physicalArrayId =
          routine.physicalTileId * arraysPerTile + endpoint.laneIndex;
      function->setAttr(tile_runtime_attrs::kTaskLocalArrayIdAttrName,
                        builder.getI64IntegerAttr(endpoint.laneIndex));
      function->setAttr(tile_runtime_attrs::kTaskPhysicalArrayIdAttrName,
                        builder.getI64IntegerAttr(physicalArrayId));
    }

    Block *entry = function.addEntryBlock();
    IRMapping mapping;
    for (auto [index, input] : llvm::enumerate(routine.inputs))
      mapping.map(input, entry->getArgument(index));
    builder.setInsertionPointToStart(entry);
    for (Operation &operation : sourceBlock) {
      if (!routine.selectedOperations.contains(&operation))
        continue;
      Operation *cloned = builder.clone(operation, mapping);
      cloned->setAttr(kDeploymentPhysicalTileIdAttr,
                      builder.getI64IntegerAttr(routine.physicalTileId));
      auto analogLane = analogLaneByOperation.find(&operation);
      if (analogLane != analogLaneByOperation.end()) {
        int64_t physicalArrayId =
            routine.physicalTileId * arraysPerTile + analogLane->second;
        cloned->setAttr(tile_runtime_attrs::kTaskLocalArrayIdAttrName,
                        builder.getI64IntegerAttr(analogLane->second));
        cloned->setAttr(tile_runtime_attrs::kTaskPhysicalArrayIdAttrName,
                        builder.getI64IntegerAttr(physicalArrayId));
      }
    }

    SmallVector<Value> returns;
    for (Value output : routine.outputs) {
      Value mapped = mapping.lookupOrNull(output);
      if (!mapped) {
        return source.emitError("failed to map an outlined routine output");
      }
      returns.push_back(mapped);
    }
    builder.create<func::ReturnOp>(source.getLoc(), returns);

    SmallVector<int64_t> inputResourceIds;
    SmallVector<int64_t> outputResourceIds;
    for (Value input : routine.inputs) {
      auto found = resourceIdByValue.find(input);
      if (found == resourceIdByValue.end())
        return source.emitError("routine input has no global resource ID");
      inputResourceIds.push_back(found->second);
    }
    for (Value output : routine.outputs) {
      auto found = resourceIdByValue.find(output);
      if (found == resourceIdByValue.end())
        return source.emitError("routine output has no global resource ID");
      outputResourceIds.push_back(found->second);
    }
    function->setAttr(kDeploymentInputResourceIdsAttr,
                      getI64Array(builder, inputResourceIds));
    function->setAttr(kDeploymentOutputResourceIdsAttr,
                      getI64Array(builder, outputResourceIds));
    routine.function = function;
  }
  return success();
}

LogicalResult attachDeploymentManifest(
    ModuleOp outer, func::FuncOp source, ArrayRef<RoutinePlan> routines,
    ArrayRef<RouteRecord> routes, ArrayRef<LocalBindingRecord> localBindings,
    ArrayRef<std::pair<unsigned, Value>> modelOutputs,
    ArrayRef<int64_t> modelInputTensorIds,
    ArrayRef<int64_t> modelOutputTensorIds,
    const DenseMap<Value, int64_t> &resourceIdByValue,
    const LogicalTilePlacementPlan &placement,
    DenseMap<int64_t, ModuleOp> &tileModules) {
  OpBuilder builder(outer.getContext());
  SmallVector<int64_t> activeTiles;
  activeTiles.reserve(tileModules.size());
  for (const auto &entry : tileModules)
    activeTiles.push_back(entry.first);
  llvm::sort(activeTiles);

  outer->setAttr(kDeploymentKindAttr, builder.getStringAttr("tile_routines"));
  outer->setAttr(kDeploymentVersionAttr, builder.getI64IntegerAttr(1));
  outer->setAttr(kDeploymentMeshRowsAttr,
                 builder.getI64IntegerAttr(placement.mesh.rows));
  outer->setAttr(kDeploymentMeshColsAttr,
                 builder.getI64IntegerAttr(placement.mesh.columns));
  outer->setAttr(kDeploymentArraysPerTileAttr,
                 builder.getI64IntegerAttr(placement.mesh.arraysPerCore));
  outer->setAttr(kDeploymentActiveTileIdsAttr,
                 getI64Array(builder, activeTiles));
  outer->setAttr(kDeploymentRoutesAttr,
                 buildRouteAttrs(builder, routes, routines,
                                 [](const RouteRecord &) { return true; }));
  outer->setAttr(kDeploymentLocalBindingsAttr,
                 buildLocalBindingAttrs(builder, localBindings, routines));

  SmallVector<Attribute> modelInputs;
  for (const RoutinePlan &routine : routines) {
    for (auto [inputPort, value] : llvm::enumerate(routine.inputs)) {
      auto argument = dyn_cast<BlockArgument>(value);
      if (!argument)
        continue;
      int64_t inputIndex = argument.getArgNumber();
      int64_t tensorId =
          inputIndex < static_cast<int64_t>(modelInputTensorIds.size())
              ? modelInputTensorIds[inputIndex]
              : -1;
      FailureOr<int64_t> byteSize = getStaticByteSize(value.getType(), source);
      if (failed(byteSize))
        return failure();
      modelInputs.push_back(TileRoutineModelIOAttr::get(
          builder.getContext(), builder.getI64IntegerAttr(inputIndex),
          builder.getI64IntegerAttr(routine.physicalTileId),
          builder.getI64IntegerAttr(routine.globalId),
          builder.getI64IntegerAttr(inputPort),
          builder.getI64IntegerAttr(resourceIdByValue.lookup(value)),
          builder.getI64IntegerAttr(tensorId),
          builder.getI64IntegerAttr(*byteSize)));
    }
  }
  outer->setAttr(kDeploymentModelInputsAttr, builder.getArrayAttr(modelInputs));

  SmallVector<Attribute> outputAttributes;
  for (auto [outputIndex, ownerAndValue] : llvm::enumerate(modelOutputs)) {
    const RoutinePlan &routine = routines[ownerAndValue.first];
    FailureOr<unsigned> port =
        findOutputPort(routine, ownerAndValue.second, source);
    FailureOr<int64_t> byteSize =
        getStaticByteSize(ownerAndValue.second.getType(), source);
    if (failed(port) || failed(byteSize))
      return failure();
    int64_t tensorId = outputIndex < modelOutputTensorIds.size()
                           ? modelOutputTensorIds[outputIndex]
                           : -1;
    outputAttributes.push_back(TileRoutineModelIOAttr::get(
        builder.getContext(), builder.getI64IntegerAttr(outputIndex),
        builder.getI64IntegerAttr(routine.physicalTileId),
        builder.getI64IntegerAttr(routine.globalId),
        builder.getI64IntegerAttr(*port),
        builder.getI64IntegerAttr(
            resourceIdByValue.lookup(ownerAndValue.second)),
        builder.getI64IntegerAttr(tensorId),
        builder.getI64IntegerAttr(*byteSize)));
  }
  outer->setAttr(kDeploymentModelOutputsAttr,
                 builder.getArrayAttr(outputAttributes));

  for (int64_t physicalTileId : activeTiles) {
    ModuleOp tile = tileModules.lookup(physicalTileId);
    tile->setAttr(
        kDeploymentIncomingRoutesAttr,
        buildRouteAttrs(
            builder, routes, routines, [&](const RouteRecord &route) {
              return routines[route.destinationRoutine].physicalTileId ==
                     physicalTileId;
            }));
    tile->setAttr(kDeploymentOutgoingRoutesAttr,
                  buildRouteAttrs(
                      builder, routes, routines, [&](const RouteRecord &route) {
                        return routines[route.sourceRoutine].physicalTileId ==
                               physicalTileId;
                      }));
    tile->setAttr(kDeploymentLocalBindingsAttr,
                  buildLocalBindingAttrs(builder, localBindings, routines,
                                         physicalTileId));
  }
  return success();
}

LogicalResult
verifyOutlinedDeployment(ModuleOp outer, ArrayRef<RoutinePlan> routines,
                         ArrayRef<RouteRecord> routes,
                         ArrayRef<LocalBindingRecord> localBindings,
                         const DenseMap<Value, int64_t> &resourceIdByValue,
                         DenseMap<int64_t, ModuleOp> &tileModules) {
  llvm::SmallDenseSet<int64_t> routineIds;
  for (const RoutinePlan &routine : routines) {
    if (!routine.function || routine.function->getParentOfType<ModuleOp>() !=
                                 tileModules.lookup(routine.physicalTileId))
      return outer.emitError("outlined routine is not nested in its owning "
                             "physical tile module");
    if (!routineIds.insert(routine.globalId).second)
      return outer.emitError("duplicate global routine ID ")
             << routine.globalId;
    if (routine.boot && routine.endpointIndices.size() != 1)
      return outer.emitError("boot routine contains multiple mapping "
                             "endpoints");
  }
  llvm::SmallDenseSet<int64_t> routeIds;
  for (const RouteRecord &route : routes) {
    if (route.sourceRoutine >= routines.size() ||
        route.destinationRoutine >= routines.size()) {
      return outer.emitError(
          "remote route references an unknown outlined routine");
    }
    if (route.id < 0 || !routeIds.insert(route.id).second)
      return outer.emitError("duplicate deployment route ID ") << route.id;
    if (routines[route.sourceRoutine].physicalTileId ==
        routines[route.destinationRoutine].physicalTileId)
      return outer.emitError("remote route remains within one physical tile");
    if (route.byteSize <= 0)
      return outer.emitError("remote route requires a positive byte size");
  }
  for (const LocalBindingRecord &binding : localBindings) {
    if (binding.sourceRoutine >= routines.size() ||
        binding.destinationRoutine >= routines.size()) {
      return outer.emitError(
          "local binding references an unknown outlined routine");
    }
    if (routines[binding.sourceRoutine].physicalTileId !=
        routines[binding.destinationRoutine].physicalTileId) {
      return outer.emitError("local binding crosses physical tiles");
    }
    if (binding.byteSize < 0)
      return outer.emitError("local binding requires a non-negative byte size");
  }
  llvm::SmallDenseSet<int64_t> resourceIds;
  for (const auto &entry : resourceIdByValue) {
    if (entry.second < 0 || !resourceIds.insert(entry.second).second)
      return outer.emitError("duplicate or invalid global resource ID");
  }
  for (const RouteRecord &route : routes) {
    if (!resourceIds.contains(route.resourceId))
      return outer.emitError("remote route references an unknown resource ID ")
             << route.resourceId;
  }
  for (const LocalBindingRecord &binding : localBindings) {
    if (!resourceIds.contains(binding.resourceId)) {
      return outer.emitError("local binding references an unknown resource ID ")
             << binding.resourceId;
    }
  }
  return verifyRoutineDependencyDAG(outer, routines, routes, localBindings);
}

LogicalResult outlineFunction(ModuleOp outer, func::FuncOp source) {
  auto treeAttr = source->getAttrOfType<RATreeAttr>(kRATreeAttrName);
  auto tileGraphAttr =
      source->getAttrOfType<LogicalTileGraphAttr>(kLogicalTileGraphAttrName);
  auto placementAttr = source->getAttrOfType<LogicalTilePlacementAttr>(
      kLogicalTilePlacementAttrName);
  if (!treeAttr || !tileGraphAttr || !placementAttr) {
    return source.emitError(
        "tile-routine outlining requires RA tree, logical-tile graph, and "
        "locked physical placement attributes");
  }

  FailureOr<ComputeGraph> graph = buildComputeGraph(source);
  if (failed(graph))
    return failure();
  FailureOr<ResourceAllocationTree> tree =
      deserializeResourceAllocationTree(treeAttr, *graph, source);
  if (failed(tree))
    return failure();
  FailureOr<LogicalTileGraph> tileGraph =
      deserializeLogicalTileGraph(tileGraphAttr, *graph, *tree, source);
  if (failed(tileGraph))
    return failure();
  LogicalTilePlacementProblem placementProblem{*tileGraph, {}, source};
  MappingCostProfile resolvedProfile;
  if (failed(initializeLogicalTilePlacementProblemFromPlan(
          placementAttr, *graph, *tree, resolvedProfile, placementProblem)))
    return failure();
  FailureOr<LogicalTilePlacementPlan> placement =
      deserializeLogicalTilePlacement(placementAttr, placementProblem);
  if (failed(placement) ||
      failed(validateLockedMapping(*tileGraph, *placement, source)))
    return failure();

  SmallVector<int64_t> modelInputTensorIds;
  SmallVector<int64_t> modelOutputTensorIds;
  for (const ComputeTensor &tensor : graph->tensors) {
    if (tensor.isFunctionInput)
      modelInputTensorIds.push_back(tensor.id);
    if (tensor.isFunctionOutput)
      modelOutputTensorIds.push_back(tensor.id);
  }
  llvm::sort(modelInputTensorIds);
  llvm::sort(modelOutputTensorIds);

  FailureOr<std::map<EndpointKey, EndpointPlan>> orderedEndpoints =
      buildEndpoints(*tileGraph, *placement, *graph, source);
  if (failed(orderedEndpoints))
    return failure();
  if (failed(materializeWorkUnits(source, *graph, *tree, *orderedEndpoints)))
    return failure();
  FailureOr<SmallVector<EndpointPlan>> endpoints =
      flattenEndpoints(std::move(*orderedEndpoints), source);
  if (failed(endpoints))
    return failure();
  FailureOr<SmallVector<RoutinePlan, 0>> routines =
      buildRoutineRegions(*endpoints);
  if (failed(routines))
    return failure();

  DenseMap<Operation *, unsigned> mappedOwner;
  if (failed(buildRoutineClosures(source, *routines, mappedOwner)))
    return failure();
  SmallVector<std::pair<unsigned, Value>> modelOutputs;
  if (failed(attachModelOutputs(source, *routines, mappedOwner, modelOutputs)))
    return failure();

  SmallVector<unsigned> routineOrder = getRoutineOrder(*routines, *endpoints);
  if (failed(assignRoutineIds(*routines, routineOrder)))
    return failure();

  DenseMap<Value, int64_t> resourceIdByValue;
  SmallVector<RouteRecord> routes;
  SmallVector<LocalBindingRecord> localBindings;
  if (failed(assignResourcesAndConnections(
          source, *routines, routineOrder, *endpoints, *tileGraph, mappedOwner,
          resourceIdByValue, routes, localBindings)))
    return failure();

  DenseMap<int64_t, ModuleOp> tileModules;
  if (failed(createRoutineFunctions(outer, source, *routines, routineOrder,
                                    *endpoints, placement->mesh.arraysPerCore,
                                    resourceIdByValue, tileModules)) ||
      failed(attachDeploymentManifest(
          outer, source, *routines, routes, localBindings, modelOutputs,
          modelInputTensorIds, modelOutputTensorIds, resourceIdByValue,
          *placement, tileModules)) ||
      failed(verifyOutlinedDeployment(outer, *routines, routes, localBindings,
                                      resourceIdByValue, tileModules)))
    return failure();

  source.erase();
  outer->removeAttr("sculptor.mapping.expanded_digital_operation_count");
  outer->removeAttr("sculptor.mapping.expanded_digital_work_unit_count");
  return success();
}

} // namespace

namespace mlir {
namespace sculptor {

void OutlineTileRoutinesPass::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<func::FuncOp> mappedFunctions;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function->hasAttr(mapping::kLogicalTilePlacementAttrName))
      mappedFunctions.push_back(function);
  }
  if (mappedFunctions.empty()) {
    module.emitError(
        "expected one function with a locked logical-tile placement");
    signalPassFailure();
    return;
  }
  if (mappedFunctions.size() != 1) {
    module.emitError(
        "tile-routine outlining currently requires exactly one mapped "
        "entry function");
    signalPassFailure();
    return;
  }
  if (failed(outlineFunction(module, mappedFunctions.front()))) {
    signalPassFailure();
    return;
  }
  if (failed(verify(module)))
    signalPassFailure();
}

void registerOutlineTileRoutinesPass() {
  PassRegistration<OutlineTileRoutinesPass>();
}

} // namespace sculptor
} // namespace mlir
