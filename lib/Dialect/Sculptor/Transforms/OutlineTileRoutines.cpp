#include "sculptor-mlir/Dialect/Sculptor/Transforms/OutlineTileRoutines.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTaskGraphAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileMemoryPlan.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileRuntimeAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/GolemTilingAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/LogicalTilePlacement.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingCostProfile.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ReductionTree.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ResourceAllocationTree.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ShardDataflow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
constexpr StringLiteral kDeploymentLayerRegionIdsAttr =
    "sculptor.deployment.layer_region_ids";
constexpr StringLiteral kDeploymentSemanticLayerIdsAttr =
    "sculptor.deployment.semantic_layer_ids";
constexpr StringLiteral kDeploymentLayerRegionEpochsAttr =
    "sculptor.deployment.layer_region_epochs";
constexpr StringLiteral kDeploymentInputResourceIdsAttr =
    "sculptor.deployment.input_resource_ids";
constexpr StringLiteral kDeploymentOutputResourceIdsAttr =
    "sculptor.deployment.output_resource_ids";
constexpr StringLiteral kDeploymentControlDependencyIdsAttr =
    "sculptor.deployment.control_dependency_ids";
constexpr StringLiteral kSequenceShardIndexAttr =
    "sculptor.sequence_shard_index";
constexpr StringLiteral kFusionCountAttr =
    "sculptor.memory.fused_producer_consumer_count";
constexpr StringLiteral kFusionBoundaryBytesAttr =
    "sculptor.memory.fused_producer_consumer_boundary_bytes";
constexpr StringLiteral kRoutineCountBeforeFusionAttr =
    "sculptor.memory.routine_count_before_fusion";
constexpr StringLiteral kRoutineCountAfterFusionAttr =
    "sculptor.memory.routine_count_after_fusion";
constexpr StringLiteral kLayerRegionConsolidationCountAttr =
    "sculptor.memory.layer_region_consolidation_count";
constexpr StringLiteral kLayerRegionConsolidationBoundaryBytesAttr =
    "sculptor.memory.layer_region_consolidation_boundary_bytes";
constexpr StringLiteral kStreamedConvPatchCountAttr =
    "sculptor.memory.streamed_conv_patch_count";
constexpr StringLiteral kStreamedConvPatchRowCountAttr =
    "sculptor.memory.streamed_conv_patch_row_count";

struct EndpointPlan {
  EndpointKey key{-1, -1};
  int64_t leafId = -1;
  int64_t logicalTileId = -1;
  PhysicalTileLocation location;
  LogicalLaneKind laneKind = LogicalLaneKind::Digital;
  int64_t laneIndex = -1;
  ComputeOperationKind operationKind = ComputeOperationKind::Structured;
  std::optional<int64_t> laneBindingGroup;
  int64_t layerRegionId = -1;
  std::optional<int64_t> semanticLayerId;
  SmallVector<Operation *> mappedOperations;
  SmallVector<Value> producedValues;
};

struct TilePart {
  int64_t operationId = -1;
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
  SmallVector<int64_t> layerRegionIds;
  SmallVector<int64_t> semanticLayerIds;
  SmallVector<int64_t> layerRegionEpochs;
  llvm::SetVector<Operation *> selectedOperations;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  SmallVector<unsigned> controlPredecessors;
  std::optional<int64_t> remoteControlOutputResourceId;
  SmallVector<int64_t> remoteControlInputResourceIds;
  int64_t sequenceSourceMVMId = -1;
  int64_t sequenceShardIndex = -1;
  bool sequenceVectorPreparation = false;
  bool sequencePhysicalMVM = false;
  bool sequenceTerminal = false;
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

enum class RoutineDependencyKind { Route, LocalBinding, Control };

struct RoutineDependencyEdge {
  unsigned sourceRoutine = 0;
  unsigned destinationRoutine = 0;
  RoutineDependencyKind kind = RoutineDependencyKind::LocalBinding;
  int64_t id = -1;
  int64_t byteSize = -1;
};

struct EndpointDependencyRecord {
  unsigned source = 0;
  unsigned destination = 0;
  int64_t byteSize = 0;
  bool remote = false;
  int64_t tensorId = -1;
};

struct RoutineFusionStats {
  int64_t initialRoutineCount = 0;
  int64_t finalRoutineCount = 0;
  int64_t fusedBoundaryCount = 0;
  int64_t fusedBoundaryBytes = 0;
};

class DisjointSet {
public:
  explicit DisjointSet(unsigned size) : parent(size) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  unsigned find(unsigned value) {
    if (parent[value] != value)
      parent[value] = find(parent[value]);
    return parent[value];
  }

  void unite(unsigned lhs, unsigned rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (rhs < lhs)
      std::swap(lhs, rhs);
    parent[rhs] = lhs;
  }

private:
  SmallVector<unsigned> parent;
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

FailureOr<bool> registerStaticConcatParts(
    tensor::ConcatOp concat,
    DenseMap<Value, SmallVector<TilePart>> &partsByFullValue,
    const std::map<EndpointKey, EndpointPlan> &endpoints) {
  auto resultType = dyn_cast<RankedTensorType>(concat.getType());
  if (!resultType || !resultType.hasStaticShape())
    return false;
  int64_t concatDimension = concat.getDim();
  if (concatDimension < 0 || concatDimension >= resultType.getRank())
    return false;

  SmallVector<TilePart> concatParts;
  int64_t concatOffset = 0;
  for (Value input : concat.getInputs()) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape() ||
        inputType.getRank() != resultType.getRank())
      return false;

    auto existing = partsByFullValue.find(input);
    if (existing != partsByFullValue.end()) {
      for (const TilePart &part : existing->second) {
        if (part.offsets.size() !=
                static_cast<size_t>(resultType.getRank()) ||
            part.sizes.size() != static_cast<size_t>(resultType.getRank()))
          return concat.emitOpError(
              "nested concat shard has invalid static geometry");
        TilePart rebased = part;
        rebased.offsets[concatDimension] += concatOffset;
        concatParts.push_back(std::move(rebased));
      }
    } else {
      Operation *producer = input.getDefiningOp();
      auto operationId = producer ? producer->getAttrOfType<IntegerAttr>(
                                        kMappingOperationIdAttrName)
                                  : IntegerAttr{};
      if (!operationId)
        return false;
      auto workUnitId =
          producer->getAttrOfType<IntegerAttr>(kMappingWorkUnitIdAttrName);
      EndpointKey endpointKey{operationId.getInt(),
                              workUnitId ? workUnitId.getInt() : -1};
      if (!endpoints.contains(endpointKey))
        return false;
      SmallVector<int64_t> offsets(resultType.getRank(), 0);
      offsets[concatDimension] = concatOffset;
      concatParts.push_back({endpointKey.first, endpointKey.second,
                             std::move(offsets),
                             SmallVector<int64_t>(inputType.getShape()),
                             input});
    }

    if (concatOffset > std::numeric_limits<int64_t>::max() -
                           inputType.getDimSize(concatDimension))
      return concat.emitOpError("static concat offset overflow");
    concatOffset += inputType.getDimSize(concatDimension);
  }
  if (concatOffset != resultType.getDimSize(concatDimension))
    return concat.emitOpError("static concat parts do not cover its result");
  partsByFullValue[concat.getResult()] = std::move(concatParts);
  return true;
}

struct LinearSliceRange {
  int64_t offset = 0;
  int64_t length = 0;
};

std::optional<LinearSliceRange>
getContiguousLinearRange(RankedTensorType sourceType,
                         ArrayRef<int64_t> offsets,
                         ArrayRef<int64_t> sizes,
                         ArrayRef<int64_t> strides) {
  int64_t rank = sourceType.getRank();
  if (!sourceType.hasStaticShape() || rank == 0 ||
      offsets.size() != static_cast<size_t>(rank) ||
      sizes.size() != static_cast<size_t>(rank) ||
      strides.size() != static_cast<size_t>(rank) ||
      llvm::any_of(offsets, ShapedType::isDynamic) ||
      llvm::any_of(sizes, ShapedType::isDynamic) ||
      llvm::any_of(strides, ShapedType::isDynamic) ||
      !llvm::all_of(strides, [](int64_t stride) { return stride == 1; }))
    return std::nullopt;

  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    if (offsets[dimension] < 0 || sizes[dimension] <= 0 ||
        offsets[dimension] >
            sourceType.getDimSize(dimension) - sizes[dimension])
      return std::nullopt;
  }

  bool contiguous = false;
  for (int64_t pivot = 0; pivot < rank && !contiguous; ++pivot) {
    contiguous = true;
    for (int64_t dimension = 0; dimension < pivot; ++dimension)
      contiguous &= sizes[dimension] == 1;
    for (int64_t dimension = pivot + 1; dimension < rank; ++dimension)
      contiguous &= offsets[dimension] == 0 &&
                    sizes[dimension] == sourceType.getDimSize(dimension);
  }
  if (!contiguous)
    return std::nullopt;

  int64_t linearOffset = 0;
  int64_t linearLength = 1;
  int64_t dimensionStride = 1;
  for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
    std::optional<int64_t> offsetContribution =
        llvm::checkedMul(offsets[dimension], dimensionStride);
    if (!offsetContribution)
      return std::nullopt;
    std::optional<int64_t> nextOffset =
        llvm::checkedAdd(linearOffset, *offsetContribution);
    std::optional<int64_t> nextLength =
        llvm::checkedMul(linearLength, sizes[dimension]);
    std::optional<int64_t> nextStride = llvm::checkedMul(
        dimensionStride, sourceType.getDimSize(dimension));
    if (!nextOffset || !nextLength || !nextStride)
      return std::nullopt;
    linearOffset = *nextOffset;
    linearLength = *nextLength;
    dimensionStride = *nextStride;
  }
  return LinearSliceRange{linearOffset, linearLength};
}

std::optional<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
getRectangularRange(RankedTensorType type, LinearSliceRange range) {
  if (!type.hasStaticShape() || type.getRank() == 0 || range.offset < 0 ||
      range.length <= 0 || range.offset > type.getNumElements() - range.length)
    return std::nullopt;

  SmallVector<int64_t> dimensionStrides(type.getRank(), 1);
  for (int64_t dimension = type.getRank() - 2; dimension >= 0; --dimension) {
    std::optional<int64_t> stride = llvm::checkedMul(
        dimensionStrides[dimension + 1], type.getDimSize(dimension + 1));
    if (!stride)
      return std::nullopt;
    dimensionStrides[dimension] = *stride;
  }
  SmallVector<int64_t> coordinates(type.getRank(), 0);
  int64_t remainder = range.offset;
  for (int64_t dimension = 0; dimension < type.getRank(); ++dimension) {
    coordinates[dimension] = remainder / dimensionStrides[dimension];
    remainder %= dimensionStrides[dimension];
  }

  for (int64_t pivot = 0; pivot < type.getRank(); ++pivot) {
    int64_t stride = dimensionStrides[pivot];
    if (range.offset % stride != 0 || range.length % stride != 0)
      continue;
    int64_t extent = range.length / stride;
    if (extent <= 0 ||
        extent > type.getDimSize(pivot) - coordinates[pivot])
      continue;
    SmallVector<int64_t> offsets = coordinates;
    SmallVector<int64_t> sizes(type.getRank(), 1);
    sizes[pivot] = extent;
    for (int64_t dimension = pivot + 1; dimension < type.getRank();
         ++dimension) {
      if (coordinates[dimension] != 0)
        return std::nullopt;
      sizes[dimension] = type.getDimSize(dimension);
    }
    return std::make_pair(std::move(offsets), std::move(sizes));
  }
  return std::nullopt;
}

tensor::ExtractSliceOp canonicalizeContiguousViewSlice(
    tensor::ExtractSliceOp slice, IRRewriter &rewriter) {
  auto sourceType = dyn_cast<RankedTensorType>(slice.getSourceType());
  auto resultType = dyn_cast<RankedTensorType>(slice.getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return slice;
  std::optional<LinearSliceRange> range = getContiguousLinearRange(
      sourceType, slice.getStaticOffsets(), slice.getStaticSizes(),
      slice.getStaticStrides());
  if (!range)
    return slice;

  Value root = slice.getSource();
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (Operation *operation = root.getDefiningOp()) {
    if (!visited.insert(operation).second)
      return slice;
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(operation)) {
      root = expand.getSrc();
      continue;
    }
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(operation)) {
      root = collapse.getSrc();
      continue;
    }
    if (auto sourceSlice = dyn_cast<tensor::ExtractSliceOp>(operation)) {
      auto parentType = dyn_cast<RankedTensorType>(sourceSlice.getSourceType());
      if (!parentType || !parentType.hasStaticShape())
        return slice;
      std::optional<LinearSliceRange> parentRange =
          getContiguousLinearRange(parentType, sourceSlice.getStaticOffsets(),
                                   sourceSlice.getStaticSizes(),
                                   sourceSlice.getStaticStrides());
      if (!parentRange)
        return slice;
      std::optional<int64_t> composedOffset =
          llvm::checkedAdd(parentRange->offset, range->offset);
      if (!composedOffset || range->offset > parentRange->length - range->length)
        return slice;
      range->offset = *composedOffset;
      root = sourceSlice.getSource();
      continue;
    }
    break;
  }

  if (!root.getDefiningOp<tensor::InsertSliceOp>() &&
      !root.getDefiningOp<tensor::ConcatOp>())
    return slice;
  auto rootType = dyn_cast<RankedTensorType>(root.getType());
  if (!rootType || rootType.getRank() != resultType.getRank())
    return slice;
  auto rectangle = getRectangularRange(rootType, *range);
  if (!rectangle || rectangle->second != resultType.getShape())
    return slice;

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(rootType.getRank(),
                                    rewriter.getIndexAttr(1));
  for (int64_t value : rectangle->first)
    offsets.push_back(rewriter.getIndexAttr(value));
  for (int64_t value : rectangle->second)
    sizes.push_back(rewriter.getIndexAttr(value));
  rewriter.setInsertionPoint(slice);
  auto replacement = rewriter.create<tensor::ExtractSliceOp>(
      slice.getLoc(), resultType, root, offsets, sizes, strides);
  replacement->setAttrs(slice->getAttrs());
  rewriter.replaceOp(slice, replacement.getResult());
  return replacement;
}

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
      rebasedParts.push_back({piece.part->operationId,
                              piece.part->workUnitId, destinationOffsets,
                              piece.sizes, pieceValue});
    }

    // A mapped vector-tile stage is a real routine boundary even when its
    // subset can be sourced directly from producer shards. Preserve that
    // boundary with a tile-sized materialization operation instead of
    // replacing the stage with a value owned by the producer work unit.
    if (slice->hasAttr(kMappingOperationIdAttrName)) {
      Value materialized = rewriter.create<tensor::EmptyOp>(
          slice.getLoc(), demandType.getShape(), demandType.getElementType());
      SmallVector<OpFoldResult> zeroOffsets(
          demandType.getRank(), rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> fullSizes;
      for (int64_t size : demandType.getShape())
        fullSizes.push_back(rewriter.getIndexAttr(size));
      auto boundary = rewriter.create<tensor::InsertSliceOp>(
          slice.getLoc(), replacement, materialized, zeroOffsets, fullSizes,
          unitStrides);
      for (NamedAttribute attribute : slice->getAttrs()) {
        if (attribute.getName().strref().starts_with("sculptor."))
          boundary->setAttr(attribute.getName(), attribute.getValue());
      }
      replacement = boundary.getResult();
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
  endpoint.laneBindingGroup = operation.laneBindingGroup;
  endpoint.layerRegionId = operation.layerRegionId;
  endpoint.semanticLayerId = operation.semanticLayerId;
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
    DenseMap<Value, SmallVector<TilePart>> &partsByFullValue,
    std::map<EndpointKey, EndpointPlan> &endpoints,
    bool enableShardRouting) {
  for (Operation *generated : generatedSlices) {
    auto slice = dyn_cast_or_null<tensor::ExtractSliceOp>(generated);
    if (!slice || !slice->getBlock())
      continue;
    slice = canonicalizeContiguousViewSlice(slice, rewriter);
    auto found = partsByFullValue.find(slice.getSource());
    if (found == partsByFullValue.end() && enableShardRouting) {
      if (auto concat = slice.getSource().getDefiningOp<tensor::ConcatOp>()) {
        FailureOr<bool> registered = registerStaticConcatParts(
            concat, partsByFullValue, endpoints);
        if (failed(registered))
          return failure();
        if (*registered)
          found = partsByFullValue.find(slice.getSource());
      }
    }
    if (found == partsByFullValue.end() && enableShardRouting) {
      SmallVector<TilePart> assemblyParts;
      Value cursor = slice.getSource();
      while (auto insert = cursor.getDefiningOp<tensor::InsertSliceOp>()) {
        ArrayRef<int64_t> offsets = insert.getStaticOffsets();
        ArrayRef<int64_t> sizes = insert.getStaticSizes();
        ArrayRef<int64_t> strides = insert.getStaticStrides();
        auto sourceType = dyn_cast<RankedTensorType>(insert.getSourceType());
        if (!sourceType || !sourceType.hasStaticShape() ||
            llvm::any_of(offsets, ShapedType::isDynamic) ||
            llvm::any_of(sizes, ShapedType::isDynamic) ||
            llvm::any_of(strides, ShapedType::isDynamic) ||
            !llvm::all_of(strides,
                          [](int64_t stride) { return stride == 1; }) ||
            sourceType.getShape() != sizes) {
          assemblyParts.clear();
          break;
        }
        auto existing = partsByFullValue.find(insert.getSource());
        if (existing != partsByFullValue.end()) {
          for (const TilePart &part : existing->second) {
            if (part.offsets.size() != offsets.size() ||
                part.sizes.size() != sizes.size()) {
              assemblyParts.clear();
              break;
            }
            TilePart rebased = part;
            for (auto [offset, insertionOffset] :
                 llvm::zip_equal(rebased.offsets, offsets))
              offset += insertionOffset;
            assemblyParts.push_back(std::move(rebased));
          }
          if (assemblyParts.empty())
            break;
        } else {
          int64_t operationId = -1;
          int64_t workUnitId = -1;
          if (Operation *producer = insert.getSource().getDefiningOp()) {
            if (auto id = producer->getAttrOfType<IntegerAttr>(
                    kMappingOperationIdAttrName)) {
              int64_t candidateWorkUnitId = -1;
              if (auto id = producer->getAttrOfType<IntegerAttr>(
                      kMappingWorkUnitIdAttrName))
                candidateWorkUnitId = id.getInt();
              if (endpoints.contains({id.getInt(), candidateWorkUnitId})) {
                operationId = id.getInt();
                workUnitId = candidateWorkUnitId;
              }
            }
          }
          assemblyParts.push_back(
              {operationId, workUnitId, SmallVector<int64_t>(offsets),
               SmallVector<int64_t>(sizes), insert.getSource()});
        }
        cursor = insert.getDest();
      }
      if (!assemblyParts.empty() && cursor.getDefiningOp<tensor::EmptyOp>()) {
        partsByFullValue[slice.getSource()] = std::move(assemblyParts);
        found = partsByFullValue.find(slice.getSource());
      }
    }
    if (found != partsByFullValue.end()) {
      ArrayRef<int64_t> demandOffsets = slice.getStaticOffsets();
      ArrayRef<int64_t> demandSizes = slice.getStaticSizes();
      ArrayRef<int64_t> demandStrides = slice.getStaticStrides();
      auto demandType = dyn_cast<RankedTensorType>(slice.getType());
      if (!demandType ||
          demandOffsets.size() != static_cast<size_t>(demandType.getRank()) ||
          demandSizes.size() != static_cast<size_t>(demandType.getRank()) ||
          llvm::any_of(demandOffsets, ShapedType::isDynamic) ||
          llvm::any_of(demandSizes, ShapedType::isDynamic) ||
          llvm::any_of(demandStrides, ShapedType::isDynamic) ||
          !llvm::all_of(demandStrides,
                        [](int64_t stride) { return stride == 1; })) {
        return slice.emitOpError(
            "shard-routed tile slice requires static unit-stride geometry");
      }

      SmallVector<DemandPiece> pieces;
      int64_t coveredElements = 0;
      for (const TilePart &part : found->second) {
        DemandPiece piece;
        piece.part = &part;
        bool overlaps = true;
        for (int64_t dimension = 0; dimension < demandType.getRank();
             ++dimension) {
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
        if (llvm::any_of(pieces, [&](const DemandPiece &existing) {
              return regionsOverlap(existing.offsets, existing.sizes,
                                    piece.offsets, piece.sizes);
            })) {
          return slice.emitOpError(
              "producer shards overlap within one generated tile slice");
        }
        FailureOr<int64_t> pieceElements =
            getStaticRegionElementCount(piece.sizes, slice);
        if (failed(pieceElements) ||
            coveredElements >
                std::numeric_limits<int64_t>::max() - *pieceElements) {
          if (succeeded(pieceElements))
            slice.emitOpError("generated tile slice coverage overflow");
          return failure();
        }
        coveredElements += *pieceElements;
        pieces.push_back(std::move(piece));
      }

      FailureOr<int64_t> demandedElements =
          getStaticRegionElementCount(demandSizes, slice);
      if (failed(demandedElements))
        return failure();
      if (pieces.empty() || coveredElements != *demandedElements) {
        return slice.emitOpError(
            "generated tile slice is not completely covered by producer "
            "shards");
      }
      llvm::sort(pieces, [](const DemandPiece &lhs, const DemandPiece &rhs) {
        return std::lexicographical_compare(
            lhs.offsets.begin(), lhs.offsets.end(), rhs.offsets.begin(),
            rhs.offsets.end());
      });

      rewriter.setInsertionPoint(slice);
      SmallVector<OpFoldResult> unitStrides(demandType.getRank(),
                                            rewriter.getIndexAttr(1));
      SmallVector<TilePart> rebasedParts;
      Value replacement;
      if (pieces.size() > 1) {
        replacement = rewriter.create<tensor::EmptyOp>(
            slice.getLoc(), demandType.getShape(),
            demandType.getElementType());
      }

      for (const DemandPiece &piece : pieces) {
        SmallVector<int64_t> sourceOffsets;
        SmallVector<int64_t> destinationOffsets;
        SmallVector<OpFoldResult> sourceOffsetAttrs;
        SmallVector<OpFoldResult> destinationOffsetAttrs;
        SmallVector<OpFoldResult> sizeAttrs;
        for (int64_t dimension = 0; dimension < demandType.getRank();
             ++dimension) {
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
        if (sourceOffsets !=
                SmallVector<int64_t>(demandType.getRank(), 0) ||
            piece.sizes != piece.part->sizes) {
          auto pieceType = RankedTensorType::get(
              piece.sizes, demandType.getElementType(),
              demandType.getEncoding());
          auto sourceSlice = rewriter.create<tensor::ExtractSliceOp>(
              slice.getLoc(), pieceType, pieceValue, sourceOffsetAttrs,
              sizeAttrs, unitStrides);
          pieceValue = sourceSlice.getResult();
          if (piece.part->operationId >= 0) {
            auto endpoint = endpoints.find(
                {piece.part->operationId, piece.part->workUnitId});
            if (endpoint == endpoints.end()) {
              return slice.emitOpError(
                  "cannot resolve the producer endpoint for a shard slice");
            }
            sourceSlice->setAttr(
                kMappingOperationIdAttrName,
                rewriter.getI64IntegerAttr(piece.part->operationId));
            if (piece.part->workUnitId >= 0) {
              sourceSlice->setAttr(
                  kMappingWorkUnitIdAttrName,
                  rewriter.getI64IntegerAttr(piece.part->workUnitId));
            }
            sourceSlice->setAttr(
                kRALeafIdAttrName,
                rewriter.getI64IntegerAttr(endpoint->second.leafId));
            endpoint->second.mappedOperations.push_back(sourceSlice);
            appendUnique(endpoint->second.producedValues, pieceValue);
          }
        }

        if (pieces.size() == 1) {
          if (pieceValue.getType() != demandType) {
            return slice.emitOpError(
                "generated tile slice type disagrees with its shard view");
          }
          replacement = pieceValue;
        } else {
          replacement = rewriter.create<tensor::InsertSliceOp>(
              slice.getLoc(), pieceValue, replacement,
              destinationOffsetAttrs, sizeAttrs, unitStrides);
        }
        rebasedParts.push_back({piece.part->operationId,
                                piece.part->workUnitId, destinationOffsets,
                                piece.sizes, pieceValue});
      }

      rewriter.replaceOp(slice, replacement);
      partsByFullValue[replacement] = std::move(rebasedParts);
      continue;
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

bool canRouteResultWithoutFullAssembly(
    int64_t sourceOperationId, int64_t sourceResultNumber, Value result,
    const DenseMap<Operation *, int64_t> &operationIds,
    const ResourceAllocationTree &tree) {
  if (result.use_empty())
    return true;
  for (OpOperand &use : result.getUses()) {
    auto targetOperation = operationIds.find(use.getOwner());
    if (targetOperation == operationIds.end())
      return false;
    bool refined = llvm::any_of(
        tree.workUnitEdges, [&](const MappingWorkUnitEdge &edge) {
          return edge.sourceOperationId == sourceOperationId &&
                 edge.sourceResultNumber == sourceResultNumber &&
                 edge.targetOperationId == targetOperation->second &&
                 edge.targetOperandNumber ==
                     static_cast<int64_t>(use.getOperandNumber()) &&
                 edge.sourceWorkUnitId >= 0 && edge.targetWorkUnitId >= 0;
        });
    if (!refined)
      return false;
  }
  return true;
}

void hoistNestedConstantsForTiling(IRRewriter &rewriter,
                                   Operation *operation) {
  SmallVector<Operation *> constants;
  operation->walk([&](Operation *nested) {
    if (nested != operation && nested->hasTrait<OpTrait::ConstantLike>() &&
        nested->getNumOperands() == 0 && nested->getNumRegions() == 0)
      constants.push_back(nested);
  });
  if (constants.empty())
    return;

  rewriter.setInsertionPoint(operation);
  for (Operation *constant : constants) {
    Operation *hoisted = rewriter.clone(*constant);
    for (auto [original, replacement] :
         llvm::zip_equal(constant->getResults(), hoisted->getResults()))
      original.replaceAllUsesWith(replacement);
  }
}

/// Move producer-owned static tensor views out of a newly tiled region.
///
/// Region-bearing tiling interfaces can clone an existing extract_slice into
/// their implementation region.  Leaving that pure view nested there hides
/// its mapping identity from routine outlining: the consumer then captures and
/// routes the complete source tensor even when it needs only the static view.
/// Hoisting is legal only when every operand dominates the tiled root and the
/// view has fully static, unit-stride geometry.  Index-dependent slices remain
/// in their original region.
SmallVector<Operation *> hoistInvariantStaticSlicesForRouting(
    IRRewriter &rewriter, ArrayRef<Operation *> tiledRoots,
    const DenseMap<Value, SmallVector<TilePart>> &partsByFullValue,
    const std::map<EndpointKey, EndpointPlan> &endpoints,
    bool enableShardRouting) {
  if (!enableShardRouting)
    return {};

  auto isNestedWithin = [](Operation *root, Operation *operation) {
    for (Operation *current = operation; current;
         current = current->getParentOp()) {
      if (current == root)
        return true;
    }
    return false;
  };
  auto valueIsDefinedWithin = [&](Operation *root, Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation *parent = argument.getOwner()->getParentOp();
      return isNestedWithin(root, parent);
    }
    return isNestedWithin(root, value.getDefiningOp());
  };
  auto resolveEndpoint = [&](Operation *operation)
      -> std::optional<EndpointKey> {
    if (!operation)
      return std::nullopt;
    auto operationId =
        operation->getAttrOfType<IntegerAttr>(kMappingOperationIdAttrName);
    if (!operationId)
      return std::nullopt;
    auto workUnitId =
        operation->getAttrOfType<IntegerAttr>(kMappingWorkUnitIdAttrName);
    EndpointKey key{operationId.getInt(),
                    workUnitId ? workUnitId.getInt() : -1};
    if (!endpoints.contains(key))
      return std::nullopt;
    return key;
  };

  SmallVector<Operation *> hoistedSlices;
  for (Operation *root : tiledRoots) {
    SmallVector<tensor::ExtractSliceOp> candidates;
    root->walk([&](tensor::ExtractSliceOp slice) {
      if (slice == root || slice->getBlock() == root->getBlock())
        return;
      auto sourceType = dyn_cast<RankedTensorType>(slice.getSourceType());
      auto resultType = dyn_cast<RankedTensorType>(slice.getType());
      ArrayRef<int64_t> offsets = slice.getStaticOffsets();
      ArrayRef<int64_t> sizes = slice.getStaticSizes();
      ArrayRef<int64_t> strides = slice.getStaticStrides();
      if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
          !resultType.hasStaticShape() || resultType.getShape() != sizes ||
          llvm::any_of(offsets, ShapedType::isDynamic) ||
          llvm::any_of(sizes, ShapedType::isDynamic) ||
          llvm::any_of(strides, ShapedType::isDynamic) ||
          !llvm::all_of(strides,
                        [](int64_t stride) { return stride == 1; }) ||
          llvm::any_of(slice->getOperands(), [&](Value operand) {
            return valueIsDefinedWithin(root, operand);
          }))
        return;

      std::optional<EndpointKey> endpoint = resolveEndpoint(slice);
      if (!endpoint)
        endpoint = resolveEndpoint(slice.getSource().getDefiningOp());
      if (!endpoint && !partsByFullValue.contains(slice.getSource()))
        return;
      candidates.push_back(slice);
    });

    for (tensor::ExtractSliceOp slice : candidates) {
      if (!slice->getBlock())
        continue;
      std::optional<EndpointKey> endpoint = resolveEndpoint(slice);
      if (!endpoint)
        endpoint = resolveEndpoint(slice.getSource().getDefiningOp());

      rewriter.setInsertionPoint(root);
      Operation *hoisted = rewriter.clone(*slice);
      if (endpoint) {
        const EndpointPlan &plan = endpoints.at(*endpoint);
        hoisted->setAttr(kMappingOperationIdAttrName,
                         rewriter.getI64IntegerAttr(endpoint->first));
        if (endpoint->second >= 0) {
          hoisted->setAttr(kMappingWorkUnitIdAttrName,
                           rewriter.getI64IntegerAttr(endpoint->second));
        } else {
          hoisted->removeAttr(kMappingWorkUnitIdAttrName);
        }
        hoisted->setAttr(kRALeafIdAttrName,
                         rewriter.getI64IntegerAttr(plan.leafId));
      }
      slice.getResult().replaceAllUsesWith(hoisted->getResult(0));
      rewriter.eraseOp(slice);
      hoistedSlices.push_back(hoisted);
    }
  }
  return hoistedSlices;
}

LogicalResult
routeConvPatchChannelsWithoutAssembly(
    IRRewriter &rewriter, linalg::GenericOp generic,
    const DenseMap<Value, SmallVector<TilePart>> &partsByFullValue) {
  auto semanticSection =
      generic->getAttrOfType<StringAttr>("sculptor.semantic.section");
  if (!semanticSection || semanticSection.getValue() != "digital.conv_patch")
    return success();

  SmallVector<OpOperand *> inputs = generic.getDpsInputOperands();
  auto resultType = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  if (inputs.size() != 1 || generic.getNumDpsInits() != 1 ||
      generic.getNumLoops() != 2 || !resultType ||
      !resultType.hasStaticShape() || resultType.getRank() != 2)
    return success();

  Value fullActivation = inputs.front()->get();
  auto activationType = dyn_cast<RankedTensorType>(fullActivation.getType());
  auto parts = partsByFullValue.find(fullActivation);
  if (!activationType || !activationType.hasStaticShape() ||
      activationType.getRank() != 4 || parts == partsByFullValue.end() ||
      parts->second.size() < 2)
    return success();

  SmallVector<tensor::ExtractOp> activationReads;
  generic.getRegion().walk([&](tensor::ExtractOp read) {
    if (read.getTensor() == fullActivation)
      activationReads.push_back(read);
  });
  if (activationReads.size() != 1 ||
      activationReads.front().getIndices().size() != 4)
    return generic.emitOpError(
        "conv patch channel routing expected one rank-4 activation read");

  tensor::ExtractOp oldRead = activationReads.front();
  Value globalChannel = oldRead.getIndices()[1];
  auto channelDiv = globalChannel.getDefiningOp<arith::DivUIOp>();
  auto kernelPlaneConstant =
      channelDiv ? channelDiv.getRhs().getDefiningOp<arith::ConstantIndexOp>()
                 : arith::ConstantIndexOp{};
  if (!channelDiv || !kernelPlaneConstant || kernelPlaneConstant.value() <= 0)
    return success();
  int64_t kernelPlane = kernelPlaneConstant.value();
  Value flattenedIndex = channelDiv.getLhs();

  std::function<std::optional<int64_t>(Value, int64_t)> evaluateIndex =
      [&](Value value, int64_t localFeature) -> std::optional<int64_t> {
    if (auto index = value.getDefiningOp<linalg::IndexOp>()) {
      if (index.getDim() == 1)
        return localFeature;
      return std::nullopt;
    }
    if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
      return constant.value();
    auto apply = value.getDefiningOp<affine::AffineApplyOp>();
    if (!apply || apply.getMap().getNumResults() != 1)
      return std::nullopt;

    SmallVector<Attribute> operandConstants;
    operandConstants.reserve(apply.getMapOperands().size());
    for (Value operand : apply.getMapOperands()) {
      std::optional<int64_t> evaluated =
          evaluateIndex(operand, localFeature);
      if (!evaluated)
        return std::nullopt;
      operandConstants.push_back(rewriter.getIndexAttr(*evaluated));
    }
    SmallVector<Attribute> results;
    if (failed(apply.getMap().constantFold(operandConstants, results)) ||
        results.size() != 1)
      return std::nullopt;
    auto integer = dyn_cast<IntegerAttr>(results.front());
    if (!integer)
      return std::nullopt;
    return integer.getInt();
  };

  int64_t featureSize = resultType.getDimSize(1);
  if (featureSize <= 0)
    return success();
  std::optional<int64_t> featureBegin =
      evaluateIndex(flattenedIndex, /*localFeature=*/0);
  std::optional<int64_t> featureLast =
      evaluateIndex(flattenedIndex, featureSize - 1);
  if (!featureBegin || !featureLast || *featureBegin < 0 ||
      *featureLast != *featureBegin + featureSize - 1)
    return success();

  int64_t channels = activationType.getDimSize(1);
  int64_t flattenedWidth = channels * kernelPlane;
  if (channels <= 0 || *featureBegin > flattenedWidth - featureSize)
    return generic.emitOpError(
        "conv patch feature range exceeds its activation channels");
  int64_t channelBegin = *featureBegin / kernelPlane;
  int64_t channelEnd =
      (*featureBegin + featureSize + kernelPlane - 1) / kernelPlane;

  SmallVector<const TilePart *> channelParts;
  channelParts.reserve(parts->second.size());
  for (const TilePart &part : parts->second) {
    if (part.offsets.size() != 4 || part.sizes.size() != 4)
      return success();
    channelParts.push_back(&part);
  }
  llvm::sort(channelParts, [](const TilePart *lhs, const TilePart *rhs) {
    return lhs->offsets[1] < rhs->offsets[1];
  });

  // Only complete channel partitions are rewritten. Spatial partitions need
  // halo ownership and are intentionally left to the existing assembly path.
  int64_t coveredChannels = 0;
  for (const TilePart *part : channelParts) {
    auto partType = dyn_cast<RankedTensorType>(part->value.getType());
    if (!partType || !partType.hasStaticShape() || partType.getRank() != 4 ||
        part->offsets[0] != 0 || part->offsets[1] != coveredChannels ||
        part->offsets[2] != 0 || part->offsets[3] != 0 ||
        part->sizes[0] != activationType.getDimSize(0) ||
        part->sizes[2] != activationType.getDimSize(2) ||
        part->sizes[3] != activationType.getDimSize(3) ||
        partType.getShape() != ArrayRef<int64_t>(part->sizes))
      return success();
    coveredChannels += part->sizes[1];
  }
  if (coveredChannels != channels)
    return success();

  SmallVector<const TilePart *> demandedParts;
  for (const TilePart *part : channelParts) {
    int64_t partBegin = part->offsets[1];
    int64_t partEnd = partBegin + part->sizes[1];
    if (partBegin < channelEnd && channelBegin < partEnd)
      demandedParts.push_back(part);
  }
  if (demandedParts.empty())
    return generic.emitOpError(
        "conv patch feature tile has no activation channel source");

  SmallVector<Value> originalIndices(oldRead.getIndices());
  auto buildDirectRead = [&](OpBuilder &builder,
                             const TilePart &part) -> Value {
    SmallVector<Value> indices = originalIndices;
    if (part.offsets[1] != 0) {
      Value offset = builder.create<arith::ConstantIndexOp>(
          oldRead.getLoc(), part.offsets[1]);
      indices[1] = builder.create<arith::SubIOp>(
          oldRead.getLoc(), globalChannel, offset);
    }
    auto read =
        builder.create<tensor::ExtractOp>(oldRead.getLoc(), part.value, indices);
    read->setAttrs(oldRead->getAttrs());
    return read.getResult();
  };

  rewriter.setInsertionPoint(oldRead);
  std::function<Value(unsigned)> buildRead = [&](unsigned index) -> Value {
    const TilePart &part = *demandedParts[index];
    if (index + 1 == demandedParts.size())
      return buildDirectRead(rewriter, part);

    int64_t upperChannel = part.offsets[1] + part.sizes[1];
    Value upper = rewriter.create<arith::ConstantIndexOp>(
        oldRead.getLoc(), upperChannel);
    Value inThisPart = rewriter.create<arith::CmpIOp>(
        oldRead.getLoc(), arith::CmpIPredicate::ult, globalChannel, upper);
    auto choose = rewriter.create<scf::IfOp>(
        oldRead.getLoc(), oldRead.getType(), inThisPart,
        /*addThenBlock=*/true, /*addElseBlock=*/true);
    OpBuilder thenBuilder = choose.getThenBodyBuilder();
    Value direct = buildDirectRead(thenBuilder, part);
    thenBuilder.create<scf::YieldOp>(oldRead.getLoc(), direct);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&choose.getElseRegion().front());
      Value remaining = buildRead(index + 1);
      rewriter.setInsertionPointToEnd(&choose.getElseRegion().front());
      rewriter.create<scf::YieldOp>(oldRead.getLoc(), remaining);
    }
    return choose.getResult(0);
  };

  Value replacement = buildRead(0);
  generic.getDpsInputOperand(0)->set(demandedParts.front()->value);
  rewriter.replaceOp(oldRead, replacement);
  generic->setAttr("sculptor.memory.full_activation_assembly_elided",
                   rewriter.getUnitAttr());
  generic->setAttr("sculptor.memory.routed_activation_channel_shards",
                   rewriter.getI64IntegerAttr(demandedParts.size()));
  return success();
}

LogicalResult routeOutputLayoutPiecesWithoutAssembly(
    IRRewriter &rewriter, linalg::GenericOp generic,
    DenseMap<Value, SmallVector<TilePart>> &partsByFullValue,
    std::map<EndpointKey, EndpointPlan> &endpoints) {
  auto semanticSection =
      generic->getAttrOfType<StringAttr>("sculptor.semantic.section");
  if (!semanticSection ||
      semanticSection.getValue() != "digital.output_recombine")
    return success();

  SmallVector<OpOperand *> inputs = generic.getDpsInputOperands();
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (inputs.size() != 1 || generic.getNumDpsInits() != 1 ||
      generic->getNumResults() != 1 || maps.size() != 2)
    return success();

  Value fullInput = inputs.front()->get();
  auto inputType = dyn_cast<RankedTensorType>(fullInput.getType());
  auto parts = partsByFullValue.find(fullInput);
  AffineMap inputMap = maps.front();
  AffineMap outputMap = maps.back();
  if (!inputType || !inputType.hasStaticShape() || inputType.getRank() != 2 ||
      parts == partsByFullValue.end() || parts->second.size() < 2 ||
      inputMap.getNumResults() != 2 || inputMap.getNumSymbols() != 0 ||
      inputMap.getNumDims() != generic.getNumLoops())
    return success();

  SmallVector<const TilePart *> rowParts;
  rowParts.reserve(parts->second.size());
  for (const TilePart &part : parts->second) {
    if (part.offsets.size() != 2 || part.sizes.size() != 2)
      return success();
    rowParts.push_back(&part);
  }
  llvm::sort(rowParts, [](const TilePart *lhs, const TilePart *rhs) {
    return lhs->offsets[0] < rhs->offsets[0];
  });

  int64_t coveredRows = 0;
  for (const TilePart *part : rowParts) {
    auto partType = dyn_cast<RankedTensorType>(part->value.getType());
    if (!partType || !partType.hasStaticShape() || partType.getRank() != 2 ||
        part->offsets[0] != coveredRows || part->offsets[1] != 0 ||
        part->sizes[1] != inputType.getDimSize(1) ||
        partType.getShape() != ArrayRef<int64_t>(part->sizes))
      return success();
    coveredRows += part->sizes[0];
  }
  if (coveredRows != inputType.getDimSize(0))
    return success();

  rewriter.setInsertionPoint(generic);
  SmallVector<utils::IteratorType> iteratorTypes =
      generic.getIteratorTypesArray();
  auto replacement = rewriter.create<linalg::GenericOp>(
      generic.getLoc(), generic.getResultTypes(), ValueRange{},
      generic.getDpsInits(), ArrayRef<AffineMap>{outputMap}, iteratorTypes,
      [&](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange) {
        SmallVector<Value> indices;
        indices.reserve(generic.getNumLoops());
        for (unsigned dimension = 0; dimension < generic.getNumLoops();
             ++dimension) {
          indices.push_back(
              bodyBuilder.create<linalg::IndexOp>(bodyLoc, dimension));
        }

        auto applyResult = [&](AffineExpr expression) -> Value {
          AffineMap map =
              AffineMap::get(inputMap.getNumDims(), /*symbolCount=*/0,
                             expression, bodyBuilder.getContext());
          return bodyBuilder.create<affine::AffineApplyOp>(bodyLoc, map,
                                                           indices);
        };
        Value globalRow = applyResult(inputMap.getResult(0));
        Value column = applyResult(inputMap.getResult(1));

        std::function<Value(OpBuilder &, unsigned, unsigned)> buildRead =
            [&](OpBuilder &builder, unsigned begin, unsigned end) -> Value {
          if (end - begin == 1) {
            const TilePart &part = *rowParts[begin];
            Value localRow = globalRow;
            if (part.offsets[0] != 0) {
              Value offset = builder.create<arith::ConstantIndexOp>(
                  bodyLoc, part.offsets[0]);
              localRow = builder.create<arith::SubIOp>(bodyLoc, globalRow,
                                                        offset);
            }
            return builder.create<tensor::ExtractOp>(
                bodyLoc, part.value, ValueRange{localRow, column});
          }

          unsigned middle = begin + (end - begin) / 2;
          Value boundary = builder.create<arith::ConstantIndexOp>(
              bodyLoc, rowParts[middle]->offsets[0]);
          Value beforeBoundary = builder.create<arith::CmpIOp>(
              bodyLoc, arith::CmpIPredicate::ult, globalRow, boundary);
          auto choose = builder.create<scf::IfOp>(
              bodyLoc, inputType.getElementType(), beforeBoundary,
              /*addThenBlock=*/true, /*addElseBlock=*/true);
          OpBuilder thenBuilder = choose.getThenBodyBuilder();
          Value before = buildRead(thenBuilder, begin, middle);
          thenBuilder.create<scf::YieldOp>(bodyLoc, before);
          OpBuilder elseBuilder =
              OpBuilder::atBlockEnd(&choose.getElseRegion().front());
          Value after = buildRead(elseBuilder, middle, end);
          elseBuilder.create<scf::YieldOp>(bodyLoc, after);
          return choose.getResult(0);
        };

        Value selected =
            buildRead(bodyBuilder, /*begin=*/0, rowParts.size());
        bodyBuilder.create<linalg::YieldOp>(bodyLoc, selected);
      });
  for (NamedAttribute attribute : generic->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name == "indexing_maps" || name == "iterator_types" ||
        name == "operandSegmentSizes")
      continue;
    replacement->setAttr(attribute.getName(), attribute.getValue());
  }
  replacement->setAttr("sculptor.memory.full_output_assembly_elided",
                       rewriter.getUnitAttr());
  replacement->setAttr("sculptor.memory.routed_output_sequence_shards",
                       rewriter.getI64IntegerAttr(rowParts.size()));

  Value oldResult = generic.getResult(0);
  Value newResult = replacement.getResult(0);
  std::optional<SmallVector<TilePart>> keyedParts;
  if (auto keyed = partsByFullValue.find(oldResult);
      keyed != partsByFullValue.end()) {
    keyedParts = keyed->second;
    partsByFullValue.erase(keyed);
  }
  for (auto &entry : partsByFullValue) {
    for (TilePart &part : entry.second) {
      if (part.value == oldResult)
        part.value = newResult;
    }
  }
  if (keyedParts)
    partsByFullValue[newResult] = std::move(*keyedParts);
  for (auto &[key, endpoint] : endpoints) {
    (void)key;
    for (Value &produced : endpoint.producedValues) {
      if (produced == oldResult)
        produced = newResult;
    }
  }
  rewriter.replaceOp(generic, newResult);
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

  DenseMap<Operation *, int64_t> operationIdByMember;
  for (const ComputeOperation &operation : graph.operations) {
    operationIdByMember[operation.operation] = operation.id;
    for (Operation *member : operation.members)
      operationIdByMember[member] = operation.id;
  }

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
  auto dataflowMode =
      function->getAttrOfType<StringAttr>(kShardDataflowModeAttrName);
  bool enableShardRouting = dataflowMode && dataflowMode.getValue() == "sharded";
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
    // Some region-bearing tiling interfaces use a region's invariant yield
    // value while constructing a tiled root.  Keeping that value nested in the
    // original operation creates an illegal cross-region SSA capture.  Hoist
    // only operand-free constants; index-dependent region logic is preserved.
    hoistNestedConstantsForTiling(rewriter, operation);
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

      Operation *insertionPredecessor = operation->getPrevNode();
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
                                        partsByFullValue, endpoints,
                                        enableShardRouting)))
        return failure();

      EndpointPlan &plan = endpoint->second;
      SmallVector<Operation *> tiledRoots;
      for (Operation *tiledOperation : tiled->tiledOps) {
        Operation *root = tiledOperation;
        while (root->getBlock() != &function.getBody().front()) {
          Operation *parent = root->getParentOp();
          if (!parent || parent == function) {
            root = nullptr;
            break;
          }
          root = parent;
        }
        if (!root) {
          return tiledOperation->emitError("materialized work unit ")
                 << workUnit->id << " for operation " << operationId
                 << " is not owned by the function being outlined";
        }
        // Tiling interfaces such as tensor.pad report both the new root op and
        // operations in its region body.  The root is the independently
        // schedulable operation; its nested operations are implementation
        // details that clone with it into the same routine.
        bool insertedAtTilingPoint =
            root != operation && root->isBeforeInBlock(operation) &&
            (!insertionPredecessor ||
             insertionPredecessor->isBeforeInBlock(root));
        if (!insertedAtTilingPoint) {
          return tiledOperation->emitError("materialized work unit ")
                 << workUnit->id << " for operation " << operationId
                 << " is nested under a pre-existing operation";
        }
        if (!llvm::is_contained(tiledRoots, root))
          tiledRoots.push_back(root);
      }
      if (tiledRoots.empty())
        return operation->emitError("materialized mapping work unit has no "
                                    "top-level tiled operation");
      SmallVector<Operation *> hoistedSlices =
          hoistInvariantStaticSlicesForRouting(
              rewriter, tiledRoots, partsByFullValue, endpoints,
              enableShardRouting);
      if (failed(replaceGeneratedSlices(rewriter, hoistedSlices,
                                        partsByFullValue, endpoints,
                                        enableShardRouting)))
        return failure();
      plan.mappedOperations.append(tiledRoots.begin(), tiledRoots.end());
      Value tiledValue = tiled->tiledValues[workUnit->resultNumber];
      plan.producedValues.push_back(tiledValue);
      for (Operation *tiledRoot : tiledRoots) {
        tiledRoot->setAttr(kMappingOperationIdAttrName,
                           rewriter.getI64IntegerAttr(operationId));
        tiledRoot->setAttr(kMappingWorkUnitIdAttrName,
                           rewriter.getI64IntegerAttr(workUnit->id));
        tiledRoot->setAttr(kRALeafIdAttrName,
                           rewriter.getI64IntegerAttr(plan.leafId));
      }
      partsByResult[workUnit->resultNumber].push_back(
          {operationId, workUnit->id, workUnit->resultOffsets,
           workUnit->resultSizes, tiledValue});
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

      if (canRouteResultWithoutFullAssembly(operationId, resultNumber,
                                            originalResult,
                                            operationIdByMember,
                                            tree)) {
        rewriter.setInsertionPoint(operation);
        Value proxy = rewriter.create<tensor::EmptyOp>(
            operation->getLoc(), resultType.getShape(),
            resultType.getElementType());
        originalResult.replaceAllUsesWith(proxy);
        partsByFullValue[proxy] = parts;
        continue;
      }

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

  // MVM vector tiles are already independently mapped operations, so they do
  // not appear in the work-unit materialization loop above. Once all digital
  // producer parts are known, rewrite their patch readers to capture only the
  // channel shards that the vector tile can actually address.
  SmallVector<linalg::GenericOp> patchOperations;
  for (Operation &operation : function.getBody().front().without_terminator()) {
    auto generic = dyn_cast<linalg::GenericOp>(&operation);
    if (!generic)
      continue;
    auto section =
        generic->getAttrOfType<StringAttr>("sculptor.semantic.section");
    if (section && section.getValue() == "digital.conv_patch")
      patchOperations.push_back(generic);
  }
  for (linalg::GenericOp generic : patchOperations) {
    if (failed(routeConvPatchChannelsWithoutAssembly(
            rewriter, generic, partsByFullValue)))
      return failure();
  }
  SmallVector<linalg::GenericOp> outputLayoutOperations;
  for (Operation &operation : function.getBody().front().without_terminator()) {
    auto generic = dyn_cast<linalg::GenericOp>(&operation);
    if (!generic)
      continue;
    auto section =
        generic->getAttrOfType<StringAttr>("sculptor.semantic.section");
    if (section && section.getValue() == "digital.output_recombine")
      outputLayoutOperations.push_back(generic);
  }
  for (linalg::GenericOp generic : outputLayoutOperations) {
    if (failed(routeOutputLayoutPiecesWithoutAssembly(
            rewriter, generic, partsByFullValue, endpoints)))
      return failure();
  }

  // Tiling interfaces may return helper operations that are subsequently
  // replaced while producer/consumer slices are wired together. Endpoint
  // plans must not retain those erased Operation pointers: allocator reuse
  // otherwise makes routine construction nondeterministic and can turn the
  // stale pointer into an unrelated nested operation. Rebuild every work-unit
  // endpoint from stable mapping IDs on the operations that survived all
  // rewrites.
  for (auto &[key, endpoint] : endpoints)
    endpoint.mappedOperations.clear();
  Block &entry = function.getBody().front();
  for (Operation &operation : entry.without_terminator()) {
    auto operationId =
        operation.getAttrOfType<IntegerAttr>(kMappingOperationIdAttrName);
    auto workUnitId =
        operation.getAttrOfType<IntegerAttr>(kMappingWorkUnitIdAttrName);
    if (!operationId)
      continue;
    int64_t resolvedWorkUnitId = workUnitId ? workUnitId.getInt() : -1;
    auto endpoint =
        endpoints.find({operationId.getInt(), resolvedWorkUnitId});
    if (endpoint != endpoints.end())
      endpoint->second.mappedOperations.push_back(&operation);
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

FailureOr<SmallVector<EndpointDependencyRecord>>
buildEndpointDependencies(ArrayRef<EndpointPlan> endpoints,
                          const LogicalTileGraph &tileGraph,
                          Operation *anchor) {
  std::map<EndpointKey, unsigned> indexByEndpoint;
  for (auto [index, endpoint] : llvm::enumerate(endpoints))
    indexByEndpoint[endpoint.key] = index;

  auto resolve = [&](int64_t operationId, int64_t workUnitId,
                     int64_t logicalTileId) {
    SmallVector<unsigned> matches;
    auto exact = indexByEndpoint.find({operationId, workUnitId});
    if (exact != indexByEndpoint.end() &&
        endpoints[exact->second].logicalTileId == logicalTileId)
      matches.push_back(exact->second);
    if (workUnitId >= 0 || !matches.empty())
      return matches;
    for (auto [index, endpoint] : llvm::enumerate(endpoints)) {
      if (endpoint.key.first == operationId &&
          endpoint.logicalTileId == logicalTileId)
        matches.push_back(index);
    }
    return matches;
  };

  SmallVector<EndpointDependencyRecord> dependencies;
  auto append = [&](const LogicalTileDependency &dependency,
                    int64_t sourceTileId,
                    int64_t destinationTileId) -> LogicalResult {
    SmallVector<unsigned> sources =
        resolve(dependency.sourceOperationId, dependency.sourceWorkUnitId,
                sourceTileId);
    SmallVector<unsigned> destinations =
        resolve(dependency.targetOperationId, dependency.targetWorkUnitId,
                destinationTileId);
    if (sources.empty() || destinations.empty()) {
      return anchor->emitError(
                 "logical-tile dependency references an unknown mapping "
                 "endpoint: source (")
             << dependency.sourceOperationId << ", "
             << dependency.sourceWorkUnitId << "), target ("
             << dependency.targetOperationId << ", "
             << dependency.targetWorkUnitId << ")";
    }
    for (unsigned source : sources) {
      for (unsigned destination : destinations) {
        if (source == destination)
          continue;
        dependencies.push_back(
            {source, destination, dependency.byteSize,
             endpoints[source].location.physicalTileId !=
                 endpoints[destination].location.physicalTileId,
             dependency.tensorId});
      }
    }
    return success();
  };

  for (const LogicalTile &tile : tileGraph.tiles) {
    for (const LogicalTileDependency &dependency : tile.internalDependencies) {
      if (failed(append(dependency, tile.id, tile.id)))
        return failure();
    }
  }
  for (const LogicalTileEdge &edge : tileGraph.edges) {
    for (const LogicalTileDependency &dependency : edge.dependencies) {
      if (failed(append(dependency, edge.sourceTileId, edge.targetTileId)))
        return failure();
    }
  }
  llvm::sort(dependencies, [](const EndpointDependencyRecord &left,
                              const EndpointDependencyRecord &right) {
    return std::tuple{left.source, left.destination, left.byteSize,
                      left.remote, left.tensorId} <
           std::tuple{right.source, right.destination, right.byteSize,
                      right.remote, right.tensorId};
  });
  return dependencies;
}

bool isFusibleDigitalEndpoint(const EndpointPlan &endpoint) {
  return endpoint.laneKind == LogicalLaneKind::Digital &&
         (endpoint.operationKind == ComputeOperationKind::Structured ||
          endpoint.operationKind == ComputeOperationKind::DigitalStage);
}

bool sharesSemanticLayerRegion(const EndpointPlan &source,
                               const EndpointPlan &destination) {
  return source.semanticLayerId && destination.semanticLayerId &&
         source.layerRegionId >= 0 &&
         source.layerRegionId == destination.layerRegionId &&
         source.semanticLayerId == destination.semanticLayerId;
}

bool mayFuseWithoutCrossingSemanticBoundary(const EndpointPlan &source,
                                            const EndpointPlan &destination) {
  if (!source.semanticLayerId && !destination.semanticLayerId)
    return true;
  return sharesSemanticLayerRegion(source, destination);
}

bool isConvPatchEndpoint(const EndpointPlan &endpoint) {
  if (endpoint.operationKind != ComputeOperationKind::VectorTile)
    return false;
  return llvm::any_of(endpoint.mappedOperations, [](Operation *operation) {
    auto generic = dyn_cast_or_null<linalg::GenericOp>(operation);
    auto section =
        generic
            ? generic->getAttrOfType<StringAttr>("sculptor.semantic.section")
            : StringAttr{};
    return section && section.getValue() == "digital.conv_patch";
  });
}

FailureOr<SmallVector<RoutinePlan, 0>>
buildRoutineRegions(ArrayRef<EndpointPlan> endpoints,
                    ArrayRef<EndpointDependencyRecord> dependencies,
                    bool fuseProducerConsumer, bool consolidateLayerRegions,
                    const llvm::SmallDenseSet<int64_t> &protectedOperationIds,
                    RoutineFusionStats &stats, Operation *anchor) {
  stats.initialRoutineCount = endpoints.size();
  DisjointSet components(endpoints.size());
  SmallVector<int64_t> layerRegionEpochs;

  // Preserve layers as strategic boundaries while turning connected local
  // digital work inside each layer into a real execution region. Every edge
  // that cannot be internalized advances a communication epoch. Contracting
  // only zero-barrier edges in one epoch cannot create a routine-level cycle,
  // including the common send-to-another-tile-and-return pattern.
  if (consolidateLayerRegions) {
    SmallVector<SmallVector<const EndpointDependencyRecord *>> successors(
        endpoints.size());
    SmallVector<unsigned> indegree(endpoints.size(), 0);
    for (const EndpointDependencyRecord &dependency : dependencies) {
      successors[dependency.source].push_back(&dependency);
      ++indegree[dependency.destination];
    }

    std::set<unsigned> ready;
    for (unsigned index = 0; index < endpoints.size(); ++index)
      if (indegree[index] == 0)
        ready.insert(index);
    layerRegionEpochs.assign(endpoints.size(), 0);
    unsigned processed = 0;
    while (!ready.empty()) {
      unsigned sourceIndex = *ready.begin();
      ready.erase(ready.begin());
      ++processed;
      for (const EndpointDependencyRecord *dependency :
           successors[sourceIndex]) {
        const EndpointPlan &source = endpoints[dependency->source];
        const EndpointPlan &destination = endpoints[dependency->destination];
        bool internalizable = !dependency->remote &&
                              isFusibleDigitalEndpoint(source) &&
                              isFusibleDigitalEndpoint(destination) &&
                              source.location.physicalTileId ==
                                  destination.location.physicalTileId &&
                              sharesSemanticLayerRegion(source, destination);
        layerRegionEpochs[dependency->destination] =
            std::max(layerRegionEpochs[dependency->destination],
                     layerRegionEpochs[dependency->source] +
                         static_cast<int64_t>(!internalizable));
        if (--indegree[dependency->destination] == 0)
          ready.insert(dependency->destination);
      }
    }
    if (processed != endpoints.size())
      return anchor->emitError(
          "cannot consolidate layer regions in a cyclic endpoint graph");

    for (const EndpointDependencyRecord &dependency : dependencies) {
      const EndpointPlan &source = endpoints[dependency.source];
      const EndpointPlan &destination = endpoints[dependency.destination];
      if (dependency.remote || !isFusibleDigitalEndpoint(source) ||
          !isFusibleDigitalEndpoint(destination) ||
          source.location.physicalTileId !=
              destination.location.physicalTileId ||
          !sharesSemanticLayerRegion(source, destination) ||
          layerRegionEpochs[dependency.source] !=
              layerRegionEpochs[dependency.destination])
        continue;
      components.unite(dependency.source, dependency.destination);
    }
  }

  // A convolution patch and its physical MVM consumers form one naturally
  // streaming tile-local unit.  Internalize that boundary only when every
  // consumer is an MVM on the same physical tile and in the same semantic
  // layer.  This deliberately leaves mixed, remote, and externally observed
  // patch values on the existing materialized-tensor path.
  SmallVector<SmallVector<const EndpointDependencyRecord *>> outgoing(
      endpoints.size());
  for (const EndpointDependencyRecord &dependency : dependencies)
    outgoing[dependency.source].push_back(&dependency);
  for (unsigned sourceIndex = 0; sourceIndex < endpoints.size();
       ++sourceIndex) {
    const EndpointPlan &source = endpoints[sourceIndex];
    if (!isConvPatchEndpoint(source) || outgoing[sourceIndex].empty() ||
        protectedOperationIds.contains(source.key.first))
      continue;

    bool legal = llvm::all_of(
        outgoing[sourceIndex], [&](const EndpointDependencyRecord *dependency) {
          const EndpointPlan &destination = endpoints[dependency->destination];
          return !dependency->remote && dependency->byteSize > 0 &&
                 destination.operationKind ==
                     ComputeOperationKind::PhysicalMVM &&
                 source.location.physicalTileId ==
                     destination.location.physicalTileId &&
                 sharesSemanticLayerRegion(source, destination);
        });
    if (!legal)
      continue;
    for (const EndpointDependencyRecord *dependency : outgoing[sourceIndex])
      components.unite(sourceIndex, dependency->destination);
  }

  if (fuseProducerConsumer) {
    SmallVector<bool> touchesRemoteBoundary(endpoints.size(), false);
    for (const EndpointDependencyRecord &dependency : dependencies) {
      if (!dependency.remote)
        continue;
      touchesRemoteBoundary[dependency.source] = true;
      touchesRemoteBoundary[dependency.destination] = true;
    }

    while (true) {
      std::map<unsigned, std::set<unsigned>> outgoing;
      std::map<unsigned, std::set<unsigned>> incoming;
      std::map<std::pair<unsigned, unsigned>, int64_t> boundaryBytes;
      for (const EndpointDependencyRecord &dependency : dependencies) {
        unsigned source = components.find(dependency.source);
        unsigned destination = components.find(dependency.destination);
        if (source == destination)
          continue;
        outgoing[source].insert(destination);
        incoming[destination].insert(source);
        std::pair<unsigned, unsigned> key{source, destination};
        std::optional<int64_t> sum =
            llvm::checkedAdd(boundaryBytes[key], dependency.byteSize);
        if (!sum)
          return anchor->emitError(
              "producer-consumer fusion boundary byte count overflow");
        boundaryBytes[key] = *sum;
      }

      auto componentIsEligible = [&](unsigned root) {
        std::optional<int64_t> physicalTile;
        for (unsigned index = 0; index < endpoints.size(); ++index) {
          if (components.find(index) != root)
            continue;
          const EndpointPlan &endpoint = endpoints[index];
          if (!isFusibleDigitalEndpoint(endpoint) ||
              touchesRemoteBoundary[index])
            return false;
          if (!physicalTile)
            physicalTile = endpoint.location.physicalTileId;
          else if (*physicalTile != endpoint.location.physicalTileId)
            return false;
        }
        return physicalTile.has_value();
      };
      auto componentContainsProtectedOutput = [&](unsigned root) {
        for (unsigned index = 0; index < endpoints.size(); ++index) {
          if (components.find(index) == root &&
              protectedOperationIds.contains(endpoints[index].key.first))
            return true;
        }
        return false;
      };

      std::optional<std::pair<unsigned, unsigned>> selected;
      for (unsigned index = 0; index < endpoints.size(); ++index) {
        unsigned source = components.find(index);
        if (source != index || outgoing[source].size() != 1)
          continue;
        unsigned destination = *outgoing[source].begin();
        if (incoming[destination].size() != 1 ||
            *incoming[destination].begin() != source ||
            endpoints[source].location.physicalTileId !=
                endpoints[destination].location.physicalTileId ||
            !mayFuseWithoutCrossingSemanticBoundary(endpoints[source],
                                                    endpoints[destination]) ||
            boundaryBytes[{source, destination}] <= 0 ||
            componentContainsProtectedOutput(source) ||
            !componentIsEligible(source) || !componentIsEligible(destination))
          continue;
        selected = {source, destination};
        break;
      }
      if (!selected)
        break;
      components.unite(selected->first, selected->second);
    }
  }

  std::map<unsigned, unsigned> routineByRoot;
  SmallVector<RoutinePlan, 0> routines;
  routines.reserve(endpoints.size());
  for (unsigned index = 0; index < endpoints.size(); ++index) {
    unsigned root = components.find(index);
    auto [found, inserted] =
        routineByRoot.emplace(root, static_cast<unsigned>(routines.size()));
    if (inserted)
      routines.emplace_back();
    RoutinePlan &routine = routines[found->second];
    const EndpointPlan &endpoint = endpoints[index];
    if (routine.endpointIndices.empty()) {
      routine.physicalTileId = endpoint.location.physicalTileId;
      routine.tileRow = endpoint.location.row;
      routine.tileCol = endpoint.location.column;
    } else if (routine.physicalTileId != endpoint.location.physicalTileId) {
      return anchor->emitError(
          "producer-consumer fusion spans multiple physical tiles");
    }
    routine.endpointIndices.push_back(index);
    appendUnique(routine.logicalTileIds, endpoint.logicalTileId);
    appendUnique(routine.sourceLeafIds, endpoint.leafId);
    appendUnique(routine.layerRegionIds, endpoint.layerRegionId);
    if (endpoint.semanticLayerId)
      appendUnique(routine.semanticLayerIds, *endpoint.semanticLayerId);
    if (!layerRegionEpochs.empty())
      appendUnique(routine.layerRegionEpochs, layerRegionEpochs[index]);
    routine.boot |= endpoint.operationKind == ComputeOperationKind::MatrixSetup;
    for (Operation *operation : endpoint.mappedOperations)
      routine.selectedOperations.insert(operation);
  }

  stats.finalRoutineCount = routines.size();
  stats.fusedBoundaryCount =
      stats.initialRoutineCount - stats.finalRoutineCount;
  for (const EndpointDependencyRecord &dependency : dependencies) {
    if (components.find(dependency.source) !=
        components.find(dependency.destination))
      continue;
    std::optional<int64_t> sum =
        llvm::checkedAdd(stats.fusedBoundaryBytes, dependency.byteSize);
    if (!sum)
      return anchor->emitError("fused producer-consumer byte count overflow");
    stats.fusedBoundaryBytes = *sum;
  }
  return routines;
}

LogicalResult copyReductionMetadata(func::FuncOp function,
                                    const RoutinePlan &routine) {
  // Function-level reduction metadata promises that the complete routine is
  // a pure, shape-preserving N-to-1 reduction.  Shard export slices may be
  // co-located with the reduction endpoint so downstream consumers receive
  // only their demanded regions; such a routine still computes the reduction
  // correctly, but it is no longer legal for task-graph reassociation.
  if (routine.inputs.size() < 2 || routine.outputs.size() != 1)
    return success();
  Type reductionType = routine.outputs.front().getType();
  if (!llvm::all_of(routine.inputs, [&](Value input) {
        return input.getType() == reductionType;
      }))
    return success();

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

bool isOperationNestedWithin(Operation *root, Operation *operation) {
  for (Operation *current = operation; current;
       current = current->getParentOp()) {
    if (current == root)
      return true;
  }
  return false;
}

SmallVector<Value> getExternalValuesUsedBy(Operation *root) {
  SmallVector<Value> values;
  auto isDefinedInside = [&](Value value) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation *parent = argument.getOwner()->getParentOp();
      return isOperationNestedWithin(root, parent);
    }
    Operation *defining = value.getDefiningOp();
    return isOperationNestedWithin(root, defining);
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
  auto isOwnedByRoutineRoot = [&](Operation *operation) {
    return operation && llvm::any_of(
                            routine.selectedOperations, [&](Operation *root) {
                              return isOperationNestedWithin(root, operation);
                            });
  };
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *parent = argument.getOwner()->getParentOp();
    if (isOwnedByRoutineRoot(parent))
      return success();
    if (parent != source.getOperation())
      return parent->emitError("routine ")
             << routine.globalId << " closure captured block argument "
             << argument.getArgNumber() << " from nested operation '"
             << parent->getName() << "'";
    appendUnique(routine.inputs, value);
    return success();
  }

  Operation *defining = value.getDefiningOp();
  if (!defining)
    return source.emitError("routine closure encountered a value without a "
                            "definition");
  if (isOwnedByRoutineRoot(defining))
    return success();
  auto owner = mappedOwner.find(defining);
  if (owner != mappedOwner.end()) {
    if (owner->second != routineIndex) {
      appendUnique(routine.inputs, value);
      appendUnique(routines[owner->second].outputs, value);
    }
    return success();
  }
  if (defining->getBlock() != &source.getBody().front()) {
    InFlightDiagnostic diagnostic = defining->emitError(
        "cannot outline a support operation outside the source entry block");
    for (Operation *root : routine.selectedOperations)
      diagnostic.attachNote(root->getLoc())
          << "selected routine root is '" << root->getName() << "'";
    for (Operation *parent = defining->getParentOp(); parent && parent != source;
         parent = parent->getParentOp())
      diagnostic.attachNote(parent->getLoc())
          << "support operation parent is '" << parent->getName() << "'";
    return failure();
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
    SmallVector<Operation *> seeds;
    Block *sourceEntry = &source.getBody().front();
    for (Operation *operation : routines[routineIndex].selectedOperations) {
      // Mapping stages own their complete nested region so dependency lookup
      // can resolve values produced inside an scf loop. Closure construction,
      // however, must start from the top-level stage roots that will actually
      // be cloned into the routine. Treating nested members as independent
      // roots exposes loop induction and iter_args as illegal routine inputs
      // and makes the result depend on member insertion order.
      if (operation->getBlock() == sourceEntry)
        seeds.push_back(operation);
    }
    if (seeds.empty())
      return source.emitError("outlined routine has no top-level operation");
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

void collectMappedBoundaryValues(
    Value value, const DenseMap<Operation *, unsigned> &mappedOwner,
    SmallVectorImpl<std::pair<unsigned, Value>> &boundaries,
    llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (isa<BlockArgument>(value))
    return;
  Operation *defining = value.getDefiningOp();
  if (!defining || !visited.insert(defining).second)
    return;
  auto owner = mappedOwner.find(defining);
  if (owner != mappedOwner.end()) {
    auto boundary = std::make_pair(owner->second, value);
    if (!llvm::is_contained(boundaries, boundary))
      boundaries.push_back(boundary);
    return;
  }
  // Region-bearing support operations (for example scf.for output assembly)
  // can capture mapped values inside their bodies without listing those
  // values as direct operation operands.  Follow the complete external-value
  // closure so model outputs retain their mapped producer ownership.
  for (Value operand : getExternalValuesUsedBy(defining))
    collectMappedBoundaryValues(operand, mappedOwner, boundaries, visited);
}

LogicalResult
attachModelOutputs(func::FuncOp source, SmallVectorImpl<RoutinePlan> &routines,
                   DenseMap<Operation *, unsigned> &mappedOwner,
                   SmallVectorImpl<std::pair<unsigned, Value>> &modelOutputs,
                   const LogicalTilePlacementPlan &placement) {
  auto returnOp =
      dyn_cast<func::ReturnOp>(source.getBody().front().getTerminator());
  if (!returnOp)
    return source.emitError("expected a func.return terminator");

  std::optional<int64_t> physicalTileCount =
      llvm::checkedMul(placement.mesh.rows, placement.mesh.columns);
  if (!physicalTileCount || *physicalTileCount <= 0)
    return source.emitError("model-output placement has invalid mesh size");

  // The logical placement estimate describes the burden already assigned to
  // each physical tile.  Synthetic output routines are not represented in
  // that graph, so account for their external destinations and route inputs
  // here while placing each output.  Keeping this state across outputs avoids
  // independently selecting the same apparently-empty tile for every gather.
  std::map<int64_t, int64_t> baseBytesByPhysicalTile;
  std::map<int64_t, int64_t> addedOutputBytesByPhysicalTile;
  for (const LogicalTilePhysicalAssignment &assignment :
       placement.assignments) {
    auto estimateIndex =
        placement.memoryEstimateIndexByTileId.find(assignment.logicalTileId);
    if (estimateIndex == placement.memoryEstimateIndexByTileId.end())
      return source.emitError("model-output placement is missing the memory "
                              "estimate for logical tile ")
             << assignment.logicalTileId;
    int64_t bytes =
        placement.memoryEstimates[estimateIndex->second].requiredBytes;
    int64_t &physicalBytes =
        baseBytesByPhysicalTile[assignment.location.physicalTileId];
    std::optional<int64_t> total = llvm::checkedAdd(physicalBytes, bytes);
    if (!total)
      return source.emitError(
          "model-output base physical-tile byte count overflowed");
    physicalBytes = *total;
  }
  for (const RoutinePlan &routine : routines) {
    for (Value input : routine.inputs) {
      if (!isa<BlockArgument>(input))
        continue;
      FailureOr<int64_t> bytes = getStaticByteSize(input.getType(), source);
      if (failed(bytes))
        return failure();
      int64_t &physicalBytes = baseBytesByPhysicalTile[routine.physicalTileId];
      std::optional<int64_t> total = llvm::checkedAdd(physicalBytes, *bytes);
      if (!total)
        return source.emitError(
            "model-output external-input byte count overflowed");
      physicalBytes = *total;
    }
  }

  for (Value value : returnOp.getOperands()) {
    if (Operation *defining = value.getDefiningOp()) {
      auto directOwner = mappedOwner.find(defining);
      if (directOwner != mappedOwner.end()) {
        appendUnique(routines[directOwner->second].outputs, value);
        modelOutputs.emplace_back(directOwner->second, value);
        continue;
      }
    }

    SmallVector<std::pair<unsigned, Value>> boundaries;
    llvm::SmallPtrSet<Operation *, 32> visited;
    collectMappedBoundaryValues(value, mappedOwner, boundaries, visited);
    if (boundaries.empty()) {
      return source.emitError(
          "model output is not derived from a mapped compute operation");
    }

    // Put otherwise-unmapped output assembly in a real digital routine.  The
    // old behavior captured it into the lowest-numbered mapped producer.  A
    // sequence's first analog shard could then depend on every later shard
    // merely because it happened to own the final concat, widening lifetimes
    // and making valid bounded-wave ordering cyclic.
    FailureOr<int64_t> outputBytes = getStaticByteSize(value.getType(), source);
    if (failed(outputBytes))
      return failure();

    std::map<unsigned, int64_t> boundaryBytesByOwner;
    std::map<int64_t, int64_t> localBoundaryBytes;
    int64_t totalBoundaryBytes = 0;
    for (auto [owner, boundary] : boundaries) {
      FailureOr<int64_t> bytes = getStaticByteSize(boundary.getType(), source);
      if (failed(bytes))
        return failure();
      int64_t sourceTile = routines[owner].physicalTileId;
      std::optional<int64_t> ownerTotal =
          llvm::checkedAdd(boundaryBytesByOwner[owner], *bytes);
      if (!ownerTotal)
        return source.emitError(
            "model-output owner boundary byte count overflowed");
      boundaryBytesByOwner[owner] = *ownerTotal;
      std::optional<int64_t> total =
          llvm::checkedAdd(localBoundaryBytes[sourceTile], *bytes);
      if (!total)
        return source.emitError("model-output boundary byte count overflowed");
      localBoundaryBytes[sourceTile] = *total;
      total = llvm::checkedAdd(totalBoundaryBytes, *bytes);
      if (!total)
        return source.emitError(
            "model-output total boundary byte count overflowed");
      totalBoundaryBytes = *total;
    }

    // Preserve the established owner-local policy when memory capacity is not
    // enabled.  Capacity-constrained deployments opt into the stronger search
    // below without unexpectedly changing unconstrained deployment topology.
    unsigned legacyAnchor = boundaries.front().first;
    for (const auto &[candidate, bytes] : boundaryBytesByOwner) {
      if (bytes > boundaryBytesByOwner.at(legacyAnchor) ||
          (bytes == boundaryBytesByOwner.at(legacyAnchor) &&
           candidate < legacyAnchor))
        legacyAnchor = candidate;
    }
    int64_t legacyPhysicalTile = routines[legacyAnchor].physicalTileId;

    struct OutputPlacementCandidate {
      int64_t physicalTileId = -1;
      int64_t projectedBytes = 0;
      int64_t communicationByteHops = 0;
      int64_t remoteBoundaryBytes = 0;
    };
    std::optional<OutputPlacementCandidate> best;
    int64_t minimumProjectedBytes = std::numeric_limits<int64_t>::max();
    for (int64_t candidateTile = 0; candidateTile < *physicalTileCount;
         ++candidateTile) {
      if (placement.tileMemoryCapacityBytes == 0 &&
          candidateTile != legacyPhysicalTile)
        continue;
      int64_t remoteBoundaryBytes =
          totalBoundaryBytes - localBoundaryBytes[candidateTile];
      int64_t projectedBytes = baseBytesByPhysicalTile[candidateTile];
      for (int64_t bytes : {addedOutputBytesByPhysicalTile[candidateTile],
                            *outputBytes, remoteBoundaryBytes}) {
        std::optional<int64_t> total = llvm::checkedAdd(projectedBytes, bytes);
        if (!total)
          return source.emitError(
              "model-output projected tile byte count overflowed");
        projectedBytes = *total;
      }
      minimumProjectedBytes = std::min(minimumProjectedBytes, projectedBytes);
      if (placement.tileMemoryCapacityBytes > 0 &&
          projectedBytes > placement.tileMemoryCapacityBytes)
        continue;

      int64_t candidateRow = candidateTile / placement.mesh.columns;
      int64_t candidateCol = candidateTile % placement.mesh.columns;
      int64_t communicationByteHops = 0;
      for (const auto &[sourceTile, bytes] : localBoundaryBytes) {
        int64_t sourceRow = sourceTile / placement.mesh.columns;
        int64_t sourceCol = sourceTile % placement.mesh.columns;
        int64_t hops = std::abs(sourceRow - candidateRow) +
                       std::abs(sourceCol - candidateCol);
        std::optional<int64_t> contribution = llvm::checkedMul(bytes, hops);
        std::optional<int64_t> total =
            contribution
                ? llvm::checkedAdd(communicationByteHops, *contribution)
                : std::nullopt;
        if (!total)
          return source.emitError(
              "model-output communication byte-hop count overflowed");
        communicationByteHops = *total;
      }

      OutputPlacementCandidate candidate{candidateTile, projectedBytes,
                                         communicationByteHops,
                                         remoteBoundaryBytes};
      if (!best ||
          std::tie(candidate.projectedBytes, candidate.communicationByteHops,
                   candidate.physicalTileId) <
              std::tie(best->projectedBytes, best->communicationByteHops,
                       best->physicalTileId))
        best = candidate;
    }
    if (!best)
      return source.emitError("no physical tile can hold model-output "
                              "assembly: minimum estimated requirement is ")
             << minimumProjectedBytes << " bytes, capacity is "
             << placement.tileMemoryCapacityBytes << " bytes, output is "
             << *outputBytes << " bytes, and assembly boundaries total "
             << totalBoundaryBytes << " bytes";

    std::optional<int64_t> addedBytes =
        llvm::checkedAdd(*outputBytes, best->remoteBoundaryBytes);
    std::optional<int64_t> totalAdded =
        addedBytes ? llvm::checkedAdd(
                         addedOutputBytesByPhysicalTile[best->physicalTileId],
                         *addedBytes)
                   : std::nullopt;
    if (!totalAdded)
      return source.emitError(
          "model-output cumulative tile byte count overflowed");
    addedOutputBytesByPhysicalTile[best->physicalTileId] = *totalAdded;

    RoutinePlan outputRoutine;
    outputRoutine.physicalTileId = best->physicalTileId;
    outputRoutine.tileRow = best->physicalTileId / placement.mesh.columns;
    outputRoutine.tileCol = best->physicalTileId % placement.mesh.columns;
    outputRoutine.syntheticOutput = true;
    unsigned outputRoutineIndex = routines.size();
    routines.push_back(std::move(outputRoutine));

    llvm::SmallDenseSet<std::pair<unsigned, Value>, 32> active;
    if (failed(collectRoutineValue(outputRoutineIndex, value, source, routines,
                                   mappedOwner, active)))
      return failure();
    appendUnique(routines[outputRoutineIndex].outputs, value);
    for (Operation *operation :
         routines[outputRoutineIndex].selectedOperations) {
      if (!mappedOwner.try_emplace(operation, outputRoutineIndex).second)
        return operation->emitError(
            "model-output support operation already has a routine owner");
    }
    modelOutputs.emplace_back(outputRoutineIndex, value);
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
    auto leftKey = key(left);
    auto rightKey = key(right);
    return leftKey != rightKey ? leftKey < rightKey : lhs < rhs;
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

/// Add the bounded in-flight contract for sequence-sharded MVMs.
///
/// Sequence shards are independent in the tensor dataflow, but a realistic
/// tile has bounded staging storage.  Without an explicit control edge, route
/// timing can make several vector/patch routines run before their analog
/// consumers and force every shard buffer to remain live. Gate each local
/// wave's vector preparation on completion of the wave one bounded window
/// behind it, using that wave's terminal recombination routines (or its
/// physical MVMs when no recombination is required). The bound is
/// deployment-wide for each persistent analog binding group: a per-source-MVM
/// or per-tile bound lets parallel invocations of tied weights run ahead of
/// their one shared analog lane and leaves every partial result resident at
/// the reduction tile. A binding-group wave-credit window preserves parallel
/// execution within and across admitted waves, as well as across unrelated
/// matrices, while bounding staging at recombination tiles.
LogicalResult buildBoundedSequenceDependencies(
    SmallVectorImpl<RoutinePlan> &routines, ArrayRef<EndpointPlan> endpoints,
    int64_t wavesInFlight, Operation *anchor) {
  struct Wave {
    int64_t sourceMVMId = -1;
    int64_t shardIndex = -1;
    SmallVector<int64_t> bindingGroups;
    SmallVector<unsigned> vectorPreparations;
    SmallVector<unsigned> physicalMVMs;
    SmallVector<unsigned> terminals;
  };
  std::map<int64_t, std::map<int64_t, Wave>> waves;
  const size_t waveWindow = static_cast<size_t>(wavesInFlight);

  for (auto [routineIndex, routine] : llvm::enumerate(routines)) {
    std::optional<int64_t> sourceMVMId;
    std::optional<int64_t> shardIndex;
    for (unsigned endpointIndex : routine.endpointIndices) {
      const EndpointPlan &endpoint = endpoints[endpointIndex];
      bool vectorPreparation =
          endpoint.operationKind == ComputeOperationKind::VectorTile;
      bool physicalMVM =
          endpoint.operationKind == ComputeOperationKind::PhysicalMVM;
      bool sequenceOperation = false;

      for (Operation *operation : endpoint.mappedOperations) {
        auto operationShard =
            operation->getAttrOfType<IntegerAttr>(kSequenceShardIndexAttr);
        if (!operationShard)
          continue;
        sequenceOperation = true;
        auto operationSource = operation->getAttrOfType<IntegerAttr>(
            golem_tiling_attrs::kSourceMVMIdAttrName);
        if (!operationSource || operationSource.getInt() < 0 ||
            operationShard.getInt() < 0) {
          return operation->emitError(
              "sequence-sharded mapping stage has incomplete identity");
        }
        if ((sourceMVMId && *sourceMVMId != operationSource.getInt()) ||
            (shardIndex && *shardIndex != operationShard.getInt())) {
          return operation->emitError(
              "one outlined routine spans multiple MVM sequence waves");
        }
        sourceMVMId = operationSource.getInt();
        shardIndex = operationShard.getInt();
      }
      if (!sequenceOperation)
        continue;
      routine.sequenceVectorPreparation |= vectorPreparation;
      routine.sequencePhysicalMVM |= physicalMVM;
      routine.sequenceTerminal |= !vectorPreparation && !physicalMVM;
    }

    if (!shardIndex)
      continue;
    if (!sourceMVMId)
      return anchor->emitError(
          "sequence-sharded routine has no source MVM identity");
    routine.sequenceSourceMVMId = *sourceMVMId;
    routine.sequenceShardIndex = *shardIndex;
    Wave &wave = waves[*sourceMVMId][*shardIndex];
    wave.sourceMVMId = *sourceMVMId;
    wave.shardIndex = *shardIndex;
    if (routine.sequenceVectorPreparation)
      wave.vectorPreparations.push_back(routineIndex);
    if (routine.sequencePhysicalMVM) {
      wave.physicalMVMs.push_back(routineIndex);
      for (unsigned endpointIndex : routine.endpointIndices) {
        const EndpointPlan &endpoint = endpoints[endpointIndex];
        if (endpoint.operationKind != ComputeOperationKind::PhysicalMVM)
          continue;
        if (!endpoint.laneBindingGroup) {
          return anchor->emitError(
              "sequence-sharded physical MVM has no analog binding group");
        }
        if (!llvm::is_contained(wave.bindingGroups,
                                *endpoint.laneBindingGroup))
          wave.bindingGroups.push_back(*endpoint.laneBindingGroup);
      }
    }
    if (routine.sequenceTerminal)
      wave.terminals.push_back(routineIndex);
  }

  // The nested maps provide a deterministic global order by source operation
  // and then sequence shard. Every binding group observes that same order, so
  // a wave using several arrays cannot acquire them in conflicting orders.
  std::map<int64_t, SmallVector<Wave *>> wavesByBindingGroup;
  for (auto &[sourceMVMId, sourceWaves] : waves) {
    for (auto &[shardIndex, wave] : sourceWaves) {
      if (wave.physicalMVMs.empty())
        continue;
      if (wave.bindingGroups.empty()) {
        return anchor->emitError(
            "sequence-sharded MVM wave has no analog binding groups");
      }
      llvm::sort(wave.bindingGroups);
      for (int64_t bindingGroup : wave.bindingGroups)
        wavesByBindingGroup[bindingGroup].push_back(&wave);
      (void)shardIndex;
    }
    (void)sourceMVMId;
  }

  for (auto &[bindingGroup, orderedWaves] : wavesByBindingGroup) {
    for (size_t waveIndex = waveWindow; waveIndex < orderedWaves.size();
         ++waveIndex) {
      Wave &current = *orderedWaves[waveIndex];
      Wave &predecessor = *orderedWaves[waveIndex - waveWindow];
      ArrayRef<unsigned> gated =
          current.vectorPreparations.empty()
              ? ArrayRef<unsigned>(current.physicalMVMs)
              : ArrayRef<unsigned>(current.vectorPreparations);
      ArrayRef<unsigned> predecessors =
          predecessor.terminals.empty()
              ? ArrayRef<unsigned>(predecessor.physicalMVMs)
              : ArrayRef<unsigned>(predecessor.terminals);
      for (unsigned target : gated) {
        for (unsigned predecessorRoutine : predecessors) {
          if (!llvm::is_contained(routines[target].controlPredecessors,
                                  predecessorRoutine))
            routines[target].controlPredecessors.push_back(
                predecessorRoutine);
        }
      }
    }
    (void)bindingGroup;
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

LogicalResult assignResourcesAndConnections(
    func::FuncOp source, SmallVectorImpl<RoutinePlan> &routines,
    ArrayRef<unsigned> routineOrder, ArrayRef<EndpointPlan> endpoints,
    ArrayRef<EndpointDependencyRecord> endpointDependencies,
    const DenseMap<Operation *, unsigned> &mappedOwner,
    DenseMap<Value, int64_t> &resourceIdByValue,
    SmallVectorImpl<RouteRecord> &routes,
    SmallVectorImpl<LocalBindingRecord> &localBindings) {
  SmallVector<unsigned> routineByEndpoint(endpoints.size());
  for (auto [routineIndex, routine] : llvm::enumerate(routines)) {
    for (unsigned endpointIndex : routine.endpointIndices)
      routineByEndpoint[endpointIndex] = routineIndex;
  }
  std::map<std::pair<unsigned, unsigned>,
           SmallVector<const EndpointDependencyRecord *>>
      dependenciesByRoutinePair;
  for (const EndpointDependencyRecord &dependency : endpointDependencies) {
    dependenciesByRoutinePair[{routineByEndpoint[dependency.source],
                               routineByEndpoint[dependency.destination]}]
        .push_back(&dependency);
  }
  auto findTensorId = [&](unsigned sourceRoutine, unsigned targetRoutine,
                          int64_t byteSize) {
    int64_t best = -1;
    auto found =
        dependenciesByRoutinePair.find({sourceRoutine, targetRoutine});
    if (found == dependenciesByRoutinePair.end())
      return best;
    for (const EndpointDependencyRecord *dependency : found->second) {
      if (dependency->byteSize != byteSize && dependency->byteSize != 0 &&
          byteSize != 0)
        continue;
      if (dependency->tensorId >= 0 &&
          (best < 0 || dependency->tensorId < best))
        best = dependency->tensorId;
    }
    return best;
  };

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
                        findTensorId(*sourceIndex, targetIndex, *byteSize),
                        *byteSize});
    }
  }

  // A cross-tile control dependency is a real synchronization message, not a
  // local task dependency. Materialize one four-byte token output per source
  // routine and fan it out to token inputs on the remote successors. This
  // keeps the ordinary runtime ABI honest: the successor becomes ready only
  // after the source executes and the network delivers the token.
  for (unsigned targetIndex : routineOrder) {
    RoutinePlan &target = routines[targetIndex];
    llvm::sort(target.controlPredecessors, [&](unsigned left,
                                               unsigned right) {
      return routines[left].globalId < routines[right].globalId;
    });
    for (unsigned sourceIndex : target.controlPredecessors) {
      RoutinePlan &producer = routines[sourceIndex];
      if (producer.physicalTileId == target.physicalTileId)
        continue;
      if (!producer.remoteControlOutputResourceId)
        producer.remoteControlOutputResourceId = nextResourceId++;
      int64_t resourceId = *producer.remoteControlOutputResourceId;
      unsigned sourceOutput = producer.outputs.size();
      unsigned destinationInput =
          target.inputs.size() + target.remoteControlInputResourceIds.size();
      target.remoteControlInputResourceIds.push_back(resourceId);
      routes.push_back({-1, sourceIndex, sourceOutput, targetIndex,
                        destinationInput, resourceId, -1, 4});
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
  for (auto [destination, routine] : llvm::enumerate(routines)) {
    for (unsigned source : routine.controlPredecessors) {
      edges.push_back({source, static_cast<unsigned>(destination),
                       RoutineDependencyKind::Control,
                       routines[source].globalId, 0});
    }
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
    } else if (edge.kind == RoutineDependencyKind::Control) {
      diagnostic << "bounded-wave control dependency (source MVM "
                 << source.sequenceSourceMVMId << ", shard "
                 << source.sequenceShardIndex << " -> "
                 << destination.sequenceShardIndex << ")";
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
  DenseMap<Operation *, unsigned> sourcePosition;
  for (auto [position, operation] : llvm::enumerate(sourceBlock))
    sourcePosition[&operation] = position;
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
    auto controlTokenType =
        RankedTensorType::get({1}, IntegerType::get(outer.getContext(), 32));
    inputTypes.append(routine.remoteControlInputResourceIds.size(),
                      controlTokenType);
    if (routine.remoteControlOutputResourceId)
      outputTypes.push_back(controlTokenType);

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
    function->setAttr(kDeploymentLayerRegionIdsAttr,
                      getI64Array(builder, routine.layerRegionIds));
    if (!routine.semanticLayerIds.empty())
      function->setAttr(kDeploymentSemanticLayerIdsAttr,
                        getI64Array(builder, routine.semanticLayerIds));
    if (!routine.layerRegionEpochs.empty())
      function->setAttr(kDeploymentLayerRegionEpochsAttr,
                        getI64Array(builder, routine.layerRegionEpochs));
    if (!routine.controlPredecessors.empty()) {
      SmallVector<int64_t> predecessorIds;
      predecessorIds.reserve(routine.controlPredecessors.size());
      for (unsigned predecessor : routine.controlPredecessors) {
        if (routines[predecessor].physicalTileId == routine.physicalTileId)
          predecessorIds.push_back(routines[predecessor].globalId);
      }
      llvm::sort(predecessorIds);
      if (!predecessorIds.empty())
        function->setAttr(kDeploymentControlDependencyIdsAttr,
                          getI64Array(builder, predecessorIds));
    }
    if (routine.sequenceShardIndex >= 0) {
      function->setAttr(golem_tiling_attrs::kSourceMVMIdAttrName,
                        builder.getI64IntegerAttr(routine.sequenceSourceMVMId));
      function->setAttr(kSequenceShardIndexAttr,
                        builder.getI64IntegerAttr(routine.sequenceShardIndex));
    }
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
    SmallVector<Operation *> orderedOperations(
        routine.selectedOperations.begin(), routine.selectedOperations.end());
    for (Operation *operation : orderedOperations) {
      if (operation->getBlock() != &sourceBlock)
        return operation->emitError(
            "routine selected a non-top-level operation for cloning");
    }
    llvm::sort(orderedOperations, [&](Operation *lhs, Operation *rhs) {
      return sourcePosition.lookup(lhs) < sourcePosition.lookup(rhs);
    });
    for (Operation *operation : orderedOperations) {
      Operation *cloned = builder.clone(*operation, mapping);
      cloned->setAttr(kDeploymentPhysicalTileIdAttr,
                      builder.getI64IntegerAttr(routine.physicalTileId));
      auto analogLane = analogLaneByOperation.find(operation);
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
    if (routine.remoteControlOutputResourceId) {
      auto value = DenseElementsAttr::get(controlTokenType,
                                          builder.getI32IntegerAttr(0));
      returns.push_back(builder.create<arith::ConstantOp>(
          source.getLoc(), controlTokenType, value));
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
    inputResourceIds.append(routine.remoteControlInputResourceIds.begin(),
                            routine.remoteControlInputResourceIds.end());
    if (routine.remoteControlOutputResourceId)
      outputResourceIds.push_back(*routine.remoteControlOutputResourceId);
    function->setAttr(kDeploymentInputResourceIdsAttr,
                      getI64Array(builder, inputResourceIds));
    function->setAttr(kDeploymentOutputResourceIdsAttr,
                      getI64Array(builder, outputResourceIds));
    routine.function = function;
  }
  return success();
}

struct ConvPatchStreamMatch {
  linalg::GenericOp patch;
  SmallVector<scf::ForOp> consumers;
  SmallVector<tensor::ExtractSliceOp> rowSlices;
  int64_t rows = 0;
  int64_t columns = 0;
};

std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  return std::nullopt;
}

std::optional<ConvPatchStreamMatch>
matchLocalConvPatchStream(linalg::GenericOp patch) {
  auto section = patch->getAttrOfType<StringAttr>("sculptor.semantic.section");
  auto patchType = dyn_cast<RankedTensorType>(patch.getResult(0).getType());
  if (!section || section.getValue() != "digital.conv_patch" ||
      patch->getNumResults() != 1 || patch.getNumDpsInits() != 1 ||
      !patchType || !patchType.hasStaticShape() || patchType.getRank() != 2 ||
      patchType.getDimSize(0) <= 0 || patchType.getDimSize(1) <= 0)
    return std::nullopt;

  ConvPatchStreamMatch match;
  match.patch = patch;
  match.rows = patchType.getDimSize(0);
  match.columns = patchType.getDimSize(1);
  for (Operation *user : patch.getResult(0).getUsers()) {
    auto slice = dyn_cast<tensor::ExtractSliceOp>(user);
    auto loop = slice ? dyn_cast_or_null<scf::ForOp>(slice->getParentOp())
                      : scf::ForOp{};
    if (!slice || !loop || slice->getBlock() != loop.getBody() ||
        loop.getNumRegionIterArgs() != 1 || loop->getNumResults() != 1 ||
        getConstantIndexValue(loop.getLowerBound()) != 0 ||
        getConstantIndexValue(loop.getUpperBound()) != match.rows ||
        getConstantIndexValue(loop.getStep()) != 1 ||
        slice.getStaticSizes() != ArrayRef<int64_t>({1, match.columns}) ||
        slice.getStaticStrides() != ArrayRef<int64_t>({1, 1}))
      return std::nullopt;

    SmallVector<OpFoldResult> offsets = slice.getMixedOffsets();
    if (offsets.size() != 2 || !llvm::isa<Value>(offsets[0]) ||
        llvm::cast<Value>(offsets[0]) != loop.getInductionVar() ||
        getConstantIntValue(offsets[1]) != 0)
      return std::nullopt;
    if (llvm::is_contained(match.consumers, loop))
      return std::nullopt;

    match.consumers.push_back(loop);
    match.rowSlices.push_back(slice);
  }
  if (match.consumers.empty())
    return std::nullopt;

  SmallVector<unsigned> order(match.consumers.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::sort(order, [&](unsigned lhs, unsigned rhs) {
    return match.consumers[lhs]->isBeforeInBlock(match.consumers[rhs]);
  });
  SmallVector<scf::ForOp> sortedConsumers;
  SmallVector<tensor::ExtractSliceOp> sortedSlices;
  for (unsigned index : order) {
    sortedConsumers.push_back(match.consumers[index]);
    sortedSlices.push_back(match.rowSlices[index]);
  }
  match.consumers = std::move(sortedConsumers);
  match.rowSlices = std::move(sortedSlices);
  return match;
}

LogicalResult streamLocalConvPatchRows(func::FuncOp function,
                                       int64_t &streamedPatches,
                                       int64_t &streamedRows) {
  SmallVector<linalg::GenericOp> candidates;
  function.walk([&](linalg::GenericOp generic) {
    auto section =
        generic->getAttrOfType<StringAttr>("sculptor.semantic.section");
    if (section && section.getValue() == "digital.conv_patch")
      candidates.push_back(generic);
  });
  IRRewriter rewriter(function.getContext());
  for (linalg::GenericOp patch : candidates) {
    if (!patch || !patch->getBlock())
      continue;
    std::optional<ConvPatchStreamMatch> matched =
        matchLocalConvPatchStream(patch);
    if (!matched)
      continue;
    ConvPatchStreamMatch &match = *matched;

    SmallVector<Value> initialValues;
    rewriter.setInsertionPoint(match.consumers.front());
    auto rowType = RankedTensorType::get(
        {1, match.columns},
        cast<RankedTensorType>(patch.getResult(0).getType()).getElementType());
    initialValues.push_back(rewriter.create<tensor::EmptyOp>(
        patch.getLoc(), rowType.getShape(), rowType.getElementType()));
    for (scf::ForOp consumer : match.consumers) {
      auto outputType =
          dyn_cast<RankedTensorType>(consumer.getInitArgs().front().getType());
      if (!outputType || !outputType.hasStaticShape())
        return consumer.emitOpError(
            "streamed convolution MVM requires a static tensor result");
      initialValues.push_back(rewriter.create<tensor::EmptyOp>(
          consumer.getLoc(), outputType.getShape(),
          outputType.getElementType()));
    }
    Value lower = rewriter.create<arith::ConstantIndexOp>(patch.getLoc(), 0);
    Value upper =
        rewriter.create<arith::ConstantIndexOp>(patch.getLoc(), match.rows);
    Value step = rewriter.create<arith::ConstantIndexOp>(patch.getLoc(), 1);
    auto rowLoop = rewriter.create<scf::ForOp>(patch.getLoc(), lower, upper,
                                               step, initialValues);
    rowLoop->setAttr("sculptor.semantic.section",
                     rewriter.getStringAttr("digital.conv_patch_stream"));
    rowLoop->setAttr("sculptor.memory.streamed_conv_patch",
                     rewriter.getUnitAttr());

    Block *body = rowLoop.getBody();
    scf::YieldOp defaultYield;
    if (!body->empty())
      defaultYield = dyn_cast<scf::YieldOp>(body->getTerminator());
    if (!defaultYield) {
      rewriter.setInsertionPointToEnd(body);
      defaultYield = rewriter.create<scf::YieldOp>(patch.getLoc(),
                                                   rowLoop.getRegionIterArgs());
    }
    rewriter.setInsertionPoint(defaultYield);
    ValueRange dpsInputs = patch.getDpsInputs();
    SmallVector<Value> patchInputs(dpsInputs.begin(), dpsInputs.end());
    auto tiledPatch = rewriter.create<linalg::GenericOp>(
        patch.getLoc(), rowType, patchInputs,
        ValueRange{rowLoop.getRegionIterArgs().front()},
        patch.getIndexingMapsArray(), patch.getIteratorTypesArray(),
        [&](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange arguments) {
          IRMapping mapping;
          for (auto [oldArgument, newArgument] :
               llvm::zip_equal(patch.getBody()->getArguments(), arguments))
            mapping.map(oldArgument, newArgument);
          for (Operation &operation : patch.getBody()->without_terminator()) {
            auto index = dyn_cast<linalg::IndexOp>(&operation);
            if (!index || index.getDim() != 0) {
              bodyBuilder.clone(operation, mapping);
              continue;
            }
            auto localIndex = bodyBuilder.create<linalg::IndexOp>(
                index.getLoc(), index.getDim());
            localIndex->setAttrs(index->getAttrs());
            Value globalIndex = bodyBuilder.create<arith::AddIOp>(
                index.getLoc(), localIndex, rowLoop.getInductionVar());
            mapping.map(index.getResult(), globalIndex);
          }
          auto oldYield =
              cast<linalg::YieldOp>(patch.getBody()->getTerminator());
          SmallVector<Value> results;
          for (Value result : oldYield.getValues())
            results.push_back(mapping.lookupOrDefault(result));
          bodyBuilder.create<linalg::YieldOp>(bodyLoc, results);
        });
    tiledPatch->setAttrs(patch->getAttrs());
    Value streamedRow = tiledPatch.getResult(0);

    SmallVector<Value> yieldedValues{streamedRow};
    rewriter.setInsertionPoint(defaultYield);
    for (auto [consumerIndex, consumer] : llvm::enumerate(match.consumers)) {
      tensor::ExtractSliceOp oldSlice = match.rowSlices[consumerIndex];
      IRMapping mapping;
      mapping.map(consumer.getInductionVar(), rowLoop.getInductionVar());
      mapping.map(consumer.getRegionIterArgs().front(),
                  rowLoop.getRegionIterArgs()[consumerIndex + 1]);
      mapping.map(oldSlice.getResult(), streamedRow);
      for (Operation &operation : consumer.getBody()->without_terminator()) {
        if (&operation == oldSlice.getOperation())
          continue;
        rewriter.clone(operation, mapping);
      }
      auto oldYield = cast<scf::YieldOp>(consumer.getBody()->getTerminator());
      Value yielded = mapping.lookupOrDefault(oldYield.getOperand(0));
      yieldedValues.push_back(yielded);
    }

    defaultYield.getResultsMutable().assign(yieldedValues);
    for (auto [consumerIndex, consumer] : llvm::enumerate(match.consumers))
      consumer.getResult(0).replaceAllUsesWith(
          rowLoop.getResult(consumerIndex + 1));

    SmallVector<Operation *> deadInitializers;
    for (scf::ForOp consumer : match.consumers) {
      if (Operation *initializer =
              consumer.getInitArgs().front().getDefiningOp())
        deadInitializers.push_back(initializer);
      rewriter.eraseOp(consumer);
    }
    Operation *patchDestination = patch.getOutputs().front().getDefiningOp();
    rewriter.eraseOp(patch);
    for (Operation *operation : deadInitializers)
      if (operation && operation->getBlock() && operation->use_empty() &&
          isa<tensor::EmptyOp>(operation))
        rewriter.eraseOp(operation);
    if (patchDestination && patchDestination->getBlock() &&
        patchDestination->use_empty() && isa<tensor::EmptyOp>(patchDestination))
      rewriter.eraseOp(patchDestination);

    ++streamedPatches;
    streamedRows += match.rows;
  }

  if (streamedPatches > 0) {
    OpBuilder builder(function.getContext());
    function->setAttr(kStreamedConvPatchCountAttr,
                      builder.getI64IntegerAttr(streamedPatches));
    function->setAttr(kStreamedConvPatchRowCountAttr,
                      builder.getI64IntegerAttr(streamedRows));
  }
  return success();
}

LogicalResult
streamLocalConvPatchRowsInRoutines(MutableArrayRef<RoutinePlan> routines,
                                   ModuleOp outer) {
  int64_t streamedPatches = 0;
  int64_t streamedRows = 0;
  for (RoutinePlan &routine : routines) {
    if (routine.boot)
      continue;
    int64_t routinePatches = 0;
    int64_t routineRows = 0;
    if (failed(streamLocalConvPatchRows(routine.function, routinePatches,
                                        routineRows)))
      return failure();
    streamedPatches += routinePatches;
    streamedRows += routineRows;
  }
  OpBuilder builder(outer.getContext());
  outer->setAttr(kStreamedConvPatchCountAttr,
                 builder.getI64IntegerAttr(streamedPatches));
  outer->setAttr(kStreamedConvPatchRowCountAttr,
                 builder.getI64IntegerAttr(streamedRows));
  return success();
}

LogicalResult
fuseElementwiseOperationsInRoutines(MutableArrayRef<RoutinePlan> routines,
                                    bool enabled) {
  if (!enabled)
    return success();
  for (RoutinePlan &routine : routines) {
    if (routine.boot)
      continue;
    RewritePatternSet patterns(routine.function.getContext());
    linalg::populateElementwiseOpsFusionPatterns(
        patterns, [](OpOperand *) { return true; });
    if (failed(applyPatternsGreedily(routine.function, std::move(patterns)))) {
      return routine.function.emitError(
          "failed to fuse elementwise producer-consumer operations");
    }
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
  outer->setAttr(tile_memory::kConfiguredCapacityAttrName,
                 builder.getI64IntegerAttr(placement.tileMemoryCapacityBytes));
  for (const auto &[tileId, tile] : tileModules) {
    tile->setAttr(tile_memory::kConfiguredCapacityAttrName,
                  builder.getI64IntegerAttr(placement.tileMemoryCapacityBytes));
    (void)tileId;
  }
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
  std::map<int64_t, std::pair<Type, int64_t>> routeResourceTypes;
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

    const RoutinePlan &source = routines[route.sourceRoutine];
    const RoutinePlan &destination = routines[route.destinationRoutine];
    func::FuncOp sourceFunction = source.function;
    func::FuncOp destinationFunction = destination.function;
    if (route.sourceOutput >= sourceFunction.getNumResults() ||
        route.destinationInput >= destinationFunction.getNumArguments()) {
      return outer.emitError("remote route references an invalid routine port");
    }
    Type sourceType = sourceFunction.getResultTypes()[route.sourceOutput];
    Type destinationType =
        destinationFunction.getArgumentTypes()[route.destinationInput];
    if (sourceType != destinationType) {
      return outer.emitError("remote route type mismatch: source routine ")
             << source.globalId << " output " << route.sourceOutput << " has "
             << sourceType << ", but destination routine "
             << destination.globalId << " input " << route.destinationInput
             << " expects " << destinationType;
    }
    FailureOr<int64_t> staticByteSize =
        getStaticByteSize(sourceType, sourceFunction);
    if (failed(staticByteSize))
      return failure();
    if (*staticByteSize != route.byteSize) {
      return outer.emitError("remote route byte-size mismatch for resource ")
             << route.resourceId << ": route declares " << route.byteSize
             << " bytes but " << sourceType << " requires "
             << *staticByteSize;
    }

    auto [entry, inserted] = routeResourceTypes.emplace(
        route.resourceId, std::make_pair(sourceType, route.byteSize));
    if (!inserted && (entry->second.first != sourceType ||
                      entry->second.second != route.byteSize)) {
      return outer.emitError(
          "fan-out routes sharing one resource disagree on payload type or "
          "byte size");
    }
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
  for (const RoutinePlan &routine : routines) {
    if (!routine.remoteControlOutputResourceId)
      continue;
    if (*routine.remoteControlOutputResourceId < 0 ||
        !resourceIds.insert(*routine.remoteControlOutputResourceId).second)
      return outer.emitError(
          "duplicate or invalid remote-control resource ID");
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

LogicalResult outlineFunction(ModuleOp outer, func::FuncOp source,
                              bool fuseProducerConsumer,
                              bool consolidateLayerRegions,
                              int64_t sequenceWavesInFlight) {
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
  FailureOr<SmallVector<EndpointDependencyRecord>> endpointDependencies =
      buildEndpointDependencies(*endpoints, *tileGraph, source);
  if (failed(endpointDependencies))
    return failure();
  RoutineFusionStats fusionStats;
  llvm::SmallDenseSet<int64_t> protectedOperationIds;
  for (const ComputeTensor &tensor : graph->tensors) {
    if (!tensor.isFunctionOutput)
      continue;
    protectedOperationIds.insert(tensor.producerOperations.begin(),
                                 tensor.producerOperations.end());
  }
  auto assemblyBoundaryCount = source->getAttrOfType<IntegerAttr>(
      mapping::kAssemblyBoundaryCountAttrName);
  bool enableProducerConsumerFusion =
      fuseProducerConsumer &&
      (!assemblyBoundaryCount || assemblyBoundaryCount.getInt() == 0);
  FailureOr<SmallVector<RoutinePlan, 0>> routines =
      buildRoutineRegions(*endpoints, *endpointDependencies,
                          enableProducerConsumerFusion,
                          consolidateLayerRegions, protectedOperationIds,
                          fusionStats, source);
  if (failed(routines))
    return failure();

  if (fuseProducerConsumer || consolidateLayerRegions) {
    OpBuilder outerBuilder(outer.getContext());
    outer->setAttr(
        kRoutineCountBeforeFusionAttr,
        outerBuilder.getI64IntegerAttr(fusionStats.initialRoutineCount));
    outer->setAttr(
        kRoutineCountAfterFusionAttr,
        outerBuilder.getI64IntegerAttr(fusionStats.finalRoutineCount));
    outer->setAttr(kFusionCountAttr, outerBuilder.getI64IntegerAttr(
                                         fusionStats.fusedBoundaryCount));
    outer->setAttr(
        kFusionBoundaryBytesAttr,
        outerBuilder.getI64IntegerAttr(fusionStats.fusedBoundaryBytes));
    if (consolidateLayerRegions) {
      outer->setAttr(
          kLayerRegionConsolidationCountAttr,
          outerBuilder.getI64IntegerAttr(fusionStats.fusedBoundaryCount));
      outer->setAttr(kLayerRegionConsolidationBoundaryBytesAttr,
                     outerBuilder.getI64IntegerAttr(
                         fusionStats.fusedBoundaryBytes));
    }
  }

  DenseMap<Operation *, unsigned> mappedOwner;
  if (failed(buildRoutineClosures(source, *routines, mappedOwner)))
    return failure();
  SmallVector<std::pair<unsigned, Value>> modelOutputs;
  if (failed(attachModelOutputs(source, *routines, mappedOwner, modelOutputs,
                                *placement)))
    return failure();

  SmallVector<unsigned> routineOrder = getRoutineOrder(*routines, *endpoints);
  if (failed(assignRoutineIds(*routines, routineOrder)))
    return failure();
  if (failed(buildBoundedSequenceDependencies(
          *routines, *endpoints, sequenceWavesInFlight, source)))
    return failure();

  OpBuilder deploymentBuilder(outer.getContext());
  outer->setAttr("sculptor.deployment.sequence_waves_in_flight",
                 deploymentBuilder.getI64IntegerAttr(sequenceWavesInFlight));

  DenseMap<Value, int64_t> resourceIdByValue;
  SmallVector<RouteRecord> routes;
  SmallVector<LocalBindingRecord> localBindings;
  if (failed(assignResourcesAndConnections(
          source, *routines, routineOrder, *endpoints, *endpointDependencies,
          mappedOwner, resourceIdByValue, routes, localBindings)))
    return failure();

  DenseMap<int64_t, ModuleOp> tileModules;
  if (failed(createRoutineFunctions(outer, source, *routines, routineOrder,
                                    *endpoints, placement->mesh.arraysPerCore,
                                    resourceIdByValue, tileModules)))
    return failure();
  if (failed(streamLocalConvPatchRowsInRoutines(*routines, outer)))
    return failure();
  if (failed(fuseElementwiseOperationsInRoutines(
          *routines, enableProducerConsumerFusion)))
    return failure();
  if (failed(attachDeploymentManifest(
          outer, source, *routines, routes, localBindings, modelOutputs,
          modelInputTensorIds, modelOutputTensorIds, resourceIdByValue,
          *placement, tileModules)))
    return failure();
  if (failed(verifyOutlinedDeployment(outer, *routines, routes, localBindings,
                                      resourceIdByValue, tileModules)))
    return failure();
  if (failed(tile_memory::buildAndAttachTileMemoryPlan(outer)))
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
  if (sequenceWavesInFlight <= 0) {
    module.emitError("sequence-waves-in-flight must be positive");
    signalPassFailure();
    return;
  }
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
  if (failed(outlineFunction(module, mappedFunctions.front(),
                             fuseProducerConsumer, consolidateLayerRegions,
                             sequenceWavesInFlight))) {
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
