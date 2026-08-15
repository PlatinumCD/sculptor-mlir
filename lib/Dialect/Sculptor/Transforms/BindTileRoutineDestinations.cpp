#include "sculptor-mlir/Dialect/Sculptor/Transforms/BindTileRoutineDestinations.h"

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/TileMemoryPlan.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <map>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
namespace tile_memory = mlir::sculptor::tile_memory;

constexpr StringLiteral kGlobalRoutineIdAttr =
    "sculptor.deployment.global_routine_id";
constexpr StringLiteral kOutputResourceIdsAttr =
    "sculptor.deployment.output_resource_ids";
constexpr StringLiteral kRoutineKindAttr = "sculptor.deployment.routine_kind";
constexpr StringLiteral kDestinationBindingsAttr =
    "sculptor.memory.destination_bindings";
constexpr StringLiteral kDestinationBoundAttr =
    "sculptor.memory.destination_bound";

template <typename AttrTy>
FailureOr<SmallVector<AttrTy>> getTypedMemoryArray(ModuleOp module,
                                                   StringRef name) {
  auto values = module->getAttrOfType<ArrayAttr>(name);
  if (!values)
    return module.emitError("expected tile memory-plan attribute '")
           << name << "'";
  SmallVector<AttrTy> result;
  for (Attribute value : values) {
    auto typed = dyn_cast<AttrTy>(value);
    if (!typed)
      return module.emitError("invalid typed tile memory-plan record");
    result.push_back(typed);
  }
  return result;
}

FailureOr<int64_t> getOutputCount(func::FuncOp function) {
  auto resources = function->getAttrOfType<ArrayAttr>(kOutputResourceIdsAttr);
  if (!resources)
    return function.emitError("outlined routine has no output resource IDs");
  for (Attribute value : resources)
    if (!isa<IntegerAttr>(value))
      return function.emitError("routine output resource IDs must be integers");
  return static_cast<int64_t>(resources.size());
}

IntegerAttr getArgumentIntegerAttr(BlockArgument argument, StringRef name) {
  return cast<FunctionOpInterface>(argument.getOwner()->getParentOp())
      .getArgAttrOfType<IntegerAttr>(argument.getArgNumber(), name);
}

BlockArgument stripToBlockArgument(Value value) {
  while (true) {
    if (auto argument = dyn_cast<BlockArgument>(value))
      return argument;
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    return {};
  }
}

bool staticSubviewMatches(memref::SubViewOp subview, TileMemoryViewAttr view) {
  if (llvm::is_contained(subview.getStaticOffsets(), ShapedType::kDynamic) ||
      llvm::is_contained(subview.getStaticSizes(), ShapedType::kDynamic) ||
      llvm::is_contained(subview.getStaticStrides(), ShapedType::kDynamic))
    return false;
  auto matches = [](ArrayRef<int64_t> values, ArrayAttr attributes) {
    if (values.size() != attributes.size())
      return false;
    return llvm::all_of(llvm::zip(values, attributes), [](auto pair) {
      return std::get<0>(pair) == cast<IntegerAttr>(std::get<1>(pair)).getInt();
    });
  };
  return matches(subview.getStaticOffsets(), view.getOffsets()) &&
         matches(subview.getStaticSizes(), view.getSizes()) &&
         llvm::all_of(subview.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; });
}

bool isAssemblyPair(int64_t sourceView, int64_t destinationView,
                    ArrayRef<TileMemoryAssemblyAttr> assemblies,
                    ArrayRef<TileMemoryMovementAttr> movements) {
  for (TileMemoryAssemblyAttr assembly : assemblies) {
    ArrayAttr sources = assembly.getContributingViewIds();
    ArrayAttr destinations = assembly.getDestinationViewIds();
    for (auto [source, destination] : llvm::zip(sources, destinations)) {
      if (cast<IntegerAttr>(source).getInt() != sourceView ||
          cast<IntegerAttr>(destination).getInt() != destinationView)
        continue;
      return llvm::any_of(movements, [&](TileMemoryMovementAttr movement) {
        return movement.getAssemblyId().getInt() == assembly.getId().getInt() &&
               movement.getSourceViewId().getInt() == sourceView &&
               movement.getDestinationViewId().getInt() == destinationView &&
               (movement.getMode() == MemoryMovementMode::Contiguous ||
                movement.getMode() == MemoryMovementMode::Segmented);
      });
    }
  }
  return false;
}

int64_t
foldRouteAssemblyCopies(func::FuncOp function,
                        const std::map<int64_t, TileMemoryOwnerAttr> &owners,
                        const std::map<int64_t, TileMemoryViewAttr> &views,
                        ArrayRef<TileMemoryAssemblyAttr> assemblies,
                        ArrayRef<TileMemoryMovementAttr> movements) {
  int64_t eliminated = 0;
  SmallVector<memref::CopyOp> copies;
  function.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
  for (memref::CopyOp copy : copies) {
    BlockArgument source = stripToBlockArgument(copy.getSource());
    auto subview = copy.getTarget().getDefiningOp<memref::SubViewOp>();
    if (!source || !subview || !source.hasOneUse())
      continue;
    BlockArgument destination = stripToBlockArgument(subview.getSource());
    if (!destination)
      continue;
    IntegerAttr sourceOwner =
        getArgumentIntegerAttr(source, tile_memory::kOwnerIdAttrName);
    IntegerAttr sourceView =
        getArgumentIntegerAttr(source, tile_memory::kViewIdAttrName);
    IntegerAttr destinationOwner =
        getArgumentIntegerAttr(destination, tile_memory::kOwnerIdAttrName);
    if (!sourceOwner || !sourceView || !destinationOwner)
      continue;
    auto owner = owners.find(sourceOwner.getInt());
    if (owner == owners.end() ||
        owner->second.getKind() != MemoryOwnerKind::RouteInput)
      continue;
    auto sourcePlanView = views.find(sourceView.getInt());
    if (sourcePlanView == views.end())
      continue;
    for (const auto &[viewId, view] : views) {
      if (view.getOwnerId().getInt() != destinationOwner.getInt() ||
          view.getByteSize().getInt() !=
              sourcePlanView->second.getByteSize().getInt() ||
          !staticSubviewMatches(subview, view) ||
          !isAssemblyPair(sourceView.getInt(), viewId, assemblies, movements))
        continue;
      copy.erase();
      if (subview->use_empty())
        subview.erase();
      ++eliminated;
      break;
    }
  }
  return eliminated;
}

bool isBefore(Operation *operation, Operation *limit) {
  Operation *ancestor = operation;
  while (ancestor && ancestor->getBlock() != limit->getBlock())
    ancestor = ancestor->getParentOp();
  return ancestor && ancestor != limit && ancestor->isBeforeInBlock(limit);
}

bool foldAllocationCopy(memref::CopyOp copy, BlockArgument destination) {
  if (copy.getTarget() != destination || !destination.hasOneUse())
    return false;
  Value sourceStorage = copy.getSource();
  if (auto result = dyn_cast<OpResult>(sourceStorage)) {
    if (auto loop = dyn_cast<scf::ForOp>(result.getOwner())) {
      unsigned index = result.getResultNumber();
      auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
      if (index >= loop.getInitArgs().size() || !yield ||
          index >= yield.getNumOperands() ||
          yield.getOperand(index) != loop.getRegionIterArgs()[index])
        return false;
      sourceStorage = loop.getInitArgs()[index];
    }
  }
  auto allocation = sourceStorage.getDefiningOp<memref::AllocOp>();
  if (!allocation || allocation.getType() != destination.getType() ||
      !allocation.getDynamicSizes().empty() ||
      allocation->getBlock() != copy->getBlock() ||
      !allocation->isBeforeInBlock(copy))
    return false;

  SmallVector<memref::DeallocOp> deallocations;
  for (OpOperand &use : allocation.getMemref().getUses()) {
    Operation *user = use.getOwner();
    if (user == copy)
      continue;
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      if (dealloc.getMemref() != allocation.getMemref() ||
          dealloc->getBlock() != copy->getBlock() ||
          !copy->isBeforeInBlock(dealloc))
        return false;
      deallocations.push_back(dealloc);
      continue;
    }
    if (!isBefore(user, copy))
      return false;
  }

  for (OpOperand &use :
       llvm::make_early_inc_range(allocation.getMemref().getUses())) {
    if (use.getOwner() != copy && !isa<memref::DeallocOp>(use.getOwner()))
      use.set(destination);
  }
  for (memref::DeallocOp dealloc : deallocations)
    dealloc.erase();
  copy.erase();
  if (allocation.getMemref().use_empty())
    allocation.erase();
  return true;
}

// Bufferization preserves a tensor destination operand by computing into the
// corresponding input memref and copying the result to the routine's output
// argument.  That is incorrect at a task boundary when the same SSA tensor is
// consumed by more than one routine: the first routine mutates the shared
// input before a later routine reads its initializer.  Seed the private output
// from the input, redirect the routine-local computation to the output, and
// remove the trailing copy.
bool foldInputResultCopy(memref::CopyOp copy, BlockArgument destination) {
  if (copy.getTarget() != destination || !destination.hasOneUse())
    return false;
  auto source = dyn_cast<BlockArgument>(copy.getSource());
  if (!source || source == destination ||
      source.getOwner() != destination.getOwner() ||
      source.getType() != destination.getType() ||
      copy->getBlock() != source.getOwner())
    return false;

  Operation *firstUse = nullptr;
  for (OpOperand &use : source.getUses()) {
    Operation *user = use.getOwner();
    if (user == copy)
      continue;
    if (user->getBlock() != copy->getBlock() || !user->isBeforeInBlock(copy))
      return false;
    if (!firstUse || user->isBeforeInBlock(firstUse))
      firstUse = user;
  }
  if (!firstUse)
    return false;

  OpBuilder builder(firstUse);
  memref::CopyOp seed =
      builder.create<memref::CopyOp>(copy.getLoc(), source, destination);
  for (OpOperand &use : llvm::make_early_inc_range(source.getUses()))
    if (use.getOwner() != copy && use.getOwner() != seed)
      use.set(destination);
  copy.erase();
  return true;
}

// Tensor concatenation followed by a reshape frequently bufferizes as a
// hierarchy of temporary allocations.  The leaves are copied into an
// allocation, the allocation is collapsed, and the collapsed value is copied
// into a slice of the runtime-owned result.  Expand that result slice back to
// the allocation's shape and retarget the leaf views to it.  This preserves
// the exact strided geometry while removing both the allocation and the
// terminal copy.
bool foldNestedAssemblyCopy(memref::CopyOp terminalCopy,
                            BlockArgument destination) {
  auto destinationView =
      terminalCopy.getTarget().getDefiningOp<memref::SubViewOp>();
  auto collapse =
      terminalCopy.getSource().getDefiningOp<memref::CollapseShapeOp>();
  if (!destinationView || destinationView.getSource() != destination ||
      !destinationView->hasOneUse() || !collapse || !collapse->hasOneUse() ||
      terminalCopy->getBlock() != destination.getOwner())
    return false;

  auto allocation = collapse.getSrc().getDefiningOp<memref::AllocOp>();
  auto allocationType =
      allocation ? dyn_cast<MemRefType>(allocation.getType()) : MemRefType();
  auto collapsedType = dyn_cast<MemRefType>(collapse.getType());
  auto destinationViewType = dyn_cast<MemRefType>(destinationView.getType());
  if (!allocation || !allocationType || !collapsedType ||
      !destinationViewType || !allocationType.hasStaticShape() ||
      !allocationType.getLayout().isIdentity() ||
      !allocation.getDynamicSizes().empty() ||
      collapsedType.getShape() != destinationViewType.getShape() ||
      collapsedType.getElementType() != destinationViewType.getElementType() ||
      allocation->getBlock() != terminalCopy->getBlock() ||
      collapse->getBlock() != terminalCopy->getBlock() ||
      destinationView->getBlock() != terminalCopy->getBlock() ||
      !allocation->isBeforeInBlock(collapse) ||
      !collapse->isBeforeInBlock(terminalCopy) ||
      llvm::is_contained(destinationView.getStaticOffsets(),
                         ShapedType::kDynamic) ||
      llvm::is_contained(destinationView.getStaticSizes(),
                         ShapedType::kDynamic) ||
      llvm::is_contained(destinationView.getStaticStrides(),
                         ShapedType::kDynamic))
    return false;

  SmallVector<memref::SubViewOp> assemblyViews;
  SmallVector<memref::DeallocOp> deallocations;
  for (OpOperand &use : allocation.getMemref().getUses()) {
    Operation *user = use.getOwner();
    if (user == collapse)
      continue;
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      if (dealloc.getMemref() != allocation.getMemref() ||
          dealloc->getBlock() != terminalCopy->getBlock() ||
          !terminalCopy->isBeforeInBlock(dealloc))
        return false;
      deallocations.push_back(dealloc);
      continue;
    }
    auto view = dyn_cast<memref::SubViewOp>(user);
    if (!view || view.getSource() != allocation.getMemref() ||
        view->getBlock() != terminalCopy->getBlock() ||
        !view->isBeforeInBlock(collapse) ||
        llvm::is_contained(view.getStaticOffsets(), ShapedType::kDynamic) ||
        llvm::is_contained(view.getStaticSizes(), ShapedType::kDynamic) ||
        llvm::is_contained(view.getStaticStrides(), ShapedType::kDynamic))
      return false;
    for (Operation *viewUser : view->getUsers())
      if (!isBefore(viewUser, collapse))
        return false;
    assemblyViews.push_back(view);
  }
  if (assemblyViews.empty())
    return false;

  // Complete validation before changing the routine.  In particular, the
  // inferred expanded layout must exist for the destination's strided view.
  OpBuilder builder(allocation);
  Location loc = terminalCopy.getLoc();
  auto directCollapsed = builder.create<memref::SubViewOp>(
      loc, destination, destinationView.getMixedOffsets(),
      destinationView.getMixedSizes(), destinationView.getMixedStrides());
  auto expanded = builder.create<memref::ExpandShapeOp>(
      loc, allocationType.getShape(), directCollapsed,
      collapse.getReassociationIndices());
  expanded->setAttr("sculptor.memory.direct_nested_assembly",
                    builder.getUnitAttr());

  for (memref::SubViewOp oldView : assemblyViews) {
    auto oldType = cast<MemRefType>(oldView.getType());
    MemRefType directType = memref::SubViewOp::inferRankReducedResultType(
        oldType.getShape(), expanded.getType(), oldView.getStaticOffsets(),
        oldView.getStaticSizes(), oldView.getStaticStrides());
    OpBuilder viewBuilder(oldView);
    auto directView = viewBuilder.create<memref::SubViewOp>(
        oldView.getLoc(), directType, expanded, oldView.getMixedOffsets(),
        oldView.getMixedSizes(), oldView.getMixedStrides());
    oldView.getResult().replaceAllUsesWith(directView.getResult());
    oldView.erase();
  }

  terminalCopy.erase();
  if (destinationView->use_empty())
    destinationView.erase();
  collapse.erase();
  for (memref::DeallocOp dealloc : deallocations)
    dealloc.erase();
  allocation.erase();
  return true;
}

struct AssemblySegment {
  memref::SubViewOp target;
  memref::CopyOp copy;
  int64_t offset = 0;
  int64_t size = 0;
};

// Fuse a concat-like assembly allocation into a consuming reshape/subview/
// transpose.  Each contiguous input segment is decomposed at expanded-shape
// row boundaries and transposed directly into its exact destination slice.
// This is useful for channel-sharded feature maps: a tile that needs only a
// spatial output shard must never materialize the complete channel concat.
int64_t foldAssembledTranspose(linalg::TransposeOp transpose,
                               BlockArgument destination) {
  if (transpose.getInit() != destination || transpose.getNumResults() != 0)
    return 0;
  SmallVector<memref::SubViewOp> consumedViews;
  Value consumed = transpose.getInput();
  while (auto view = consumed.getDefiningOp<memref::SubViewOp>()) {
    if (!view->hasOneUse() ||
        llvm::is_contained(view.getStaticOffsets(), ShapedType::kDynamic) ||
        llvm::is_contained(view.getStaticSizes(), ShapedType::kDynamic) ||
        llvm::is_contained(view.getStaticStrides(), ShapedType::kDynamic) ||
        !llvm::all_of(view.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; }) ||
        view.getSourceType().getRank() != view.getType().getRank())
      return 0;
    consumedViews.push_back(view);
    consumed = view.getSource();
  }
  auto expand = consumed.getDefiningOp<memref::ExpandShapeOp>();
  auto allocation = expand ? expand.getSrc().getDefiningOp<memref::AllocOp>()
                           : memref::AllocOp();
  if (consumedViews.empty() || !expand || !expand->hasOneUse() || !allocation ||
      !allocation.getDynamicSizes().empty())
    return 0;

  auto allocationType = dyn_cast<MemRefType>(allocation.getType());
  auto expandedType = dyn_cast<MemRefType>(expand.getType());
  auto consumedType = dyn_cast<MemRefType>(transpose.getInput().getType());
  auto destinationType = dyn_cast<MemRefType>(destination.getType());
  if (!allocationType || !expandedType || !consumedType || !destinationType ||
      !allocationType.hasStaticShape() ||
      !allocationType.getLayout().isIdentity() ||
      !expandedType.hasStaticShape() || !consumedType.hasStaticShape() ||
      !destinationType.hasStaticShape() ||
      allocationType.getElementType() != destinationType.getElementType() ||
      allocation->getBlock() != transpose->getBlock() ||
      expand->getBlock() != transpose->getBlock() ||
      !allocation->isBeforeInBlock(expand) ||
      !expand->isBeforeInBlock(consumedViews.back()) ||
      !consumedViews.front()->isBeforeInBlock(transpose))
    return 0;
  for (memref::SubViewOp view : consumedViews)
    if (view->getBlock() != transpose->getBlock())
      return 0;

  SmallVector<int64_t> consumedOffsets(expandedType.getRank(), 0);
  SmallVector<int64_t> consumedSizes(expandedType.getShape());
  for (memref::SubViewOp view : llvm::reverse(consumedViews)) {
    for (unsigned dim = 0; dim < expandedType.getRank(); ++dim) {
      consumedOffsets[dim] += view.getStaticOffsets()[dim];
      consumedSizes[dim] = view.getStaticSizes()[dim];
    }
  }

  SmallVector<ReassociationIndices> reassociation =
      expand.getReassociationIndices();
  if (reassociation.size() != static_cast<size_t>(allocationType.getRank()) ||
      consumedType.getRank() != expandedType.getRank() ||
      destinationType.getRank() != expandedType.getRank())
    return 0;

  ArrayRef<int64_t> permutation = transpose.getPermutation();
  if (permutation.size() != static_cast<size_t>(expandedType.getRank()))
    return 0;
  SmallVector<bool> seenPermutation(permutation.size(), false);
  for (auto [outputDim, inputDim] : llvm::enumerate(permutation)) {
    if (inputDim < 0 || inputDim >= expandedType.getRank() ||
        seenPermutation[inputDim] ||
        destinationType.getDimSize(outputDim) != consumedSizes[inputDim])
      return 0;
    seenPermutation[inputDim] = true;
  }

  SmallVector<memref::SubViewOp> assemblyViews;
  SmallVector<memref::DeallocOp> deallocations;
  for (OpOperand &use : allocation.getMemref().getUses()) {
    Operation *user = use.getOwner();
    if (user == expand)
      continue;
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      if (dealloc.getMemref() != allocation.getMemref())
        return 0;
      deallocations.push_back(dealloc);
      continue;
    }
    auto view = dyn_cast<memref::SubViewOp>(user);
    if (!view || view.getSource() != allocation.getMemref() ||
        !view->hasOneUse() || view->getBlock() != transpose->getBlock() ||
        !view->isBeforeInBlock(expand) ||
        llvm::is_contained(view.getStaticOffsets(), ShapedType::kDynamic) ||
        llvm::is_contained(view.getStaticSizes(), ShapedType::kDynamic) ||
        llvm::is_contained(view.getStaticStrides(), ShapedType::kDynamic) ||
        !llvm::all_of(view.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; }))
      return 0;
    assemblyViews.push_back(view);
  }
  if (assemblyViews.size() < 2)
    return 0;

  std::optional<unsigned> concatDimension;
  SmallVector<AssemblySegment> segments;
  for (memref::SubViewOp view : assemblyViews) {
    std::optional<unsigned> viewConcatDimension;
    for (unsigned dim = 0; dim < allocationType.getRank(); ++dim) {
      if (view.getStaticOffsets()[dim] == 0 &&
          view.getStaticSizes()[dim] == allocationType.getDimSize(dim))
        continue;
      if (viewConcatDimension)
        return 0;
      viewConcatDimension = dim;
    }
    if (!viewConcatDimension ||
        (concatDimension && *concatDimension != *viewConcatDimension))
      return 0;
    concatDimension = viewConcatDimension;
    auto copy = dyn_cast<memref::CopyOp>(*view->getUsers().begin());
    auto sourceType =
        copy ? dyn_cast<MemRefType>(copy.getSource().getType()) : MemRefType();
    if (!copy || copy.getTarget() != view.getResult() || !sourceType ||
        !sourceType.hasStaticShape() ||
        sourceType.getShape() != view.getStaticSizes() ||
        sourceType.getElementType() != allocationType.getElementType())
      return 0;
    segments.push_back({view, copy, view.getStaticOffsets()[*concatDimension],
                        view.getStaticSizes()[*concatDimension]});
  }
  if (!concatDimension)
    return 0;

  llvm::sort(segments,
             [](const AssemblySegment &left, const AssemblySegment &right) {
               return left.offset < right.offset;
             });
  int64_t covered = 0;
  for (const AssemblySegment &segment : segments) {
    if (segment.offset != covered || segment.size <= 0)
      return 0;
    covered += segment.size;
  }
  if (covered != allocationType.getDimSize(*concatDimension))
    return 0;

  // A partial range in a non-concat expanded group is representable directly
  // only when that group contains one dimension.  Multi-dimensional groups
  // are accepted when consumed in full.
  for (auto [sourceDim, group] : llvm::enumerate(reassociation)) {
    if (sourceDim == *concatDimension)
      continue;
    if (group.size() == 1)
      continue;
    for (int64_t expandedDim : group)
      if (consumedOffsets[expandedDim] != 0 ||
          consumedSizes[expandedDim] != expandedType.getDimSize(expandedDim))
        return 0;
  }
  for (int64_t expandedDim : reassociation[*concatDimension])
    if (consumedOffsets[expandedDim] != 0 ||
        consumedSizes[expandedDim] != expandedType.getDimSize(expandedDim))
      return 0;

  SmallVector<NamedAttribute> transposeAttributes;
  for (NamedAttribute attribute : transpose->getAttrs())
    if (attribute.getName().strref() != "permutation" &&
        !attribute.getName().strref().starts_with("sculptor.mapping."))
      transposeAttributes.push_back(attribute);

  OpBuilder builder(transpose);
  Location loc = transpose.getLoc();
  int64_t directTransposeCount = 0;
  ArrayRef<int64_t> concatGroup = reassociation[*concatDimension];
  int64_t innermostExtent = expandedType.getDimSize(concatGroup.back());
  for (AssemblySegment &segment : segments) {
    auto segmentSourceType =
        cast<MemRefType>(segment.copy.getSource().getType());
    int64_t globalOffset = segment.offset;
    int64_t localOffset = 0;
    int64_t remaining = segment.size;
    while (remaining > 0) {
      SmallVector<int64_t> concatCoordinates(concatGroup.size(), 0);
      int64_t coordinateRemainder = globalOffset;
      for (int64_t index = static_cast<int64_t>(concatGroup.size()) - 1;
           index >= 0; --index) {
        int64_t extent = expandedType.getDimSize(concatGroup[index]);
        concatCoordinates[index] = coordinateRemainder % extent;
        coordinateRemainder /= extent;
      }
      if (coordinateRemainder != 0)
        return 0;
      int64_t run =
          std::min(remaining, innermostExtent - concatCoordinates.back());

      SmallVector<OpFoldResult> sourceOffsets;
      SmallVector<OpFoldResult> sourceSizes;
      SmallVector<OpFoldResult> sourceStrides;
      sourceOffsets.reserve(allocationType.getRank());
      sourceSizes.reserve(allocationType.getRank());
      sourceStrides.reserve(allocationType.getRank());
      for (unsigned sourceDim = 0; sourceDim < allocationType.getRank();
           ++sourceDim) {
        int64_t offset = 0;
        int64_t size = segmentSourceType.getDimSize(sourceDim);
        if (sourceDim == *concatDimension) {
          offset = localOffset;
          size = run;
        } else if (reassociation[sourceDim].size() == 1) {
          int64_t expandedDim = reassociation[sourceDim].front();
          offset = consumedOffsets[expandedDim];
          size = consumedSizes[expandedDim];
        }
        sourceOffsets.push_back(builder.getIndexAttr(offset));
        sourceSizes.push_back(builder.getIndexAttr(size));
        sourceStrides.push_back(builder.getIndexAttr(1));
      }
      Value source = builder.create<memref::SubViewOp>(
          loc, segment.copy.getSource(), sourceOffsets, sourceSizes,
          sourceStrides);

      SmallVector<int64_t> pieceExpandedShape(expandedType.getRank());
      for (auto [sourceDim, group] : llvm::enumerate(reassociation)) {
        if (sourceDim == *concatDimension) {
          for (int64_t expandedDim : group)
            pieceExpandedShape[expandedDim] = 1;
          pieceExpandedShape[group.back()] = run;
          continue;
        }
        for (int64_t expandedDim : group)
          pieceExpandedShape[expandedDim] = consumedSizes[expandedDim];
      }
      Value expandedSource = builder.create<memref::ExpandShapeOp>(
          loc, pieceExpandedShape, source, reassociation);

      SmallVector<OpFoldResult> targetOffsets(destinationType.getRank());
      SmallVector<OpFoldResult> targetSizes(destinationType.getRank());
      SmallVector<OpFoldResult> targetStrides(destinationType.getRank(),
                                              builder.getIndexAttr(1));
      SmallVector<int64_t> pieceInputOffsets(expandedType.getRank(), 0);
      for (auto [index, expandedDim] : llvm::enumerate(concatGroup))
        pieceInputOffsets[expandedDim] = concatCoordinates[index];
      for (auto [outputDim, inputDim] : llvm::enumerate(permutation)) {
        targetOffsets[outputDim] =
            builder.getIndexAttr(pieceInputOffsets[inputDim]);
        targetSizes[outputDim] =
            builder.getIndexAttr(pieceExpandedShape[inputDim]);
      }
      Value target = builder.create<memref::SubViewOp>(
          loc, destination, targetOffsets, targetSizes, targetStrides);
      auto direct = builder.create<linalg::TransposeOp>(
          loc, expandedSource, target, permutation, transposeAttributes);
      direct->setAttr("sculptor.memory.direct_assembled_transpose",
                      builder.getUnitAttr());
      ++directTransposeCount;

      globalOffset += run;
      localOffset += run;
      remaining -= run;
    }
  }

  transpose.erase();
  for (memref::SubViewOp view : consumedViews)
    view.erase();
  expand.erase();
  for (AssemblySegment &segment : segments) {
    segment.copy.erase();
    segment.target.erase();
  }
  for (memref::DeallocOp dealloc : deallocations)
    dealloc.erase();
  allocation.erase();
  cast<FunctionOpInterface>(destination.getOwner()->getParentOp())
      ->setAttr("sculptor.memory.assembled_transpose_direct",
                builder.getUnitAttr());
  return directTransposeCount;
}

struct PartitionedResultCopy {
  BlockArgument destination;
  memref::CopyOp terminalCopy;
  memref::SubViewOp resultView;
  int64_t columnOffset = 0;
  int64_t columnCount = 0;
};

// A sharded routine result bufferizes as one full temporary followed by one
// full-height copy per result shard.  When the routine builds that temporary
// one row at a time, scatter each completed row directly to the runtime-owned
// shard destinations.  This preserves the routed shard layout while removing
// the full sequence allocation and the terminal second pass over every byte.
int64_t foldPartitionedSequenceResultCopies(func::FuncOp function,
                                            unsigned firstDestination,
                                            int64_t outputCount) {
  if (outputCount < 2)
    return 0;

  SmallVector<PartitionedResultCopy> partitions;
  Value fullResult;
  for (int64_t port = 0; port < outputCount; ++port) {
    BlockArgument destination = function.getArgument(firstDestination + port);
    auto destinationType = dyn_cast<MemRefType>(destination.getType());
    if (!destinationType || destinationType.getRank() != 2 ||
        !destinationType.hasStaticShape())
      return 0;

    SmallVector<memref::CopyOp> terminalCopies;
    function.walk([&](memref::CopyOp copy) {
      if (copy.getTarget() == destination)
        terminalCopies.push_back(copy);
    });
    if (terminalCopies.size() != 1)
      return 0;
    memref::CopyOp terminalCopy = terminalCopies.front();
    auto resultView =
        terminalCopy.getSource().getDefiningOp<memref::SubViewOp>();
    if (!resultView || !resultView->hasOneUse())
      return 0;
    if (fullResult && resultView.getSource() != fullResult)
      return 0;
    fullResult = resultView.getSource();

    ArrayRef<int64_t> offsets = resultView.getStaticOffsets();
    ArrayRef<int64_t> sizes = resultView.getStaticSizes();
    ArrayRef<int64_t> strides = resultView.getStaticStrides();
    if (offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2 ||
        llvm::is_contained(offsets, ShapedType::kDynamic) ||
        llvm::is_contained(sizes, ShapedType::kDynamic) || offsets[0] != 0 ||
        strides[0] != 1 || strides[1] != 1 ||
        sizes[0] != destinationType.getDimSize(0) ||
        sizes[1] != destinationType.getDimSize(1))
      return 0;
    partitions.push_back(
        {destination, terminalCopy, resultView, offsets[1], sizes[1]});
  }

  if (!fullResult)
    return 0;
  auto fullType = dyn_cast<MemRefType>(fullResult.getType());
  if (!fullType || fullType.getRank() != 2 || !fullType.hasStaticShape())
    return 0;

  auto rowLoop = fullResult.getDefiningOp<scf::ForOp>();
  if (!rowLoop || rowLoop.getNumResults() != 1 ||
      rowLoop.getInitArgs().size() != 1)
    return 0;
  auto fullAllocation =
      rowLoop.getInitArgs().front().getDefiningOp<memref::AllocOp>();
  if (!fullAllocation || !fullAllocation.getDynamicSizes().empty() ||
      fullAllocation.getType() != fullType)
    return 0;
  BlockArgument rowStorage = rowLoop.getRegionIterArgs().front();
  llvm::sort(partitions, [](const PartitionedResultCopy &left,
                            const PartitionedResultCopy &right) {
    return left.columnOffset < right.columnOffset;
  });
  int64_t coveredColumns = 0;
  for (const PartitionedResultCopy &partition : partitions) {
    if (partition.columnOffset != coveredColumns)
      return 0;
    coveredColumns += partition.columnCount;
  }
  if (coveredColumns != fullType.getDimSize(1))
    return 0;

  memref::SubViewOp rowView;
  memref::CopyOp rowCopy;
  for (OpOperand &use : rowStorage.getUses()) {
    Operation *user = use.getOwner();
    if (isa<scf::YieldOp>(user))
      continue;
    auto candidate = dyn_cast<memref::SubViewOp>(user);
    if (!candidate || rowView)
      return 0;
    ArrayRef<int64_t> offsets = candidate.getStaticOffsets();
    ArrayRef<int64_t> sizes = candidate.getStaticSizes();
    ArrayRef<int64_t> strides = candidate.getStaticStrides();
    if (offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2 ||
        offsets[0] != ShapedType::kDynamic || offsets[1] != 0 ||
        sizes[0] != 1 || sizes[1] != fullType.getDimSize(1) ||
        strides[0] != 1 || strides[1] != 1 || !candidate->hasOneUse())
      return 0;
    auto copy = dyn_cast<memref::CopyOp>(*candidate->getUsers().begin());
    if (!copy || copy.getTarget() != candidate.getResult())
      return 0;
    auto rowType = dyn_cast<MemRefType>(copy.getSource().getType());
    if (!rowType || !rowType.hasStaticShape() || rowType.getRank() != 2 ||
        rowType.getDimSize(0) != 1 ||
        rowType.getDimSize(1) != fullType.getDimSize(1))
      return 0;
    rowView = candidate;
    rowCopy = copy;
  }
  if (!rowView || !rowCopy)
    return 0;

  // Complete every structural check before mutating the routine.  A failed
  // match must leave the original bufferized program intact.
  auto oldYield = dyn_cast<scf::YieldOp>(rowLoop.getBody()->getTerminator());
  if (!oldYield || oldYield.getNumOperands() != 1 ||
      oldYield.getOperand(0) != rowStorage)
    return 0;
  for (OpOperand &use : fullResult.getUses()) {
    auto view = dyn_cast<memref::SubViewOp>(use.getOwner());
    if (!view ||
        !llvm::any_of(partitions, [&](const PartitionedResultCopy &partition) {
          return partition.resultView == view;
        }))
      return 0;
  }
  SmallVector<memref::DeallocOp> deallocations;
  for (OpOperand &use : fullAllocation.getMemref().getUses()) {
    if (use.getOwner() == rowLoop)
      continue;
    if (auto dealloc = dyn_cast<memref::DeallocOp>(use.getOwner())) {
      deallocations.push_back(dealloc);
      continue;
    }
    return 0;
  }

  SmallVector<OpFoldResult> rowOffsets = rowView.getMixedOffsets();
  if (rowOffsets.size() != 2)
    return 0;
  OpBuilder builder(rowCopy);
  Location loc = rowCopy.getLoc();
  for (const PartitionedResultCopy &partition : partitions) {
    SmallVector<OpFoldResult> sourceOffsets{
        builder.getIndexAttr(0), builder.getIndexAttr(partition.columnOffset)};
    SmallVector<OpFoldResult> targetOffsets{rowOffsets[0],
                                            builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{
        builder.getIndexAttr(1), builder.getIndexAttr(partition.columnCount)};
    SmallVector<OpFoldResult> strides{builder.getIndexAttr(1),
                                      builder.getIndexAttr(1)};
    Value source = builder.create<memref::SubViewOp>(
        loc, rowCopy.getSource(), sourceOffsets, sizes, strides);
    Value target = builder.create<memref::SubViewOp>(
        loc, partition.destination, targetOffsets, sizes, strides);
    auto direct = builder.create<memref::CopyOp>(loc, source, target);
    direct->setAttr("sculptor.memory.direct_partitioned_result",
                    builder.getUnitAttr());
  }

  rowCopy.erase();
  if (rowView->use_empty())
    rowView.erase();
  for (PartitionedResultCopy &partition : partitions) {
    partition.terminalCopy.erase();
    if (partition.resultView->use_empty())
      partition.resultView.erase();
  }

  OpBuilder loopBuilder(rowLoop);
  auto replacementLoop = loopBuilder.create<scf::ForOp>(
      rowLoop.getLoc(), rowLoop.getLowerBound(), rowLoop.getUpperBound(),
      rowLoop.getStep());
  for (NamedAttribute attribute : rowLoop->getAttrs())
    if (attribute.getName().strref() != "operandSegmentSizes")
      replacementLoop->setAttr(attribute.getName(), attribute.getValue());
  Block *replacementBody = replacementLoop.getBody();
  rowLoop.getInductionVar().replaceAllUsesWith(
      replacementLoop.getInductionVar());
  replacementBody->getTerminator()->erase();
  oldYield.erase();
  replacementBody->getOperations().splice(replacementBody->end(),
                                          rowLoop.getBody()->getOperations());
  OpBuilder::atBlockEnd(replacementBody).create<scf::YieldOp>(rowLoop.getLoc());
  rowLoop.erase();

  for (memref::DeallocOp dealloc : deallocations)
    dealloc.erase();
  fullAllocation.erase();
  function->setAttr("sculptor.memory.partitioned_result_direct",
                    builder.getUnitAttr());
  return partitions.size();
}

LogicalResult bindRoutineDestinations(
    func::FuncOp function, ArrayRef<TileMemoryBindingAttr> allBindings,
    const std::map<int64_t, TileMemoryOwnerAttr> &owners,
    const std::map<int64_t, TileMemoryViewAttr> &views,
    ArrayRef<TileMemoryAssemblyAttr> assemblies,
    ArrayRef<TileMemoryMovementAttr> movements, int64_t &eliminatedCopies) {
  auto routineId = function->getAttrOfType<IntegerAttr>(kGlobalRoutineIdAttr);
  if (!routineId)
    return success();
  if (auto kind = function->getAttrOfType<StringAttr>(kRoutineKindAttr);
      kind && kind.getValue() == "boot")
    return success();
  FailureOr<int64_t> outputCount = getOutputCount(function);
  if (failed(outputCount))
    return failure();
  if (*outputCount == 0)
    return success();
  if (function.getNumResults() != 0) {
    return function.emitError(
        "destination binding requires buffer-results-to-out-params first");
  }
  if (*outputCount > static_cast<int64_t>(function.getNumArguments()))
    return function.emitError("routine has fewer arguments than outputs");

  SmallVector<TileMemoryBindingAttr> outputBindings;
  outputBindings.reserve(*outputCount);
  for (int64_t port = 0; port < *outputCount; ++port) {
    TileMemoryBindingAttr match;
    for (TileMemoryBindingAttr binding : allBindings) {
      if (binding.getRoutine().getInt() != routineId.getInt() ||
          binding.getInput().getValue() || binding.getPort().getInt() != port ||
          (binding.getEffect() != MemoryAccessEffect::Write &&
           binding.getEffect() != MemoryAccessEffect::ReadWrite))
        continue;
      if (match)
        return function.emitError("duplicate planned output binding for port ")
               << port;
      match = binding;
    }
    if (!match)
      return function.emitError("missing planned output binding for port ")
             << port;
    outputBindings.push_back(match);
  }

  unsigned firstDestination = function.getNumArguments() - *outputCount;
  Builder builder(function.getContext());
  SmallVector<TileMemoryBindingAttr> inputBindings;
  for (TileMemoryBindingAttr binding : allBindings) {
    if (binding.getRoutine().getInt() != routineId.getInt() ||
        !binding.getInput().getValue() ||
        (binding.getEffect() != MemoryAccessEffect::Read &&
         binding.getEffect() != MemoryAccessEffect::ReadWrite))
      continue;
    auto owner = owners.find(binding.getOwnerId().getInt());
    if (owner == owners.end())
      return function.emitError("input binding names an unknown owner");
    if (owner->second.getKind() == MemoryOwnerKind::Persistent ||
        owner->second.getByteSize().getInt() == 0)
      continue;
    inputBindings.push_back(binding);
  }
  llvm::sort(inputBindings,
             [](TileMemoryBindingAttr left, TileMemoryBindingAttr right) {
               return left.getPort().getInt() < right.getPort().getInt();
             });
  if (inputBindings.size() != firstDestination)
    return function.emitError(
        "runtime tensor input count disagrees with planned bindings");
  for (auto [argumentIndex, binding] : llvm::enumerate(inputBindings)) {
    function.setArgAttr(argumentIndex, tile_memory::kOwnerIdAttrName,
                        binding.getOwnerId());
    function.setArgAttr(argumentIndex, tile_memory::kViewIdAttrName,
                        binding.getViewId());
  }

  for (auto [port, binding] : llvm::enumerate(outputBindings)) {
    unsigned argumentIndex = firstDestination + port;
    BlockArgument destination = function.getArgument(argumentIndex);
    if (!isa<BaseMemRefType>(destination.getType())) {
      return function.emitError("routine destination is not a memref at port ")
             << port;
    }
    function.setArgAttr(argumentIndex, tile_memory::kOwnerIdAttrName,
                        binding.getOwnerId());
    function.setArgAttr(argumentIndex, tile_memory::kViewIdAttrName,
                        binding.getViewId());
    function.setArgAttr(argumentIndex, kDestinationBoundAttr,
                        builder.getUnitAttr());

    SmallVector<memref::CopyOp> copies;
    function.walk([&](memref::CopyOp copy) {
      if (copy.getTarget() == destination)
        copies.push_back(copy);
    });
    for (memref::CopyOp copy : copies) {
      if (foldAllocationCopy(copy, destination) ||
          foldInputResultCopy(copy, destination))
        ++eliminatedCopies;
    }

    SmallVector<memref::CopyOp> nestedAssemblyCopies;
    function.walk([&](memref::CopyOp copy) {
      auto target = copy.getTarget().getDefiningOp<memref::SubViewOp>();
      if (target && target.getSource() == destination)
        nestedAssemblyCopies.push_back(copy);
    });
    for (memref::CopyOp copy : nestedAssemblyCopies)
      if (foldNestedAssemblyCopy(copy, destination))
        ++eliminatedCopies;

    SmallVector<linalg::TransposeOp> assembledTransposes;
    function.walk([&](linalg::TransposeOp transpose) {
      if (transpose.getInit() == destination)
        assembledTransposes.push_back(transpose);
    });
    for (linalg::TransposeOp transpose : assembledTransposes)
      eliminatedCopies += foldAssembledTranspose(transpose, destination);

    bool deallocated = false;
    function.walk([&](memref::DeallocOp dealloc) {
      deallocated |= dealloc.getMemref() == destination;
    });
    if (deallocated)
      return function.emitError("routine deallocates runtime-owned output ")
             << port;
  }
  eliminatedCopies += foldPartitionedSequenceResultCopies(
      function, firstDestination, *outputCount);
  eliminatedCopies +=
      foldRouteAssemblyCopies(function, owners, views, assemblies, movements);
  for (auto [port, binding] : llvm::enumerate(outputBindings)) {
    BlockArgument destination = function.getArgument(firstDestination + port);
    if (!destination.use_empty())
      continue;
    bool runtimeAssemblyDestination =
        llvm::any_of(assemblies, [&](TileMemoryAssemblyAttr assembly) {
          return assembly.getOwnerId().getInt() ==
                 binding.getOwnerId().getInt();
        });
    if (!runtimeAssemblyDestination)
      return function.emitError("routine does not write planned output ")
             << port;
  }
  SmallVector<Attribute> bindingAttrs;
  bindingAttrs.reserve(outputBindings.size());
  llvm::append_range(bindingAttrs, outputBindings);
  function->setAttr(kDestinationBindingsAttr,
                    builder.getArrayAttr(bindingAttrs));
  function->setAttr(kDestinationBoundAttr, builder.getUnitAttr());
  return success();
}

LogicalResult bindDestinations(ModuleOp module) {
  if (failed(tile_memory::verifyTileMemoryPlan(module)))
    return failure();
  auto bindings = getTypedMemoryArray<TileMemoryBindingAttr>(
      module, tile_memory::kBindingsAttrName);
  auto ownerAttrs = getTypedMemoryArray<TileMemoryOwnerAttr>(
      module, tile_memory::kOwnersAttrName);
  auto viewAttrs = getTypedMemoryArray<TileMemoryViewAttr>(
      module, tile_memory::kViewsAttrName);
  auto assemblies = getTypedMemoryArray<TileMemoryAssemblyAttr>(
      module, tile_memory::kAssembliesAttrName);
  auto movements = getTypedMemoryArray<TileMemoryMovementAttr>(
      module, tile_memory::kMovementsAttrName);
  if (failed(bindings) || failed(ownerAttrs) || failed(viewAttrs) ||
      failed(assemblies) || failed(movements))
    return failure();
  std::map<int64_t, TileMemoryOwnerAttr> owners;
  std::map<int64_t, TileMemoryViewAttr> views;
  for (TileMemoryOwnerAttr owner : *ownerAttrs)
    owners.emplace(owner.getId().getInt(), owner);
  for (TileMemoryViewAttr view : *viewAttrs)
    views.emplace(view.getId().getInt(), view);
  int64_t eliminatedCopies = 0;
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (failed(bindRoutineDestinations(function, *bindings, owners, views,
                                       *assemblies, *movements,
                                       eliminatedCopies)))
      return failure();
  Builder builder(module.getContext());
  module->setAttr("sculptor.memory.eliminated_result_copies",
                  builder.getI64IntegerAttr(eliminatedCopies));
  if (failed(verify(module)))
    return module.emitError(
        "destination-bound tile module failed verification");
  return success();
}

} // namespace

namespace mlir::sculptor {

void BindTileRoutineDestinationsPass::runOnOperation() {
  if (failed(bindDestinations(getOperation())))
    signalPassFailure();
}

void registerBindTileRoutineDestinationsPass() {
  PassRegistration<BindTileRoutineDestinationsPass>();
}

} // namespace mlir::sculptor
