#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_RESOURCEALLOCATIONTREE_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_RESOURCEALLOCATIONTREE_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorAttrs.h"
#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/ComputeGraph.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {
namespace mapping {

inline constexpr StringLiteral kRATreeAttrName = "sculptor.mapping.ra_tree";
inline constexpr StringLiteral kMappingWorkUnitIdAttrName =
    "sculptor.mapping.work_unit_id";

struct MappingWorkUnit {
  int64_t id = -1;
  int64_t operationId = -1;
  int64_t resultNumber = -1;
  SmallVector<int64_t> resultOffsets;
  SmallVector<int64_t> resultSizes;
  SmallVector<int64_t> iterationOffsets;
  SmallVector<int64_t> iterationSizes;
  int64_t shardGroupId = -1;
  int64_t shardIndex = -1;
  int64_t shardCount = -1;
};

// An exact dependency refinement for a tiled endpoint. A work-unit ID of -1
// denotes an operation-wide endpoint.
struct MappingWorkUnitEdge {
  int64_t sourceOperationId = -1;
  int64_t sourceWorkUnitId = -1;
  int64_t targetOperationId = -1;
  int64_t targetWorkUnitId = -1;
  int64_t tensorId = -1;
  int64_t sourceResultNumber = -1;
  int64_t targetOperandNumber = -1;
  int64_t byteSize = -1;
};

struct StructuralRATreeNode {
  int64_t id = -1;
  RATreeNodeKind kind = RATreeNodeKind::Leaf;
  int64_t parentId = -1;
  SmallVector<int64_t> childIds;
  int64_t operationId = -1;
  int64_t workUnitId = -1;
  int64_t workGroupCount = 1;
};

struct ResourceAllocationTree {
  int64_t rootId = -1;
  SmallVector<StructuralRATreeNode, 0> nodes;
  SmallVector<MappingWorkUnit, 0> workUnits;
  SmallVector<MappingWorkUnitEdge, 0> workUnitEdges;
};

FailureOr<ResourceAllocationTree>
buildTemporalBaselineRATree(const ComputeGraph &graph, Operation *anchor);

FailureOr<ResourceAllocationTree>
deserializeResourceAllocationTree(RATreeAttr attr, const ComputeGraph &graph,
                                  Operation *anchor);

ResourceAllocationTree
cloneResourceAllocationTree(const ResourceAllocationTree &tree);

FailureOr<ResourceAllocationTree>
reindexResourceAllocationTree(const ResourceAllocationTree &tree,
                              Operation *anchor);

LogicalResult verifyResourceAllocationTree(const ResourceAllocationTree &tree,
                                           const ComputeGraph &graph,
                                           Operation *anchor);

std::string computeGraphFingerprint(const ComputeGraph &graph);
std::string computeRATreeFingerprint(const ResourceAllocationTree &tree);

RATreeAttr serializeResourceAllocationTree(MLIRContext *context,
                                           const ResourceAllocationTree &tree,
                                           const ComputeGraph &graph,
                                           StringRef graphFingerprint);

} // namespace mapping
} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_MAPPING_RESOURCEALLOCATIONTREE_H
