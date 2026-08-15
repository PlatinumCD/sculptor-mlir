#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILEMEMORYPLAN_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILEMEMORYPLAN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::sculptor::tile_memory {

inline constexpr StringLiteral kPlanVersionAttrName =
    "sculptor.memory.plan_version";
inline constexpr StringLiteral kOwnersAttrName = "sculptor.memory.owners";
inline constexpr StringLiteral kViewsAttrName = "sculptor.memory.views";
inline constexpr StringLiteral kBindingsAttrName = "sculptor.memory.bindings";
inline constexpr StringLiteral kMovementsAttrName = "sculptor.memory.movements";
inline constexpr StringLiteral kSegmentsAttrName = "sculptor.memory.segments";
inline constexpr StringLiteral kAssembliesAttrName =
    "sculptor.memory.assemblies";
inline constexpr StringLiteral kCompletionEventsAttrName =
    "sculptor.memory.completion_events";
inline constexpr StringLiteral kEventEdgesAttrName =
    "sculptor.memory.event_edges";
inline constexpr StringLiteral kLifetimesAttrName = "sculptor.memory.lifetimes";
inline constexpr StringLiteral kInterferencesAttrName =
    "sculptor.memory.interferences";
inline constexpr StringLiteral kInterferenceDefaultAttrName =
    "sculptor.memory.interference_default";
inline constexpr StringLiteral kInterferenceExceptionsAttrName =
    "sculptor.memory.interference_exceptions";
inline constexpr StringLiteral kInPlaceAliasesAttrName =
    "sculptor.memory.in_place_aliases";
inline constexpr StringLiteral kCapacityAttrName = "sculptor.memory.capacity";
inline constexpr StringLiteral kConfiguredCapacityAttrName =
    "sculptor.deployment.tile_memory_capacity_bytes";
inline constexpr StringLiteral kOwnerIdAttrName = "sculptor.memory.owner_id";
inline constexpr StringLiteral kViewIdAttrName = "sculptor.memory.view_id";
inline constexpr StringLiteral kZeroCopyViewAttrName =
    "sculptor.memory.zero_copy_view";

struct WorkspaceAllocationRequest {
  int64_t lifetimeId = -1;
  int64_t byteSize = 0;
  int64_t alignment = 1;
};

struct WorkspaceAllocation {
  int64_t lifetimeId = -1;
  int64_t offset = -1;
};

struct ExactWorkspaceLayout {
  llvm::SmallVector<WorkspaceAllocation> allocations;
  int64_t workspaceBytes = 0;
};

/// Builds the deterministic deployment-wide memory contract and attaches a
/// filtered contract to every nested physical-tile module.
LogicalResult buildAndAttachTileMemoryPlan(ModuleOp deployment);

/// Verifies either a deployment-wide plan or one extracted tile-local plan.
LogicalResult verifyTileMemoryPlan(ModuleOp module);

/// Recomputes pairwise lifetime relations from the attached completion DAG.
LogicalResult rebuildTileMemoryInterference(ModuleOp module);

/// Recomputes tile-local capacity categories and the conservative peak.
LogicalResult rebuildTileMemoryCapacitySummary(ModuleOp module);

/// Enforces the configured physical-tile capacity against the latest exact
/// capacity summary. A zero or absent configured capacity disables the check.
LogicalResult validateTileMemoryCapacity(ModuleOp module);

/// Packs workspace lifetimes using exact happens-before queries against the
/// completion DAG. Unlike serialized pairwise interference metadata, this
/// remains exact for large tile programs without constructing O(N^2) records.
FailureOr<ExactWorkspaceLayout>
buildExactWorkspaceLayout(ModuleOp module,
                          llvm::ArrayRef<WorkspaceAllocationRequest> requests);

} // namespace mlir::sculptor::tile_memory

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_TILEMEMORYPLAN_H
