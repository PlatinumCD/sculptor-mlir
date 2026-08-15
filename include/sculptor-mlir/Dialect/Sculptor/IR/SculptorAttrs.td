#ifndef SCULPTOR_ATTRS
#define SCULPTOR_ATTRS

include "mlir/IR/EnumAttr.td"

include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.td"

class Sculptor_Attr<string name, string attrMnemonic>
    : AttrDef<Sculptor_Dialect, name> {
  let mnemonic = attrMnemonic;
}

def Sculptor_TaskReductionKindAdd
    : I32EnumAttrCase<"Add", 0, "add">;
def Sculptor_TaskReductionKindMax
    : I32EnumAttrCase<"Max", 1, "max">;
def Sculptor_TaskReductionKindMin
    : I32EnumAttrCase<"Min", 2, "min">;

def Sculptor_TaskReductionKind
    : I32EnumAttr<"TaskReductionKind", "Task reduction kind", [
        Sculptor_TaskReductionKindAdd,
        Sculptor_TaskReductionKindMax,
        Sculptor_TaskReductionKindMin
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_TaskReductionAttr
    : Sculptor_Attr<"TaskReduction", "task_reduction"> {
  let summary = "associative task-graph reduction metadata";
  let description = [{
    Marks a pure, elementwise task reduction whose input order may be
    reassociated. The task-graph reduction balancing pass uses this metadata
    instead of guessing reduction semantics from task names or callee bodies.
  }];
  let parameters = (ins
    EnumParameter<Sculptor_TaskReductionKind>:$kind,
    "BoolAttr":$reassociate
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_RATreeNodeKindSpatialCut
    : I32EnumAttrCase<"SpatialCut", 0, "spatial_cut">;
def Sculptor_RATreeNodeKindTemporalCut
    : I32EnumAttrCase<"TemporalCut", 1, "temporal_cut">;
def Sculptor_RATreeNodeKindLeaf
    : I32EnumAttrCase<"Leaf", 2, "leaf">;
def Sculptor_RATreeNodeKindLayer
    : I32EnumAttrCase<"Layer", 3, "layer">;

def Sculptor_RATreeNodeKind
    : I32EnumAttr<"RATreeNodeKind", "Resource Allocation Tree node kind", [
        Sculptor_RATreeNodeKindSpatialCut,
        Sculptor_RATreeNodeKindTemporalCut,
        Sculptor_RATreeNodeKindLeaf,
        Sculptor_RATreeNodeKindLayer
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_MappingWorkUnitAttr
    : Sculptor_Attr<"MappingWorkUnit", "mapping_work_unit"> {
  let summary = "one statically bounded tile of a compute operation";
  let description = [{
    Identifies one result tile selected by a mapping planner and the
    corresponding iteration-domain tile. All offsets and sizes are static.
    The work unit remains declarative until a mapping-application pass invokes
    the operation's TilingInterface implementation.
  }];
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$operationId,
    "IntegerAttr":$resultNumber,
    "ArrayAttr":$resultOffsets,
    "ArrayAttr":$resultSizes,
    "ArrayAttr":$iterationOffsets,
    "ArrayAttr":$iterationSizes,
    "IntegerAttr":$shardGroupId,
    "IntegerAttr":$shardIndex,
    "IntegerAttr":$shardCount
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_MappingWorkUnitEdgeAttr
    : Sculptor_Attr<"MappingWorkUnitEdge", "mapping_work_unit_edge"> {
  let summary = "an exact dependency between mapping work units";
  let description = [{
    Refines one operation-level compute-graph dependency to identify the
    producer and consumer work units that actually communicate. A work-unit
    ID of -1 denotes an operation-wide endpoint. The operation-level DAG
    remains the fallback for dependency pairs with no explicit refinement.
  }];
  let parameters = (ins
    "IntegerAttr":$sourceOperationId,
    "IntegerAttr":$sourceWorkUnitId,
    "IntegerAttr":$targetOperationId,
    "IntegerAttr":$targetWorkUnitId,
    "IntegerAttr":$tensorId,
    "IntegerAttr":$sourceResultNumber,
    "IntegerAttr":$targetOperandNumber,
    "IntegerAttr":$byteSize
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_RATreeNodeAttr
    : Sculptor_Attr<"RATreeNode", "ra_tree_node"> {
  let summary = "one structural Resource Allocation Tree node";
  let description = [{
    Describes one node in an unevaluated Resource Allocation Tree. Parent,
    child, and operation references use deterministic integer IDs. A value of
    -1 denotes an inapplicable parent or operation. Only leaves reference
    operations.
  }];
  let parameters = (ins
    "IntegerAttr":$id,
    EnumParameter<Sculptor_RATreeNodeKind>:$kind,
    "IntegerAttr":$parentId,
    "ArrayAttr":$childIds,
    "IntegerAttr":$operationId,
    "IntegerAttr":$workUnitId,
    "IntegerAttr":$workGroupCount
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_RATreeAttr
    : Sculptor_Attr<"RATree", "ra_tree"> {
  let summary = "structural Resource Allocation Tree for one compute graph";
  let description = [{
    Stores a deterministic structural RA Tree over supported compute
    operations. These include generic structured MLIR operations and explicit
    backend-neutral Sculptor primitives such as sculptor.mvm. The computation
    DAG is deliberately not copied into this attribute; later passes rebuild
    it from SSA and validate the graph fingerprint.
  }];
  let parameters = (ins
    "IntegerAttr":$version,
    "IntegerAttr":$rootId,
    "ArrayAttr":$nodes,
    "ArrayAttr":$workUnits,
    "ArrayAttr":$workUnitEdges,
    "IntegerAttr":$operationCount,
    "IntegerAttr":$tensorCount,
    "StringAttr":$graphFingerprint
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_MappingPlanObjectiveLatency
    : I32EnumAttrCase<"Latency", 0, "latency">;

def Sculptor_MappingPlanObjective
    : I32EnumAttr<"MappingPlanObjective", "Mapping plan objective", [
        Sculptor_MappingPlanObjectiveLatency
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_RATreeNodeEvaluationAttr
    : Sculptor_Attr<"RATreeNodeEvaluation", "ra_tree_node_evaluation"> {
  let summary = "reference evaluation for one Resource Allocation Tree node";
  let parameters = (ins
    "IntegerAttr":$nodeId,
    "BoolAttr":$feasible,
    "FloatAttr":$estimatedLatencyNs,
    "IntegerAttr":$crossingBytes,
    "FloatAttr":$estimatedCommunicationNs,
    "IntegerAttr":$requiredResourceUnits,
    "IntegerAttr":$pipelineStages,
    "StringAttr":$infeasibilityReason
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_MappingCandidateEvaluationAttr
    : Sculptor_Attr<"MappingCandidateEvaluation",
                    "mapping_candidate_evaluation"> {
  let summary = "root evaluation summary for one mapping candidate";
  let parameters = (ins
    "StringAttr":$name,
    "BoolAttr":$selected,
    "BoolAttr":$feasible,
    "FloatAttr":$estimatedLatencyNs,
    "IntegerAttr":$crossingBytes,
    "FloatAttr":$estimatedCommunicationNs,
    "IntegerAttr":$requiredResourceUnits,
    "IntegerAttr":$pipelineStages,
    "StringAttr":$infeasibilityReason
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_MappingLaneKindDigital
    : I32EnumAttrCase<"Digital", 0, "digital">;
def Sculptor_MappingLaneKindAnalog
    : I32EnumAttrCase<"Analog", 1, "analog">;

def Sculptor_MappingLaneKind
    : I32EnumAttr<"MappingLaneKind", "Realized logical lane kind", [
        Sculptor_MappingLaneKindDigital,
        Sculptor_MappingLaneKindAnalog
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_MappingNodeAllocationAttr
    : Sculptor_Attr<"MappingNodeAllocation", "mapping_node_allocation"> {
  let summary = "logical resources inherited by one RA-tree node";
  let parameters = (ins
    "IntegerAttr":$nodeId,
    "ArrayAttr":$digitalTileIds,
    "ArrayAttr":$analogTileIds,
    "ArrayAttr":$analogLaneIndices
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_MappingLeafAssignmentAttr
    : Sculptor_Attr<"MappingLeafAssignment", "mapping_leaf_assignment"> {
  let summary = "realized logical tile and lane assignment for one RA leaf";
  let parameters = (ins
    "IntegerAttr":$leafId,
    "IntegerAttr":$operationId,
    "IntegerAttr":$tileId,
    EnumParameter<Sculptor_MappingLaneKind>:$laneKind,
    "IntegerAttr":$laneIndex,
    "FloatAttr":$startNs,
    "FloatAttr":$finishNs
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_MappingLayerTilePoolAttr
    : Sculptor_Attr<"MappingLayerTilePool", "mapping_layer_tile_pool"> {
  let summary = "allowed logical-tile neighborhood for one semantic layer";
  let parameters = (ins
    "IntegerAttr":$layerRegionId,
    "ArrayAttr":$tileIds
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_MappingRealizationAttr
    : Sculptor_Attr<"MappingRealization", "mapping_realization"> {
  let summary = "explicit logical-tile realization of an RA tree";
  let description = [{
    Records the logical tile and lane resources inherited by every RA-tree
    node and the concrete lane assigned to every leaf. It remains independent
    of physical mesh coordinates and runtime core identities.
  }];
  let parameters = (ins
    "IntegerAttr":$version,
    "IntegerAttr":$logicalTileCount,
    "IntegerAttr":$analogLanesPerTile,
    "ArrayAttr":$digitalWorkPerTile,
    "ArrayAttr":$layerTilePools,
    "ArrayAttr":$nodeAllocations,
    "ArrayAttr":$leafAssignments
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_LogicalTileAssignmentAttr
    : Sculptor_Attr<"LogicalTileAssignment", "logical_tile_assignment"> {
  let summary = "RA leaf assignment retained by one logical tile";
  let parameters = (ins
    "IntegerAttr":$leafId,
    "IntegerAttr":$operationId,
    "IntegerAttr":$workUnitId,
    EnumParameter<Sculptor_MappingLaneKind>:$laneKind,
    "IntegerAttr":$laneIndex,
    "ArrayAttr":$raNodePath,
    "FloatAttr":$startNs,
    "FloatAttr":$finishNs
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_LogicalTileAnalogLaneAttr
    : Sculptor_Attr<"LogicalTileAnalogLane", "logical_tile_analog_lane"> {
  let summary = "one analog lane in a logical tile";
  let parameters = (ins
    "IntegerAttr":$laneIndex,
    "IntegerAttr":$laneBindingGroup,
    "ArrayAttr":$assignments
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_LogicalTileDependencyAttr
    : Sculptor_Attr<"LogicalTileDependency", "logical_tile_dependency"> {
  let summary = "ST-graph dependency retained by logical-tile formation";
  let parameters = (ins
    "IntegerAttr":$sourceOperationId,
    "IntegerAttr":$sourceWorkUnitId,
    "IntegerAttr":$targetOperationId,
    "IntegerAttr":$targetWorkUnitId,
    "IntegerAttr":$tensorId,
    "IntegerAttr":$targetOperandNumber,
    "IntegerAttr":$byteSize
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_LogicalTileAttr
    : Sculptor_Attr<"LogicalTile", "logical_tile"> {
  let summary = "RA-planned logical tile before physical placement";
  let parameters = (ins
    "IntegerAttr":$tileId,
    "IntegerAttr":$digitalWork,
    "ArrayAttr":$digitalAssignments,
    "ArrayAttr":$analogLanes,
    "ArrayAttr":$modelInputTensorIds,
    "ArrayAttr":$modelOutputTensorIds,
    "ArrayAttr":$internalDependencies
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_LogicalTileEdgeAttr
    : Sculptor_Attr<"LogicalTileEdge", "logical_tile_edge"> {
  let summary = "weighted communication edge between logical tiles";
  let parameters = (ins
    "IntegerAttr":$edgeId,
    "IntegerAttr":$sourceTileId,
    "IntegerAttr":$targetTileId,
    "IntegerAttr":$byteSize,
    "ArrayAttr":$dependencies
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_LogicalTileGraphAttr
    : Sculptor_Attr<"LogicalTileGraph", "logical_tile_graph"> {
  let summary = "logical placement units and their weighted communication";
  let parameters = (ins
    "IntegerAttr":$version,
    "IntegerAttr":$plannedMeshRows,
    "IntegerAttr":$plannedMeshCols,
    "IntegerAttr":$logicalTileCapacity,
    "IntegerAttr":$analogLanesPerTile,
    "ArrayAttr":$tiles,
    "ArrayAttr":$edges
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_PhysicalTileAssignmentAttr
    : Sculptor_Attr<"PhysicalTileAssignment", "physical_tile_assignment"> {
  let summary = "physical mesh location assigned to one logical tile";
  let parameters = (ins
    "IntegerAttr":$logicalTileId,
    "IntegerAttr":$physicalTileId,
    "IntegerAttr":$row,
    "IntegerAttr":$column
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_PlacedLogicalTileEdgeAttr
    : Sculptor_Attr<"PlacedLogicalTileEdge", "placed_logical_tile_edge"> {
  let summary = "placement cost of one logical-tile communication edge";
  let parameters = (ins
    "IntegerAttr":$edgeId,
    "IntegerAttr":$sourceTileId,
    "IntegerAttr":$targetTileId,
    "IntegerAttr":$byteSize,
    "IntegerAttr":$manhattanHops,
    "IntegerAttr":$transferCost
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_LogicalTileMemoryEstimateAttr
    : Sculptor_Attr<"LogicalTileMemoryEstimate", "logical_tile_memory_estimate"> {
  let summary = "schedule-aware pre-outlining peak-memory estimate for one logical tile";
  let parameters = (ins
    "IntegerAttr":$logicalTileId,
    "IntegerAttr":$persistentBytes,
    "IntegerAttr":$producedBytes,
    "IntegerAttr":$incomingBytes,
    "IntegerAttr":$requiredBytes,
    "BoolAttr":$complete
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_LogicalTilePlacementAttr
    : Sculptor_Attr<"LogicalTilePlacement", "logical_tile_placement"> {
  let summary = "physical placement of an RA-planned logical-tile graph";
  let parameters = (ins
    "IntegerAttr":$version,
    "StringAttr":$schedule,
    "StringAttr":$objective,
    "StringAttr":$networkMode,
    "StringAttr":$timingScope,
    "StringAttr":$routePolicy,
    "StringAttr":$costProfileName,
    "StringAttr":$costProfileHash,
    "IntegerAttr":$meshRows,
    "IntegerAttr":$meshCols,
    "IntegerAttr":$arraysPerCore,
    "IntegerAttr":$tileMemoryCapacityBytes,
    "ArrayAttr":$memoryEstimates,
    "IntegerAttr":$initialScore,
    "IntegerAttr":$objectiveScore,
    "IntegerAttr":$totalTransferCost,
    "FloatAttr":$predictedMakespanNs,
    "IntegerAttr":$evaluations,
    "ArrayAttr":$assignments,
    "ArrayAttr":$edges
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_AnnealingScoreSampleAttr
    : Sculptor_Attr<"AnnealingScoreSample", "annealing_score_sample"> {
  let summary = "one evaluated simulated-annealing placement move";
  let parameters = (ins
    "IntegerAttr":$iteration,
    "IntegerAttr":$candidateScore,
    "IntegerAttr":$currentScore,
    "IntegerAttr":$bestScore,
    "BoolAttr":$accepted
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_MemoryOwnerKindModelInput
    : I32EnumAttrCase<"ModelInput", 0, "model_input">;
def Sculptor_MemoryOwnerKindModelOutput
    : I32EnumAttrCase<"ModelOutput", 1, "model_output">;
def Sculptor_MemoryOwnerKindIntermediate
    : I32EnumAttrCase<"Intermediate", 2, "intermediate">;
def Sculptor_MemoryOwnerKindPersistent
    : I32EnumAttrCase<"Persistent", 3, "persistent">;
def Sculptor_MemoryOwnerKindRouteInput
    : I32EnumAttrCase<"RouteInput", 4, "route_input">;
def Sculptor_MemoryOwnerKindRouteOutput
    : I32EnumAttrCase<"RouteOutput", 5, "route_output">;
def Sculptor_MemoryOwnerKindAssembly
    : I32EnumAttrCase<"Assembly", 6, "assembly">;
def Sculptor_MemoryOwnerKindLocalTemporary
    : I32EnumAttrCase<"LocalTemporary", 7, "local_temporary">;

def Sculptor_MemoryOwnerKind
    : I32EnumAttr<"MemoryOwnerKind", "tile memory owner kind", [
        Sculptor_MemoryOwnerKindModelInput,
        Sculptor_MemoryOwnerKindModelOutput,
        Sculptor_MemoryOwnerKindIntermediate,
        Sculptor_MemoryOwnerKindPersistent,
        Sculptor_MemoryOwnerKindRouteInput,
        Sculptor_MemoryOwnerKindRouteOutput,
        Sculptor_MemoryOwnerKindAssembly,
        Sculptor_MemoryOwnerKindLocalTemporary
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_MemoryAccessEffectRead
    : I32EnumAttrCase<"Read", 0, "read">;
def Sculptor_MemoryAccessEffectWrite
    : I32EnumAttrCase<"Write", 1, "write">;
def Sculptor_MemoryAccessEffectReadWrite
    : I32EnumAttrCase<"ReadWrite", 2, "read_write">;
def Sculptor_MemoryAccessEffectAsyncTransferSource
    : I32EnumAttrCase<"AsyncTransferSource", 3, "async_transfer_source">;
def Sculptor_MemoryAccessEffectAsyncTransferDestination
    : I32EnumAttrCase<"AsyncTransferDestination", 4,
                      "async_transfer_destination">;

def Sculptor_MemoryAccessEffect
    : I32EnumAttr<"MemoryAccessEffect", "tile memory access effect", [
        Sculptor_MemoryAccessEffectRead,
        Sculptor_MemoryAccessEffectWrite,
        Sculptor_MemoryAccessEffectReadWrite,
        Sculptor_MemoryAccessEffectAsyncTransferSource,
        Sculptor_MemoryAccessEffectAsyncTransferDestination
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_MemoryMovementModeLocalAlias
    : I32EnumAttrCase<"LocalAlias", 0, "local_alias">;
def Sculptor_MemoryMovementModeContiguous
    : I32EnumAttrCase<"Contiguous", 1, "contiguous">;
def Sculptor_MemoryMovementModePacked
    : I32EnumAttrCase<"Packed", 2, "packed">;
def Sculptor_MemoryMovementModeSegmented
    : I32EnumAttrCase<"Segmented", 3, "segmented">;
def Sculptor_MemoryMovementModeAssembly
    : I32EnumAttrCase<"Assembly", 4, "assembly">;

def Sculptor_MemoryMovementMode
    : I32EnumAttr<"MemoryMovementMode", "tile memory movement mode", [
        Sculptor_MemoryMovementModeLocalAlias,
        Sculptor_MemoryMovementModeContiguous,
        Sculptor_MemoryMovementModePacked,
        Sculptor_MemoryMovementModeSegmented,
        Sculptor_MemoryMovementModeAssembly
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_MemoryContiguityContiguous
    : I32EnumAttrCase<"Contiguous", 0, "contiguous">;
def Sculptor_MemoryContiguityStrided
    : I32EnumAttrCase<"Strided", 1, "strided">;
def Sculptor_MemoryContiguityNonContiguous
    : I32EnumAttrCase<"NonContiguous", 2, "non_contiguous">;

def Sculptor_MemoryContiguity
    : I32EnumAttr<"MemoryContiguity", "tile memory view contiguity", [
        Sculptor_MemoryContiguityContiguous,
        Sculptor_MemoryContiguityStrided,
        Sculptor_MemoryContiguityNonContiguous
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_MemoryCompletionKindRoutineStart
    : I32EnumAttrCase<"RoutineStart", 0, "routine_start">;
def Sculptor_MemoryCompletionKindRoutineComplete
    : I32EnumAttrCase<"RoutineComplete", 1, "routine_complete">;
def Sculptor_MemoryCompletionKindRouteArrival
    : I32EnumAttrCase<"RouteArrival", 2, "route_arrival">;
def Sculptor_MemoryCompletionKindRouteSendComplete
    : I32EnumAttrCase<"RouteSendComplete", 3, "route_send_complete">;
def Sculptor_MemoryCompletionKindDMAComplete
    : I32EnumAttrCase<"DMAComplete", 4, "dma_complete">;
def Sculptor_MemoryCompletionKindAssemblyContribution
    : I32EnumAttrCase<"AssemblyContribution", 5,
                      "assembly_contribution">;
def Sculptor_MemoryCompletionKindAssemblyReady
    : I32EnumAttrCase<"AssemblyReady", 6, "assembly_ready">;
def Sculptor_MemoryCompletionKindFinalFanOutSend
    : I32EnumAttrCase<"FinalFanOutSend", 7, "final_fan_out_send">;
def Sculptor_MemoryCompletionKindFinalConsumerComplete
    : I32EnumAttrCase<"FinalConsumerComplete", 8,
                      "final_consumer_complete">;
def Sculptor_MemoryCompletionKindOwnerRelease
    : I32EnumAttrCase<"OwnerRelease", 9, "owner_release">;
def Sculptor_MemoryCompletionKindScratchpadRelease
    : I32EnumAttrCase<"ScratchpadRelease", 10, "scratchpad_release">;

def Sculptor_MemoryCompletionKind
    : I32EnumAttr<"MemoryCompletionKind", "tile completion event kind", [
        Sculptor_MemoryCompletionKindRoutineStart,
        Sculptor_MemoryCompletionKindRoutineComplete,
        Sculptor_MemoryCompletionKindRouteArrival,
        Sculptor_MemoryCompletionKindRouteSendComplete,
        Sculptor_MemoryCompletionKindDMAComplete,
        Sculptor_MemoryCompletionKindAssemblyContribution,
        Sculptor_MemoryCompletionKindAssemblyReady,
        Sculptor_MemoryCompletionKindFinalFanOutSend,
        Sculptor_MemoryCompletionKindFinalConsumerComplete,
        Sculptor_MemoryCompletionKindOwnerRelease,
        Sculptor_MemoryCompletionKindScratchpadRelease
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_MemoryEventEdgeKindRoutineExecution
    : I32EnumAttrCase<"RoutineExecution", 0, "routine_execution">;
def Sculptor_MemoryEventEdgeKindLocalDependency
    : I32EnumAttrCase<"LocalDependency", 1, "local_dependency">;
def Sculptor_MemoryEventEdgeKindRouteSend
    : I32EnumAttrCase<"RouteSend", 2, "route_send">;
def Sculptor_MemoryEventEdgeKindNetworkTransfer
    : I32EnumAttrCase<"NetworkTransfer", 3, "network_transfer">;
def Sculptor_MemoryEventEdgeKindRouteReady
    : I32EnumAttrCase<"RouteReady", 4, "route_ready">;
def Sculptor_MemoryEventEdgeKindDMACompletion
    : I32EnumAttrCase<"DMACompletion", 5, "dma_completion">;
def Sculptor_MemoryEventEdgeKindAssemblyContribution
    : I32EnumAttrCase<"AssemblyContribution", 6,
                      "assembly_contribution">;
def Sculptor_MemoryEventEdgeKindAssemblyJoin
    : I32EnumAttrCase<"AssemblyJoin", 7, "assembly_join">;
def Sculptor_MemoryEventEdgeKindBootBarrier
    : I32EnumAttrCase<"BootBarrier", 8, "boot_barrier">;
def Sculptor_MemoryEventEdgeKindFanOutJoin
    : I32EnumAttrCase<"FanOutJoin", 9, "fan_out_join">;
def Sculptor_MemoryEventEdgeKindFinalConsumer
    : I32EnumAttrCase<"FinalConsumer", 10, "final_consumer">;
def Sculptor_MemoryEventEdgeKindLifetimeRelease
    : I32EnumAttrCase<"LifetimeRelease", 11, "lifetime_release">;
def Sculptor_MemoryEventEdgeKindScratchpadRelease
    : I32EnumAttrCase<"ScratchpadRelease", 12, "scratchpad_release">;

def Sculptor_MemoryEventEdgeKind
    : I32EnumAttr<"MemoryEventEdgeKind", "tile event dependency kind", [
        Sculptor_MemoryEventEdgeKindRoutineExecution,
        Sculptor_MemoryEventEdgeKindLocalDependency,
        Sculptor_MemoryEventEdgeKindRouteSend,
        Sculptor_MemoryEventEdgeKindNetworkTransfer,
        Sculptor_MemoryEventEdgeKindRouteReady,
        Sculptor_MemoryEventEdgeKindDMACompletion,
        Sculptor_MemoryEventEdgeKindAssemblyContribution,
        Sculptor_MemoryEventEdgeKindAssemblyJoin,
        Sculptor_MemoryEventEdgeKindBootBarrier,
        Sculptor_MemoryEventEdgeKindFanOutJoin,
        Sculptor_MemoryEventEdgeKindFinalConsumer,
        Sculptor_MemoryEventEdgeKindLifetimeRelease,
        Sculptor_MemoryEventEdgeKindScratchpadRelease
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_MemoryLifetimeSubjectKindOwner
    : I32EnumAttrCase<"Owner", 0, "owner">;
def Sculptor_MemoryLifetimeSubjectKindRoutineAllocation
    : I32EnumAttrCase<"RoutineAllocation", 1, "routine_allocation">;

def Sculptor_MemoryLifetimeSubjectKind
    : I32EnumAttr<"MemoryLifetimeSubjectKind", "memory lifetime subject kind", [
        Sculptor_MemoryLifetimeSubjectKindOwner,
        Sculptor_MemoryLifetimeSubjectKindRoutineAllocation
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_MemoryLifetimeStorageExternal
    : I32EnumAttrCase<"External", 0, "external">;
def Sculptor_MemoryLifetimeStoragePersistent
    : I32EnumAttrCase<"Persistent", 1, "persistent">;
def Sculptor_MemoryLifetimeStorageWorkspace
    : I32EnumAttrCase<"Workspace", 2, "workspace">;
def Sculptor_MemoryLifetimeStorageRoutineLocal
    : I32EnumAttrCase<"RoutineLocal", 3, "routine_local">;
def Sculptor_MemoryLifetimeStorageScratchpad
    : I32EnumAttrCase<"Scratchpad", 4, "scratchpad">;

def Sculptor_MemoryLifetimeStorage
    : I32EnumAttr<"MemoryLifetimeStorage", "memory lifetime storage class", [
        Sculptor_MemoryLifetimeStorageExternal,
        Sculptor_MemoryLifetimeStoragePersistent,
        Sculptor_MemoryLifetimeStorageWorkspace,
        Sculptor_MemoryLifetimeStorageRoutineLocal,
        Sculptor_MemoryLifetimeStorageScratchpad
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_MemoryInterferenceRelationBefore
    : I32EnumAttrCase<"Before", 0, "before">;
def Sculptor_MemoryInterferenceRelationAfter
    : I32EnumAttrCase<"After", 1, "after">;
def Sculptor_MemoryInterferenceRelationInterferes
    : I32EnumAttrCase<"Interferes", 2, "interferes">;
def Sculptor_MemoryInterferenceRelationSeparateStorage
    : I32EnumAttrCase<"SeparateStorage", 3, "separate_storage">;
def Sculptor_MemoryInterferenceRelationInPlaceAlias
    : I32EnumAttrCase<"InPlaceAlias", 4, "in_place_alias">;

def Sculptor_MemoryInterferenceRelation
    : I32EnumAttr<"MemoryInterferenceRelation",
                  "relation between two tile memory lifetimes", [
        Sculptor_MemoryInterferenceRelationBefore,
        Sculptor_MemoryInterferenceRelationAfter,
        Sculptor_MemoryInterferenceRelationInterferes,
        Sculptor_MemoryInterferenceRelationSeparateStorage,
        Sculptor_MemoryInterferenceRelationInPlaceAlias
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_TileMemoryOwnerAttr
    : Sculptor_Attr<"TileMemoryOwner", "tile_memory_owner"> {
  let summary = "one allocation-owning tile memory resource";
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$resourceId,
    "IntegerAttr":$tensorId,
    "IntegerAttr":$tile,
    "IntegerAttr":$routine,
    "IntegerAttr":$port,
    EnumParameter<Sculptor_MemoryOwnerKind>:$kind,
    "IntegerAttr":$byteSize,
    "ArrayAttr":$shape,
    "ArrayAttr":$strides
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileMemoryViewAttr
    : Sculptor_Attr<"TileMemoryView", "tile_memory_view"> {
  let summary = "one static region view into an owned tile resource";
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$ownerId,
    "IntegerAttr":$byteOffset,
    "IntegerAttr":$byteSize,
    "ArrayAttr":$offsets,
    "ArrayAttr":$sizes,
    "ArrayAttr":$strides,
    EnumParameter<Sculptor_MemoryContiguity>:$contiguity
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileMemoryBindingAttr
    : Sculptor_Attr<"TileMemoryBinding", "tile_memory_binding"> {
  let summary = "one routine port bound to an owned resource view";
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$routine,
    "IntegerAttr":$port,
    "BoolAttr":$input,
    "IntegerAttr":$ownerId,
    "IntegerAttr":$viewId,
    EnumParameter<Sculptor_MemoryAccessEffect>:$effect
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileMemoryMovementAttr
    : Sculptor_Attr<"TileMemoryMovement", "tile_memory_movement"> {
  let summary = "one local alias, route, packing, or assembly movement";
  let parameters = (ins
    "IntegerAttr":$id,
    EnumParameter<Sculptor_MemoryMovementMode>:$mode,
    "IntegerAttr":$sourceViewId,
    "IntegerAttr":$destinationViewId,
    "IntegerAttr":$routeId,
    "IntegerAttr":$sourceCompletionEventId,
    "IntegerAttr":$destinationCompletionEventId,
    "IntegerAttr":$assemblyId,
    "IntegerAttr":$byteSize
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileMemorySegmentAttr
    : Sculptor_Attr<"TileMemorySegment", "tile_memory_segment"> {
  let summary = "one deterministic scatter-gather segment";
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$movementId,
    "IntegerAttr":$ordinal,
    "IntegerAttr":$sourceByteOffset,
    "IntegerAttr":$destinationByteOffset,
    "IntegerAttr":$byteSize
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileMemoryAssemblyAttr
    : Sculptor_Attr<"TileMemoryAssembly", "tile_memory_assembly"> {
  let summary = "one full-tensor assembly owner and readiness join";
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$ownerId,
    "ArrayAttr":$contributingViewIds,
    "ArrayAttr":$destinationViewIds,
    "ArrayAttr":$completionEventIds,
    "IntegerAttr":$readinessEventId,
    "IntegerAttr":$localCopyBytes,
    "IntegerAttr":$routedBytes
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileMemoryCompletionEventAttr
    : Sculptor_Attr<"TileMemoryCompletionEvent", "tile_memory_completion"> {
  let summary = "one stable memory-lifetime or readiness completion event";
  let parameters = (ins
    "IntegerAttr":$id,
    EnumParameter<Sculptor_MemoryCompletionKind>:$kind,
    "IntegerAttr":$tile,
    "IntegerAttr":$routine,
    "IntegerAttr":$routeId,
    "IntegerAttr":$ownerId,
    "IntegerAttr":$viewId
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileMemoryEventEdgeAttr
    : Sculptor_Attr<"TileMemoryEventEdge", "tile_memory_event_edge"> {
  let summary = "one directed happens-before edge between memory events";
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$sourceEventId,
    "IntegerAttr":$targetEventId,
    EnumParameter<Sculptor_MemoryEventEdgeKind>:$kind
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileMemoryLifetimeAttr
    : Sculptor_Attr<"TileMemoryLifetime", "tile_memory_lifetime"> {
  let summary = "one allocation lifetime in the unified tile memory model";
  let parameters = (ins
    "IntegerAttr":$id,
    EnumParameter<Sculptor_MemoryLifetimeSubjectKind>:$subjectKind,
    EnumParameter<Sculptor_MemoryLifetimeStorage>:$storage,
    "IntegerAttr":$ownerId,
    "IntegerAttr":$routine,
    "IntegerAttr":$allocationOrdinal,
    "IntegerAttr":$tile,
    "IntegerAttr":$byteSize,
    "IntegerAttr":$alignment,
    "IntegerAttr":$offset,
    "ArrayAttr":$viewIds,
    "ArrayAttr":$accessEventIds
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileMemoryInterferenceAttr
    : Sculptor_Attr<"TileMemoryInterference", "tile_memory_interference"> {
  let summary = "one proven ordering or conservative interference relation";
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$leftLifetimeId,
    "IntegerAttr":$rightLifetimeId,
    EnumParameter<Sculptor_MemoryInterferenceRelation>:$relation
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileMemoryInPlaceAliasAttr
    : Sculptor_Attr<"TileMemoryInPlaceAlias", "tile_memory_in_place_alias"> {
  let summary = "one proven elementwise update that aliases input and output storage";
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$routine,
    "IntegerAttr":$inputPort,
    "IntegerAttr":$outputPort,
    "IntegerAttr":$inputOwnerId,
    "IntegerAttr":$outputOwnerId,
    "IntegerAttr":$inputViewId,
    "IntegerAttr":$outputViewId
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileMemoryCapacityAttr
    : Sculptor_Attr<"TileMemoryCapacity", "tile_memory_capacity"> {
  let summary = "one physical tile memory-capacity summary";
  let parameters = (ins
    "IntegerAttr":$tile,
    "IntegerAttr":$externalBytes,
    "IntegerAttr":$persistentBytes,
    "IntegerAttr":$workspaceBytes,
    "IntegerAttr":$scratchpadBytes,
    "IntegerAttr":$routeInputBytes,
    "IntegerAttr":$routeOutputBytes,
    "IntegerAttr":$assemblyBytes,
    "IntegerAttr":$intermediateBytes,
    "IntegerAttr":$routineTemporaryPeakBytes,
    "IntegerAttr":$routineTemporaryTotalBytes,
    "IntegerAttr":$peakLiveBytes,
    "IntegerAttr":$requiredLocalBytes,
    "IntegerAttr":$reusableBytes,
    "BoolAttr":$complete
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileScratchpadCandidateAttr
    : Sculptor_Attr<"TileScratchpadCandidate", "tile_scratchpad_candidate"> {
  let summary = "one globally evaluated tile scratchpad candidate";
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$ownerId,
    "IntegerAttr":$lifetimeId,
    "IntegerAttr":$byteSize,
    "IntegerAttr":$alignment,
    "IntegerAttr":$capacityDemand,
    "IntegerAttr":$readCount,
    "IntegerAttr":$writeCount,
    "IntegerAttr":$movementCount,
    "IntegerAttr":$reuseCount,
    "IntegerAttr":$estimatedSavings,
    "BoolAttr":$selected
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileScratchpadAllocationAttr
    : Sculptor_Attr<"TileScratchpadAllocation", "tile_scratchpad_allocation"> {
  let summary = "one selected scratchpad owner and physical byte range";
  let parameters = (ins
    "IntegerAttr":$candidateId,
    "IntegerAttr":$ownerId,
    "IntegerAttr":$lifetimeId,
    "IntegerAttr":$offset,
    "IntegerAttr":$byteSize,
    "IntegerAttr":$capacityDemand,
    "IntegerAttr":$alignment,
    "IntegerAttr":$estimatedSavings
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileRoutineRouteAttr
    : Sculptor_Attr<"TileRoutineRoute", "tile_routine_route"> {
  let summary = "one routed tensor value between outlined tile routines";
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$sourceTile,
    "IntegerAttr":$sourceRoutine,
    "IntegerAttr":$sourceOutput,
    "IntegerAttr":$destinationTile,
    "IntegerAttr":$destinationRoutine,
    "IntegerAttr":$destinationInput,
    "IntegerAttr":$resourceId,
    "IntegerAttr":$tensorId,
    "IntegerAttr":$byteSize
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileRoutineBindingAttr
    : Sculptor_Attr<"TileRoutineBinding", "tile_routine_binding"> {
  let summary = "one same-tile value binding between outlined routines";
  let parameters = (ins
    "IntegerAttr":$sourceRoutine,
    "IntegerAttr":$sourceOutput,
    "IntegerAttr":$destinationRoutine,
    "IntegerAttr":$destinationInput,
    "IntegerAttr":$resourceId,
    "IntegerAttr":$byteSize
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TileRoutineModelIOAttr
    : Sculptor_Attr<"TileRoutineModelIO", "tile_routine_model_io"> {
  let summary = "model input or output owned by one outlined tile routine";
  let parameters = (ins
    "IntegerAttr":$index,
    "IntegerAttr":$tile,
    "IntegerAttr":$routine,
    "IntegerAttr":$port,
    "IntegerAttr":$resourceId,
    "IntegerAttr":$tensorId,
    "IntegerAttr":$byteSize
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_LogicalTileAnnealingTraceAttr
    : Sculptor_Attr<"LogicalTileAnnealingTrace", "logical_tile_annealing_trace"> {
  let summary = "score trajectory for logical-tile simulated annealing";
  let parameters = (ins
    "IntegerAttr":$version,
    "IntegerAttr":$initialScore,
    "IntegerAttr":$finalScore,
    "IntegerAttr":$evaluations,
    "ArrayAttr":$samples
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_MappingPlanAttr
    : Sculptor_Attr<"MappingPlan", "mapping_plan"> {
  let summary = "selected mapping planner result and reference evaluation";
  let description = [{
    Records the planner, objective, selected RA-tree identity, root metrics,
    evaluated alternatives, and per-node evaluation. It contains planning
    decisions only and does not assign runtime task or core identities.
  }];
  let parameters = (ins
    "IntegerAttr":$version,
    "StringAttr":$planner,
    EnumParameter<Sculptor_MappingPlanObjective>:$objective,
    "StringAttr":$mvmBodyPolicy,
    "StringAttr":$setupBindingPolicy,
    "StringAttr":$costProfileName,
    "StringAttr":$costProfileHash,
    "StringAttr":$raTreeFingerprint,
    "BoolAttr":$feasible,
    "FloatAttr":$estimatedLatencyNs,
    "IntegerAttr":$crossingBytes,
    "FloatAttr":$estimatedCommunicationNs,
    "IntegerAttr":$requiredResourceUnits,
    "IntegerAttr":$pipelineStages,
    "ArrayAttr":$candidates,
    "ArrayAttr":$nodeEvaluations,
    "MappingRealizationAttr":$realization
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_DeploymentRouteAttr
    : Sculptor_Attr<"DeploymentRoute", "deployment_route"> {
  let summary = "one routed tensor edge between deployed cores";
  let parameters = (ins
    "IntegerAttr":$id,
    "IntegerAttr":$sourceCore,
    "IntegerAttr":$sourceTask,
    "IntegerAttr":$sourceOutput,
    "IntegerAttr":$destinationCore,
    "IntegerAttr":$destinationTask,
    "IntegerAttr":$destinationInput,
    "IntegerAttr":$resourceId,
    "IntegerAttr":$byteSize
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // SCULPTOR_ATTRS
