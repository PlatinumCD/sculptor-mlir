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

def Sculptor_RATreeNodeKind
    : I32EnumAttr<"RATreeNodeKind", "Resource Allocation Tree node kind", [
        Sculptor_RATreeNodeKindSpatialCut,
        Sculptor_RATreeNodeKindTemporalCut,
        Sculptor_RATreeNodeKindLeaf
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
