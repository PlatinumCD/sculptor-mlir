#ifndef SCULPTOR_ATTRS
#define SCULPTOR_ATTRS

include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.td"

class Sculptor_Attr<string name, string attrMnemonic>
    : AttrDef<Sculptor_Dialect, name> {
  let mnemonic = attrMnemonic;
}

def Sculptor_TimingModelAttr
    : Sculptor_Attr<"TimingModel", "timing_model"> {
  let summary = "hardware timing model used for task graph analysis";
  let parameters = (ins
    "IntegerAttr":$analogMVMLatencyNs,
    "IntegerAttr":$analogIOBitsPerCycle,
    "BoolAttr":$analogIOShared,
    "FloatAttr":$digitalClockGHz,
    "IntegerAttr":$digitalIssueWidth,
    "IntegerAttr":$digitalVectorBitsPerCycle,
    "IntegerAttr":$networkLinkBitsPerCycle,
    "IntegerAttr":$networkHopLatencyCycles,
    "BoolAttr":$networkPipelined
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_IslandTimingAttr
    : Sculptor_Attr<"IslandTiming", "island_timing"> {
  let summary = "timing profile for one logical placement island";
  let parameters = (ins
    "IntegerAttr":$islandId,
    "IntegerAttr":$taskCount,
    "FloatAttr":$totalWorkNs,
    "FloatAttr":$analogWorkNs,
    "FloatAttr":$digitalWorkNs,
    "FloatAttr":$earliestStartNs,
    "FloatAttr":$earliestFinishNs,
    "FloatAttr":$criticalPathRemainingNs,
    "FloatAttr":$slackNs,
    "BoolAttr":$isCritical
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_TimedIslandEdgeAttr
    : Sculptor_Attr<"TimedIslandEdge", "timed_island_edge"> {
  let summary = "timing profile for one directed island dependency";
  let parameters = (ins
    "IntegerAttr":$producerIsland,
    "IntegerAttr":$consumerIsland,
    "IntegerAttr":$bytes,
    "FloatAttr":$estimatedTransferNsPerHop,
    "FloatAttr":$estimatedAdditionalHopNs,
    "FloatAttr":$criticality,
    "FloatAttr":$producerReadyTimeNs,
    "FloatAttr":$consumerTimingPressure
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Sculptor_NetworkEdgeTimingAttr
    : Sculptor_Attr<"NetworkEdgeTiming", "network_edge_timing"> {
  let summary = "placement-aware timing for one task graph network edge";
  let parameters = (ins
    "IntegerAttr":$producerTask,
    "IntegerAttr":$consumerTask,
    "IntegerAttr":$sourceCore,
    "IntegerAttr":$destinationCore,
    "IntegerAttr":$meshHops,
    "FloatAttr":$transferStartNs,
    "FloatAttr":$transferFinishNs,
    "FloatAttr":$networkLatencyNs,
    "FloatAttr":$contentionDelayNs
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // SCULPTOR_ATTRS
