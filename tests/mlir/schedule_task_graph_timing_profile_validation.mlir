// RUN: not sculptor-mlir-opt %s --sculptor-build-task-graph-islands --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" 2>&1 | FileCheck %s

module {
  func.func private @generate_task_graph() -> !sculptor.task_graph attributes {
    sculptor.timing.critical_path_ns = 0.000000e+00 : f64,
    sculptor.timing.island_edges = [],
    sculptor.timing.islands = [#sculptor.island_timing<islandId = 0 : i64, taskCount = 0 : i64, totalWorkNs = 0.000000e+00 : f64, analogWorkNs = 0.000000e+00 : f64, digitalWorkNs = 0.000000e+00 : f64, earliestStartNs = 0.000000e+00 : f64, earliestFinishNs = 0.000000e+00 : f64, criticalPathRemainingNs = 0.000000e+00 : f64, slackNs = 0.000000e+00 : f64, isCritical = false>],
    sculptor.timing.model = #sculptor.timing_model<analogMVMLatencyNs = 100 : i64, analogIOBitsPerCycle = 256 : i64, analogIOShared = true, digitalClockGHz = 1.000000e+00 : f64, digitalIssueWidth = 2 : i64, digitalVectorBitsPerCycle = 256 : i64, networkLinkBitsPerCycle = 32 : i64, networkHopLatencyCycles = 1 : i64, networkPipelined = true>,
    sculptor.timing.placement_aware = false,
    sculptor.timing.task_count = 0 : i64
  } {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    return %graph : !sculptor.task_graph
  }
}

// CHECK: timing profile attribute 'sculptor.timing.islands' has 1 entries; expected 0
// CHECK: failed to load pre-placement scheduling timing profile
