// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=analog digital-clock-ghz=1 digital-issue-width=2 digital-vector-bits-per-cycle=256" | FileCheck %s --check-prefix=ANALOG-TIMING
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital digital-clock-ghz=1 digital-issue-width=2 digital-vector-bits-per-cycle=256" | FileCheck %s --check-prefix=DIGITAL-TIMING
// RUN: not sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=invalid" 2>&1 | FileCheck %s --check-prefix=INVALID-MODE
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" | FileCheck %s --check-prefix=COST-SCHEDULE
// RUN: rm -f %t.summary.csv
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing summary-output=%t.summary.csv" -o /dev/null
// RUN: FileCheck %s --check-prefix=SUMMARY --input-file=%t.summary.csv
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=analog" --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=ANALOG-COST-ANALOG-EXEC
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=DIGITAL-COST-ANALOG-EXEC
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=analog" --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" --sculptor-lower-scheduled-mvm-to-digital --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=ANALOG-COST-DIGITAL-EXEC
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" --sculptor-lower-scheduled-mvm-to-digital --sculptor-analyze-task-graph-timing | FileCheck %s --check-prefix=DIGITAL-COST-DIGITAL-EXEC
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" --sculptor-lower-scheduled-mvm-to-digital --sculptor-analyze-task-graph-timing --sculptor-partition-task-graph-by-core | FileCheck %s --check-prefix=PARTITIONED
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" --sculptor-lower-scheduled-mvm-to-digital --sculptor-analyze-task-graph-timing --sculptor-fuse-task-graph --sculptor-analyze-task-graph-timing --sculptor-lower-golem-to-llvm-shims --sculptor-partition-task-graph-by-core --sculptor-extract-core-module="core-id=0" | FileCheck %s --check-prefix=EXTRACTED
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" --sculptor-export-task-graph-vis="output=%t.graphml format=graphml" -o /dev/null
// RUN: FileCheck %s --check-prefix=GRAPHML --input-file=%t.graphml
// RUN: sculptor-mlir-opt %s --sculptor-analyze-task-graph-timing="mvm-cost-mode=digital" --sculptor-schedule-task-graph="cores=1 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=1 schedule=greedy-timing" --sculptor-finalize-task-graph-resources --sculptor-export-task-graph-sim-model="output=%t.json" -o /dev/null
// RUN: FileCheck %s --check-prefix=JSON --input-file=%t.json

module {
  func.func private @task_setup() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<8x8xf32>
    %array = sculptor.array.set %matrix : tensor<8x8xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<1x8xf32>, %arg1: !sculptor.logical.array) -> tensor<1x8xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x8xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x8xf32>
    return %stored : tensor<1x8xf32>
  }

  func.func private @task_digital(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
    return %arg0 : tensor<1x8xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %output = sculptor.task_graph.output %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %mvm_out = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x8xf32>>
    %array = sculptor.task_graph.intermediate %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup = sculptor.task.create %graph, @task_setup, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "setup", source_layer = "linear", source_task_ordinal = 0, inputs[], outputs[%array], deps[] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "mvm", source_layer = "linear", source_task_ordinal = 1, inputs[%input, %array], outputs[%mvm_out], deps[%setup] {sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task) -> !sculptor.task
    %digital = sculptor.task.create %graph, @task_digital, domain = "digital", task_kind = "digital.compute", task_name = "digital", source_layer = "linear", source_task_ordinal = 2, inputs[%mvm_out], outputs[%output], deps[%mvm] {sculptor.runtime.digital_ops = 16 : i64, sculptor.schedule.island_id = 0 : i64} : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task_resource<tensor<1x8xf32>>, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// ANALOG-TIMING-LABEL: func.func private @generate_task_graph
// ANALOG-TIMING-SAME: sculptor.timing.critical_path_ns = 1.040000e+02 : f64
// ANALOG-TIMING-SAME: sculptor.timing.mvm_cost_mode = "analog"
// ANALOG-TIMING-SAME: sculptor.timing.total_digital_replacement_ops = 128 : i64
// ANALOG-TIMING: task_name = "mvm"
// ANALOG-TIMING-SAME: sculptor.timing.analog_execute_latency_ns = 1.000000e+02 : f64
// ANALOG-TIMING-SAME: sculptor.timing.analog_load_latency_ns = 1.000000e+00 : f64
// ANALOG-TIMING-SAME: sculptor.timing.analog_store_latency_ns = 1.000000e+00 : f64
// ANALOG-TIMING-SAME: sculptor.timing.digital_ops = 0 : i64
// ANALOG-TIMING-SAME: sculptor.timing.digital_replacement_ops = 128 : i64
// ANALOG-TIMING-SAME: sculptor.timing.intrinsic_latency_ns = 1.020000e+02 : f64
// ANALOG-TIMING: task_name = "digital"
// ANALOG-TIMING-SAME: sculptor.timing.digital_ops = 16 : i64
// ANALOG-TIMING-SAME: sculptor.timing.digital_replacement_ops = 0 : i64
// ANALOG-TIMING-SAME: sculptor.timing.intrinsic_latency_ns = 2.000000e+00 : f64

// DIGITAL-TIMING-LABEL: func.func private @generate_task_graph
// DIGITAL-TIMING-SAME: sculptor.timing.critical_path_ns = 1.800000e+01 : f64
// DIGITAL-TIMING-SAME: sculptor.timing.mvm_cost_mode = "digital"
// DIGITAL-TIMING-SAME: sculptor.timing.total_digital_replacement_ops = 128 : i64
// DIGITAL-TIMING: task_name = "mvm"
// DIGITAL-TIMING-SAME: sculptor.timing.analog_execute_latency_ns = 0.000000e+00 : f64
// DIGITAL-TIMING-SAME: sculptor.timing.analog_load_latency_ns = 0.000000e+00 : f64
// DIGITAL-TIMING-SAME: sculptor.timing.analog_store_latency_ns = 0.000000e+00 : f64
// DIGITAL-TIMING-SAME: sculptor.timing.digital_ops = 0 : i64
// DIGITAL-TIMING-SAME: sculptor.timing.digital_replacement_ops = 128 : i64
// DIGITAL-TIMING-SAME: sculptor.timing.intrinsic_latency_ns = 1.600000e+01 : f64
// DIGITAL-TIMING: task_name = "digital"
// DIGITAL-TIMING-SAME: sculptor.timing.digital_ops = 16 : i64
// DIGITAL-TIMING-SAME: sculptor.timing.digital_replacement_ops = 0 : i64
// DIGITAL-TIMING-SAME: sculptor.timing.intrinsic_latency_ns = 2.000000e+00 : f64

// INVALID-MODE: unknown MVM cost mode 'invalid'; expected 'analog' or 'digital'

// COST-SCHEDULE: module attributes
// COST-SCHEDULE-SAME: sculptor.schedule.placement_cost_mode = "digital"
// COST-SCHEDULE-SAME: sculptor.schedule.predicted_makespan_ns

// SUMMARY: generate_task_graph,greedy-timing,{{.*}},digital,{{[0-9.e+-]+}},{{[0-9.e+-]+}},{{[0-9.e+-]+}}
// COST-SCHEDULE-LABEL: func.func private @generate_task_graph
// COST-SCHEDULE-SAME: sculptor.schedule.critical_communication_ns
// COST-SCHEDULE-SAME: sculptor.schedule.maximum_resource_work_ns
// COST-SCHEDULE-SAME: sculptor.schedule.placement_cost_mode = "digital"
// COST-SCHEDULE-SAME: sculptor.schedule.predicted_makespan_ns

// ANALOG-COST-ANALOG-EXEC: module attributes
// ANALOG-COST-ANALOG-EXEC-SAME: sculptor.schedule.placement_cost_mode = "analog"
// ANALOG-COST-ANALOG-EXEC: func.func private @task_mvm
// ANALOG-COST-ANALOG-EXEC-LABEL: func.func private @generate_task_graph
// ANALOG-COST-ANALOG-EXEC-SAME: sculptor.schedule.placement_cost_mode = "analog"
// ANALOG-COST-ANALOG-EXEC-SAME: sculptor.timing.mvm_cost_mode = "analog"
// ANALOG-COST-ANALOG-EXEC: task_kind = "sculptor.mvm"
// ANALOG-COST-ANALOG-EXEC-SAME: sculptor.timing.analog_execute_latency_ns = 1.000000e+02 : f64

// DIGITAL-COST-ANALOG-EXEC: module attributes
// DIGITAL-COST-ANALOG-EXEC-SAME: sculptor.schedule.placement_cost_mode = "digital"
// DIGITAL-COST-ANALOG-EXEC: func.func private @task_mvm
// DIGITAL-COST-ANALOG-EXEC-LABEL: func.func private @generate_task_graph
// DIGITAL-COST-ANALOG-EXEC-SAME: sculptor.schedule.placement_cost_mode = "digital"
// DIGITAL-COST-ANALOG-EXEC-SAME: sculptor.timing.mvm_cost_mode = "analog"
// DIGITAL-COST-ANALOG-EXEC: task_kind = "sculptor.mvm"
// DIGITAL-COST-ANALOG-EXEC-SAME: sculptor.timing.analog_execute_latency_ns = 1.000000e+02 : f64

// ANALOG-COST-DIGITAL-EXEC: module attributes
// ANALOG-COST-DIGITAL-EXEC-SAME: sculptor.schedule.placement_cost_mode = "analog"
// ANALOG-COST-DIGITAL-EXEC: func.func private @task_mvm(%arg0: tensor<1x8xf32>)
// ANALOG-COST-DIGITAL-EXEC-SAME: sculptor.runtime.digital_ops = 128 : i64
// ANALOG-COST-DIGITAL-EXEC-SAME: sculptor.task_kind = "digital.matmul"
// ANALOG-COST-DIGITAL-EXEC-LABEL: func.func private @generate_task_graph
// ANALOG-COST-DIGITAL-EXEC-SAME: sculptor.schedule.placement_cost_mode = "analog"
// ANALOG-COST-DIGITAL-EXEC-SAME: sculptor.timing.mvm_cost_mode = "analog"

// DIGITAL-COST-DIGITAL-EXEC: module attributes
// DIGITAL-COST-DIGITAL-EXEC-SAME: sculptor.schedule.placement_cost_mode = "digital"
// DIGITAL-COST-DIGITAL-EXEC: func.func private @task_mvm(%arg0: tensor<1x8xf32>)
// DIGITAL-COST-DIGITAL-EXEC-SAME: sculptor.runtime.digital_ops = 128 : i64
// DIGITAL-COST-DIGITAL-EXEC-SAME: sculptor.task_kind = "digital.matmul"
// DIGITAL-COST-DIGITAL-EXEC-LABEL: func.func private @generate_task_graph
// DIGITAL-COST-DIGITAL-EXEC-SAME: sculptor.schedule.placement_cost_mode = "digital"
// DIGITAL-COST-DIGITAL-EXEC-SAME: sculptor.timing.mvm_cost_mode = "analog"

// PARTITIONED: module attributes
// PARTITIONED-SAME: sculptor.schedule.placement_cost_mode = "digital"
// PARTITIONED: module @core_0
// PARTITIONED: func.func private @generate_task_graph
// PARTITIONED-SAME: sculptor.schedule.placement_cost_mode = "digital"

// EXTRACTED: module attributes
// EXTRACTED-SAME: sculptor.runtime.core_id = 0 : i64
// EXTRACTED-SAME: sculptor.schedule.placement_cost_mode = "digital"
// EXTRACTED: func.func private @generate_task_graph
// EXTRACTED-SAME: sculptor.schedule.placement_cost_mode = "digital"

// GRAPHML: <key id="digital_replacement_ops" for="node"
// GRAPHML: <key id="placement_cost_mode" for="graph"
// GRAPHML: <data key="placement_cost_mode">digital</data>
// GRAPHML: <data key="mvm_cost_mode">digital</data>
// GRAPHML: <data key="digital_replacement_ops">128</data>

// JSON: "digital_replacement_ops": 128
// JSON: "placement_cost_mode": "digital"
// JSON: "total_digital_replacement_ops": 128
// JSON: "mvm_cost_mode": "digital"
