// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=random" | FileCheck %s --check-prefix=RANDOM
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=snake" | FileCheck %s --check-prefix=SNAKE
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=annealing annealing-initial-schedule=snake annealing-move-set=segment-reverse" | FileCheck %s --check-prefix=ANNEALING
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=annealing annealing-initial-schedule=random annealing-move-set=basic-wide annealing-move-radius=1" | FileCheck %s --check-prefix=ANNEALING-WIDE
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=annealing annealing-initial-schedule=greedy greedy-heuristic=transfer-cost,boundary-regret,compact-region,lookahead=3,beam=1,scope=diagonal annealing-move-set=basic" | FileCheck %s --check-prefix=ANNEALING-GREEDY
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=annealing annealing-move-set=move-one-position" | FileCheck %s --check-prefix=MOVE-ONE-POSITION
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=annealing annealing-move-set=move-one-relocation" | FileCheck %s --check-prefix=MOVE-ONE-RELOCATION
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=annealing annealing-move-set=swap-two-positions" | FileCheck %s --check-prefix=SWAP-TWO-POSITIONS
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=annealing annealing-move-set=adjacent-swap" | FileCheck %s --check-prefix=ADJACENT-SWAP
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=annealing annealing-move-set=segment-relocation" | FileCheck %s --check-prefix=SEGMENT-RELOCATION
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=annealing annealing-move-set=block-swap" | FileCheck %s --check-prefix=BLOCK-SWAP
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy greedy-heuristic=boundary-regret,lookahead=1,beam=1,scope=cardinal" | FileCheck %s --check-prefix=GREEDY
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy greedy-heuristic=transfer-cost,compact-region,lookahead=2,beam=1,scope=cardinal" | FileCheck %s --check-prefix=COMPACT
// RUN: sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy greedy-heuristic=transfer-cost,scope=producer-consumer" | FileCheck %s --check-prefix=PRODUCER-CONSUMER
// RUN: not sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy greedy-heuristic=unknown" 2>&1 | FileCheck %s --check-prefix=INVALID-GREEDY-TERM
// RUN: not sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=greedy greedy-heuristic=transfer-cost,lookahead=0" 2>&1 | FileCheck %s --check-prefix=INVALID-GREEDY-LOOKAHEAD
// RUN: not sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=annealing annealing-move-set=unknown" 2>&1 | FileCheck %s --check-prefix=INVALID-ANNEALING-MOVE
// RUN: not sculptor-mlir-opt %s --sculptor-schedule-task-graph="cores=2 arrays-per-core=1 topology=mesh mesh-rows=1 mesh-cols=2 schedule=annealing annealing-cooling-rate=1" 2>&1 | FileCheck %s --check-prefix=INVALID-ANNEALING-COOLING

module {
  func.func private @task_matrix() -> !sculptor.logical.array {
    %matrix = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
    %array = sculptor.array.set %matrix : tensor<2x2xf32> -> !sculptor.logical.array
    return %array : !sculptor.logical.array
  }

  func.func private @task_mvm(%arg0: tensor<1x2xf32>, %arg1: !sculptor.logical.array) -> tensor<1x2xf32> {
    sculptor.array.load %arg0, %arg1 : tensor<1x2xf32>, !sculptor.logical.array
    %result = sculptor.array.execute %arg1 : !sculptor.logical.array -> !sculptor.array.result
    %stored = sculptor.array.store %result : !sculptor.array.result -> tensor<1x2xf32>
    return %stored : tensor<1x2xf32>
  }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %input = sculptor.task_graph.input %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %output = sculptor.task_graph.output %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %tmp = sculptor.task_graph.temporary %graph {sculptor.runtime.byte_size = 8 : i64} : !sculptor.task_graph -> !sculptor.task_resource<tensor<1x2xf32>>
    %array0 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %array1 = sculptor.task_graph.temporary %graph : !sculptor.task_graph -> !sculptor.task_resource<!sculptor.logical.array>
    %setup0 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer0_matrix", source_layer = "layer0", source_task_ordinal = 0, inputs[], outputs[%array0], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %setup1 = sculptor.task.create %graph, @task_matrix, domain = "analog", task_kind = "sculptor.matrix_setup", task_name = "layer1_matrix", source_layer = "layer1", source_task_ordinal = 0, inputs[], outputs[%array1], deps[] : (!sculptor.task_graph, !sculptor.task_resource<!sculptor.logical.array>) -> !sculptor.task
    %mvm0 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "layer0_mvm", source_layer = "layer0", source_task_ordinal = 1, inputs[%input, %array0], outputs[%tmp], deps[%setup0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task) -> !sculptor.task
    %mvm1 = sculptor.task.create %graph, @task_mvm, domain = "analog", task_kind = "sculptor.mvm", task_name = "layer1_mvm", source_layer = "layer1", source_task_ordinal = 1, inputs[%tmp, %array1], outputs[%output], deps[%setup1, %mvm0] : (!sculptor.task_graph, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task_resource<!sculptor.logical.array>, !sculptor.task_resource<tensor<1x2xf32>>, !sculptor.task, !sculptor.task) -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// RANDOM-LABEL: func.func private @generate_task_graph
// RANDOM-SAME: sculptor.schedule.graph_score = 8 : i64
// RANDOM-SAME: sculptor.schedule.total_transfer_cost = 8 : i64

// SNAKE-LABEL: func.func private @generate_task_graph
// SNAKE-SAME: sculptor.schedule.graph_score = 8 : i64
// SNAKE-SAME: sculptor.schedule.total_transfer_cost = 8 : i64

// ANNEALING-LABEL: func.func private @generate_task_graph
// ANNEALING-SAME: sculptor.schedule.annealing_move_set = "segment-reverse"
// ANNEALING-SAME: sculptor.schedule.graph_score = 8 : i64
// ANNEALING-SAME: sculptor.schedule.total_transfer_cost = 8 : i64

// ANNEALING-WIDE-LABEL: func.func private @generate_task_graph
// ANNEALING-WIDE-SAME: sculptor.schedule.annealing_move_radius = 1 : i64
// ANNEALING-WIDE-SAME: sculptor.schedule.annealing_move_set = "basic-wide"
// ANNEALING-WIDE-SAME: sculptor.schedule.graph_score = 8 : i64
// ANNEALING-WIDE-SAME: sculptor.schedule.total_transfer_cost = 8 : i64

// ANNEALING-GREEDY-LABEL: func.func private @generate_task_graph
// ANNEALING-GREEDY-SAME: sculptor.schedule.annealing_move_set = "basic"
// ANNEALING-GREEDY-SAME: sculptor.schedule.graph_score = 8 : i64
// ANNEALING-GREEDY-SAME: sculptor.schedule.greedy_heuristic = "transfer-cost,boundary-regret,compact-region,lookahead=3,beam=1,scope=diagonal"
// ANNEALING-GREEDY-SAME: sculptor.schedule.total_transfer_cost = 8 : i64

// MOVE-ONE-POSITION-LABEL: func.func private @generate_task_graph
// MOVE-ONE-POSITION-SAME: sculptor.schedule.annealing_move_set = "move-one-position"
// MOVE-ONE-POSITION-SAME: sculptor.schedule.graph_score = 8 : i64

// MOVE-ONE-RELOCATION-LABEL: func.func private @generate_task_graph
// MOVE-ONE-RELOCATION-SAME: sculptor.schedule.annealing_move_set = "move-one-relocation"
// MOVE-ONE-RELOCATION-SAME: sculptor.schedule.graph_score = 8 : i64

// SWAP-TWO-POSITIONS-LABEL: func.func private @generate_task_graph
// SWAP-TWO-POSITIONS-SAME: sculptor.schedule.annealing_move_set = "swap-two-positions"
// SWAP-TWO-POSITIONS-SAME: sculptor.schedule.graph_score = 8 : i64

// ADJACENT-SWAP-LABEL: func.func private @generate_task_graph
// ADJACENT-SWAP-SAME: sculptor.schedule.annealing_move_set = "adjacent-swap"
// ADJACENT-SWAP-SAME: sculptor.schedule.graph_score = 8 : i64

// SEGMENT-RELOCATION-LABEL: func.func private @generate_task_graph
// SEGMENT-RELOCATION-SAME: sculptor.schedule.annealing_move_set = "segment-relocation"
// SEGMENT-RELOCATION-SAME: sculptor.schedule.graph_score = 8 : i64

// BLOCK-SWAP-LABEL: func.func private @generate_task_graph
// BLOCK-SWAP-SAME: sculptor.schedule.annealing_move_set = "block-swap"
// BLOCK-SWAP-SAME: sculptor.schedule.graph_score = 8 : i64

// GREEDY-LABEL: func.func private @generate_task_graph
// GREEDY-SAME: sculptor.schedule.graph_score = 8 : i64
// GREEDY-SAME: sculptor.schedule.greedy_beam_width = 1 : i64
// GREEDY-SAME: sculptor.schedule.greedy_candidate_scope = "cardinal"
// GREEDY-SAME: sculptor.schedule.greedy_heuristic = "boundary-regret,lookahead=1,beam=1,scope=cardinal"
// GREEDY-SAME: sculptor.schedule.total_transfer_cost = 8 : i64

// COMPACT-LABEL: func.func private @generate_task_graph
// COMPACT-SAME: sculptor.schedule.graph_score = 8 : i64
// COMPACT-SAME: sculptor.schedule.greedy_beam_width = 1 : i64
// COMPACT-SAME: sculptor.schedule.greedy_candidate_scope = "cardinal"
// COMPACT-SAME: sculptor.schedule.greedy_heuristic = "transfer-cost,compact-region,lookahead=2,beam=1,scope=cardinal"
// COMPACT-SAME: sculptor.schedule.greedy_lookahead = 2 : i64
// COMPACT-SAME: sculptor.schedule.total_transfer_cost = 8 : i64

// PRODUCER-CONSUMER-LABEL: func.func private @generate_task_graph
// PRODUCER-CONSUMER-SAME: sculptor.schedule.graph_score = 8 : i64
// PRODUCER-CONSUMER-SAME: sculptor.schedule.greedy_candidate_scope = "producer-consumer"
// PRODUCER-CONSUMER-SAME: sculptor.schedule.total_transfer_cost = 8 : i64

// INVALID-GREEDY-TERM: error: unknown Sculptor greedy heuristic term 'unknown'
// INVALID-GREEDY-LOOKAHEAD: error: expected Sculptor greedy heuristic term 'lookahead' to use a positive integer value
// INVALID-ANNEALING-MOVE: error: unknown Sculptor annealing move-set term 'unknown'
// INVALID-ANNEALING-COOLING: error: expected Sculptor annealing cooling rate to be greater than zero and less than one
