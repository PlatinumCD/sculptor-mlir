// RUN: sculptor-mlir-opt %s --split-input-file --verify-diagnostics --sculptor-distribute-digital-matmul="strategy=attention-heads max-shards=2 min-ops-per-shard=1"
// RUN: sculptor-mlir-opt %s --split-input-file --sculptor-distribute-digital-matmul="strategy=output-columns max-shards=2 min-ops-per-shard=1" -o /dev/null

module {
  func.func private @scores(
      %query: tensor<1x2x8xf32>,
      %key: tensor<1x2x8xf32>) -> tensor<1x4x2x2xf32>
      attributes {
        causal = true,
        sculptor.task_domain = "digital",
        sculptor.task_kind = "digital.attention_scores"
      }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %query = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x2x8xf32>>
    %key = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x2x8xf32>>
    %scores = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x4x2x2xf32>>
    // expected-error @below {{expected positive head_dim metadata on attention task callee}}
    %task = sculptor.task.create %graph, @scores,
        domain = "digital",
        task_kind = "digital.attention_scores",
        task_name = "scores",
        source_layer = "attention",
        source_task_ordinal = 0,
        inputs[%query, %key],
        outputs[%scores],
        deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x2x8xf32>>,
           !sculptor.task_resource<tensor<1x2x8xf32>>,
           !sculptor.task_resource<tensor<1x4x2x2xf32>>)
        -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @scores(
      %query: tensor<1x2x8xf32>,
      %key: tensor<1x2x8xf32>) -> tensor<1x4x2x2xf32>
      attributes {
        head_dim = 2 : i64,
        sculptor.task_domain = "digital",
        sculptor.task_kind = "digital.attention_scores"
      }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %query = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x2x8xf32>>
    %key = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x2x8xf32>>
    %scores = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x4x2x2xf32>>
    // expected-error @below {{expected causal metadata on attention score task callee}}
    %task = sculptor.task.create %graph, @scores,
        domain = "digital",
        task_kind = "digital.attention_scores",
        task_name = "scores",
        source_layer = "attention",
        source_task_ordinal = 0,
        inputs[%query, %key],
        outputs[%scores],
        deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x2x8xf32>>,
           !sculptor.task_resource<tensor<1x2x8xf32>>,
           !sculptor.task_resource<tensor<1x4x2x2xf32>>)
        -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @apply(
      %probabilities: tensor<1x4x2x2xf32>,
      %value: tensor<1x2x8xf32>) -> tensor<1x4x2x3xf32>
      attributes {
        head_dim = 2 : i64,
        sculptor.task_domain = "digital",
        sculptor.task_kind = "digital.attention_apply"
      }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %probabilities = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x4x2x2xf32>>
    %value = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x2x8xf32>>
    %output = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x4x2x3xf32>>
    // expected-error @below {{attention apply result disagrees with head_dim metadata}}
    %task = sculptor.task.create %graph, @apply,
        domain = "digital",
        task_kind = "digital.attention_apply",
        task_name = "apply",
        source_layer = "attention",
        source_task_ordinal = 0,
        inputs[%probabilities, %value],
        outputs[%output],
        deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x4x2x2xf32>>,
           !sculptor.task_resource<tensor<1x2x8xf32>>,
           !sculptor.task_resource<tensor<1x4x2x3xf32>>)
        -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}

// -----

module {
  func.func private @scores(
      %query: tensor<1x?x8xf32>,
      %key: tensor<1x2x8xf32>) -> tensor<1x4x?x2xf32>
      attributes {
        causal = false,
        head_dim = 2 : i64,
        sculptor.task_domain = "digital",
        sculptor.task_kind = "digital.attention_scores"
      }

  func.func private @generate_task_graph() -> !sculptor.task_graph {
    %graph = sculptor.task_graph.create : !sculptor.task_graph
    %query = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x?x8xf32>>
    %key = sculptor.task_graph.input %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x2x8xf32>>
    %scores = sculptor.task_graph.output %graph
        : !sculptor.task_graph
        -> !sculptor.task_resource<tensor<1x4x?x2xf32>>
    // expected-error @below {{digital attention matmul distribution requires static tensor shapes}}
    %task = sculptor.task.create %graph, @scores,
        domain = "digital",
        task_kind = "digital.attention_scores",
        task_name = "scores",
        source_layer = "attention",
        source_task_ordinal = 0,
        inputs[%query, %key],
        outputs[%scores],
        deps[]
        : (!sculptor.task_graph,
           !sculptor.task_resource<tensor<1x?x8xf32>>,
           !sculptor.task_resource<tensor<1x2x8xf32>>,
           !sculptor.task_resource<tensor<1x4x?x2xf32>>)
        -> !sculptor.task
    return %graph : !sculptor.task_graph
  }
}
