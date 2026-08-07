# Sculptor Pivot

The papers point toward one central change:

> **Sculptor currently lowers one decomposition and then places it. A stronger compiler would preserve a mapping space, evaluate alternatives, select a mapping, and only then instantiate tasks and physical arrays.**

`sculptor.mvm` is a non-negotiable part of that mapping space. Transformer and
other layer converters must preserve logical MVM work, the RA Tree must model
it directly, and Golem expansion must realize it only after a mapping is
selected. Task regions are a downstream deployment representation, not the
search representation.

LoopTree jointly explores tiling, retention, recomputation, and scheduling. SET argues that inter-layer allocation and ordering must be represented explicitly instead of hidden inside heuristics. Gemini similarly builds an explicit mapping encoding before searching it. [LoopTree](https://eems.mit.edu/wp-content/uploads/2024/09/2024_tcasai_looptree.pdf), [SET](https://dl.acm.org/doi/10.1145/3579371.3589048), [Gemini](https://arxiv.org/pdf/2312.16436)

| What you currently do | What I believe is the better idea |
|---|---|
| **Overall:** Lower first, form a fixed task graph, then search placement. | **Map first, lower second.** Preserve legal decomposition and dataflow alternatives, evaluate them, then instantiate the selected task graph. |
| **Canonicalization:** Recognize a whole Transformer and replace it with semantic Sculptor operations. | Keep this. Semantic recognition is valuable and gives the mapper structure that raw `linalg` cannot provide. |
| **Layer extraction:** Outline every Transformer block into a function early. | Preserve block hierarchy as metadata or regions, but do not make function boundaries optimization barriers until the mapping is selected. |
| **Conversion:** Immediately decompose blocks into fixed MVM and digital task regions. | Produce a hierarchical tensor-computation representation containing operations, loop/rank domains, dependencies, and legal backend choices. |
| **Transformer parallelism:** The generated graph largely reflects the decomposition chosen by the converter. | Explicitly expose token, attention-head, query/key position, input-channel, output-channel, and MLP dimensions as candidate partition ranks. |
| **Loops:** Small static sequence dimensions can become repeated operations and tasks. | Retain `scf`/affine loops or symbolic rank domains. Let mapping decide whether to unroll, tile, vectorize, distribute, or retain them. |
| **MVM expansion:** `sculptor.mvm` is expanded into physical tiles, padding, setup, load, execute, store, and recombination before placement. | Keep logical MVMs and compact physical-tile descriptors during exploration. Materialize Golem operations only after mapping decisions are fixed. |
| **Task regions:** A candidate task region becomes a private function through `materialize-tasks`. | Treat a task region as a **candidate scheduling boundary**. Outline only final dispatch units after placement, fusion, and local scheduling. |
| **Task meaning:** Affinity group, placement unit, dispatch unit, and executable function can collapse into nearly the same concept. | Separate them: computation tile, affinity group, placement cluster, dispatch task, and executable kernel must be distinct objects. |
| **Islands:** Build one irreversible island decomposition and assign surrounding digital work to it. | Represent islands as hierarchical mapping candidates. Allow alternative clustering, splitting, and fusion choices while retaining the current islands as a baseline policy. |
| **Scheduling:** Assign already-fixed islands to cores. | Jointly select partitioning, clustering, tensor residency, core allocation, array allocation, and per-core temporal order. |
| **Scheduler search:** Greedy, snake, annealing, and other methods manipulate one fixed graph. | Define a legal mapping encoding with structured transformations. Search algorithms then explore meaningful choices instead of swapping prematurely fixed tasks. Gemini demonstrates this distinction with a layer-centric encoding and custom search operators. [Gemini](https://arxiv.org/pdf/2312.16436) |
| **Cost function:** Primarily optimize transfer distance, communication, and attached timing information. | Evaluate critical-path latency, link contention, peak local storage, tensor lifetime, analog and digital utilization, transfer volume, energy, and recomputation together. |
| **Timing:** Timing annotates a task graph that already embodies one decomposition. | Make the timing model a mapping evaluator. Every candidate decomposition and placement should receive a predicted execution schedule and cost. |
| **Intermediate tensors:** Task boundaries can force complete intermediate tensors to exist. | Choose per tensor whether to retain, stream, refetch, or recompute it. LoopTree shows these choices must be explored jointly with tiling and schedule. [LoopTree](https://eems.mit.edu/wp-content/uploads/2024/09/2024_tcasai_looptree.pdf) |
| **Fusion:** Fuse connected same-island, same-core tasks after placement. | Select tiled producer-consumer fusion before physical lowering. Produce and immediately consume tensor tiles where profitable rather than materializing complete intermediates. |
| **Same-core execution:** A fused function can accidentally serialize work that could overlap. | Preserve a per-core local DAG containing digital, analog-load, analog-execute, analog-store, and network events with explicit resource constraints. |
| **Global versus local parallelism:** Global placement is the main optimization decision. | Use two levels: global spatial mapping across tiles and local temporal/dataflow scheduling within each tile. Tangram specifically motivates combining coarse inter-tile and fine intra-engine dataflow. [Tangram](https://nas.iiis.tsinghua.edu.cn/~gaomy/pubs/tangram.asplos19.pdf) |
| **Memory:** Runtime slots are assigned late, but mapping does not fully optimize residency and peak capacity. | Keep final slot assignment late, but model tensor residency, lifetime, capacity, and reuse during mapping. Deterministic software-managed memory is a central accelerator concern, not merely backend bookkeeping. [TPU analysis](https://arxiv.org/pdf/1704.04760) |
| **Backend selection:** Analog structure is established early; digital execution can replace scheduled MVMs later. | Keep operations backend-neutral during mapping and evaluate analog and digital implementations without forcing either physical expansion first. |
| **Final lowering:** Hundreds of task functions and physical resources can exist before the best execution organization is known. | After selecting the mapping, instantiate physical arrays, task functions, routes, buffers, core modules, LLVM, and ELFs exactly once. |
| **Evaluation:** Compare scheduler scores on a single decomposition. | Compare Pareto fronts across mapping choices: latency, communication, storage, utilization, energy, and compiler time. There is unlikely to be one universally optimal mapping. |

The table describes the compiler before the pivot. The current architecture is
documented in [Design](../docs/pages/design.md).

## Recommended Architecture

```text
PyTorch / Torch-MLIR
        |
Semantic Sculptor graph
        |
Hierarchical mapping representation
  - operation and tensor domains
  - legal partition ranks
  - legal fusion boundaries
  - retention/stream/recompute choices
  - analog/digital implementation choices
        |
Mapping exploration + cost model
  - spatial placement
  - local temporal schedule
  - memory capacity
  - communication/contention
  - latency/utilization
        |
Selected mapping plan
        |
Instantiate tasks and physical MVM tiles
        |
Build per-core local DAGs
        |
Outline final dispatch functions
        |
Partition, finalize resources, LLVM, ELFs
```

The analog-digital co-design paper supports keeping the end-to-end stack layered, with model optimization, mapping, per-tile linearization, and runtime generation as distinct responsibilities. [Analog-digital accelerator software stack](https://sitaohuang.com/publications/2018_icrc_analog_ml.pdf)

The practical first move is therefore **not replacing the entire compiler**. Add a non-destructive mapping representation between semantic conversion and physical MVM expansion. Make the current eager pipeline one legal baseline mapping. Then move one decision at a time into that representation, beginning with partition ranks and task-boundary selection.

## Implementation Guide

See [Sculptor Pivot RA Tree Implementation Plan](sculptor_pivot_implementation_plan.md)
for the file-level migration sequence, compatibility rules, acceptance gates,
removal criteria, and first vertical slice.
