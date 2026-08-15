# Open Questions

## General Architecture

1. What is the performance trade-off between the four planning strategies? Has empirical benchmarking been done?
2. Why was the RA tree chosen over a directed acyclic graph (DAG) representation?
3. How does the compiler handle dynamic control flow (e.g., conditional branches)?

## Mapping Algorithm

4. Is the mapping problem NP-hard? Is the greedy approach guaranteed to find a valid plan?
5. Why does the evaluator use `estimatedLatencyNs` rather than exact timing?
6. How does the `MappingEvaluator` compute crossing bytes and communication cost?

## Correctness

7. Is there a formal proof that each pipeline stage is semantically correct?
8. Are there any invariants about the RA tree that can be checked automatically?
9. How does the compiler ensure that lane bindings are respected during placement?

## Runtime

10. Does the Golem runtime handle all edge cases (e.g., buffer overflow, lost frames)?
11. How does the runtime handle multiple simultaneous receives from different sources?
12. Is the 16-entry task pool a hard limit, or does it scale with hardware?

## Implementation

13. Why is there no explicit test coverage for each planner strategy?
14. Is there an automated test suite for the RA tree construction algorithm?
15. What is the current status of the `sculptor-ra-tree-report` tool?

## Design Decisions

16. Why separate logical mapping from runtime lowering?
17. Why use an MLIR out-of-tree dialect instead of an in-tree extension?
18. What is the relationship between the "pivot" pipeline and any legacy "island" pipeline?

## Open Implementation Gaps

19. Are there known bugs or missing features in the current codebase?
20. What would be needed to support more complex models (e.g., attention, transformers)?