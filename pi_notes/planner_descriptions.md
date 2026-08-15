# Detailed Description of RA Tree Planning Strategies

## 1. Setup-First Planner

### High-Level Perspective
The Setup-First planner addresses a fundamental scheduling concern: matrix setup operations (loading weights into analog arrays) should precede their dependent execution operations. In the target architecture, each analog lane must be loaded with weights before it can execute. The planner ensures that matrix setup operations are temporally ordered before the operations that depend on them.

### Detailed Description
The planner operates on the RA tree by cloning each matrix setup leaf to the root position, creating a temporal ordering where setup operations come before execution. For each matrix setup leaf, it clones that leaf to the root, then multiplies the work group counts of the original subtree to preserve concurrency. The key insight is that matrix setup operations are independent — they can all execute in parallel before any execution happens.

The algorithm works as follows:
1. Clone each matrix setup leaf to the root
2. Clone each non-setup operation to its original position
3. Multiply work group counts to preserve concurrency
4. Create a temporal cut that puts all setup operations before execution

### Potential Insights
- This strategy is particularly effective when setup time dominates execution time
- The temporal ordering ensures no execution happens before its dependencies are loaded
- The strategy preserves all original dependencies

---

## 2. MVM Wave Planner

### High-Level Perspective
The MVM Wave planner groups independent matrix-vector multiply operations into waves that can execute in parallel. Each wave contains operations that can run concurrently without violating dependencies.

### Detailed Description
The planner builds waves of MVM operations based on topological rank. Each wave contains:
- Vector tile operations (sorted by topological rank)
- Physical MVM operations (sorted by wave member)
- Recombine and bias-add operations

The algorithm:
1. Compute topological ranks for each operation
2. Group MVM operations into waves
3. Sort within each wave for consistent ordering
4. Create spatial cuts for parallel execution within waves

### Potential Insights
- This strategy exploits data-level parallelism in MVM operations
- The wave structure allows multiple MVMs to run concurrently
- The topological ordering ensures dependencies are respected
- The strategy is particularly effective for models with many independent MVMs

---

## 3. Fan-Out Cut Planner

### High-Level Perspective
The Fan-Out Cut planner exposes parallelism after fan-out operations. A "fill" operation that produces a tensor that multiple consumers use can have its consumers executed in parallel.

### Detailed Description
The planner identifies root fill operations (those with no non-setup producers) and groups their direct consumers. It then inserts cuts to parallelize independent consumer subtrees.

The algorithm:
1. Identify all fill operations
2. Group fill leaves by their direct consumers
3. Create subtrees that bind each fill to its consumers
4. Ensure dependencies are preserved

### Potential Insights
- This strategy exposes parallelism that other strategies miss
- It's particularly effective when there are many fan-out points
- The strategy reduces memory pressure by binding fills to consumers
- It can significantly reduce overall latency

---

## 4. Consumer-Bound Fill Planner

### High-Level Perspective
The Consumer-Bound Fill planner binds fill operations to their consumers, reducing redundant computation and improving memory locality.

### Detailed Description
The planner identifies all fill operations and groups them by their direct consumers. Each fill is bound to its consumers, creating a tighter coupling between data producers and consumers.

The algorithm:
1. Identify all fill operations
2. Find direct consumers for each fill
3. Create subtrees that bind each fill to its consumers
4. Ensure the binding is preserved throughout the pipeline

### Potential Insights
- This strategy improves memory locality
- It reduces redundant computation by binding fills to consumers
- The strategy is particularly effective for models with many fill operations
- It can significantly reduce memory bandwidth requirements

---

## Combined Effect

When these four strategies are composed in sequence, they each transform the RA tree, with each subsequent planner building on the previous one's output. The mapping evaluator scores each candidate, and the best plan is selected.

The combined effect is:
1. Setup operations are properly ordered
2. Independent MVMs are grouped into parallel waves
3. Fan-out points expose parallel consumers
4. Fill operations are bound to their consumers

This creates a well-structured RA tree that the placement algorithm can use to produce good results.