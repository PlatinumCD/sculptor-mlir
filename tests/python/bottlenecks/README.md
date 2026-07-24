# Bottleneck Compiler Fixtures

These Python programs are compiler inputs for studying logical task-graph
structure before placement, fusion, resource finalization, or runtime lowering.

`large_matmul_chain.py` contains a serial chain of progressively smaller linear
projections: `8192 -> 256 -> 64 -> 8`. Each projection consumes the result of
the preceding projection; the graph does not fan out.

The self-contained timing visualization in `visualizations/index.html` shows
the pre-placement critical path using a 1,152 x 256 array, 100 ns MVM
execution, 256-bit-per-cycle analog I/O, and a 1 GHz clock.
