# Python Model Lowering Tests

These tests exercise each supported PyTorch model family through the complete
pre-placement pivot pipeline:

```text
PyTorch
  -> linalg-on-tensors
  -> canonicalize layers
  -> extract layers
  -> convert layers
  -> expand MVM to Golem
  -> build RA tree
```

The suite covers Linear, Conv1D, Conv2D, grouped Conv2D, Conv3D, RNNCell,
stacked RNN, GRUCell, stacked GRU, LSTMCell, stacked LSTM, and Transformer.

Run the complete suite from the repository root:

```bash
../../.venv/bin/python tests/python_tests/run_all.py
```

Set `SCULPTOR_MLIR_OPT`, `TORCH_MLIR_OPT`, or
`TORCH_MLIR_PYTHON_PACKAGE` to override the default build-tree locations.
