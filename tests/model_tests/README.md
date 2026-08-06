# Model Tests

Model generators are kept separate from compiler lowering tests.

## GPT-2

Generate a six-block GPT-2 Mini fixture:

```bash
python tests/model_tests/generators/gpt2_generator.py \
  --mode mlir \
  --layers 6 \
  --hidden-size 384 \
  --attention-heads 6 \
  --intermediate-size 1536 \
  --sequence-length 4 \
  --output tests/model_tests/generated/gpt2_mini.mlir
```

Generate a one-block fixture by changing `--layers 6` to `--layers 1`.
The sweep driver is in `experiments/run_gpt2_sweep.py` and defaults to this
generator.
