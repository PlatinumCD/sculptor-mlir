#!/usr/bin/env python3
"""Export a compact GPT-2-style Transformer compiler fixture."""

import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path

import torch


DEFAULT_TORCH_MLIR_PACKAGE = (
    Path(__file__).resolve().parents[5]
    / "build"
    / "torch-mlir"
    / "python_packages"
    / "torch_mlir"
)
torch_mlir_package = Path(
    os.environ.get("TORCH_MLIR_PYTHON_PACKAGE", DEFAULT_TORCH_MLIR_PACKAGE)
)
if str(torch_mlir_package) not in sys.path:
    sys.path.insert(0, str(torch_mlir_package))


BATCH_SIZE = 1
SEQUENCE_LENGTH = 4
HIDDEN_SIZE = 384
ATTENTION_HEADS = 6
TRANSFORMER_LAYERS = 6
INTERMEDIATE_SIZE = 1536


warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self.use_nested_tensor is False.*",
    category=UserWarning,
)

try:
    torch.backends.mha.set_fastpath_enabled(False)
except AttributeError:
    pass


class GPT2Mini(torch.nn.Module):
    def __init__(
        self,
        num_layers=TRANSFORMER_LAYERS,
        sequence_length=SEQUENCE_LENGTH,
        hidden_size=HIDDEN_SIZE,
        attention_heads=ATTENTION_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=intermediate_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        self.blocks = torch.nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=torch.nn.LayerNorm(HIDDEN_SIZE),
        )
        self.causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool),
            diagonal=1,
        )
        self._initialize_deterministically()

    def _initialize_deterministically(self):
        with torch.no_grad():
            for index, (name, parameter) in enumerate(self.named_parameters()):
                if "norm" in name and name.endswith("weight"):
                    parameter.fill_(1.0)
                elif "norm" in name and name.endswith("bias"):
                    parameter.zero_()
                else:
                    scale = 1000.0 + 10.0 * index if parameter.ndim == 1 else 100.0 + 10.0 * index
                    parameter.fill_((index + 1) / scale)

    def forward(self, hidden_states):
        for layer in self.blocks.layers:
            hidden_states = layer(
                hidden_states,
                src_mask=self.causal_mask,
                is_causal=False,
            )
        return self.blocks.norm(hidden_states)


def make_inputs(sequence_length=SEQUENCE_LENGTH, hidden_size=HIDDEN_SIZE):
    values = torch.arange(
        1,
        BATCH_SIZE * sequence_length * hidden_size + 1,
        dtype=torch.float32,
    )
    return (values.reshape(BATCH_SIZE, sequence_length, hidden_size) / 100.0,)


def emit_mlir(model, inputs):
    from torch_mlir import fx

    exported = torch.export.export(model, inputs).run_decompositions()
    torch_module = fx.export_and_import(
        exported,
        output_type="torch",
        func_name="forward",
    )
    torch_mlir_opt = Path(
        os.environ.get(
            "TORCH_MLIR_OPT",
            DEFAULT_TORCH_MLIR_PACKAGE.parent.parent / "bin" / "torch-mlir-opt",
        )
    )
    result = subprocess.run(
        [
            str(torch_mlir_opt),
            "-",
            "--pass-pipeline="
            "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
        ],
        input=str(torch_module),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "mlir"), default="run")
    parser.add_argument("--layers", type=int, default=TRANSFORMER_LAYERS)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--attention-heads", type=int, default=ATTENTION_HEADS)
    parser.add_argument("--intermediate-size", type=int, default=INTERMEDIATE_SIZE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for name in ("layers", "sequence_length", "hidden_size", "attention_heads", "intermediate_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.hidden_size % args.attention_heads:
        parser.error("--hidden-size must be divisible by --attention-heads")

    torch.manual_seed(0)
    model = GPT2Mini(
        num_layers=args.layers,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        attention_heads=args.attention_heads,
        intermediate_size=args.intermediate_size,
    ).eval()
    inputs = make_inputs(
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
    )

    if args.mode == "mlir":
        module = str(emit_mlir(model, inputs))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(module)
        else:
            print(module)
        return

    with torch.no_grad():
        output = model(*inputs)
    print(output)


if __name__ == "__main__":
    main()
