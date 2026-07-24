#!/usr/bin/env python3
"""Compiler fixture containing progressively smaller serial projections."""

import argparse

import torch
from torch_mlir import fx


LARGE_INPUT_SIZE = 8192
LARGE_OUTPUT_SIZE = 256
MEDIUM_OUTPUT_SIZE = 64
SMALL_OUTPUT_SIZE = 8


class LargeMatmulChain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.large_projection = torch.nn.Linear(
            LARGE_INPUT_SIZE,
            LARGE_OUTPUT_SIZE,
            bias=False,
        )
        self.medium_projection = torch.nn.Linear(
            LARGE_OUTPUT_SIZE,
            MEDIUM_OUTPUT_SIZE,
            bias=False,
        )
        self.small_projection = torch.nn.Linear(
            MEDIUM_OUTPUT_SIZE,
            SMALL_OUTPUT_SIZE,
            bias=False,
        )

        with torch.no_grad():
            self.large_projection.weight.fill_(1.0 / LARGE_INPUT_SIZE)
            self.medium_projection.weight.fill_(1.0 / LARGE_OUTPUT_SIZE)
            self.small_projection.weight.fill_(1.0 / MEDIUM_OUTPUT_SIZE)

    def forward(self, x):
        x = self.large_projection(x)
        x = self.medium_projection(x)
        return self.small_projection(x)


def make_input():
    return torch.linspace(
        0.0,
        1.0,
        LARGE_INPUT_SIZE,
        dtype=torch.float32,
    ).reshape(1, LARGE_INPUT_SIZE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "mlir"), default="run")
    return parser.parse_args()


def main():
    args = parse_args()
    model = LargeMatmulChain().eval()
    model_input = make_input()

    if args.mode == "mlir":
        mlir_module = fx.export_and_import(
            model,
            model_input,
            output_type="torch",
            func_name="forward",
        )
        print(mlir_module)
        return

    with torch.no_grad():
        output = model(model_input)
    print(output)


if __name__ == "__main__":
    main()
