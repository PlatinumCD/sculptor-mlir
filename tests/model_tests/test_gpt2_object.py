#!/usr/bin/env python3
"""Verify that a generated GPT-2 fixture reaches a RISC-V object."""

from pathlib import Path
import sys
import unittest

import torch


TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT / "python_tests"))
sys.path.insert(0, str(TEST_ROOT / "model_tests" / "generators"))

from gpt2_generator import GPT2Mini, make_inputs  # noqa: E402
from lowering_harness import LoweringCase, lower_to_tile_object  # noqa: E402


class GPT2ObjectLoweringTest(unittest.TestCase):
    def test_generated_one_block_fixture_reaches_riscv_object(self):
        output = lower_to_tile_object(
            LoweringCase(
                name="gpt2_one_block",
                model_factory=lambda: GPT2Mini(
                    num_layers=1,
                    sequence_length=4,
                    hidden_size=384,
                    attention_heads=6,
                    intermediate_size=1536,
                ),
                input_factory=lambda: make_inputs(4, 384),
                array_rows=1024,
                array_cols=512,
                external_linalg_lowering=True,
            )
        )
        self.assertEqual(output.read_bytes()[:4], b"\x7fELF")


if __name__ == "__main__":
    unittest.main()
