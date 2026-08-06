#!/usr/bin/env python3
"""Python-backed Linear lowering test."""

import torch

from lowering_harness import (
    LoweringCase,
    LoweringTestCase,
    initialize_parameters,
)


class LinearModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=bias)
        initialize_parameters(self)

    def forward(self, value):
        return self.linear(value)


class LinearLoweringTest(LoweringTestCase):
    def test_linear_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                name="linear_with_bias",
                model_factory=lambda: LinearModel(bias=True),
                input_factory=lambda: (torch.ones(1, 4),),
                expected_fragments=("tensor<1x3xf32>", "digital.bias_add"),
            )
        )

    def test_linear_without_bias(self):
        lowered = self.assert_lowers(
            LoweringCase(
                name="linear_without_bias",
                model_factory=lambda: LinearModel(bias=False),
                input_factory=lambda: (torch.ones(1, 4),),
                expected_fragments=("tensor<1x3xf32>",),
            )
        )
        self.assertNotIn("digital.bias_add", lowered)


if __name__ == "__main__":
    import unittest

    unittest.main()
