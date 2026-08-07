#!/usr/bin/env python3
"""Python-backed tests for every supported convolution form."""

import re

import torch

from lowering_harness import (
    LoweringCase,
    LoweringTestCase,
    initialize_parameters,
)


class Conv1DModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 3, 3, bias=bias)
        initialize_parameters(self)

    def forward(self, value):
        return self.conv(value)


class Conv2DModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 3, bias=bias)
        initialize_parameters(self)

    def forward(self, value):
        return self.conv(value)


class GroupedConv2DModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, groups=2, bias=bias)
        initialize_parameters(self)

    def forward(self, value):
        return self.conv(value)


class Conv3DModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 2, 2, bias=bias)
        initialize_parameters(self)

    def forward(self, value):
        return self.conv(value)


class ConvolutionLoweringTest(LoweringTestCase):
    def test_conv1d_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv1d_with_bias",
                lambda: Conv1DModel(bias=True),
                lambda: (torch.ones(1, 1, 6),),
                ("tensor<1x3x4xf32>",),
            )
        )

    def test_conv1d_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv1d_without_bias",
                lambda: Conv1DModel(bias=False),
                lambda: (torch.ones(1, 1, 6),),
                ("tensor<1x3x4xf32>",),
            )
        )

    def test_conv2d_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv2d_with_bias",
                lambda: Conv2DModel(bias=True),
                lambda: (torch.ones(1, 1, 5, 5),),
                ("tensor<1x2x3x3xf32>",),
            )
        )

    def test_conv2d_duplicate_matrices_rewrites_nested_physical_mvm(self):
        lowered = self.assert_lowers(
            LoweringCase(
                "conv2d_duplicate_matrices_nested_loop",
                lambda: Conv2DModel(bias=True),
                lambda: (torch.ones(1, 1, 5, 5),),
                ("tensor<1x2x3x3xf32>",),
                array_rows=1024,
                array_cols=512,
                duplicate_matrices=True,
            )
        )

        set_matches = re.findall(
            r"(?m)^\s*(%[\w.]+) = sculptor\.array\.set ", lowered
        )
        self.assertEqual(len(set_matches), 1)
        cloned_array = set_matches[0]

        array_lines = [
            line
            for line in lowered.splitlines()
            if any(
                operation in line
                for operation in (
                    "sculptor.array.load ",
                    "sculptor.array.execute ",
                    "sculptor.array.store ",
                )
            )
        ]
        self.assertEqual(len(array_lines), 3)
        self.assertTrue(
            all(f", {cloned_array} " in line for line in array_lines)
        )
        self.assertTrue(
            all("sculptor.mapping.stage_id" not in line for line in array_lines)
        )
        self.assertIn("matrix_setup_replica_0", lowered)
        self.assertEqual(lowered.count("sculptor.array.set "), 1)
        self.assertTrue(
            any(
                line.lstrip().startswith("} {")
                and 'sculptor.mapping.stage_kind = "physical_mvm"' in line
                for line in lowered.splitlines()
            )
        )

    def test_conv2d_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv2d_without_bias",
                lambda: Conv2DModel(bias=False),
                lambda: (torch.ones(1, 1, 5, 5),),
                ("tensor<1x2x3x3xf32>",),
            )
        )

    def test_grouped_conv2d_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "grouped_conv2d_with_bias",
                lambda: GroupedConv2DModel(bias=True),
                lambda: (torch.ones(1, 4, 5, 5),),
                ("tensor<1x4x3x3xf32>",),
            )
        )

    def test_grouped_conv2d_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "grouped_conv2d_without_bias",
                lambda: GroupedConv2DModel(bias=False),
                lambda: (torch.ones(1, 4, 5, 5),),
                ("tensor<1x4x3x3xf32>",),
            )
        )

    def test_conv3d_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv3d_with_bias",
                lambda: Conv3DModel(bias=True),
                lambda: (torch.ones(1, 1, 4, 4, 4),),
                ("tensor<1x2x3x3x3xf32>",),
            )
        )

    def test_conv3d_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv3d_without_bias",
                lambda: Conv3DModel(bias=False),
                lambda: (torch.ones(1, 1, 4, 4, 4),),
                ("tensor<1x2x3x3x3xf32>",),
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
