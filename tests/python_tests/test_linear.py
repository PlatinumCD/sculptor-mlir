#!/usr/bin/env python3
"""Python-backed Linear lowering test."""

import re

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


class TwoLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(8, 8, bias=False)
        self.second = torch.nn.Linear(8, 8, bias=False)
        initialize_parameters(self)

    def forward(self, value):
        return self.second(self.first(value))


class ReusedLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
        initialize_parameters(self)

    def forward(self, value):
        return self.linear(self.linear(value))


class LinearLoweringTest(LoweringTestCase):
    def test_two_linear_layers_have_distinct_matrix_setup_stages(self):
        lowered = self.assert_lowers(
            LoweringCase(
                name="two_linear_layers",
                model_factory=TwoLinearModel,
                input_factory=lambda: (torch.ones(1, 8),),
                expected_fragments=("forward_mvm_0_matrix_tile_0_0",),
                minimum_matrix_setups=2,
            )
        )

        setup_lines = [
            line for line in lowered.splitlines()
            if "sculptor.array.set " in line
        ]
        self.assertEqual(len(setup_lines), 2)
        setup_ids = {
            int(match.group(1))
            for line in setup_lines
            if (match := re.search(
                r'sculptor\.mapping\.stage_id = (\d+)', line
            ))
        }
        setup_names = {
            match.group(1)
            for line in setup_lines
            if (match := re.search(
                r'sculptor\.mapping\.stage_name = "([^"]+)"', line
            ))
        }
        self.assertEqual(len(setup_ids), 2)
        self.assertEqual(
            setup_names,
            {
                "forward_mvm_0_matrix_tile_0_0",
                "forward_mvm_1_matrix_tile_0_0",
            },
        )

    def test_duplicate_matrices_assigns_unique_replicas_to_shared_matrix(self):
        lowered = self.assert_lowers(
            LoweringCase(
                name="reused_linear_duplicate_matrices",
                model_factory=ReusedLinearModel,
                input_factory=lambda: (torch.ones(1, 8),),
                expected_fragments=("tensor<1x8xf32>",),
                minimum_matrix_setups=2,
                duplicate_matrices=True,
            )
        )

        set_arrays = re.findall(
            r"(?m)^\s*(%[\w.]+) = sculptor\.array\.set ", lowered
        )
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
        referenced_arrays = [
            re.search(r", (%[\w.]+) \{", line).group(1)
            for line in array_lines
        ]
        matrix_ids = {
            int(re.search(r"sculptor\.matrix_id = (\d+)", line).group(1))
            for line in array_lines
        }
        replica_ids = {
            int(
                re.search(
                    r"sculptor\.matrix_replica_id = (\d+)", line
                ).group(1)
            )
            for line in array_lines
        }

        self.assertEqual(len(set_arrays), 2)
        self.assertEqual(len(set(set_arrays)), 2)
        self.assertEqual(len(array_lines), 6)
        self.assertEqual(referenced_arrays[:3], [set_arrays[0]] * 3)
        self.assertEqual(referenced_arrays[3:], [set_arrays[1]] * 3)
        self.assertEqual(matrix_ids, {0})
        self.assertEqual(replica_ids, {0, 1})

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
