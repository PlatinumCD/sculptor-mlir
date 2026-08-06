#!/usr/bin/env python3
"""Python-backed GRU cell and stacked-GRU lowering tests."""

import torch

from lowering_harness import (
    LoweringCase,
    LoweringTestCase,
    initialize_parameters,
)


class GRUCellModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.cell = torch.nn.GRUCell(4, 3, bias=bias)
        initialize_parameters(self)

    def forward(self, value, hidden):
        return self.cell(value, hidden)


class GRUModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.gru = torch.nn.GRU(
            4, 3, num_layers=2, bias=bias, batch_first=True
        )
        initialize_parameters(self)

    def forward(self, value, hidden):
        return self.gru(value, hidden)


class GRULoweringTest(LoweringTestCase):
    def test_gru_cell_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "gru_cell_with_bias",
                lambda: GRUCellModel(bias=True),
                lambda: (torch.ones(1, 4), torch.ones(1, 3)),
                ("tensor<1x3xf32>",),
                minimum_matrix_setups=2,
            )
        )

    def test_gru_cell_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "gru_cell_without_bias",
                lambda: GRUCellModel(bias=False),
                lambda: (torch.ones(1, 4), torch.ones(1, 3)),
                ("tensor<1x3xf32>",),
                minimum_matrix_setups=2,
            )
        )

    def test_gru_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "gru_with_bias",
                lambda: GRUModel(bias=True),
                lambda: (torch.ones(1, 3, 4), torch.ones(2, 1, 3)),
                ("tensor<1x3x3xf32>", "tensor<2x1x3xf32>"),
                minimum_matrix_setups=4,
            )
        )

    def test_gru_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "gru_without_bias",
                lambda: GRUModel(bias=False),
                lambda: (torch.ones(1, 3, 4), torch.ones(2, 1, 3)),
                ("tensor<1x3x3xf32>", "tensor<2x1x3xf32>"),
                minimum_matrix_setups=4,
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
