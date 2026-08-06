#!/usr/bin/env python3
"""Python-backed RNN cell and stacked-RNN lowering tests."""

import torch

from lowering_harness import (
    LoweringCase,
    LoweringTestCase,
    initialize_parameters,
)


class RNNCellModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.cell = torch.nn.RNNCell(4, 3, bias=bias)
        initialize_parameters(self)

    def forward(self, value, hidden):
        return self.cell(value, hidden)


class RNNModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.rnn = torch.nn.RNN(
            4, 3, num_layers=2, bias=bias, batch_first=True
        )
        initialize_parameters(self)

    def forward(self, value, hidden):
        return self.rnn(value, hidden)


class RNNLoweringTest(LoweringTestCase):
    def test_rnn_cell_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "rnn_cell_with_bias",
                lambda: RNNCellModel(bias=True),
                lambda: (torch.ones(1, 4), torch.ones(1, 3)),
                ("tensor<1x3xf32>",),
                minimum_matrix_setups=1,
            )
        )

    def test_rnn_cell_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "rnn_cell_without_bias",
                lambda: RNNCellModel(bias=False),
                lambda: (torch.ones(1, 4), torch.ones(1, 3)),
                ("tensor<1x3xf32>",),
                minimum_matrix_setups=1,
            )
        )

    def test_rnn_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "rnn_with_bias",
                lambda: RNNModel(bias=True),
                lambda: (torch.ones(1, 3, 4), torch.ones(2, 1, 3)),
                ("tensor<1x3x3xf32>", "tensor<2x1x3xf32>"),
                minimum_matrix_setups=2,
            )
        )

    def test_rnn_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "rnn_without_bias",
                lambda: RNNModel(bias=False),
                lambda: (torch.ones(1, 3, 4), torch.ones(2, 1, 3)),
                ("tensor<1x3x3xf32>", "tensor<2x1x3xf32>"),
                minimum_matrix_setups=2,
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
