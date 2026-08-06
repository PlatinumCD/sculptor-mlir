#!/usr/bin/env python3
"""Python-backed LSTM cell and stacked-LSTM lowering tests."""

import torch

from lowering_harness import (
    LoweringCase,
    LoweringTestCase,
    initialize_parameters,
)


class LSTMCellModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.cell = torch.nn.LSTMCell(4, 3, bias=bias)
        initialize_parameters(self)

    def forward(self, value, hidden, cell):
        return self.cell(value, (hidden, cell))


class LSTMModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            4, 3, num_layers=2, bias=bias, batch_first=True
        )
        initialize_parameters(self)

    def forward(self, value, hidden, cell):
        return self.lstm(value, (hidden, cell))


class LSTMLoweringTest(LoweringTestCase):
    def test_lstm_cell_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "lstm_cell_with_bias",
                lambda: LSTMCellModel(bias=True),
                lambda: (
                    torch.ones(1, 4),
                    torch.ones(1, 3),
                    torch.ones(1, 3),
                ),
                ("tensor<1x3xf32>",),
                minimum_matrix_setups=2,
            )
        )

    def test_lstm_cell_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "lstm_cell_without_bias",
                lambda: LSTMCellModel(bias=False),
                lambda: (
                    torch.ones(1, 4),
                    torch.ones(1, 3),
                    torch.ones(1, 3),
                ),
                ("tensor<1x3xf32>",),
                minimum_matrix_setups=2,
            )
        )

    def test_lstm_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "lstm_with_bias",
                lambda: LSTMModel(bias=True),
                lambda: (
                    torch.ones(1, 3, 4),
                    torch.ones(2, 1, 3),
                    torch.ones(2, 1, 3),
                ),
                ("tensor<1x3x3xf32>", "tensor<2x1x3xf32>"),
                minimum_matrix_setups=4,
            )
        )

    def test_lstm_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "lstm_without_bias",
                lambda: LSTMModel(bias=False),
                lambda: (
                    torch.ones(1, 3, 4),
                    torch.ones(2, 1, 3),
                    torch.ones(2, 1, 3),
                ),
                ("tensor<1x3x3xf32>", "tensor<2x1x3xf32>"),
                minimum_matrix_setups=4,
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
