#!/usr/bin/env python3
"""Python-backed Transformer block lowering test."""

import torch

from lowering_harness import (
    LoweringCase,
    LoweringTestCase,
    initialize_parameters,
)


try:
    torch.backends.mha.set_fastpath_enabled(False)
except AttributeError:
    pass


class TransformerBlockModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        layer = torch.nn.TransformerEncoderLayer(
            d_model=384,
            nhead=6,
            dim_feedforward=1536,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=bias,
        )
        self.transformer = torch.nn.TransformerEncoder(
            layer,
            num_layers=1,
            norm=torch.nn.LayerNorm(384, bias=bias),
        )
        self.mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        initialize_parameters(self)

    def forward(self, value):
        return self.transformer(value, mask=self.mask, is_causal=True)


class TransformerLoweringTest(LoweringTestCase):
    def test_transformer_block_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "transformer_block_with_bias",
                lambda: TransformerBlockModel(bias=True),
                lambda: (torch.ones(1, 4, 384),),
                ("tensor<1x4x384xf32>", "sculptor.mapping.mvm_wave_id"),
                minimum_matrix_setups=8,
                array_rows=1024,
                array_cols=512,
                external_linalg_lowering=True,
            )
        )

    def test_transformer_block_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "transformer_block_without_bias",
                lambda: TransformerBlockModel(bias=False),
                lambda: (torch.ones(1, 4, 384),),
                ("tensor<1x4x384xf32>", "sculptor.mapping.mvm_wave_id"),
                minimum_matrix_setups=8,
                array_rows=1024,
                array_cols=512,
                external_linalg_lowering=True,
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
