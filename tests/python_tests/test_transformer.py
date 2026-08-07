#!/usr/bin/env python3
"""Python-backed Transformer block lowering test."""

import re

import torch

from lowering_harness import (
    LoweringCase,
    LoweringTestCase,
    initialize_parameters,
    lower_to_logical_tile_placement,
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
    @staticmethod
    def placement_case():
        return LoweringCase(
            "transformer_greedy_priority",
            lambda: TransformerBlockModel(bias=True),
            lambda: (torch.ones(1, 4, 384),),
            array_rows=1024,
            array_cols=512,
            external_linalg_lowering=True,
        )

    def test_greedy_priority_sum_and_max(self):
        case = self.placement_case()
        sum_ir = lower_to_logical_tile_placement(
            case, tile_order="priority", priority_mode="sum"
        )
        max_ir = lower_to_logical_tile_placement(
            case, tile_order="priority", priority_mode="max"
        )

        for mode, placed_ir in (("sum", sum_ir), ("max", max_ir)):
            self.assertIn(
                'sculptor.mapping.logical_tile_greedy_tile_order = "priority"',
                placed_ir,
            )
            self.assertIn(
                f'sculptor.mapping.logical_tile_greedy_priority_mode = "{mode}"',
                placed_ir,
            )
            self.assertRegex(
                placed_ir,
                r"logicalTileId = 0 : i64, physicalTileId = 0 : i64",
            )

        sum_score = re.search(r"totalTransferCost = ([0-9]+) : i64", sum_ir)
        max_score = re.search(r"totalTransferCost = ([0-9]+) : i64", max_ir)
        self.assertIsNotNone(sum_score)
        self.assertIsNotNone(max_score)
        self.assertGreaterEqual(int(sum_score.group(1)), 0)
        self.assertGreaterEqual(int(max_score.group(1)), 0)

    def test_greedy_candidate_scopes(self):
        case = self.placement_case()
        for scope in ("cardinal", "diagonal", "frontier"):
            with self.subTest(scope=scope):
                placed_ir = lower_to_logical_tile_placement(
                    case,
                    tile_order="priority",
                    priority_mode="max",
                    candidate_scope=scope,
                )
                self.assertIn(
                    "sculptor.mapping.logical_tile_greedy_candidate_scope = "
                    f'"{scope}"',
                    placed_ir,
                )
                assignments = re.findall(
                    r"logicalTileId = ([0-9]+) : i64, "
                    r"physicalTileId = ([0-9]+) : i64",
                    placed_ir,
                )
                self.assertTrue(assignments)
                self.assertEqual(
                    len(assignments), len({item[1] for item in assignments})
                )
                self.assertIn(("0", "0"), assignments)

    def test_invalid_greedy_candidate_scope(self):
        with self.assertRaisesRegex(
            AssertionError, "unknown greedy candidate scope 'invalid'"
        ):
            lower_to_logical_tile_placement(
                self.placement_case(),
                tile_order="priority",
                priority_mode="sum",
                candidate_scope="invalid",
            )

    def test_greedy_lookahead_depths(self):
        case = self.placement_case()
        for lookahead in (1, 2, 3):
            with self.subTest(lookahead=lookahead):
                placed_ir = lower_to_logical_tile_placement(
                    case,
                    tile_order="priority",
                    priority_mode="max",
                    candidate_scope="frontier",
                    lookahead=lookahead,
                )
                self.assertIn(
                    "sculptor.mapping.logical_tile_greedy_lookahead = "
                    f"{lookahead} : i64",
                    placed_ir,
                )
                assignments = re.findall(
                    r"logicalTileId = ([0-9]+) : i64, "
                    r"physicalTileId = ([0-9]+) : i64",
                    placed_ir,
                )
                self.assertTrue(assignments)
                self.assertEqual(
                    len(assignments), len({item[1] for item in assignments})
                )

    def test_invalid_greedy_lookahead(self):
        with self.assertRaisesRegex(AssertionError, "lookahead must be positive"):
            lower_to_logical_tile_placement(
                self.placement_case(),
                tile_order="priority",
                priority_mode="sum",
                candidate_scope="frontier",
                lookahead=0,
            )

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
