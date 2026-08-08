#!/usr/bin/env python3
"""Python-backed Transformer block lowering test."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

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


class DualUseParallelLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(8, 8, bias=False)
        self.right = torch.nn.Linear(8, 8, bias=False)
        initialize_parameters(self)

    def forward(self, value):
        left = self.left(value)
        right = self.right(value)
        return left + right, left * right


class TransformerLoweringTest(LoweringTestCase):
    @staticmethod
    def ra_tree_report(placed_ir: str) -> dict:
        repository_root = Path(__file__).resolve().parents[2]
        analog_root = repository_root.parent.parent
        report_tool = Path(
            os.environ.get(
                "SCULPTOR_RA_TREE_REPORT",
                analog_root
                / "build"
                / "sculptor-mlir-pivot"
                / "bin"
                / "sculptor-ra-tree-report",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "placed.mlir"
            json_path = Path(directory) / "report.json"
            html_path = Path(directory) / "report.html"
            input_path.write_text(placed_ir)
            result = subprocess.run(
                [
                    str(report_tool),
                    str(input_path),
                    f"--json-output={json_path}",
                    "-o",
                    str(html_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                raise AssertionError(f"RA-tree report failed:\n{result.stderr}")
            return json.loads(json_path.read_text())

    @staticmethod
    def stage_operation_id(placed_ir: str, stage_name: str) -> int:
        return TransformerLoweringTest.stage_operation_id_matching(
            placed_ir, re.escape(stage_name)
        )

    @staticmethod
    def stage_operation_id_matching(placed_ir: str, stage_name: str) -> int:
        match = re.search(
            r"sculptor\.mapping\.operation_id = ([0-9]+) : i64[^}\n]*"
            rf'sculptor\.mapping\.stage_name = "{stage_name}"',
            placed_ir,
        )
        if match is None:
            raise AssertionError(f"missing mapping stage {stage_name}")
        return int(match.group(1))

    @staticmethod
    def operation_logical_tiles(placed_ir: str, operation_id: int) -> set[int]:
        marker = "#sculptor.logical_tile<tileId = "
        result: set[int] = set()
        for chunk in placed_ir.split(marker)[1:]:
            tile_match = re.match(r"([0-9]+) : i64", chunk)
            if tile_match is None:
                continue
            tile_body = chunk.split("], edges = [", 1)[0]
            if re.search(
                rf"operationId = {operation_id} : i64(?:,|>)", tile_body
            ):
                result.add(int(tile_match.group(1)))
        return result

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

    def test_recursive_groups_frontier_equivalent_mvm_waves(self):
        placed_ir = lower_to_logical_tile_placement(
            LoweringCase(
                "frontier_equivalent_mvm_waves",
                DualUseParallelLinearModel,
                lambda: (torch.ones(1, 8),),
                array_rows=8,
                array_cols=8,
            ),
            tile_order="priority",
            priority_mode="max",
            candidate_scope="frontier",
            lookahead=1,
            mapping_strategies="setup-first,recursive-fork-join",
            mvm_body_policy="spread",
        )
        report = self.ra_tree_report(placed_ir)
        nodes = report["functions"][0]["tree"]["nodes"]
        nodes_by_id = {node["id"]: node for node in nodes}

        wave_parents = []
        for wave_id in (0, 1):
            roots = [
                node
                for node in nodes
                if node["mvm_wave_id"] == wave_id
                and node["kind"] == "temporal_cut"
                and nodes_by_id[node["parent_id"]]["mvm_wave_id"] == -1
            ]
            self.assertEqual(len(roots), 1)
            wave_parents.append(nodes_by_id[roots[0]["parent_id"]])

        self.assertEqual(wave_parents[0]["id"], wave_parents[1]["id"])
        self.assertEqual(wave_parents[0]["kind"], "spatial_cut")

    def test_recursive_mvm_body_policies(self):
        case = LoweringCase(
            "transformer_recursive_mvm_body_policy",
            lambda: TransformerBlockModel(bias=True),
            lambda: (torch.ones(1, 4, 384),),
            array_rows=1024,
            array_cols=512,
            external_linalg_lowering=True,
            duplicate_matrices=True,
        )
        for policy in ("packed", "spread"):
            with self.subTest(policy=policy):
                placed_ir = lower_to_logical_tile_placement(
                    case,
                    tile_order="priority",
                    priority_mode="max",
                    candidate_scope="frontier",
                    lookahead=3,
                    mapping_strategies="setup-first,recursive-fork-join",
                    mvm_body_policy=policy,
                )
                self.assertIn(
                    f'sculptor.mapping.mvm_body_policy = "{policy}"',
                    placed_ir,
                )
                if policy != "spread":
                    continue

                for token in range(4):
                    home_names = (
                        f"transformer_block_mlp_down_token_extract_b0_s{token}",
                        f"transformer_block_mlp_down_bias_add_b0_s{token}",
                    )
                    home_tiles = []
                    member_tiles = []
                    for member in range(3):
                        vector_id = self.stage_operation_id_matching(
                            placed_ir,
                            f"transformer_block_mlp_down_b0_s{token}_mvm_"
                            rf"[0-9]+_vector_tile_{member}",
                        )
                        mvm_id = self.stage_operation_id_matching(
                            placed_ir,
                            f"transformer_block_mlp_down_b0_s{token}_mvm_"
                            rf"[0-9]+_array_0_{member}",
                        )
                        vector_tiles = self.operation_logical_tiles(
                            placed_ir, vector_id
                        )
                        mvm_tiles = self.operation_logical_tiles(placed_ir, mvm_id)
                        self.assertEqual(vector_tiles, mvm_tiles)
                        self.assertEqual(len(vector_tiles), 1)
                        member_tiles.append(next(iter(vector_tiles)))

                    for name in home_names:
                        operation_id = self.stage_operation_id(placed_ir, name)
                        tiles = self.operation_logical_tiles(placed_ir, operation_id)
                        self.assertEqual(len(tiles), 1)
                        home_tiles.append(next(iter(tiles)))
                    recombine_id = self.stage_operation_id_matching(
                        placed_ir,
                        f"transformer_block_mlp_down_b0_s{token}_mvm_"
                        r"[0-9]+_tile_recombine",
                    )
                    recombine_tiles = self.operation_logical_tiles(
                        placed_ir, recombine_id
                    )
                    self.assertEqual(len(recombine_tiles), 1)
                    home_tiles.append(next(iter(recombine_tiles)))
                self.assertEqual(len(set(home_tiles)), 1)
                self.assertEqual(home_tiles[0], member_tiles[0])
                self.assertEqual(len(set(member_tiles)), 3)

    def test_consumer_anchored_setup_bindings_follow_ra_flow(self):
        case = LoweringCase(
            "transformer_consumer_anchored_setup_bindings",
            lambda: TransformerBlockModel(bias=True),
            lambda: (torch.ones(1, 4, 384),),
            array_rows=1024,
            array_cols=512,
            external_linalg_lowering=True,
            duplicate_matrices=True,
        )
        placed_ir = lower_to_logical_tile_placement(
            case,
            tile_order="priority",
            priority_mode="max",
            candidate_scope="frontier",
            lookahead=3,
            mapping_strategies="setup-first,recursive-fork-join",
            mvm_body_policy="spread",
            setup_binding_policy="consumer-anchored",
        )
        self.assertIn(
            'sculptor.mapping.setup_binding_policy = "consumer-anchored"',
            placed_ir,
        )

        head_recombine_id = self.stage_operation_id(
            placed_ir, "transformer_block_self_head_recombine"
        )
        head_recombine_tiles = self.operation_logical_tiles(
            placed_ir, head_recombine_id
        )
        self.assertTrue(head_recombine_tiles)

        output_body_tiles = []
        for token in range(4):
            vector_id = self.stage_operation_id_matching(
                placed_ir,
                f"transformer_block_attn_output_b0_s{token}_mvm_"
                r"[0-9]+_vector_tile_0",
            )
            setup_id = self.stage_operation_id_matching(
                placed_ir,
                f"transformer_block_attn_output_b0_s{token}_mvm_"
                r"[0-9]+_array_0_0_matrix_setup_replica_[0-9]+",
            )
            mvm_id = self.stage_operation_id_matching(
                placed_ir,
                f"transformer_block_attn_output_b0_s{token}_mvm_"
                r"[0-9]+_array_0_0",
            )
            vector_tiles = self.operation_logical_tiles(placed_ir, vector_id)
            setup_tiles = self.operation_logical_tiles(placed_ir, setup_id)
            mvm_tiles = self.operation_logical_tiles(placed_ir, mvm_id)
            self.assertEqual(vector_tiles, setup_tiles)
            self.assertEqual(vector_tiles, mvm_tiles)
            self.assertEqual(len(vector_tiles), 1)
            output_body_tiles.append(next(iter(vector_tiles)))

        self.assertEqual(len(set(output_body_tiles)), 4)
        self.assertGreater(min(output_body_tiles), max(head_recombine_tiles))

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
