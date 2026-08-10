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
    lower_to_ra_tree,
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


class TenArrayLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(384, 1152, bias=True)
        initialize_parameters(self)

    def forward(self, value):
        return self.linear(value)


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

    def assert_mvm_waves_follow_policy(
        self, report: dict, policy: str, arrays_per_core: int
    ) -> None:
        operation_tiles = {}
        for tile in report["logical_tile_graph"]["tiles"]:
            assignments = list(tile["digital_assignments"])
            for lane in tile["analog_lanes"]:
                assignments.extend(lane["assignments"])
            for assignment in assignments:
                operation_tiles.setdefault(assignment["operation_id"], set()).add(
                    tile["id"]
                )

        operations = {operation["id"]: operation for operation in report["operations"]}
        for wave in report["mvm_waves"]:
            physical_tiles = []
            member_tiles = {}
            for operation_id in wave["physical_mvm_operation_ids"]:
                self.assertIn(operation_id, operation_tiles)
                self.assertEqual(len(operation_tiles[operation_id]), 1)
                tile_id = next(iter(operation_tiles[operation_id]))
                physical_tiles.append(tile_id)
                member = operations[operation_id]["mvm_wave_member"]
                self.assertGreaterEqual(member, 0)
                member_tiles[member] = tile_id

            occupancy = {}
            for tile_id in physical_tiles:
                occupancy[tile_id] = occupancy.get(tile_id, 0) + 1
            expected_tiles = (
                len(physical_tiles)
                if policy == "spread"
                else (len(physical_tiles) + arrays_per_core - 1)
                // arrays_per_core
            )
            self.assertEqual(
                len(occupancy),
                expected_tiles,
                f'MVM wave {wave["id"]} has occupancy {occupancy}',
            )
            self.assertLessEqual(max(occupancy.values()), arrays_per_core)
            if policy == "spread":
                self.assertTrue(all(count == 1 for count in occupancy.values()))

            home_tile = member_tiles[min(member_tiles)]
            for operation_id in wave["vector_tile_operation_ids"]:
                self.assertIn(operation_id, operation_tiles)
                self.assertEqual(len(operation_tiles[operation_id]), 1)
                member = operations[operation_id]["mvm_wave_member"]
                expected_tile = (
                    member_tiles[member] if member >= 0 else home_tile
                )
                self.assertEqual(operation_tiles[operation_id], {expected_tile})
            for operation_id in (
                wave["recombine_operation_id"],
                wave["bias_add_operation_id"],
            ):
                if operation_id < 0:
                    continue
                self.assertEqual(operation_tiles[operation_id], {home_tile})

        for group in report["lane_binding_groups"]:
            group_tiles = set()
            for operation_id in group["operation_ids"]:
                self.assertIn(operation_id, operation_tiles)
                self.assertEqual(len(operation_tiles[operation_id]), 1)
                group_tiles.update(operation_tiles[operation_id])
            self.assertEqual(
                len(group_tiles),
                1,
                f'lane-binding group {group["id"]} spans tiles {group_tiles}',
            )

    def test_expand_digital_work_dissolves_grouped_digital_stages(self):
        lowered = lower_to_ra_tree(
            LoweringCase(
                "transformer_expand_digital_work",
                lambda: TransformerBlockModel(bias=True),
                lambda: (torch.ones(1, 4, 384),),
                array_rows=1024,
                array_cols=512,
                external_linalg_lowering=True,
                duplicate_matrices=True,
            ),
            digital_workers=12,
        )

        self.assertIn(
            "sculptor.mapping.digital_parallel_workers = 12 : i64", lowered
        )
        self.assertRegex(
            lowered,
            r"sculptor\.mapping\.expanded_digital_operation_count = [1-9][0-9]*",
        )
        self.assertIn("sculptor.mapping.expanded_digital_work =", lowered)
        self.assertNotIn(
            'sculptor.mapping.stage_kind = "digital_stage"', lowered
        )
        self.assertIn(
            'sculptor.mapping.stage_kind = "physical_mvm"', lowered
        )

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

    def test_recursive_serializes_shared_matrix_consumers_without_duplication(
        self,
    ):
        placed_ir = lower_to_logical_tile_placement(
            LoweringCase(
                "transformer_recursive_shared_matrices",
                lambda: TransformerBlockModel(bias=True),
                lambda: (torch.ones(1, 4, 384),),
                array_rows=1024,
                array_cols=512,
                external_linalg_lowering=True,
            ),
            tile_order="priority",
            priority_mode="max",
            candidate_scope="frontier",
            lookahead=3,
            mapping_strategies="setup-first,recursive-fork-join",
            mvm_body_policy="spread",
            setup_binding_policy="consumer-anchored",
            digital_workers=8,
            balance_digital_work=True,
        )
        report = self.ra_tree_report(placed_ir)["functions"][0]
        self.assert_mvm_waves_follow_policy(report, "spread", 4)
        nodes_by_id = {node["id"]: node for node in report["tree"]["nodes"]}
        operation_groups = {}
        for group in report["lane_binding_groups"]:
            for operation_id in group["operation_ids"]:
                operation_groups[operation_id] = group["id"]

        self.assertTrue(
            any(
                len(group["operation_ids"]) > 2
                for group in report["lane_binding_groups"]
            )
        )

        def subtree_groups(node_id):
            node = nodes_by_id[node_id]
            if node["kind"] == "leaf":
                group = operation_groups.get(node["operation_id"])
                return set() if group is None else {group}
            result = set()
            for child_id in node["child_ids"]:
                result.update(subtree_groups(child_id))
            return result

        for node in nodes_by_id.values():
            if node["kind"] != "spatial_cut":
                continue
            seen_groups = set()
            for child_id in node["child_ids"]:
                child_groups = subtree_groups(child_id)
                self.assertTrue(seen_groups.isdisjoint(child_groups))
                seen_groups.update(child_groups)

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
                report = self.ra_tree_report(placed_ir)["functions"][0]
                self.assert_mvm_waves_follow_policy(report, policy, 4)

    def test_ten_array_wave_uses_three_packed_or_ten_spread_tiles(self):
        case = LoweringCase(
            "ten_array_mvm_wave",
            TenArrayLinearModel,
            lambda: (torch.ones(1, 384),),
            array_rows=256,
            array_cols=256,
            mesh_rows=8,
            mesh_cols=8,
            arrays_per_core=4,
            external_linalg_lowering=True,
        )
        expected_occupancy = {
            "packed": [2, 4, 4],
            "spread": [1] * 10,
        }
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
                    setup_binding_policy="consumer-anchored",
                )
                report = self.ra_tree_report(placed_ir)["functions"][0]
                self.assertEqual(len(report["mvm_waves"]), 1)
                wave = report["mvm_waves"][0]
                self.assertEqual(len(wave["physical_mvm_operation_ids"]), 10)
                self.assert_mvm_waves_follow_policy(report, policy, 4)

                operation_tiles = {}
                for tile in report["logical_tile_graph"]["tiles"]:
                    for lane in tile["analog_lanes"]:
                        for assignment in lane["assignments"]:
                            operation_tiles[assignment["operation_id"]] = tile["id"]
                occupancy = {}
                for operation_id in wave["physical_mvm_operation_ids"]:
                    tile_id = operation_tiles[operation_id]
                    occupancy[tile_id] = occupancy.get(tile_id, 0) + 1
                self.assertEqual(
                    sorted(occupancy.values()), expected_occupancy[policy]
                )

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
