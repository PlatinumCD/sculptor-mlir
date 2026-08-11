#!/usr/bin/env python3
"""Focused regressions for optional mapping extensions."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import torch

from lowering_harness import (
    DEFAULT_SCULPTOR_OPT,
    LoweringCase,
    export_linalg,
    initialize_parameters,
)


THIS_DIR = Path(__file__).resolve().parent
COST_PROFILE = THIS_DIR / "data" / "calibrated_mapping_costs.json"


class FourWayReductionLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
        initialize_parameters(self)

    def forward(self, value):
        return self.linear(value)


class ElementwiseChain(torch.nn.Module):
    def forward(self, left, right):
        summed = left + right
        return torch.relu(summed)


def run_pipeline(case: LoweringCase, passes: list[str]) -> str:
    result = subprocess.run(
        [str(DEFAULT_SCULPTOR_OPT), "-", *passes],
        input=export_linalg(case),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(
            f"{case.name} failed with exit code {result.returncode}:\n"
            f"{result.stderr}"
        )
    return result.stdout


def linear_prefix(array_cols: int = 2) -> list[str]:
    return [
        "--sculptor-canonicalize-layers",
        "--sculptor-extract-layers",
        "--sculptor-convert-layers",
        (
            "--sculptor-expand-mvm-to-golem="
            f"array-rows=8 array-cols={array_cols}"
        ),
    ]


def materialize_outlined_tiles(outlined: str) -> list[str]:
    tile_ids = sorted(
        {int(value) for value in re.findall(r"module @tile_(\d+)", outlined)}
    )
    if not tile_ids:
        raise AssertionError("outlined deployment contains no tile modules")
    materialized = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "outlined.mlir"
        path.write_text(outlined)
        for tile_id in tile_ids:
            result = subprocess.run(
                [
                    str(DEFAULT_SCULPTOR_OPT),
                    str(path),
                    f"--sculptor-extract-tile-module=tile-id={tile_id}",
                    "--sculptor-materialize-tile-runtime-graph",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                raise AssertionError(
                    f"tile {tile_id} runtime materialization failed:\n"
                    f"{result.stderr}"
                )
            materialized.append(result.stdout)
    return materialized


class MappingExtensionTest(unittest.TestCase):
    def setUp(self):
        self.linear_case = LoweringCase(
            name="mapping_extension_linear",
            model_factory=FourWayReductionLinear,
            input_factory=lambda: (torch.ones(1, 8),),
            array_rows=8,
            array_cols=2,
        )

    def test_calibrated_profile_and_makespan_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "placement.csv"
            lowered = run_pipeline(
                self.linear_case,
                [
                    *linear_prefix(),
                    "--sculptor-build-ra-tree",
                    (
                        "--sculptor-plan-mapping="
                        "strategies=setup-first mesh-rows=4 mesh-cols=4 "
                        "arrays-per-core=4 array-rows=8 array-cols=2 "
                        f"cost-profile={COST_PROFILE}"
                    ),
                    (
                        "--sculptor-place-logical-tiles="
                        "schedule=greedy objective=makespan network-mode=full "
                        "timing-scope=warm temporal-candidate-limit=4 "
                        "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                        f"summary-output={summary}"
                    ),
                ],
            )
            summary_text = summary.read_text()

        expected_hash = hashlib.sha256(COST_PROFILE.read_bytes()).hexdigest()
        self.assertIn('costProfileName = "test-calibrated-v1"', lowered)
        self.assertIn(f'costProfileHash = "{expected_hash}"', lowered)
        self.assertIn("cost_profile_name,cost_profile_hash", summary_text)
        self.assertIn(f"test-calibrated-v1,{expected_hash}", summary_text)
        self.assertIn('objective = "makespan"', lowered)
        self.assertIn('networkMode = "full"', lowered)
        self.assertIn('timingScope = "warm"', lowered)
        makespan = re.search(r"predictedMakespanNs = ([0-9.eE+-]+)", lowered)
        self.assertIsNotNone(makespan)
        self.assertGreater(float(makespan.group(1)), 0.0)

    def test_default_extension_modes_remain_legacy(self):
        lowered = run_pipeline(
            self.linear_case,
            [
                *linear_prefix(),
                "--sculptor-expand-digital-work=parallel-workers=2",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4 array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
            ],
        )

        self.assertIn('costProfileName = "legacy-v1"', lowered)
        self.assertIn('objective = "transfer-cost"', lowered)
        self.assertIn('dataflow_mode = "bulk"', lowered)
        self.assertIn('reduction_tree_policy = "none"', lowered)
        self.assertRegex(lowered, r"predictedMakespanNs = 0(?:\.0+)?")

    def test_sharded_elementwise_chain_has_exact_edges(self):
        case = LoweringCase(
            name="sharded_elementwise_chain",
            model_factory=ElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8), torch.ones(4, 8)),
            minimum_matrix_setups=0,
        )
        lowered = run_pipeline(
            case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded",
                "--sculptor-build-ra-tree",
            ],
        )

        self.assertIn('dataflow_mode = "sharded"', lowered)
        self.assertRegex(lowered, r"shardGroupId = [0-9]+")
        self.assertRegex(lowered, r"shardCount = 4")
        self.assertIn("sculptor.mapping.shard_group_count = 1", lowered)
        self.assertIn("sculptor.mapping.shard_edge_count = 4", lowered)
        self.assertIn("sculptor.mapping.assembly_boundary_count = 1", lowered)
        self.assertIn("#sculptor.mapping_work_unit_edge<", lowered)
        self.assertRegex(lowered, r"targetOperandNumber = [0-9]+")
        self.assertNotIn("tensorId = -1", lowered)

        outlined = run_pipeline(
            case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        self.assertIn("byteSize = 32 : i64", outlined)
        materialized = materialize_outlined_tiles(outlined)
        self.assertTrue(
            any("task_graph.route_input" in ir for ir in materialized)
        )
        self.assertTrue(
            any("task_graph.route_output" in ir for ir in materialized)
        )

    def test_balanced_reduction_tree_reaches_outlined_routines(self):
        lowered = run_pipeline(
            self.linear_case,
            [
                *linear_prefix(),
                (
                    "--sculptor-expand-digital-work="
                    "parallel-workers=1 reduction-tree=balanced "
                    "reduction-fan-in=2 reduction-min-width=3"
                ),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )

        self.assertIn("sculptor.mapping.reduction_tree_count = 1", lowered)
        self.assertIn("sculptor.mapping.reduction_node_count = 3", lowered)
        self.assertIn("sculptor.mapping.maximum_reduction_fan_in = 2", lowered)
        node_ids = {
            int(value)
            for value in re.findall(
                r"sculptor\.mapping\.reduction_node_id = (\d+)", lowered
            )
        }
        self.assertEqual(node_ids, {0, 1, 2})
        self.assertIn("sculptor.task.reduction_tree_id", lowered)
        self.assertIn("sculptor.task.reduction_level", lowered)
        self.assertIn("sculptor.task.reduction_width", lowered)
        materialized = materialize_outlined_tiles(lowered)
        self.assertTrue(
            any('task_kind = "digital.reduction"' in ir for ir in materialized)
        )


if __name__ == "__main__":
    unittest.main()
