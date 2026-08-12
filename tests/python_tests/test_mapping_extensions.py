#!/usr/bin/env python3
"""Focused regressions for optional mapping extensions."""

from __future__ import annotations

import hashlib
import json
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


class ThreeWayReductionLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 2, bias=False)
        initialize_parameters(self)

    def forward(self, value):
        return self.linear(value)


class ElementwiseChain(torch.nn.Module):
    def forward(self, left, right):
        summed = left + right
        return torch.relu(summed)


class TransposeSandwich(torch.nn.Module):
    def forward(self, value):
        produced = value + 1.0
        transposed = produced.transpose(0, 1).contiguous()
        return transposed + 2.0


class LayerNormOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, value):
        return self.norm(value)


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


def run_serialized_pipeline(
    case: LoweringCase, producing_passes: list[str], consuming_passes: list[str]
) -> str:
    produced = run_pipeline(case, producing_passes)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "placed.mlir"
        path.write_text(produced)
        result = subprocess.run(
            [str(DEFAULT_SCULPTOR_OPT), str(path), *consuming_passes],
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode:
        raise AssertionError(
            f"{case.name} failed after serialization with exit code "
            f"{result.returncode}:\n{result.stderr}"
        )
    return result.stdout


def build_ra_tree_report(module: str) -> dict:
    report_tool = DEFAULT_SCULPTOR_OPT.parent / "sculptor-ra-tree-report"
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "placed.mlir"
        json_path = Path(directory) / "report.json"
        html_path = Path(directory) / "report.html"
        input_path.write_text(module)
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

    def test_incremental_makespan_annealing_matches_full_evaluator(self):
        prefix = [
            *linear_prefix(),
            "--sculptor-build-ra-tree",
            (
                "--sculptor-plan-mapping="
                "strategies=setup-first,recursive-fork-join "
                "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                "array-rows=8 array-cols=2 "
                f"cost-profile={COST_PROFILE}"
            ),
        ]
        placement = (
            "schedule=annealing annealing-initial-schedule=greedy "
            "annealing-iterations=200 annealing-initial-temperature=0 "
            "annealing-cooling-rate=0.995 "
            "annealing-trace-sample-interval=20 random-seed=7 "
            "objective=makespan network-mode=full timing-scope=warm "
            "temporal-candidate-limit=4 "
            "greedy-tile-order=priority greedy-priority-mode=max "
            "greedy-candidate-scope=frontier greedy-lookahead=2 "
            "mesh-rows=4 mesh-cols=4 arrays-per-core=4"
        )
        full = run_pipeline(
            self.linear_case,
            [
                *prefix,
                (
                    "--sculptor-place-logical-tiles="
                    f"{placement} annealing-incremental-makespan=false"
                ),
            ],
        )
        incremental = run_pipeline(
            self.linear_case,
            [
                *prefix,
                (
                    "--sculptor-place-logical-tiles="
                    f"{placement} annealing-incremental-makespan=true "
                    "annealing-makespan-verify-interval=1"
                ),
            ],
        )

        self.assertEqual(incremental, full)
        self.assertIn("evaluations = 200 : i64", incremental)
        self.assertIn("logical_tile_annealing_trace", incremental)

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

    def test_temporal_placement_survives_serialization_before_outlining(self):
        for index, (network_mode, timing_scope) in enumerate(
            (network, scope)
            for network in ("ideal", "finite", "full")
            for scope in ("warm", "cold")
        ):
            with self.subTest(
                dataflow="bulk",
                network_mode=network_mode,
                timing_scope=timing_scope,
            ):
                outlined = run_serialized_pipeline(
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
                            "schedule=greedy objective=makespan "
                            f"network-mode={network_mode} "
                            f"timing-scope={timing_scope} "
                            f"temporal-candidate-limit={1 + index} "
                            "mesh-rows=4 mesh-cols=4 arrays-per-core=4"
                        ),
                    ],
                    ["--sculptor-outline-tile-routines"],
                )
                self.assertIn("module @tile_", outlined)

        sharded_case = LoweringCase(
            name="serialized_sharded_elementwise_chain",
            model_factory=ElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8), torch.ones(4, 8)),
            minimum_matrix_setups=0,
        )
        outlined = run_serialized_pipeline(
            sharded_case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2 "
                    f"cost-profile={COST_PROFILE}"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy objective=makespan network-mode=full "
                    "timing-scope=warm temporal-candidate-limit=4 "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4"
                ),
            ],
            ["--sculptor-outline-tile-routines"],
        )
        self.assertIn("module @tile_", outlined)

        outlined = run_serialized_pipeline(
            self.linear_case,
            [
                *linear_prefix(),
                (
                    "--sculptor-expand-digital-work="
                    "parallel-workers=1 dataflow=sharded "
                    "reduction-tree=balanced reduction-fan-in=2 "
                    "reduction-min-width=3"
                ),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2 "
                    f"cost-profile={COST_PROFILE}"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy objective=makespan network-mode=finite "
                    "timing-scope=cold temporal-candidate-limit=4 "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4"
                ),
            ],
            ["--sculptor-outline-tile-routines"],
        )
        self.assertIn("sculptor.task.reduction_tree_id", outlined)

    def test_temporal_placement_reports_specific_metadata_mismatch(self):
        placed = run_pipeline(
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
                    "timing-scope=warm mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
            ],
        )
        corrupted = placed.replace(
            'costProfileName = "test-calibrated-v1"',
            'costProfileName = "incorrect-profile"',
            1,
        )
        self.assertNotEqual(corrupted, placed)
        result = subprocess.run(
            [str(DEFAULT_SCULPTOR_OPT), "-", "--sculptor-outline-tile-routines"],
            input=corrupted,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "logical-tile placement cost profile name mismatch", result.stderr
        )

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

    def test_communication_aware_tiling_preserves_transpose_shards(self):
        case = LoweringCase(
            name="communication_aware_transpose_sandwich",
            model_factory=TransposeSandwich,
            input_factory=lambda: (torch.ones(8, 8),),
            minimum_matrix_setups=0,
        )
        dimension_first = run_pipeline(
            case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded "
                "tiling-policy=dimension-first",
                "--sculptor-build-ra-tree",
            ],
        )
        communication_aware = run_pipeline(
            case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded "
                "tiling-policy=communication-aware",
                "--sculptor-build-ra-tree",
            ],
        )

        self.assertIn("sculptor.mapping.shard_edge_count = 4", dimension_first)
        self.assertIn(
            "sculptor.mapping.assembly_boundary_count = 2", dimension_first
        )
        self.assertIn(
            'sculptor.mapping.digital_tiling_policy = "communication-aware"',
            communication_aware,
        )
        self.assertIn(
            "sculptor.mapping.shard_edge_count = 8", communication_aware
        )
        self.assertIn(
            "sculptor.mapping.assembly_boundary_count = 1",
            communication_aware,
        )
        exact_edges = set(
            re.findall(
                r"#sculptor\.mapping_work_unit_edge<([^>]*)>",
                communication_aware,
            )
        )
        self.assertEqual(len(exact_edges), 8)
        self.assertEqual(
            sum(
                "sourceOperationId = 0" in edge
                and "targetOperationId = 1" in edge
                for edge in exact_edges
            ),
            4,
        )
        self.assertEqual(
            sum(
                "sourceOperationId = 1" in edge
                and "targetOperationId = 2" in edge
                for edge in exact_edges
            ),
            4,
        )
        self.assertIn("resultSizes = [8, 2]", communication_aware)
        self.assertIn("resultSizes = [2, 8]", communication_aware)

    def test_consumer_bound_fill_colocates_sharded_work_units(self):
        case = LoweringCase(
            name="consumer_bound_fill_layer_norm",
            model_factory=LayerNormOnly,
            input_factory=lambda: (torch.ones(1, 4, 8),),
            minimum_matrix_setups=0,
        )
        placed = run_pipeline(
            case,
            [
                "--sculptor-canonicalize-layers",
                "--sculptor-extract-layers",
                "--sculptor-convert-layers",
                (
                    "--sculptor-expand-digital-work="
                    "parallel-workers=4 dataflow=sharded "
                    "shard-propagation-depth=0 "
                    "require-complete-shard-chain=false"
                ),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join,consumer-bound-fill "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
            ],
        )
        report = build_ra_tree_report(placed)["functions"][0]
        operations = {
            operation["id"]: operation
            for operation in report["operations"]
        }
        nodes = {node["id"]: node for node in report["tree"]["nodes"]}
        work_units = {unit["id"]: unit for unit in report["tree"]["work_units"]}
        assignments = {
            (
                assignment["operation_id"],
                nodes[assignment["leaf_id"]]["work_unit_id"],
            ): assignment
            for assignment in report["plan"]["realization"]["leaf_assignments"]
        }
        consumers = {}
        for edge in report["edges"]:
            consumers.setdefault(edge["source"], set()).add(edge["target"])

        eligible_fill_ids = [
            operation_id
            for operation_id, operation in operations.items()
            if operation["name"] == "linalg.fill"
            and len(consumers.get(operation_id, set())) == 1
        ]
        self.assertEqual(len(eligible_fill_ids), 1)
        for fill_id in eligible_fill_ids:
            consumer_id = next(iter(consumers[fill_id]))
            fill_units = [
                unit
                for unit in work_units.values()
                if unit["operation_id"] == fill_id
            ]
            consumer_units = [
                unit
                for unit in work_units.values()
                if unit["operation_id"] == consumer_id
            ]
            self.assertEqual(len(fill_units), 4)
            self.assertEqual(len(consumer_units), 4)
            for fill_unit in fill_units:
                matching_consumers = [
                    unit for unit in consumer_units
                    if unit["result_offsets"] == fill_unit["result_offsets"]
                    and unit["result_sizes"] == fill_unit["result_sizes"]
                ]
                self.assertEqual(len(matching_consumers), 1)
                consumer_unit = matching_consumers[0]
                self.assertEqual(
                    assignments[(fill_id, fill_unit["id"])]["tile_id"],
                    assignments[(consumer_id, consumer_unit["id"])]["tile_id"],
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

    def test_packed_balanced_reduction_keeps_ready_mvm_bodies_parallel(self):
        case = LoweringCase(
            name="packed_three_way_reduction",
            model_factory=ThreeWayReductionLinear,
            input_factory=lambda: (torch.ones(1, 6),),
            array_rows=8,
            array_cols=2,
        )
        placed = run_pipeline(
            case,
            [
                *linear_prefix(array_cols=2),
                (
                    "--sculptor-expand-digital-work="
                    "parallel-workers=1 reduction-tree=balanced "
                    "reduction-fan-in=2 reduction-min-width=3"
                ),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mvm-body-policy=packed "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
            ],
        )
        report = build_ra_tree_report(placed)["functions"][0]
        nodes_by_id = {node["id"]: node for node in report["tree"]["nodes"]}
        vector_ids = [
            operation["id"]
            for operation in report["operations"]
            if operation["kind"] == "vector_tile"
        ]
        mvm_ids = [
            operation["id"]
            for operation in report["operations"]
            if operation["kind"] == "physical_mvm"
        ]
        self.assertEqual(len(vector_ids), 3)
        self.assertEqual(len(mvm_ids), 3)

        leaves_by_operation = {
            node["operation_id"]: node
            for node in nodes_by_id.values()
            if node["kind"] == "leaf" and node["operation_id"] in vector_ids
        }
        body_roots = [
            nodes_by_id[leaves_by_operation[operation_id]["parent_id"]]
            for operation_id in vector_ids
        ]
        cohort_ids = {body["parent_id"] for body in body_roots}
        self.assertEqual(len(cohort_ids), 1)
        cohort = nodes_by_id[next(iter(cohort_ids))]
        self.assertEqual(cohort["kind"], "spatial_cut")

        assignments = {
            assignment["operation_id"]: assignment
            for assignment in report["plan"]["realization"]["leaf_assignments"]
            if assignment["operation_id"] in vector_ids + mvm_ids
        }
        self.assertEqual(len(assignments), 6)
        self.assertEqual(
            {assignments[operation_id]["start_ns"] for operation_id in vector_ids},
            {assignments[vector_ids[0]]["start_ns"]},
        )
        self.assertEqual(
            {assignments[operation_id]["start_ns"] for operation_id in mvm_ids},
            {assignments[mvm_ids[0]]["start_ns"]},
        )


if __name__ == "__main__":
    unittest.main()
