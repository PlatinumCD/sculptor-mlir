#!/usr/bin/env python3
"""Python-backed Linear lowering test."""

import os
from pathlib import Path
import re
import subprocess
import tempfile

import torch

from lowering_harness import (
    DEFAULT_SCULPTOR_OPT,
    LoweringCase,
    LoweringTestCase,
    initialize_parameters,
    lower_to_tile_object,
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


class TwoTokenLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(384, 1152, bias=True)
        initialize_parameters(self)

    def forward(self, value):
        first = self.linear(value[:, 0:1, :])
        second = self.linear(value[:, 1:2, :])
        return torch.cat((first, second), dim=1)


class LinearLoweringTest(LoweringTestCase):
    def test_one_worker_spread_routine_graph_is_acyclic(self):
        case = LoweringCase(
            name="two_token_linear_one_worker",
            model_factory=TwoTokenLinear,
            input_factory=lambda: (torch.ones(1, 2, 384),),
            array_rows=1024,
            array_cols=512,
            mesh_rows=8,
            mesh_cols=8,
            arrays_per_core=4,
            external_linalg_lowering=True,
            duplicate_matrices=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            output = lower_to_tile_object(
                case,
                output_dir=output_dir,
                digital_workers=1,
                mapping_strategies="setup-first,recursive-fork-join",
                mvm_body_policy="spread",
                setup_binding_policy="consumer-anchored",
                balance_digital_work=True,
                compile_all_active_tiles=True,
            )
            self.assertEqual(output.read_bytes()[:4], b"\x7fELF")

            outlined = (output_dir / "04-outlined.mlir").read_text()
            active_tile_ids = sorted(
                {
                    int(tile_id)
                    for tile_id in re.findall(
                        r"(?m)^\s*module @tile_(\d+)(?:\s|$)", outlined
                    )
                }
            )
            self.assertEqual(active_tile_ids, [0, 1, 2, 3])
            for tile_id in active_tile_ids:
                object_path = output_dir / f"core-{tile_id}.o"
                self.assertEqual(object_path.read_bytes()[:4], b"\x7fELF")

            placed = output_dir / "03-placed.mlir"
            sculptor_opt = Path(
                os.environ.get("SCULPTOR_MLIR_OPT", DEFAULT_SCULPTOR_OPT)
            )
            repeated = subprocess.run(
                [
                    str(sculptor_opt),
                    str(placed),
                    "--sculptor-outline-tile-routines",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(repeated.returncode, 0, repeated.stderr)

        record_pattern = re.compile(
            r"#sculptor\.tile_routine_(?:route|binding)<[^>]+>"
        )
        records = sorted(set(record_pattern.findall(outlined)))
        repeated_records = sorted(set(record_pattern.findall(repeated.stdout)))
        self.assertTrue(records)
        self.assertEqual(records, repeated_records)

        edges = set()
        for record in records:
            source = re.search(r"sourceRoutine = (\d+) : i64", record)
            destination = re.search(
                r"destinationRoutine = (\d+) : i64", record
            )
            self.assertIsNotNone(source)
            self.assertIsNotNone(destination)
            edges.add((int(source.group(1)), int(destination.group(1))))

        nodes = {node for edge in edges for node in edge}
        indegree = {node: 0 for node in nodes}
        consumers = {node: [] for node in nodes}
        for source, destination in edges:
            indegree[destination] += 1
            consumers[source].append(destination)
        ready = sorted(node for node in nodes if indegree[node] == 0)
        emitted = 0
        while ready:
            current = ready.pop(0)
            emitted += 1
            for consumer in sorted(consumers[current]):
                indegree[consumer] -= 1
                if indegree[consumer] == 0:
                    ready.append(consumer)
                    ready.sort()
        self.assertEqual(emitted, len(nodes))

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
