#!/usr/bin/env python3
"""Python-backed tests for every supported convolution form."""

import re
import subprocess
import tempfile
from pathlib import Path

import torch

from lowering_harness import (
    DEFAULT_SCULPTOR_OPT,
    LoweringCase,
    LoweringTestCase,
    export_linalg,
    initialize_parameters,
    lower_to_ra_tree,
    lower_to_tile_object,
)


class Conv1DModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 3, 3, bias=bias)
        initialize_parameters(self)

    def forward(self, value):
        return self.conv(value)


class Conv2DModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 3, bias=bias)
        initialize_parameters(self)

    def forward(self, value):
        return self.conv(value)


class GroupedConv2DModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, groups=2, bias=bias)
        initialize_parameters(self)

    def forward(self, value):
        return self.conv(value)


class Conv3DModel(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 2, 2, bias=bias)
        initialize_parameters(self)

    def forward(self, value):
        return self.conv(value)


class ThreeColumnPointwiseConv2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(17, 1, 1, bias=True)
        initialize_parameters(self)

    def forward(self, value):
        return torch.relu(self.conv(value))


class ChannelShardedConv2DChain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.producer = torch.nn.Conv2d(1, 4, 3, padding=1, bias=True)
        self.consumer = torch.nn.Conv2d(4, 2, 3, padding=1, bias=True)
        initialize_parameters(self)

    def forward(self, value):
        return self.consumer(torch.relu(self.producer(value)))


class PartitionedResultConv2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, bias=False)
        initialize_parameters(self)

    def forward(self, value):
        return self.conv(value)


class ConvolutionLoweringTest(LoweringTestCase):
    def test_conv2d_static_zero_padding_is_absorbed(self):
        case = LoweringCase(
            "conv2d_absorbed_padding",
            ChannelShardedConv2DChain,
            lambda: (torch.ones(1, 1, 5, 5),),
        )
        canonical = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-canonicalize-layers",
            ],
            input=export_linalg(case),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(canonical.returncode, 0, canonical.stderr)
        self.assertNotIn("tensor.pad", canonical.stdout)
        self.assertEqual(canonical.stdout.count("sculptor.nn.conv2d"), 2)
        self.assertEqual(canonical.stdout.count("padding = [1, 1]"), 2)

        decomposed = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-canonicalize-layers",
                "--sculptor-extract-layers",
                "--sculptor-convert-layers",
            ],
            input=export_linalg(case),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(decomposed.returncode, 0, decomposed.stderr)
        self.assertNotIn("tensor.pad", decomposed.stdout)
        self.assertIn("scf.if", decomposed.stdout)

    def test_conv2d_layer_identity_survives_semantic_decomposition(self):
        case = LoweringCase(
            "conv2d_layer_identity",
            ChannelShardedConv2DChain,
            lambda: (torch.ones(1, 1, 5, 5),),
        )
        command = [
            str(DEFAULT_SCULPTOR_OPT),
            "-",
            "--sculptor-canonicalize-layers",
            "--sculptor-extract-layers",
            "--sculptor-convert-layers",
        ]
        result = subprocess.run(
            command,
            input=export_linalg(case),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        identity_pairs = set(
            re.findall(
                r"sculptor\.semantic\.layer_id = (\d+) : i64, "
                r'sculptor\.semantic\.layer_kind = "([^"]+)"',
                result.stdout,
            )
        )
        self.assertEqual(identity_pairs, {("0", "conv2d"), ("1", "conv2d")})

        mvm_parent_ids = {
            match.group(1)
            for line in result.stdout.splitlines()
            if "sculptor.mvm " in line
            and (
                match := re.search(
                    r"sculptor\.semantic\.layer_id = (\d+)", line
                )
            )
        }
        self.assertEqual(mvm_parent_ids, {"0", "1"})

        for semantic_name in (
            "conv2d_patch_sequence",
            "conv2d_mvm_sequence",
            "conv2d_output_assembly",
        ):
            matching_lines = [
                line
                for line in result.stdout.splitlines()
                if f'sculptor.semantic.name = "{semantic_name}"' in line
            ]
            self.assertTrue(matching_lines, semantic_name)
            self.assertTrue(
                all("sculptor.semantic.layer_id" in line for line in matching_lines),
                semantic_name,
            )

    def test_conv2d_layer_identity_survives_execution_expansion(self):
        case = LoweringCase(
            "conv2d_expanded_layer_identity",
            ChannelShardedConv2DChain,
            lambda: (torch.ones(1, 1, 5, 5),),
        )
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-canonicalize-layers",
                "--sculptor-extract-layers",
                "--sculptor-convert-layers",
                "--sculptor-expand-mvm-to-golem="
                "array-rows=8 array-cols=8 sequence-shard-rows=4",
                "--sculptor-duplicate-matrices",
                "--sculptor-expand-digital-work=parallel-workers=2",
            ],
            input=export_linalg(case),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        array_lines = [
            line
            for line in result.stdout.splitlines()
            if re.search(r"sculptor\.array\.(set|load|execute|store)", line)
        ]
        self.assertTrue(array_lines)
        self.assertTrue(
            all("sculptor.semantic.layer_id" in line for line in array_lines)
        )
        self.assertEqual(
            {
                match.group(1)
                for line in array_lines
                if (
                    match := re.search(
                        r"sculptor\.semantic\.layer_id = (\d+)", line
                    )
                )
            },
            {"0", "1"},
        )

        digital_stage_lines = [
            line
            for line in result.stdout.splitlines()
            if 'sculptor.semantic.section = "digital.' in line
        ]
        self.assertTrue(digital_stage_lines)
        self.assertTrue(
            all(
                "sculptor.semantic.layer_id" in line
                for line in digital_stage_lines
            )
        )

    def test_three_column_recombine_is_one_variadic_kernel(self):
        lowered = lower_to_ra_tree(
            LoweringCase(
                "conv2d_three_column_recombine",
                ThreeColumnPointwiseConv2D,
                lambda: (torch.ones(1, 17, 1, 1),),
                array_rows=8,
                array_cols=8,
                mesh_rows=2,
                mesh_cols=2,
            )
        )
        self.assertEqual(lowered.count("sculptor.array.set "), 3)
        self.assertIn("forward_mvm_0_tile_recombine", lowered)
        self.assertNotIn("linalg.add", lowered)
        self.assertRegex(
            lowered,
            r"linalg\.generic[^\n]*ins\([^\n]*, [^\n]*, [^\n]* :",
        )

    def test_conv1d_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv1d_with_bias",
                lambda: Conv1DModel(bias=True),
                lambda: (torch.ones(1, 1, 6),),
                ("tensor<1x3x4xf32>",),
            )
        )

    def test_conv1d_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv1d_without_bias",
                lambda: Conv1DModel(bias=False),
                lambda: (torch.ones(1, 1, 6),),
                ("tensor<1x3x4xf32>",),
            )
        )

    def test_conv2d_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv2d_with_bias",
                lambda: Conv2DModel(bias=True),
                lambda: (torch.ones(1, 1, 5, 5),),
                ("tensor<1x2x3x3xf32>",),
            )
        )

    def test_conv2d_duplicate_matrices_rewrites_nested_physical_mvm(self):
        lowered = self.assert_lowers(
            LoweringCase(
                "conv2d_duplicate_matrices_nested_loop",
                lambda: Conv2DModel(bias=True),
                lambda: (torch.ones(1, 1, 5, 5),),
                ("tensor<1x2x3x3xf32>",),
                array_rows=1024,
                array_cols=512,
                duplicate_matrices=True,
            )
        )

        set_matches = re.findall(
            r"(?m)^\s*(%[\w.]+) = sculptor\.array\.set ", lowered
        )
        self.assertEqual(len(set_matches), 1)
        cloned_array = set_matches[0]

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
        self.assertEqual(len(array_lines), 3)
        self.assertTrue(
            all(f", {cloned_array} " in line for line in array_lines)
        )
        self.assertTrue(
            all("sculptor.mapping.stage_id" not in line for line in array_lines)
        )
        self.assertIn("matrix_setup_replica_0", lowered)
        self.assertEqual(lowered.count("sculptor.array.set "), 1)
        self.assertTrue(
            any(
                line.lstrip().startswith("} {")
                and 'sculptor.mapping.stage_kind = "physical_mvm"' in line
                for line in lowered.splitlines()
            )
        )

    def test_conv2d_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv2d_without_bias",
                lambda: Conv2DModel(bias=False),
                lambda: (torch.ones(1, 1, 5, 5),),
                ("tensor<1x2x3x3xf32>",),
            )
        )

    def test_conv2d_sequence_shards_bound_partial_activation_rows(self):
        case = LoweringCase(
            "conv2d_sequence_shards",
            lambda: Conv2DModel(bias=True),
            lambda: (torch.ones(1, 1, 5, 5),),
            ("tensor<1x2x3x3xf32>",),
            array_rows=8,
            array_cols=8,
        )
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-canonicalize-layers",
                "--sculptor-extract-layers",
                "--sculptor-convert-layers",
                "--sculptor-expand-mvm-to-golem="
                "array-rows=8 array-cols=8 sequence-shard-rows=4",
                "--sculptor-build-ra-tree",
            ],
            input=export_linalg(case),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lowered = result.stdout
        self.assertIn("sculptor.sequence_shard_count = 3", lowered)
        self.assertEqual(
            set(re.findall(r"sculptor.sequence_shard_index = (\d+)", lowered)),
            {"0", "1", "2"},
        )
        self.assertEqual(
            set(re.findall(r"sculptor.sequence_shard_rows = (\d+)", lowered)),
            {"1", "4"},
        )
        self.assertIn("tensor<4x2xf32>", lowered)
        self.assertIn("tensor<1x2x3x3xf32>", lowered)
        self.assertIn(
            'sculptor.mapping.stage_name = '
            '"forward_mvm_0_array_0_0_sequence_shard_0"',
            lowered,
        )
        self.assertIn(
            'sculptor.mapping.stage_name = '
            '"forward_mvm_0_array_0_0_sequence_shard_2"',
            lowered,
        )
        self.assertIn(
            'sculptor.semantic.name = '
            '"conv2d_patch_sequence_sequence_shard_0_vector_tile_0"',
            lowered,
        )
        self.assertNotRegex(lowered, r"linalg\.generic[^\n]*tensor<9x9xf32>")

    def test_conv2d_sequence_shards_lower_through_tile_object(self):
        case = LoweringCase(
            "conv2d_sequence_shards_tile_object",
            lambda: Conv2DModel(bias=True),
            lambda: (torch.ones(1, 1, 5, 5),),
            ("sculptor.sequence_shard_count = 3",),
            array_rows=8,
            array_cols=8,
            sequence_shard_rows=4,
        )
        lowered = lower_to_ra_tree(case)
        self.assertIn("sculptor.sequence_shard_count = 3", lowered)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_path = lower_to_tile_object(
                case, root, compile_all_active_tiles=True
            )
            self.assertEqual(object_path.read_bytes()[:4], b"\x7fELF")

            ra_tree = (root / "01-ra.mlir").read_text()
            self.assertIn(
                "sculptor.memory.physical_vector_padding_generated", ra_tree
            )
            self.assertNotIn(
                "dense<0.000000e+00> : tensor<1x8xf32>", ra_tree
            )

            outlined = (root / "04-outlined.mlir").read_text()
            controlled_routines = re.findall(
                r"func\.func private @routine_(\d+)[^\n]*"
                r"sculptor\.deployment\.control_dependency_ids = \[[^\]]+\]",
                outlined,
            )
            self.assertEqual(len(controlled_routines), 2)
            remote_wave_tokens = re.findall(
                r"#sculptor\.tile_routine_route<[^>]*"
                r"tensorId = -1 : i64, byteSize = 4 : i64>",
                outlined,
            )
            self.assertEqual(len(remote_wave_tokens), 6)
            self.assertIn("tensor<1xi32>", outlined)

            runtime_modules = "\n".join(
                path.read_text()
                for path in sorted(root.glob("06-runtime*.mlir"))
            )
            for routine_id in controlled_routines:
                self.assertRegex(
                    runtime_modules,
                    rf"sculptor\.task\.create [^\n]*@routine_{routine_id}"
                    rf"[^\n]*deps\[%[^\]]*\]",
                )

    def test_conv2d_sequence_byte_budget_includes_patch_storage(self):
        case = LoweringCase(
            "conv2d_sequence_byte_budget",
            lambda: Conv2DModel(bias=True),
            lambda: (torch.ones(1, 1, 5, 5),),
            array_rows=8,
            array_cols=8,
            sequence_shard_bytes=128,
        )
        lowered = lower_to_ra_tree(case)
        self.assertIn("sculptor.sequence_shard_count = 5", lowered)
        self.assertEqual(
            set(re.findall(r"sculptor.sequence_shard_rows = (\d+)", lowered)),
            {"1", "2"},
        )

    def test_conv2d_sequence_wave_window_allows_two_in_flight(self):
        case = LoweringCase(
            "conv2d_sequence_two_waves_in_flight",
            lambda: Conv2DModel(bias=True),
            lambda: (torch.ones(1, 1, 5, 5),),
            array_rows=8,
            array_cols=8,
            sequence_shard_rows=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lower_to_tile_object(
                case,
                root,
                sequence_waves_in_flight=2,
            )
            outlined = (root / "04-outlined.mlir").read_text()

        self.assertIn(
            "sculptor.deployment.sequence_waves_in_flight = 2", outlined
        )
        controlled_routines = re.findall(
            r"func\.func private @routine_(\d+)[^\n]*"
            r"sculptor\.deployment\.control_dependency_ids = \[[^\]]+\]",
            outlined,
        )
        self.assertEqual(len(controlled_routines), 1)
        remote_wave_tokens = re.findall(
            r"#sculptor\.tile_routine_route<[^>]*"
            r"tensorId = -1 : i64, byteSize = 4 : i64>",
            outlined,
        )
        self.assertEqual(len(remote_wave_tokens), 3)

    def test_conv2d_sequence_wave_window_rejects_zero(self):
        case = LoweringCase(
            "conv2d_sequence_zero_waves_in_flight",
            lambda: Conv2DModel(bias=True),
            lambda: (torch.ones(1, 1, 5, 5),),
            array_rows=8,
            array_cols=8,
            sequence_shard_rows=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                AssertionError, "sequence-waves-in-flight must be positive"
            ):
                lower_to_tile_object(
                    case,
                    Path(directory),
                    sequence_waves_in_flight=0,
                )

    def test_conv2d_sequence_shards_route_through_output_layout(self):
        case = LoweringCase(
            "conv2d_sequence_sharded_output_layout",
            lambda: Conv2DModel(bias=True),
            lambda: (torch.ones(1, 1, 8, 8),),
            array_rows=8,
            array_cols=8,
            sequence_shard_rows=4,
            mesh_rows=2,
            mesh_cols=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lower_to_tile_object(
                case,
                root,
                digital_workers=2,
                digital_dataflow="sharded",
                compile_all_active_tiles=True,
            )
            outlined = (root / "04-outlined.mlir").read_text()

        self.assertNotIn("mvm_sequence_shard_assembly", outlined)
        self.assertNotIn("tensor.concat", outlined)
        self.assertRegex(
            outlined,
            r"func\.func private @routine_\d+\([^\n]+\) -> "
            r"\(tensor<4x1xf32>, tensor<4x1xf32>\)",
        )
        self.assertRegex(
            outlined,
            r"func\.func private @routine_\d+\("
            + r"%arg\d+: tensor<4x1xf32>, " * 8
            + r"%arg\d+: tensor<4x1xf32>\) -> tensor<1x1x6x6xf32>",
        )

    def test_partitioned_analog_results_write_directly_to_destinations(self):
        case = LoweringCase(
            "conv2d_direct_partitioned_results",
            PartitionedResultConv2D,
            lambda: (torch.ones(1, 3, 64, 64),),
            array_rows=1024,
            array_cols=512,
            sequence_shard_rows=1024,
            mesh_rows=2,
            mesh_cols=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_path = lower_to_tile_object(
                case,
                root,
                digital_workers=2,
                digital_dataflow="sharded",
            )
            object_magic = object_path.read_bytes()[:4]
            llvm_dialect = (root / "08-llvm.mlir").read_text()

        self.assertEqual(object_magic, b"\x7fELF")
        self.assertEqual(
            llvm_dialect.count("sculptor.memory.partitioned_result_direct"),
            4,
        )
        self.assertEqual(
            llvm_dialect.count(
                "sculptor.memory.direct_physical_store_view_count = 1"
            ),
            4,
        )

    def test_conv2d_patch_reads_channel_shards_without_full_assembly(self):
        case = LoweringCase(
            "conv2d_channel_sharded_patch_input",
            ChannelShardedConv2DChain,
            lambda: (torch.ones(1, 1, 8, 8),),
            array_rows=8,
            array_cols=8,
            sequence_shard_rows=4,
            mesh_rows=4,
            mesh_cols=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lower_to_tile_object(
                case,
                root,
                digital_workers=2,
                digital_dataflow="sharded",
                compile_all_active_tiles=True,
            )
            outlined = (root / "04-outlined.mlir").read_text()

        rewritten = outlined.count(
            "sculptor.memory.full_activation_assembly_elided"
        )
        self.assertGreater(rewritten, 0)
        self.assertIn(
            "sculptor.memory.routed_activation_channel_shards = 1",
            outlined,
        )
        self.assertNotRegex(
            outlined,
            r"tensor\.empty\(\)[^\n]*tensor<1x4x10x10xf32>"
            r"[\s\S]{0,500}tensor\.insert_slice",
        )

    def test_grouped_conv2d_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "grouped_conv2d_with_bias",
                lambda: GroupedConv2DModel(bias=True),
                lambda: (torch.ones(1, 4, 5, 5),),
                ("tensor<1x4x3x3xf32>",),
            )
        )

    def test_grouped_conv2d_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "grouped_conv2d_without_bias",
                lambda: GroupedConv2DModel(bias=False),
                lambda: (torch.ones(1, 4, 5, 5),),
                ("tensor<1x4x3x3xf32>",),
            )
        )

    def test_grouped_conv2d_large_spatial_shape_stays_loop_compact(self):
        case = LoweringCase(
            "grouped_conv2d_large_spatial_compact",
            lambda: GroupedConv2DModel(bias=True),
            lambda: (torch.ones(1, 4, 64, 64),),
        )
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-canonicalize-layers",
                "--sculptor-extract-layers",
                "--sculptor-convert-layers",
            ],
            input=export_linalg(case),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("sculptor.mvm "), 1)
        self.assertEqual(result.stdout.count("scf.for "), 1)
        self.assertIn(
            'sculptor.semantic.name = "conv2d_grouped_patch_sequence"',
            result.stdout,
        )
        self.assertIn(
            'sculptor.semantic.name = "conv2d_grouped_output_assembly"',
            result.stdout,
        )
        self.assertIn("linalg.generic", result.stdout)
        self.assertNotIn("tensor.concat", result.stdout)
        self.assertIn("tensor<1x4x62x62xf32>", result.stdout)

    def test_conv3d_with_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv3d_with_bias",
                lambda: Conv3DModel(bias=True),
                lambda: (torch.ones(1, 1, 4, 4, 4),),
                ("tensor<1x2x3x3x3xf32>",),
            )
        )

    def test_conv3d_without_bias(self):
        self.assert_lowers(
            LoweringCase(
                "conv3d_without_bias",
                lambda: Conv3DModel(bias=False),
                lambda: (torch.ones(1, 1, 4, 4, 4),),
                ("tensor<1x2x3x3x3xf32>",),
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
