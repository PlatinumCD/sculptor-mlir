#!/usr/bin/env python3
"""Shared end-to-end lowering support for Python-backed pivot tests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Callable
import unittest
import warnings


warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The tensor attributes .*_flat_weights.* were assigned during export.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self.use_nested_tensor is False.*",
    category=UserWarning,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANALOG_ROOT = REPOSITORY_ROOT.parent.parent
DEFAULT_SCULPTOR_OPT = Path(
    os.environ.get(
        "SCULPTOR_MLIR_OPT",
        ANALOG_ROOT
        / "build"
        / "sculptor-mlir-pivot"
        / "bin"
        / "sculptor-mlir-opt",
    )
)
DEFAULT_TORCH_MLIR_PACKAGE = (
    ANALOG_ROOT / "build" / "torch-mlir" / "python_packages" / "torch_mlir"
)
DEFAULT_TORCH_MLIR_OPT = ANALOG_ROOT / "build" / "torch-mlir" / "bin" / "torch-mlir-opt"

torch_mlir_package = Path(
    os.environ.get("TORCH_MLIR_PYTHON_PACKAGE", DEFAULT_TORCH_MLIR_PACKAGE)
)
if str(torch_mlir_package) not in sys.path:
    sys.path.insert(0, str(torch_mlir_package))

import torch
from torch_mlir import fx


@dataclass(frozen=True)
class LoweringCase:
    """One Python model and its observable pivot-lowering contract."""

    name: str
    model_factory: Callable[[], torch.nn.Module]
    input_factory: Callable[[], tuple[torch.Tensor, ...]]
    expected_fragments: tuple[str, ...] = ()
    minimum_matrix_setups: int = 1
    array_rows: int = 8
    array_cols: int = 8
    mesh_rows: int = 8
    mesh_cols: int = 8
    arrays_per_core: int = 4
    run_decompositions: bool = True
    preserve_layer_norm: bool = False
    external_linalg_lowering: bool = False
    duplicate_matrices: bool = False
    matrix_replication_array_capacity: int = 0
    matrix_minimum_mvms_per_replica: int = 1
    matrix_maximum_replicas_per_setup: int = 0
    sequence_shard_rows: int = 0
    sequence_shard_bytes: int = 0


def initialize_parameters(model: torch.nn.Module) -> None:
    """Give every fixture deterministic, nontrivial f32 parameters."""

    with torch.no_grad():
        for index, (name, parameter) in enumerate(model.named_parameters()):
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1.0)
            elif "norm" in name and name.endswith("bias"):
                parameter.zero_()
            else:
                values = torch.arange(
                    1, parameter.numel() + 1, dtype=torch.float32
                ).reshape_as(parameter)
                parameter.copy_(values / (100.0 + 10.0 * index))


def export_linalg(case: LoweringCase) -> str:
    """Import a Python model directly as linalg-on-tensors MLIR."""

    torch.manual_seed(0)
    model = case.model_factory().eval()
    inputs = case.input_factory()
    exported = torch.export.export(model, inputs)
    if case.run_decompositions:
        decompositions = None
        if case.preserve_layer_norm:
            decompositions = torch.export.default_decompositions()
            decompositions.pop(torch.ops.aten.layer_norm.default, None)
        exported = exported.run_decompositions(decomp_table=decompositions)
    output_type = "torch" if case.external_linalg_lowering else "linalg-on-tensors"
    module = fx.export_and_import(
        exported,
        output_type=output_type,
        func_name="forward",
    )
    module_text = str(module)
    if not case.external_linalg_lowering:
        return module_text

    torch_mlir_opt = Path(
        os.environ.get("TORCH_MLIR_OPT", DEFAULT_TORCH_MLIR_OPT)
    )
    result = subprocess.run(
        [
            str(torch_mlir_opt),
            "-",
            (
                "--pass-pipeline="
                "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)"
            ),
        ],
        input=module_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(
            f"{case.name} Torch backend lowering failed:\n{result.stderr}"
        )
    return result.stdout


def duplicate_matrices_option(case: LoweringCase) -> str:
    option = "--sculptor-duplicate-matrices"
    values: list[str] = []
    if case.matrix_replication_array_capacity:
        values.append(
            f"array-capacity={case.matrix_replication_array_capacity}"
        )
    if case.matrix_minimum_mvms_per_replica != 1:
        values.append(
            "minimum-mvms-per-replica="
            f"{case.matrix_minimum_mvms_per_replica}"
        )
    if case.matrix_maximum_replicas_per_setup:
        values.append(
            "maximum-replicas-per-setup="
            f"{case.matrix_maximum_replicas_per_setup}"
        )
    if values:
        option += "=" + " ".join(values)
    return option


def lower_to_ra_tree(case: LoweringCase, digital_workers: int | None = None) -> str:
    """Run the complete pre-placement pivot lowering for one model."""

    sculptor_opt = Path(os.environ.get("SCULPTOR_MLIR_OPT", DEFAULT_SCULPTOR_OPT))
    if not sculptor_opt.is_file():
        raise RuntimeError(f"missing sculptor-mlir-opt: {sculptor_opt}")

    command = [
        str(sculptor_opt),
        "-",
        "--sculptor-canonicalize-layers",
        "--sculptor-extract-layers",
        "--sculptor-convert-layers",
        (
            "--sculptor-expand-mvm-to-golem="
            f"array-rows={case.array_rows} array-cols={case.array_cols} "
            f"sequence-shard-rows={case.sequence_shard_rows} "
            f"sequence-shard-bytes={case.sequence_shard_bytes}"
        ),
    ]
    if case.duplicate_matrices:
        command.append(duplicate_matrices_option(case))
    if digital_workers is not None:
        command.append(
            "--sculptor-expand-digital-work="
            f"parallel-workers={digital_workers} require-change"
        )
    command.append("--sculptor-build-ra-tree")
    result = subprocess.run(
        command,
        input=export_linalg(case),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(
            f"{case.name} lowering failed with exit code {result.returncode}:\n"
            f"{result.stderr}"
        )
    return result.stdout


def lower_to_logical_tile_placement(
    case: LoweringCase,
    *,
    tile_order: str,
    priority_mode: str,
    candidate_scope: str = "cardinal",
    lookahead: int = 1,
    mapping_strategies: str = "setup-first",
    mvm_body_policy: str = "spread",
    setup_binding_policy: str = "global",
    digital_workers: int | None = None,
    balance_digital_work: bool = False,
    digital_scheduling_policy: str | None = None,
    tile_memory_capacity_bytes: int = 0,
) -> str:
    """Lower one model through configurable greedy logical-tile placement."""

    sculptor_opt = Path(os.environ.get("SCULPTOR_MLIR_OPT", DEFAULT_SCULPTOR_OPT))
    if not sculptor_opt.is_file():
        raise RuntimeError(f"missing sculptor-mlir-opt: {sculptor_opt}")

    command = [
        str(sculptor_opt),
        "-",
        "--sculptor-canonicalize-layers",
        "--sculptor-extract-layers",
        "--sculptor-convert-layers",
        (
            "--sculptor-expand-mvm-to-golem="
            f"array-rows={case.array_rows} array-cols={case.array_cols} "
            f"sequence-shard-rows={case.sequence_shard_rows} "
            f"sequence-shard-bytes={case.sequence_shard_bytes}"
        ),
    ]
    if case.duplicate_matrices:
        command.append(duplicate_matrices_option(case))
    if digital_workers is not None:
        command.append(
            "--sculptor-expand-digital-work="
            f"parallel-workers={digital_workers} require-change"
        )
    plan_options = (
        f"strategies={mapping_strategies} "
        f"mvm-body-policy={mvm_body_policy} "
        f"setup-binding-policy={setup_binding_policy} "
        f"mesh-rows={case.mesh_rows} mesh-cols={case.mesh_cols} "
        f"arrays-per-core={case.arrays_per_core} "
        f"array-rows={case.array_rows} array-cols={case.array_cols}"
    )
    if digital_scheduling_policy is not None:
        plan_options += (
            f" digital-scheduling-policy={digital_scheduling_policy}"
        )
    elif balance_digital_work:
        plan_options += " balance-digital-work"
    placement_options = (
        "--sculptor-place-logical-tiles=schedule=greedy "
        f"mesh-rows={case.mesh_rows} mesh-cols={case.mesh_cols} "
        f"arrays-per-core={case.arrays_per_core} "
        f"greedy-tile-order={tile_order} "
        f"greedy-priority-mode={priority_mode} "
        f"greedy-candidate-scope={candidate_scope} "
        f"greedy-lookahead={lookahead}"
    )
    if tile_memory_capacity_bytes:
        placement_options += (
            f" tile-memory-capacity-bytes={tile_memory_capacity_bytes}"
        )
    command.extend(
        [
            "--sculptor-build-ra-tree",
            f"--sculptor-plan-mapping={plan_options}",
            placement_options,
        ]
    )
    result = subprocess.run(
        command,
        input=export_linalg(case),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(
            f"{case.name} placement failed with exit code {result.returncode}:\n"
            f"{result.stderr}"
        )
    return result.stdout


def _run_sculptor_stage(command: list[str], stage: str) -> None:
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(
            f"{stage} failed with exit code {result.returncode}:\n"
            f"{result.stderr}"
        )


def lower_to_tile_object(
    case: LoweringCase,
    output_dir: Path | None = None,
    digital_workers: int | None = None,
    digital_dataflow: str = "bulk",
    mapping_strategies: str = "setup-first",
    mvm_body_policy: str = "spread",
    setup_binding_policy: str = "global",
    balance_digital_work: bool = False,
    digital_scheduling_policy: str | None = None,
    sequence_waves_in_flight: int = 1,
    compile_all_active_tiles: bool = False,
) -> Path:
    """Lower one Python fixture through the pivot tile ABI to a RISC-V object.

    The compiler owns the MLIR-to-object path. ELF linking remains a platform
    concern because it requires the Golem CRT, runtime library, and linker
    configuration from golem-riscv-sim.
    """

    sculptor_opt = Path(os.environ.get("SCULPTOR_MLIR_OPT", DEFAULT_SCULPTOR_OPT))
    if not sculptor_opt.is_file():
        raise RuntimeError(f"missing sculptor-mlir-opt: {sculptor_opt}")

    translate = Path(
        os.environ.get(
            "GOLEM_MLIR_TRANSLATE",
            ANALOG_ROOT.parent / "simulation" / "golem-riscv-sim" /
            "install" / "llvm" / "bin" / "mlir-translate",
        )
    )
    clang = Path(
        os.environ.get(
            "GOLEM_CLANG",
            ANALOG_ROOT.parent / "simulation" / "golem-riscv-sim" /
            "install" / "llvm" / "bin" / "clang",
        )
    )
    if not translate.is_file() or not clang.is_file():
        raise RuntimeError(
            "missing Golem LLVM toolchain; set GOLEM_MLIR_TRANSLATE and "
            "GOLEM_CLANG"
        )

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="sculptor-tile-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    def path(name: str) -> str:
        return str(output_dir / name)

    input_path = path("00-input.mlir")
    Path(input_path).write_text(export_linalg(case))
    ra_command = [
        str(sculptor_opt),
        input_path,
        "--sculptor-canonicalize-layers",
        "--sculptor-extract-layers",
        "--sculptor-convert-layers",
        (
            "--sculptor-expand-mvm-to-golem="
            f"array-rows={case.array_rows} array-cols={case.array_cols} "
            f"sequence-shard-rows={case.sequence_shard_rows} "
            f"sequence-shard-bytes={case.sequence_shard_bytes}"
        ),
    ]
    if case.duplicate_matrices:
        ra_command.append(duplicate_matrices_option(case))
    if digital_workers is not None:
        ra_command.append(
            "--sculptor-expand-digital-work="
            f"parallel-workers={digital_workers} dataflow={digital_dataflow}"
        )
    ra_command.extend(["--sculptor-build-ra-tree", "-o", path("01-ra.mlir")])
    _run_sculptor_stage(
        ra_command,
        f"{case.name} RA-tree lowering",
    )
    plan_options = (
        f"strategies={mapping_strategies} "
        f"mvm-body-policy={mvm_body_policy} "
        f"setup-binding-policy={setup_binding_policy} "
        f"mesh-rows={case.mesh_rows} mesh-cols={case.mesh_cols} "
        f"arrays-per-core={case.arrays_per_core} "
        f"array-rows={case.array_rows} array-cols={case.array_cols}"
    )
    if digital_scheduling_policy is not None:
        plan_options += (
            f" digital-scheduling-policy={digital_scheduling_policy}"
        )
    elif balance_digital_work:
        plan_options += " balance-digital-work"
    _run_sculptor_stage(
        [
            str(sculptor_opt),
            path("01-ra.mlir"),
            f"--sculptor-plan-mapping={plan_options}",
            "-o",
            path("02-plan.mlir"),
        ],
        f"{case.name} mapping planning",
    )
    _run_sculptor_stage(
        [
            str(sculptor_opt),
            path("02-plan.mlir"),
            (
                "--sculptor-place-logical-tiles=schedule=greedy "
                f"mesh-rows={case.mesh_rows} mesh-cols={case.mesh_cols} "
                f"arrays-per-core={case.arrays_per_core}"
            ),
            "-o",
            path("03-placed.mlir"),
        ],
        f"{case.name} tile placement",
    )
    _run_sculptor_stage(
        [
            str(sculptor_opt),
            path("03-placed.mlir"),
            (
                "--sculptor-outline-tile-routines="
                f"sequence-waves-in-flight={sequence_waves_in_flight}"
            ),
            "-o",
            path("04-outlined.mlir"),
        ],
        f"{case.name} routine outlining",
    )
    target = os.environ.get("GOLEM_TARGET", "riscv64-unknown-elf")
    cpu = os.environ.get("GOLEM_CPU", "golem-analog")
    abi = os.environ.get("GOLEM_ABI", "lp64d")
    tile_ids = [0]
    if compile_all_active_tiles:
        outlined = Path(path("04-outlined.mlir")).read_text()
        tile_ids = sorted(
            {
                int(tile_id)
                for tile_id in re.findall(
                    r"(?m)^\s*module @tile_(\d+)(?:\s|$)", outlined
                )
            }
        )
        if not tile_ids:
            raise AssertionError(f"{case.name} deployment has no active tiles")

    for tile_id in tile_ids:
        suffix = "" if tile_id == 0 else f"-tile-{tile_id}"
        extracted = path(f"05-extracted{suffix}.mlir")
        runtime = path(f"06-runtime{suffix}.mlir")
        finalized = path(f"07-finalized{suffix}.mlir")
        llvm_dialect = path(f"08-llvm{suffix}.mlir")
        tile_abi = path(f"09-abi{suffix}.mlir")
        llvm_ir = path(f"10{suffix}.ll")
        object_path = output_dir / f"core-{tile_id}.o"

        _run_sculptor_stage(
            [
                str(sculptor_opt),
                path("04-outlined.mlir"),
                f"--sculptor-extract-tile-module=tile-id={tile_id}",
                "-o",
                extracted,
            ],
            f"{case.name} tile {tile_id} extraction",
        )
        _run_sculptor_stage(
            [
                str(sculptor_opt),
                extracted,
                "--sculptor-materialize-tile-runtime-graph",
                "-o",
                runtime,
            ],
            f"{case.name} tile {tile_id} runtime graph materialization",
        )
        _run_sculptor_stage(
            [
                str(sculptor_opt),
                runtime,
                "--sculptor-finalize-tile-runtime-graph",
                "--sculptor-report-tile-memory=stage=finalized",
                "-o",
                finalized,
            ],
            f"{case.name} tile {tile_id} resource finalization",
        )
        _run_sculptor_stage(
            [
                str(sculptor_opt),
                finalized,
                "--sculptor-lower-golem-to-llvm-shims",
                "--canonicalize",
                "--cse",
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=bufferize-function-boundaries "
                "function-boundary-type-conversion=identity-layout-map",
                "--buffer-results-to-out-params=hoist-static-allocs",
                "--convert-bufferization-to-memref",
                "--sculptor-bind-tile-routine-destinations",
                "--buffer-hoisting",
                "--buffer-loop-hoisting",
                "--buffer-deallocation-pipeline",
                "--optimize-allocation-liveness",
                "--sculptor-audit-tile-bufferization",
                "--sculptor-vectorize-tile-copies=vector-bits=256",
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--convert-scf-to-cf",
                "--convert-vector-to-llvm",
                "--convert-math-to-libm",
                "--convert-math-to-llvm",
                "--expand-strided-metadata",
                "--lower-affine",
                "--convert-arith-to-llvm",
                "--convert-index-to-llvm",
                "--convert-cf-to-llvm",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
                "--reconcile-unrealized-casts",
                "--sculptor-report-tile-memory=stage=llvm",
                "-o",
                llvm_dialect,
            ],
            f"{case.name} tile {tile_id} LLVM dialect lowering",
        )
        _run_sculptor_stage(
            [
                str(sculptor_opt),
                llvm_dialect,
                "--sculptor-emit-golem-tile-abi",
                "--sculptor-finalize-golem-intrinsics",
                "-o",
                tile_abi,
            ],
            f"{case.name} tile {tile_id} Golem tile ABI emission",
        )
        _run_sculptor_stage(
            [str(translate), "--mlir-to-llvmir", tile_abi, "-o", llvm_ir],
            f"{case.name} tile {tile_id} LLVM translation",
        )
        _run_sculptor_stage(
            [
                str(clang),
                f"--target={target}",
                f"-mcpu={cpu}",
                f"-mabi={abi}",
                "-mcmodel=medany",
                "-ffreestanding",
                "-fno-stack-protector",
                "-O3",
                "-c",
                llvm_ir,
                "-o",
                str(object_path),
            ],
            f"{case.name} tile {tile_id} RISC-V object compilation",
        )
    return output_dir / "core-0.o"


class LoweringTestCase(unittest.TestCase):
    """Base assertion shared by all model-family tests."""

    def assert_lowers(self, case: LoweringCase) -> str:
        lowered = lower_to_ra_tree(case)

        self.assertIn("sculptor.mapping.ra_tree", lowered)
        self.assertGreaterEqual(
            lowered.count("sculptor.array.set "), case.minimum_matrix_setups
        )
        self.assertGreaterEqual(
            lowered.count("sculptor.array.execute "), case.minimum_matrix_setups
        )
        self.assertNotIn("sculptor.nn.", lowered)
        self.assertIsNone(
            re.search(r"(?m)^\s*%[^=]+\s*=\s*sculptor\.mvm(?:\s|$)", lowered)
        )
        self.assertNotIn("sculptor.task_region", lowered)
        for fragment in case.expected_fragments:
            self.assertIn(fragment, lowered)
        object_path = lower_to_tile_object(case)
        self.assertTrue(object_path.is_file())
        self.assertEqual(object_path.read_bytes()[:4], b"\x7fELF")
        return lowered
