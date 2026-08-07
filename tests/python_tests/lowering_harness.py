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
DEFAULT_SCULPTOR_OPT = (
    ANALOG_ROOT / "build" / "sculptor-mlir-pivot" / "bin" / "sculptor-mlir-opt"
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


def lower_to_ra_tree(case: LoweringCase) -> str:
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
            f"array-rows={case.array_rows} array-cols={case.array_cols}"
        ),
    ]
    if case.duplicate_matrices:
        command.append("--sculptor-duplicate-matrices")
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


def lower_to_tile_object(case: LoweringCase, output_dir: Path | None = None) -> Path:
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
            f"array-rows={case.array_rows} array-cols={case.array_cols}"
        ),
    ]
    if case.duplicate_matrices:
        ra_command.append("--sculptor-duplicate-matrices")
    ra_command.extend(["--sculptor-build-ra-tree", "-o", path("01-ra.mlir")])
    _run_sculptor_stage(
        ra_command,
        f"{case.name} RA-tree lowering",
    )
    _run_sculptor_stage(
        [
            str(sculptor_opt),
            path("01-ra.mlir"),
            "--sculptor-plan-mapping=strategies=setup-first "
            f"mesh-rows={case.mesh_rows} mesh-cols={case.mesh_cols} "
            f"arrays-per-core={case.arrays_per_core} "
            f"array-rows={case.array_rows} array-cols={case.array_cols}",
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
            "--sculptor-outline-tile-routines",
            "-o",
            path("04-outlined.mlir"),
        ],
        f"{case.name} routine outlining",
    )
    _run_sculptor_stage(
        [
            str(sculptor_opt),
            path("04-outlined.mlir"),
            "--sculptor-extract-tile-module=tile-id=0",
            "-o",
            path("05-extracted.mlir"),
        ],
        f"{case.name} tile extraction",
    )
    _run_sculptor_stage(
        [
            str(sculptor_opt),
            path("05-extracted.mlir"),
            "--sculptor-materialize-tile-runtime-graph",
            "-o",
            path("06-runtime.mlir"),
        ],
        f"{case.name} runtime graph materialization",
    )
    _run_sculptor_stage(
        [
            str(sculptor_opt),
            path("06-runtime.mlir"),
            "--sculptor-finalize-tile-runtime-graph",
            "-o",
            path("07-finalized.mlir"),
        ],
        f"{case.name} tile resource finalization",
    )
    _run_sculptor_stage(
        [
            str(sculptor_opt),
            path("07-finalized.mlir"),
            "--sculptor-lower-golem-to-llvm-shims",
            "--canonicalize",
            "--cse",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=bufferize-function-boundaries "
            "function-boundary-type-conversion=identity-layout-map",
            "--buffer-results-to-out-params=hoist-static-allocs",
            "--convert-bufferization-to-memref",
            "--buffer-hoisting",
            "--buffer-loop-hoisting",
            "--buffer-deallocation-pipeline",
            "--optimize-allocation-liveness",
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
            "-o",
            path("08-llvm.mlir"),
        ],
        f"{case.name} LLVM dialect lowering",
    )
    _run_sculptor_stage(
        [
            str(sculptor_opt),
            path("08-llvm.mlir"),
            "--sculptor-emit-golem-tile-abi",
            "--sculptor-finalize-golem-intrinsics",
            "-o",
            path("09-abi.mlir"),
        ],
        f"{case.name} Golem tile ABI emission",
    )
    _run_sculptor_stage(
        [str(translate), "--mlir-to-llvmir", path("09-abi.mlir"), "-o", path("10.ll")],
        f"{case.name} LLVM translation",
    )

    target = os.environ.get("GOLEM_TARGET", "riscv64-unknown-elf")
    cpu = os.environ.get("GOLEM_CPU", "golem-analog")
    abi = os.environ.get("GOLEM_ABI", "lp64d")
    object_path = output_dir / "core-0.o"
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
            path("10.ll"),
            "-o",
            str(object_path),
        ],
        f"{case.name} RISC-V object compilation",
    )
    return object_path


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
