#!/usr/bin/env python3
"""Focused tests for outlined tile runtime graph materialization."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from lowering_harness import ANALOG_ROOT, DEFAULT_SCULPTOR_OPT


FAN_OUT_FIXTURE = r"""
module attributes {
  sculptor.deployment.kind = "tile_routine_core",
  sculptor.deployment.physical_tile_id = 86 : i64,
  sculptor.deployment.incoming_routes = [],
  sculptor.deployment.local_bindings = [],
  sculptor.deployment.model_inputs = [],
  sculptor.deployment.model_outputs = [],
  sculptor.deployment.outgoing_routes = [
    #sculptor.tile_routine_route<id = 143 : i64, sourceTile = 86 : i64, sourceRoutine = 95 : i64, sourceOutput = 0 : i64, destinationTile = 4 : i64, destinationRoutine = 203 : i64, destinationInput = 0 : i64, resourceId = 96 : i64, tensorId = 80 : i64, byteSize = 16 : i64>,
    #sculptor.tile_routine_route<id = 140 : i64, sourceTile = 86 : i64, sourceRoutine = 95 : i64, sourceOutput = 0 : i64, destinationTile = 1 : i64, destinationRoutine = 200 : i64, destinationInput = 0 : i64, resourceId = 96 : i64, tensorId = 86 : i64, byteSize = 16 : i64>,
    #sculptor.tile_routine_route<id = 142 : i64, sourceTile = 86 : i64, sourceRoutine = 95 : i64, sourceOutput = 0 : i64, destinationTile = 3 : i64, destinationRoutine = 202 : i64, destinationInput = 0 : i64, resourceId = 96 : i64, tensorId = 68 : i64, byteSize = 16 : i64>,
    #sculptor.tile_routine_route<id = 141 : i64, sourceTile = 86 : i64, sourceRoutine = 95 : i64, sourceOutput = 0 : i64, destinationTile = 2 : i64, destinationRoutine = 201 : i64, destinationInput = 0 : i64, resourceId = 96 : i64, tensorId = 74 : i64, byteSize = 16 : i64>
  ]
} {
  func.func private @routine_95() -> tensor<1x4xf32> attributes {
    sculptor.deployment.global_routine_id = 95 : i64,
    sculptor.deployment.input_resource_ids = [],
    sculptor.deployment.local_routine_index = 0 : i64,
    sculptor.deployment.output_resource_ids = [96],
    sculptor.deployment.physical_tile_id = 86 : i64,
    sculptor.deployment.routine_kind = "compute"
  } {
    %value = arith.constant dense<0.0> : tensor<1x4xf32>
    return %value : tensor<1x4xf32>
  }
}
"""


class TileRuntimeMaterializationTest(unittest.TestCase):
    def run_stage(self, command: list[str], name: str) -> None:
        result = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        self.assertEqual(result.returncode, 0, f"{name} failed:\n{result.stderr}")

    def test_fan_out_routes_with_distinct_tensor_ids_share_one_resource(self):
        sculptor_opt = Path(
            os.environ.get("SCULPTOR_MLIR_OPT", DEFAULT_SCULPTOR_OPT)
        )
        translate = Path(
            os.environ.get(
                "GOLEM_MLIR_TRANSLATE",
                ANALOG_ROOT.parent
                / "simulation"
                / "golem-riscv-sim"
                / "install"
                / "llvm"
                / "bin"
                / "mlir-translate",
            )
        )
        self.assertTrue(sculptor_opt.is_file(), sculptor_opt)
        self.assertTrue(translate.is_file(), translate)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fan-out.mlir"
            runtime = root / "runtime.mlir"
            finalized = root / "finalized.mlir"
            llvm_dialect = root / "llvm.mlir"
            tile_abi = root / "tile-abi.mlir"
            llvm_ir = root / "tile.ll"
            source.write_text(FAN_OUT_FIXTURE)

            self.run_stage(
                [
                    str(sculptor_opt),
                    str(source),
                    "--sculptor-materialize-tile-runtime-graph",
                    "-o",
                    str(runtime),
                ],
                "runtime graph materialization",
            )
            runtime_text = runtime.read_text()
            self.assertEqual(
                runtime_text.count("sculptor.task_graph.route_output"), 1
            )
            self.assertEqual(
                runtime_text.count("#sculptor.deployment_route<"), 4
            )
            self.assertRegex(
                runtime_text,
                re.compile(
                    r"sculptor\.task_graph\.route_output.*?"
                    r"sculptor\.deployment\.route_id = 140 : i64",
                    re.DOTALL,
                ),
            )
            for route_id in range(140, 144):
                self.assertIn(f"id = {route_id} : i64", runtime_text)

            self.run_stage(
                [
                    str(sculptor_opt),
                    str(runtime),
                    "--sculptor-finalize-tile-runtime-graph",
                    "-o",
                    str(finalized),
                ],
                "runtime resource finalization",
            )
            self.run_stage(
                [
                    str(sculptor_opt),
                    str(finalized),
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
                    str(llvm_dialect),
                ],
                "LLVM dialect lowering",
            )
            self.run_stage(
                [
                    str(sculptor_opt),
                    str(llvm_dialect),
                    "--sculptor-emit-golem-tile-abi",
                    "--sculptor-finalize-golem-intrinsics",
                    "-o",
                    str(tile_abi),
                ],
                "Golem tile ABI emission",
            )
            self.run_stage(
                [
                    str(translate),
                    "--mlir-to-llvmir",
                    str(tile_abi),
                    "-o",
                    str(llvm_ir),
                ],
                "strict LLVM translation",
            )
            self.assertIn("golem_tile_dispatch_tasks", llvm_ir.read_text())


if __name__ == "__main__":
    unittest.main()
