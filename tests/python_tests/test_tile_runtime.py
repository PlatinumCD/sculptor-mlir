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
    #sculptor.tile_routine_route<id = 143 : i64, sourceTile = 86 : i64, sourceRoutine = 95 : i64, sourceOutput = 0 : i64, destinationTile = 4 : i64, destinationRoutine = 203 : i64, destinationInput = 0 : i64, resourceId = 96 : i64, tensorId = 80 : i64, byteSize = 32 : i64>,
    #sculptor.tile_routine_route<id = 140 : i64, sourceTile = 86 : i64, sourceRoutine = 95 : i64, sourceOutput = 0 : i64, destinationTile = 1 : i64, destinationRoutine = 200 : i64, destinationInput = 0 : i64, resourceId = 96 : i64, tensorId = 86 : i64, byteSize = 32 : i64>,
    #sculptor.tile_routine_route<id = 142 : i64, sourceTile = 86 : i64, sourceRoutine = 95 : i64, sourceOutput = 0 : i64, destinationTile = 3 : i64, destinationRoutine = 202 : i64, destinationInput = 0 : i64, resourceId = 96 : i64, tensorId = 68 : i64, byteSize = 32 : i64>,
    #sculptor.tile_routine_route<id = 141 : i64, sourceTile = 86 : i64, sourceRoutine = 95 : i64, sourceOutput = 0 : i64, destinationTile = 2 : i64, destinationRoutine = 201 : i64, destinationInput = 0 : i64, resourceId = 96 : i64, tensorId = 74 : i64, byteSize = 32 : i64>
  ]
} {
  func.func private @routine_95() -> tensor<1x4xi64> attributes {
    sculptor.deployment.global_routine_id = 95 : i64,
    sculptor.deployment.input_resource_ids = [],
    sculptor.deployment.local_routine_index = 0 : i64,
    sculptor.deployment.output_resource_ids = [96],
    sculptor.deployment.physical_tile_id = 86 : i64,
    sculptor.deployment.routine_kind = "compute"
  } {
    %value = arith.constant dense<0> : tensor<1x4xi64>
    return %value : tensor<1x4xi64>
  }
}
"""


MEMORY_REPORT_FIXTURE = r"""
module attributes {
  sculptor.runtime.core_id = 7 : i64
} {
  func.func @copy_buffers() {
    %source = memref.alloc() : memref<4xf32>
    %destination = memref.alloc() : memref<4xf32>
    memref.copy %source, %destination
      : memref<4xf32> to memref<4xf32>
    memref.dealloc %destination : memref<4xf32>
    memref.dealloc %source : memref<4xf32>
    return
  }
}
"""


HEAP_INSTRUMENTATION_FIXTURE = r"""
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @allocate_and_free(%size: i64) {
    %allocation = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
    llvm.call @free(%allocation) : (!llvm.ptr) -> ()
    llvm.return
  }
}
"""


BUFFERIZATION_AUDIT_GOOD_FIXTURE = r"""
module {
  func.func @local_temporary() {
    %buffer = memref.alloc() : memref<4xf32>
    memref.dealloc %buffer : memref<4xf32>
    return
  }
}
"""


BUFFERIZATION_AUDIT_BAD_ALLOCATION_FIXTURE = r"""
module {
  func.func @leaked_temporary() {
    %buffer = memref.alloc() : memref<4xf32>
    return
  }
}
"""


BUFFERIZATION_AUDIT_BAD_COPY_FIXTURE = r"""
module {
  func.func @unplanned_copy(%source: memref<4xf32>,
                            %destination: memref<4xf32>) {
    memref.copy %source, %destination
      : memref<4xf32> to memref<4xf32>
    return
  }
}
"""


BUFFERIZATION_AUDIT_PADDING_FIXTURE = r"""
module {
  func.func @padding(%source: memref<2x3xf32>) {
    %zero = arith.constant 0.0 : f32
    %buffer = memref.alloc() : memref<2x5xf32>
    linalg.fill ins(%zero : f32) outs(%buffer : memref<2x5xf32>)
    %valid = memref.subview %buffer[0, 1] [2, 3] [1, 1]
      : memref<2x5xf32> to memref<2x3xf32, strided<[5, 1], offset: 1>>
    memref.copy %source, %valid
      : memref<2x3xf32>
        to memref<2x3xf32, strided<[5, 1], offset: 1>>
    memref.dealloc %buffer : memref<2x5xf32>
    return
  }
  func.func @physical_tail_padding(%source: memref<1x3xf32>) {
    %zero = arith.constant 0.0 : f32
    %buffer = memref.alloc() : memref<1x5xf32>
    %tail = memref.subview %buffer[0, 3] [1, 2] [1, 1]
      : memref<1x5xf32> to memref<1x2xf32, strided<[5, 1], offset: 3>>
    linalg.fill ins(%zero : f32)
      outs(%tail : memref<1x2xf32, strided<[5, 1], offset: 3>>)
    %valid = memref.subview %buffer[0, 0] [1, 3] [1, 1]
      : memref<1x5xf32> to memref<1x3xf32, strided<[5, 1]>>
    memref.copy %source, %valid
      : memref<1x3xf32> to memref<1x3xf32, strided<[5, 1]>>
    memref.dealloc %buffer : memref<1x5xf32>
    return
  }
}
"""


BUFFERIZATION_AUDIT_STRIDED_ASSEMBLY_FIXTURE = r"""
module {
  func.func @column_concat(%left: memref<2x3xf32>,
                           %right: memref<2x2xf32>) {
    %buffer = memref.alloc() : memref<2x5xf32>
    %left_view = memref.subview %buffer[0, 0] [2, 3] [1, 1]
      : memref<2x5xf32> to memref<2x3xf32, strided<[5, 1]>>
    memref.copy %left, %left_view
      : memref<2x3xf32> to memref<2x3xf32, strided<[5, 1]>>
    %right_view = memref.subview %buffer[0, 3] [2, 2] [1, 1]
      : memref<2x5xf32> to memref<2x2xf32, strided<[5, 1], offset: 3>>
    memref.copy %right, %right_view
      : memref<2x2xf32>
        to memref<2x2xf32, strided<[5, 1], offset: 3>>
    memref.dealloc %buffer : memref<2x5xf32>
    return
  }
}
"""


VECTORIZED_COPY_FIXTURE = r"""
module {
  func.func @contiguous_copy() {
    %source = memref.alloc() : memref<4x10xf32>
    %destination = memref.alloc() : memref<4x10xf32>
    memref.copy %source, %destination
      : memref<4x10xf32> to memref<4x10xf32>
    memref.dealloc %destination : memref<4x10xf32>
    memref.dealloc %source : memref<4x10xf32>
    return
  }
  func.func @row_contiguous_copy() {
    %source = memref.alloc()
      : memref<4x10xf32, strided<[16, 1], offset: 0>>
    %destination = memref.alloc()
      : memref<4x10xf32, strided<[16, 1], offset: 0>>
    memref.copy %source, %destination
      : memref<4x10xf32, strided<[16, 1], offset: 0>>
        to memref<4x10xf32, strided<[16, 1], offset: 0>>
    memref.dealloc %destination
      : memref<4x10xf32, strided<[16, 1], offset: 0>>
    memref.dealloc %source
      : memref<4x10xf32, strided<[16, 1], offset: 0>>
    return
  }
  func.func @non_contiguous_copy() {
    %source = memref.alloc()
      : memref<4x10xf32, strided<[20, 2], offset: 0>>
    %destination = memref.alloc()
      : memref<4x10xf32, strided<[20, 2], offset: 0>>
    memref.copy %source, %destination
      : memref<4x10xf32, strided<[20, 2], offset: 0>>
        to memref<4x10xf32, strided<[20, 2], offset: 0>>
    memref.dealloc %destination
      : memref<4x10xf32, strided<[20, 2], offset: 0>>
    memref.dealloc %source
      : memref<4x10xf32, strided<[20, 2], offset: 0>>
    return
  }
  func.func @rank_zero_copy() {
    %source = memref.alloc() : memref<f32>
    %destination = memref.alloc() : memref<f32>
    memref.copy %source, %destination : memref<f32> to memref<f32>
    memref.dealloc %destination : memref<f32>
    memref.dealloc %source : memref<f32>
    return
  }
  func.func @possible_overlap(%buffer: memref<20xf32>) {
    memref.copy %buffer, %buffer : memref<20xf32> to memref<20xf32>
    return
  }
  func.func @dynamic_copy(%source: memref<?xf32>,
                          %destination: memref<?xf32>) {
    memref.copy %source, %destination
      : memref<?xf32> to memref<?xf32>
    return
  }
}
"""


ROUTE_PACK_FIXTURE = r"""
module attributes {
  sculptor.memory.owners = [
    #sculptor.tile_memory_owner<id = 0 : i64, resourceId = 7 : i64, tensorId = -1 : i64, tile = 0 : i64, routine = 0 : i64, port = 0 : i64, kind = route_input, byteSize = 40 : i64, shape = [1, 10], strides = [10, 1]>,
    #sculptor.tile_memory_owner<id = 1 : i64, resourceId = 8 : i64, tensorId = -1 : i64, tile = 0 : i64, routine = 0 : i64, port = 0 : i64, kind = intermediate, byteSize = 40 : i64, shape = [1, 10], strides = [10, 1]>
  ]
} {
  func.func @route_unpack(
      %source: memref<1x10xf32> {sculptor.memory.owner_id = 0 : i64},
      %destination: memref<1x10xf32> {
        sculptor.memory.owner_id = 1 : i64}) {
    memref.copy %source, %destination
      : memref<1x10xf32> to memref<1x10xf32>
    return
  }
}
"""


DIGITAL_KERNEL_FIXTURE = r"""
#identity_2d = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @activation(
      %input: memref<2x10xf32, strided<[16, 1], offset: 3>>,
      %output: memref<2x10xf32>) {
    linalg.generic {
      indexing_maps = [#identity_2d, #identity_2d],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<2x10xf32, strided<[16, 1], offset: 3>>)
      outs(%output : memref<2x10xf32>)
      attrs = {
        activation = "gelu",
        sculptor.semantic.section = "digital.activation"
      } {
    ^bb0(%value: f32, %old: f32):
      %half = arith.constant 5.000000e-01 : f32
      %one = arith.constant 1.000000e+00 : f32
      %sqrt2 = arith.constant 1.41421354 : f32
      %scaled = arith.divf %value, %sqrt2 : f32
      %erf = math.erf %scaled : f32
      %factor = arith.addf %one, %erf : f32
      %half_input = arith.mulf %half, %value : f32
      %result = arith.mulf %half_input, %factor : f32
      linalg.yield %result : f32
    }
    return
  }

  func.func @bias_add(%left: memref<2x16xf32>,
                      %right: memref<2x16xf32>,
                      %output: memref<2x16xf32>) {
    linalg.add {
      sculptor.semantic.section = "digital.bias_add"
    } ins(%left, %right : memref<2x16xf32>, memref<2x16xf32>)
      outs(%output : memref<2x16xf32>)
    return
  }
}
"""


DIGITAL_KERNEL_UNSUPPORTED_LAYOUT_FIXTURE = r"""
#identity_2d = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @activation(
      %input: memref<1x10xf32, strided<[20, 2]>>,
      %output: memref<1x10xf32, strided<[20, 2]>>) {
    linalg.generic {
      indexing_maps = [#identity_2d, #identity_2d],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<1x10xf32, strided<[20, 2]>>)
      outs(%output : memref<1x10xf32, strided<[20, 2]>>)
      attrs = {
        activation = "gelu",
        sculptor.semantic.section = "digital.activation"
      } {
    ^bb0(%value: f32, %old: f32):
      %sqrt2 = arith.constant 1.41421354 : f32
      %scaled = arith.divf %value, %sqrt2 : f32
      %erf = math.erf %scaled : f32
      linalg.yield %erf : f32
    }
    return
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
        clang = Path(
            os.environ.get(
                "GOLEM_CLANG",
                ANALOG_ROOT.parent
                / "simulation"
                / "golem-riscv-sim"
                / "install"
                / "llvm"
                / "bin"
                / "clang",
            )
        )
        self.assertTrue(sculptor_opt.is_file(), sculptor_opt)
        self.assertTrue(translate.is_file(), translate)
        self.assertTrue(clang.is_file(), clang)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fan-out.mlir"
            runtime = root / "runtime.mlir"
            finalized = root / "finalized.mlir"
            reported = root / "reported.mlir"
            llvm_dialect = root / "llvm.mlir"
            tile_abi = root / "tile-abi.mlir"
            llvm_ir = root / "tile.ll"
            object_path = root / "tile.o"
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
                    "--sculptor-report-tile-memory=stage=finalized",
                    "-o",
                    str(reported),
                ],
                "finalized memory report",
            )
            report_text = reported.read_text()
            for expected in (
                'stage = "finalized"',
                "core_id = 86 : i64",
                "resource_count = 1 : i64",
                "route_output_count = 1 : i64",
                "route_output_bytes = 32 : i64",
                "route_record_count = 4 : i64",
                "workspace_bytes = 32 : i64",
                "runtime_descriptor_bytes = 56 : i64",
                "abi_table_bytes = 396 : i64",
                "conservative_peak_live_bytes = 88 : i64",
                "peak_estimate_complete = false",
            ):
                self.assertIn(expected, report_text)
            self.run_stage(
                [
                    str(sculptor_opt),
                    str(reported),
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
            llvm_text = llvm_ir.read_text()
            for symbol in (
                "golem_tile_dispatch_tasks",
                "golem_tile_task_id_index",
                "golem_tile_incoming_route_id_index",
                "golem_tile_outgoing_route_id_index",
                "golem_tile_static_tasks",
                "golem_tile_static_resources",
                "golem_tile_static_runtime_data",
            ):
                self.assertIn(symbol, llvm_text)
            self.run_stage(
                [
                    str(clang),
                    "--target=riscv64-unknown-elf",
                    "-mcpu=golem-analog",
                    "-mabi=lp64d",
                    "-mcmodel=medany",
                    "-ffreestanding",
                    "-fno-stack-protector",
                    "-O3",
                    "-c",
                    str(llvm_ir),
                    "-o",
                    str(object_path),
                ],
                "RISC-V object compilation",
            )
            self.assertEqual(object_path.read_bytes()[:4], b"\x7fELF")

    def test_heap_instrumentation_rewrites_malloc_and_free(self):
        sculptor_opt = Path(
            os.environ.get("SCULPTOR_MLIR_OPT", DEFAULT_SCULPTOR_OPT)
        )
        self.assertTrue(sculptor_opt.is_file(), sculptor_opt)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "heap.mlir"
            instrumented = root / "heap-instrumented.mlir"
            source.write_text(HEAP_INSTRUMENTATION_FIXTURE)

            self.run_stage(
                [
                    str(sculptor_opt),
                    str(source),
                    "--sculptor-instrument-tile-heap",
                    "-o",
                    str(instrumented),
                ],
                "tile heap instrumentation",
            )
            text = instrumented.read_text()
            self.assertIn("@golem_runtime_profiled_malloc", text)
            self.assertIn("@golem_runtime_profiled_free", text)
            self.assertNotRegex(text, r"(?:func|call) @malloc(?:\W|$)")
            self.assertNotRegex(text, r"(?:func|call) @free(?:\W|$)")

    def test_bufferized_memory_report_counts_allocations_and_copies(self):
        sculptor_opt = Path(
            os.environ.get("SCULPTOR_MLIR_OPT", DEFAULT_SCULPTOR_OPT)
        )
        self.assertTrue(sculptor_opt.is_file(), sculptor_opt)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "memory.mlir"
            reported = root / "memory-reported.mlir"
            source.write_text(MEMORY_REPORT_FIXTURE)

            self.run_stage(
                [
                    str(sculptor_opt),
                    str(source),
                    "--sculptor-report-tile-memory=stage=bufferized",
                    "-o",
                    str(reported),
                ],
                "bufferized memory report",
            )
            report_text = reported.read_text()
            for expected in (
                'stage = "bufferized"',
                "core_id = 7 : i64",
                "static_alloc_site_count = 2 : i64",
                "static_alloc_bytes = 32 : i64",
                "max_routine_static_alloc_bytes = 32 : i64",
                "copy_op_count = 1 : i64",
                "known_copy_bytes = 16 : i64",
                "conservative_peak_live_bytes = 32 : i64",
                "peak_estimate_complete = false",
            ):
                self.assertIn(expected, report_text)

    def test_strict_bufferization_audit_accepts_owned_local_temporary(self):
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-audit-tile-bufferization="
                "strict=true print=true",
            ],
            input=BUFFERIZATION_AUDIT_GOOD_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("approved_local_allocation_count = 1 : i64", result.stdout)
        self.assertIn("unplanned_allocation_count = 0 : i64", result.stdout)

    def test_strict_bufferization_audit_rejects_missing_deallocation(self):
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-audit-tile-bufferization=strict=true",
            ],
            input=BUFFERIZATION_AUDIT_BAD_ALLOCATION_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("allocation without a deallocation", result.stderr)

    def test_strict_bufferization_audit_rejects_unplanned_full_copy(self):
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-audit-tile-bufferization=strict=true",
            ],
            input=BUFFERIZATION_AUDIT_BAD_COPY_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unplanned full-tensor copy", result.stderr)

    def test_strict_bufferization_audit_accepts_initialized_padding(self):
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-audit-tile-bufferization=strict=true",
            ],
            input=BUFFERIZATION_AUDIT_PADDING_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("planned_padding_copy_count = 2 : i64", result.stdout)
        self.assertIn("unplanned_full_tensor_copy_count = 0 : i64", result.stdout)

    def test_strict_bufferization_audit_accepts_strided_column_assembly(self):
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-audit-tile-bufferization=strict=true",
            ],
            input=BUFFERIZATION_AUDIT_STRIDED_ASSEMBLY_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "planned_local_assembly_copy_count = 2 : i64", result.stdout
        )
        self.assertIn("unplanned_full_tensor_copy_count = 0 : i64", result.stdout)

    def test_vectorized_copy_fallback_classifies_geometry_and_masks_tails(self):
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-vectorize-tile-copies=vector-bits=256",
            ],
            input=VECTORIZED_COPY_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "copy_count = 6 : i64",
            "vectorized_copy_count = 2 : i64",
            "vectorized_copy_bytes = 320 : i64",
            "masked_tail_copy_count = 2 : i64",
            "scalar_rank_zero_copy_count = 1 : i64",
            "fallback_copy_count = 3 : i64",
            "fallback_copy_bytes = 240 : i64",
            "unknown_fallback_bytes_count = 1 : i64",
            'geometry = "contiguous"',
            'geometry = "row_contiguous"',
            'geometry = "non_contiguous"',
            'reason = "non_unit_innermost_stride"',
            'reason = "possible_alias_or_overlap"',
            'reason = "dynamic_shape"',
            "vector.create_mask",
            "vector.maskedload",
            "vector.maskedstore",
        ):
            self.assertIn(expected, result.stdout)
        self.assertEqual(result.stdout.count("memref.copy"), 3)
        self.assertIn("memref.load", result.stdout)
        self.assertIn("memref.store", result.stdout)

    def test_route_pack_vectorizes_and_lowers_with_a_masked_tail(self):
        classify = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-vectorize-tile-copies=vector-bits=256",
            ],
            input=ROUTE_PACK_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(classify.returncode, 0, classify.stderr)
        for expected in (
            "route_pack_count = 1 : i64",
            "vectorized_copy_count = 1 : i64",
            "vectorized_copy_bytes = 40 : i64",
            "masked_tail_copy_count = 1 : i64",
            'materialization_kind = "route_pack"',
            "vector.maskedload",
            "vector.maskedstore",
        ):
            self.assertIn(expected, classify.stdout)
        self.assertNotIn("memref.copy", classify.stdout)

        lowered = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-vectorize-tile-copies=vector-bits=256",
                "--convert-scf-to-cf",
                "--convert-vector-to-llvm",
                "--expand-strided-metadata",
                "--convert-arith-to-llvm",
                "--convert-index-to-llvm",
                "--convert-cf-to-llvm",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
                "--reconcile-unrealized-casts",
            ],
            input=ROUTE_PACK_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        self.assertIn("llvm.func @route_unpack", lowered.stdout)
        self.assertNotIn("vector.", lowered.stdout)
        self.assertNotIn("memref.", lowered.stdout)
        self.assertNotIn("scf.", lowered.stdout)

    def test_digital_kernel_framework_vectorizes_multiple_kernel_families(self):
        sculptor_opt = Path(
            os.environ.get("SCULPTOR_MLIR_OPT", DEFAULT_SCULPTOR_OPT)
        )
        result = subprocess.run(
            [
                str(sculptor_opt),
                "-",
                "--sculptor-vectorize-digital-kernels=vector-bits=256",
            ],
            input=DIGITAL_KERNEL_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "vectorized_kernel_count = 2 : i64",
            "activation_kernel_count = 1 : i64",
            "elementwise_kernel_count = 1 : i64",
            "masked_tail_count = 1 : i64",
            'sculptor.digital.kernel_family = "activation"',
            'sculptor.digital.kernel_family = "elementwise"',
            'sculptor.digital.kernel_kind = "gelu"',
            'sculptor.digital.kernel_kind = "add"',
            "sculptor.digital.vector_bits = 256 : i64",
            "vector.maskedload",
            "vector.maskedstore",
            "vector.load",
            "vector.store",
        ):
            self.assertIn(expected, result.stdout)
        self.assertNotIn("math.erf", result.stdout)
        self.assertNotIn("linalg.add", result.stdout)

    def test_digital_kernel_framework_rejects_unsupported_layout(self):
        sculptor_opt = Path(
            os.environ.get("SCULPTOR_MLIR_OPT", DEFAULT_SCULPTOR_OPT)
        )
        result = subprocess.run(
            [
                str(sculptor_opt),
                "-",
                "--sculptor-vectorize-digital-kernels=vector-bits=256",
            ],
            input=DIGITAL_KERNEL_UNSUPPORTED_LAYOUT_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "vector GELU requires a unit-stride innermost dimension",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
