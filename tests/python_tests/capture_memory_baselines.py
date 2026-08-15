#!/usr/bin/env python3
"""Regenerate deterministic tile-memory baselines for the pivot pipeline."""

from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import tempfile

import torch

from lowering_harness import (
    DEFAULT_SCULPTOR_OPT,
    LoweringCase,
    lower_to_tile_object,
)
from memory_report_utils import parse_bufferization_audit, run_memory_report
from test_transformer import TransformerBlockModel


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "tile_memory_baseline.json"


def summarize(
    reports: dict[str, dict[str, int | bool | str]]
) -> dict[str, object]:
    numeric_fields = sorted(
        {
            name
            for report in reports.values()
            for name, value in report.items()
            if isinstance(value, int)
            and not isinstance(value, bool)
            and name not in {"schema_version", "core_id"}
        }
    )
    return {
        "totals": {
            name: sum(int(report.get(name, 0)) for report in reports.values())
            for name in numeric_fields
        },
        "maximums": {
            name: max(int(report.get(name, 0)) for report in reports.values())
            for name in numeric_fields
        },
        "tiles": dict(sorted(reports.items(), key=lambda item: int(item[0]))),
    }


def capture_transformer(sculptor_opt: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="sculptor-memory-baseline-") as directory:
        root = Path(directory)
        lower_to_tile_object(
            LoweringCase(
                "memory_baseline_transformer_block",
                lambda: TransformerBlockModel(bias=True),
                lambda: (torch.ones(1, 4, 384),),
                array_rows=1024,
                array_cols=512,
                mesh_rows=8,
                mesh_cols=8,
                arrays_per_core=4,
                external_linalg_lowering=True,
            ),
            output_dir=root,
            digital_workers=1,
            mapping_strategies="setup-first,recursive-fork-join",
            mvm_body_policy="spread",
            setup_binding_policy="consumer-anchored",
            balance_digital_work=True,
            compile_all_active_tiles=True,
        )

        stages: dict[str, dict[str, dict[str, int | bool | str]]] = {
            "finalized": {},
            "bufferization_audit": {},
            "llvm": {},
        }
        for artifact in sorted(root.glob("07-finalized*.mlir")):
            report = run_memory_report(sculptor_opt, artifact, "finalized")
            stages["finalized"][str(report["core_id"])] = report
        for artifact in sorted(root.glob("08-llvm*.mlir")):
            report = run_memory_report(sculptor_opt, artifact, "llvm")
            stages["llvm"][str(report["core_id"])] = report
            audit = parse_bufferization_audit(artifact.read_text())
            stages["bufferization_audit"][str(audit["core_id"])] = audit

        if not stages["finalized"] or (
            stages["finalized"].keys() != stages["llvm"].keys()
            or stages["finalized"].keys()
            != stages["bufferization_audit"].keys()
        ):
            raise RuntimeError("memory baseline has inconsistent active tile sets")

        return {
            "configuration": {
                "model": "one Transformer encoder block with bias",
                "input_shape": [1, 4, 384],
                "array_rows": 1024,
                "array_cols": 512,
                "arrays_per_core": 4,
                "mesh_rows": 8,
                "mesh_cols": 8,
                "digital_workers": 1,
                "mapping_strategies": [
                    "setup-first",
                    "recursive-fork-join",
                ],
                "mvm_body_policy": "spread",
                "setup_binding_policy": "consumer-anchored",
                "balance_digital_work": True,
            },
            "active_tile_count": len(stages["finalized"]),
            "stages": {
                name: summarize(reports) for name, reports in stages.items()
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare a fresh capture with --output instead of updating it",
    )
    args = parser.parse_args()

    sculptor_opt = Path(
        os.environ.get("SCULPTOR_MLIR_OPT", DEFAULT_SCULPTOR_OPT)
    )
    if not sculptor_opt.is_file():
        parser.error(f"missing sculptor-mlir-opt: {sculptor_opt}")

    baseline = {
        "schema_version": 2,
        "fixtures": {
            "transformer_block": capture_transformer(sculptor_opt),
        },
    }
    rendered = json.dumps(baseline, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file():
            parser.error(f"missing baseline: {args.output}")
        existing = args.output.read_text()
        if existing != rendered:
            print(
                "".join(
                    difflib.unified_diff(
                        existing.splitlines(keepends=True),
                        rendered.splitlines(keepends=True),
                        fromfile=str(args.output),
                        tofile="fresh capture",
                    )
                )
            )
            return 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
