#!/usr/bin/env python3
"""Run the GPT-2 logical-mapping token/worker/matrix-copy sweep."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


DEFAULT_TOKENS = (4, 8, 16, 32, 64, 128, 256)
DEFAULT_DIGITAL_WORKERS = (1, 2, 4, 8, 16)
MATRIX_COPY_MODES = (False, True)

SUMMARY_FIELDS = (
    "function",
    "schedule",
    "mesh_rows",
    "mesh_cols",
    "arrays_per_core",
    "greedy_tile_order",
    "greedy_priority_mode",
    "greedy_candidate_scope",
    "greedy_lookahead",
    "digital_workers",
    "matrix_duplication",
    "matrix_setups",
    "logical_tiles",
    "logical_edges",
    "initial_score",
    "total_transfer_cost",
    "evaluations",
    "estimated_latency_ns",
    "crossing_bytes",
    "estimated_communication_ns",
    "required_resource_units",
    "pipeline_stages",
)

RESULT_FIELDS = (
    "tokens",
    "decoder_blocks",
    "matrix_copy",
    *SUMMARY_FIELDS,
    "compiler_seconds",
    "status",
    "diagnostic",
)

STAGE_FIELDS = ("tokens", "stage", "seconds", "status", "diagnostic")


def parse_positive_csv(value):
    result = tuple(int(piece) for piece in value.split(",") if piece)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def diagnostic_tail(log_path, line_count=8):
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    diagnostics = [
        line.strip()
        for line in lines
        if "error:" in line
        or "LLVM ERROR" in line
        or "Assertion" in line
        or "timed out" in line
    ]
    selected = diagnostics[-line_count:] if diagnostics else lines[-line_count:]
    return " | ".join(line[:1000] for line in selected)[:4000]


def append_row(path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def completed_keys(results_path):
    if not results_path.exists():
        return set()
    with results_path.open(newline="") as stream:
        return {
            (
                int(row["tokens"]),
                int(row["digital_workers"]),
                row["matrix_copy"] == "on",
            )
            for row in csv.DictReader(stream)
        }


def run_command(command, log_path, timeout_seconds=0):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("wb") as log:
        try:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds or None,
                check=False,
            )
            status = "ok" if result.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
    return status, time.monotonic() - started


def export_imported_module(args, token_dir, token_count):
    output = token_dir / "00-imported.mlirbc"
    if output.exists():
        return output

    temporary = output.with_suffix(".tmp.mlirbc")
    log_path = token_dir / "00-imported.log"
    environment = os.environ.copy()
    python_packages = str(args.torch_mlir_python)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{python_packages}:{existing}" if existing else python_packages
    )

    export_command = [
        str(args.python),
        str(args.model),
        "--mode",
        "mlir",
        "--layers",
        str(args.decoder_blocks),
        "--sequence-length",
        str(token_count),
    ]
    compile_command = [
        str(args.opt),
        "-",
        "--emit-bytecode",
        "-o",
        str(temporary),
    ]

    started = time.monotonic()
    with log_path.open("wb") as log:
        exporter = subprocess.Popen(
            export_command,
            stdout=subprocess.PIPE,
            stderr=log,
            env=environment,
        )
        try:
            compiler = subprocess.run(
                compile_command,
                stdin=exporter.stdout,
                stdout=log,
                stderr=log,
                timeout=args.timeout_seconds or None,
                check=False,
            )
        except subprocess.TimeoutExpired:
            exporter.kill()
            exporter.wait()
            temporary.unlink(missing_ok=True)
            return output, "timeout", time.monotonic() - started, diagnostic_tail(log_path)
        finally:
            if exporter.stdout:
                exporter.stdout.close()
        export_return = exporter.wait()

    elapsed = time.monotonic() - started
    if export_return != 0 or compiler.returncode != 0:
        temporary.unlink(missing_ok=True)
        return output, "failed", elapsed, diagnostic_tail(log_path)
    temporary.replace(output)
    return output, "ok", elapsed, ""


def build_cached_stage(args, input_path, output_path, stage, passes, token_count):
    if output_path.exists():
        return "ok"
    temporary = output_path.with_suffix(".tmp.mlirbc")
    log_path = output_path.with_suffix(".log")
    command = [
        str(args.opt),
        str(input_path),
        *passes,
        "--emit-bytecode",
        "-o",
        str(temporary),
    ]
    status, elapsed = run_command(command, log_path, args.timeout_seconds)
    diagnostic = "" if status == "ok" else diagnostic_tail(log_path)
    append_row(
        args.output / "stage_timings.csv",
        STAGE_FIELDS,
        {
            "tokens": token_count,
            "stage": stage,
            "seconds": f"{elapsed:.6f}",
            "status": status,
            "diagnostic": diagnostic,
        },
    )
    if status != "ok":
        temporary.unlink(missing_ok=True)
        return status
    temporary.replace(output_path)
    return "ok"


def prepare_token_stages(args, token_count):
    token_dir = args.output / "cache" / f"tokens-{token_count:03d}"
    token_dir.mkdir(parents=True, exist_ok=True)

    imported_result = export_imported_module(args, token_dir, token_count)
    if isinstance(imported_result, tuple) and len(imported_result) == 4:
        imported, status, elapsed, diagnostic = imported_result
        append_row(
            args.output / "stage_timings.csv",
            STAGE_FIELDS,
            {
                "tokens": token_count,
                "stage": "import",
                "seconds": f"{elapsed:.6f}",
                "status": status,
                "diagnostic": diagnostic,
            },
        )
        if status != "ok":
            return None
    else:
        imported = imported_result

    converted = token_dir / "03-converted.mlirbc"
    if build_cached_stage(
        args,
        imported,
        converted,
        "convert",
        (
            "--sculptor-canonicalize-layers",
            "--sculptor-fold-inference-parameters",
            "--sculptor-fuse-elementwise-regions",
            "--canonicalize",
            "--cse",
            "--sculptor-extract-layers",
            "--sculptor-convert-layers",
        ),
        token_count,
    ) != "ok":
        return None

    expanded = token_dir / "04-expanded.mlirbc"
    if build_cached_stage(
        args,
        converted,
        expanded,
        "expand-mvm",
        (
            f"--sculptor-expand-mvm-to-golem=array-rows={args.array_rows} "
            f"array-cols={args.array_cols}",
        ),
        token_count,
    ) != "ok":
        return None

    duplicated = token_dir / "05-matrix-duplicated.mlirbc"
    if build_cached_stage(
        args,
        expanded,
        duplicated,
        "duplicate-matrices",
        ("--sculptor-duplicate-matrices",),
        token_count,
    ) != "ok":
        return None

    return token_dir, expanded, duplicated


def run_mapping(args, token_count, workers, matrix_copy, base_path):
    mode = "on" if matrix_copy else "off"
    run_name = f"tokens-{token_count:03d}_workers-{workers:02d}_copy-{mode}"
    summary_path = args.output / "run_summaries" / f"{run_name}.csv"
    log_path = args.output / "logs" / f"{run_name}.log"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    plan_options = (
        "strategies=setup-first,mvm-wave,fan-out-cut,consumer-bound-fill "
        f"mesh-rows={args.mesh_rows} mesh-cols={args.mesh_cols} "
        f"arrays-per-core={args.arrays_per_core} "
        f"array-rows={args.array_rows} array-cols={args.array_cols} "
        "mvm-body-policy=packed balance-digital-work verify-plan"
    )
    placement_options = (
        f"schedule=greedy mesh-rows={args.mesh_rows} mesh-cols={args.mesh_cols} "
        f"arrays-per-core={args.arrays_per_core} verify-placement "
        f"summary-output={summary_path}"
    )
    command = [
        str(args.opt),
        str(base_path),
        f"--sculptor-expand-digital-work=parallel-workers={workers}",
        "--sculptor-build-ra-tree",
        f"--sculptor-plan-mapping={plan_options}",
        f"--sculptor-place-logical-tiles={placement_options}",
        "-o",
        "/dev/null",
    ]

    status, elapsed = run_command(command, log_path, args.timeout_seconds)
    row = {
        "tokens": token_count,
        "decoder_blocks": args.decoder_blocks,
        "matrix_copy": mode,
        **{field: "" for field in SUMMARY_FIELDS},
        "schedule": "greedy",
        "mesh_rows": args.mesh_rows,
        "mesh_cols": args.mesh_cols,
        "arrays_per_core": args.arrays_per_core,
        "greedy_tile_order": "sequential",
        "greedy_priority_mode": "sum",
        "greedy_candidate_scope": "cardinal",
        "greedy_lookahead": 1,
        "digital_workers": workers,
        "matrix_duplication": mode,
        "compiler_seconds": f"{elapsed:.6f}",
        "status": status,
        "diagnostic": "" if status == "ok" else diagnostic_tail(log_path),
    }
    if status == "ok":
        try:
            with summary_path.open(newline="") as stream:
                summaries = list(csv.DictReader(stream))
            if len(summaries) != 1:
                raise ValueError(f"expected one summary row, got {len(summaries)}")
            row.update(summaries[0])
            if row["matrix_duplication"] != mode:
                raise ValueError(
                    "placement summary matrix mode does not match requested mode"
                )
        except (OSError, ValueError) as error:
            row["status"] = "invalid-summary"
            row["diagnostic"] = str(error)
    return row


def write_manifest(args):
    manifest = {
        "model": "GPT-2 Mini topology",
        "decoder_blocks": args.decoder_blocks,
        "tokens": list(args.tokens),
        "digital_workers": list(args.digital_workers),
        "matrix_copy": ["off", "on"],
        "mesh": [args.mesh_rows, args.mesh_cols],
        "arrays_per_tile": args.arrays_per_core,
        "array_shape": [args.array_rows, args.array_cols],
        "placement": {
            "schedule": "greedy",
        },
        "mapping_strategies": [
            "setup-first",
            "mvm-wave",
            "fan-out-cut",
            "consumer-bound-fill",
        ],
        "mvm_body_policy": "packed",
        "balance_digital_work": True,
        "run_count": len(args.tokens)
        * len(args.digital_workers)
        * len(MATRIX_COPY_MODES),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def parse_arguments():
    repo = Path(__file__).resolve().parents[2]
    workspace = repo.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=parse_positive_csv, default=DEFAULT_TOKENS)
    parser.add_argument(
        "--digital-workers",
        type=parse_positive_csv,
        default=DEFAULT_DIGITAL_WORKERS,
    )
    parser.add_argument("--decoder-blocks", type=int, default=12)
    parser.add_argument("--mesh-rows", type=int, default=30)
    parser.add_argument("--mesh-cols", type=int, default=30)
    parser.add_argument("--arrays-per-core", type=int, default=4)
    parser.add_argument("--array-rows", type=int, default=1024)
    parser.add_argument("--array-cols", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/sculptor-pivot-gpt2-12-block-sweep"),
    )
    parser.add_argument(
        "--opt",
        type=Path,
        default=workspace / "build/sculptor-mlir-pivot/bin/sculptor-mlir-opt",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(
            "/home/blue/PhD/simulation/golem-riscv-sim/install/"
            "compiler-python/bin/python"
        ),
    )
    parser.add_argument(
        "--torch-mlir-python",
        type=Path,
        default=Path(
            "/home/blue/PhD/simulation/golem-riscv-sim/install/"
            "torch-mlir/python_packages/torch_mlir"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=repo / "tests/model_tests/generators/gpt2_generator.py",
    )
    args = parser.parse_args()
    positive_values = (
        args.decoder_blocks,
        args.mesh_rows,
        args.mesh_cols,
        args.arrays_per_core,
        args.array_rows,
        args.array_cols,
        args.jobs,
    )
    if any(value <= 0 for value in positive_values):
        parser.error("all model, hardware, and search dimensions must be positive")
    return args


def main():
    args = parse_arguments()
    for required in (args.opt, args.python, args.model, args.torch_mlir_python):
        if not required.exists():
            raise SystemExit(f"required path does not exist: {required}")

    write_manifest(args)
    results_path = args.output / "results.csv"
    complete = completed_keys(results_path)

    for token_count in args.tokens:
        pending = [
            (workers, matrix_copy)
            for matrix_copy in MATRIX_COPY_MODES
            for workers in args.digital_workers
            if (token_count, workers, matrix_copy) not in complete
        ]
        pending.sort(key=lambda case: (case[1], case[0]), reverse=True)
        if not pending:
            print(f"tokens={token_count}: already complete", flush=True)
            continue

        print(f"tokens={token_count}: preparing shared stages", flush=True)
        prepared = prepare_token_stages(args, token_count)
        if prepared is None:
            print(f"tokens={token_count}: shared stage failed", file=sys.stderr)
            continue
        token_dir, expanded, duplicated = prepared

        with ThreadPoolExecutor(max_workers=min(args.jobs, len(pending))) as executor:
            futures = {
                executor.submit(
                    run_mapping,
                    args,
                    token_count,
                    workers,
                    matrix_copy,
                    duplicated if matrix_copy else expanded,
                ): (workers, matrix_copy)
                for workers, matrix_copy in pending
            }
            for future in as_completed(futures):
                workers, matrix_copy = futures[future]
                row = future.result()
                append_row(results_path, RESULT_FIELDS, row)
                print(
                    f"tokens={token_count} workers={workers} "
                    f"matrix_copy={'on' if matrix_copy else 'off'}: "
                    f"{row['status']} "
                    f"score={row['total_transfer_cost'] or 'N/A'} "
                    f"seconds={row['compiler_seconds']}",
                    flush=True,
                )

        if not args.keep_cache:
            shutil.rmtree(token_dir)


if __name__ == "__main__":
    main()
