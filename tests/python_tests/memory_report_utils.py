#!/usr/bin/env python3
"""Read structured output from --sculptor-report-tile-memory."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


REPORT_PREFIX = "SCULPTOR_TILE_MEMORY_REPORT "
AUDIT_ATTR = "sculptor.memory.bufferization_audit = "
FIELD_PATTERN = re.compile(
    r'(\w+) = (?:"([^"]*)"|(-?[0-9]+) : i64|(true|false))'
)

AUDIT_SCALAR_FIELDS = {
    "schema_version",
    "core_id",
    "strict",
    "allocation_count",
    "static_allocation_bytes",
    "approved_local_allocation_count",
    "unplanned_allocation_count",
    "escaping_allocation_count",
    "missing_deallocation_count",
    "routine_lifetime_allocation_count",
    "copy_count",
    "static_copy_bytes",
    "planned_assembly_copy_count",
    "planned_assembly_copy_bytes",
    "planned_boot_staging_copy_count",
    "planned_boot_staging_copy_bytes",
    "unplanned_copy_count",
    "unplanned_full_tensor_copy_count",
    "pure_copy_loop_count",
    "subview_count",
}


def parse_memory_report(text: str) -> dict[str, int | bool | str]:
    """Parse one printed report dictionary into ordinary Python values."""

    report: dict[str, int | bool | str] = {}
    for name, string_value, integer_value, bool_value in FIELD_PATTERN.findall(
        text
    ):
        if string_value:
            report[name] = string_value
        elif integer_value:
            report[name] = int(integer_value)
        else:
            report[name] = bool_value == "true"
    if not report or "schema_version" not in report or "stage" not in report:
        raise ValueError(f"invalid Sculptor tile-memory report: {text}")
    return report


def run_memory_report(
    sculptor_opt: Path, artifact: Path, stage: str
) -> dict[str, int | bool | str]:
    """Recompute and return one report without modifying the input artifact."""

    result = subprocess.run(
        [
            str(sculptor_opt),
            str(artifact),
            f"--sculptor-report-tile-memory=stage={stage} print=true",
            "-o",
            "/dev/null",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"memory report failed for {artifact}:\n{result.stderr}"
        )
    lines = [
        line.removeprefix(REPORT_PREFIX)
        for line in result.stderr.splitlines()
        if line.startswith(REPORT_PREFIX)
    ]
    if len(lines) != 1:
        raise RuntimeError(
            f"expected one memory report for {artifact}, found {len(lines)}"
        )
    return parse_memory_report(lines[0])


def parse_bufferization_audit(
    text: str,
) -> dict[str, int | bool | str]:
    """Read the persisted post-bufferization audit from an MLIR module."""

    marker = text.find(AUDIT_ATTR)
    if marker < 0:
        raise ValueError("MLIR module has no tile-bufferization audit")
    start = text.find("{", marker + len(AUDIT_ATTR))
    if start < 0:
        raise ValueError("tile-bufferization audit has no dictionary")
    depth = 0
    finish = -1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                finish = index + 1
                break
    if finish < 0:
        raise ValueError("tile-bufferization audit dictionary is incomplete")

    report: dict[str, int | bool | str] = {}
    for name, string_value, integer_value, bool_value in FIELD_PATTERN.findall(
        text[start:finish]
    ):
        if name not in AUDIT_SCALAR_FIELDS:
            continue
        if string_value:
            report[name] = string_value
        elif integer_value:
            report[name] = int(integer_value)
        else:
            report[name] = bool_value == "true"
    if (
        report.get("schema_version") != 1
        or "core_id" not in report
        or "allocation_count" not in report
    ):
        raise ValueError("invalid tile-bufferization audit")
    return report
