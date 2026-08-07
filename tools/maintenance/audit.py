#!/usr/bin/env python3
"""Audit Sculptor source registration and likely dead code.

The audit separates structural errors from review candidates. Structural errors
fail by default. Candidate findings never cause deletion and do not fail the
default invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hh",
    ".hpp",
    ".inc",
    ".md",
    ".mlir",
    ".py",
    ".td",
    ".txt",
}
CODE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hh", ".hpp", ".inc"}
IGNORED_PARTS = {
    ".agents",
    ".codex",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
}
SEVERITY_ORDER = {"error": 0, "candidate": 1, "info": 2}


@dataclass(frozen=True, order=True)
class Finding:
    severity: str
    code: str
    subject: str
    detail: str


@dataclass
class PassRecord:
    argument: str
    definition: str
    registration: str | None
    registered: bool
    test_references: int
    documentation_references: int


@dataclass
class PlannerRecord:
    name: str
    implementation: str
    planner_class: str
    registered: bool
    test_references: int
    documentation_references: int


@dataclass
class AttributeRecord:
    identifier: str
    value: str
    definition: str
    references: int
    producer_matches: int
    consumer_matches: int


@dataclass
class AuditResult:
    root: str
    build_dir: str | None
    facts: dict[str, object] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    passes: list[PassRecord] = field(default_factory=list)
    planners: list[PlannerRecord] = field(default_factory=list)
    attributes: list[AttributeRecord] = field(default_factory=list)

    def add(self, severity: str, code: str, subject: str, detail: str) -> None:
        self.findings.append(Finding(severity, code, subject, detail))


def is_ignored(path: Path) -> bool:
    return any(part in IGNORED_PARTS or part.startswith("cmake-build-")
               for part in path.parts)


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def repository_files(root: Path, suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or is_ignored(path.relative_to(root)):
            continue
        if path.suffix in suffixes or path.name == "CMakeLists.txt":
            files.append(path)
    return sorted(files)


def files_under(root: Path, directories: Sequence[str],
                suffixes: set[str]) -> list[Path]:
    result: list[Path] = []
    for directory in directories:
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in suffixes and not is_ignored(
                    path.relative_to(root)):
                result.append(path)
    return sorted(set(result))


def discover_build_dir(root: Path, requested: Path | None) -> Path | None:
    if requested is not None:
        return requested.resolve()

    from_environment = os.environ.get("SCULPTOR_BUILD_DIR")
    candidates = []
    if from_environment:
        candidates.append(Path(from_environment))
    candidates.extend([
        root / "build",
        root.parent / "build" / root.name,
        root.parent.parent / "build" / root.name,
    ])
    for candidate in candidates:
        if (candidate / "build.ninja").exists() or (
                candidate / "compile_commands.json").exists():
            return candidate.resolve()
    return None


CPP_TOKEN_RE = re.compile(r"([A-Za-z0-9_./${}+-]+\.cpp)(?![A-Za-z0-9_.])")


def resolve_cmake_source(root: Path, cmake_file: Path,
                         token: str) -> tuple[Path | None, Path | None]:
    expanded = token
    expanded = expanded.replace("${CMAKE_CURRENT_SOURCE_DIR}",
                                cmake_file.parent.as_posix())
    expanded = expanded.replace("${PROJECT_SOURCE_DIR}", root.as_posix())
    expanded = expanded.replace("${CMAKE_SOURCE_DIR}", root.as_posix())
    if "${" in expanded:
        return None, None

    raw = Path(expanded)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [
            cmake_file.parent / raw,
            cmake_file.parent.parent / raw,
            root / raw,
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), None
    return None, candidates[0].resolve()


def parse_cmake_sources(root: Path) -> tuple[set[Path], list[tuple[Path, str,
                                                                  Path]]]:
    sources: set[Path] = set()
    unresolved: list[tuple[Path, str, Path]] = []
    for cmake_file in sorted(root.rglob("CMakeLists.txt")):
        if is_ignored(cmake_file.relative_to(root)):
            continue
        for token in CPP_TOKEN_RE.findall(read_text(cmake_file)):
            source, expected = resolve_cmake_source(root, cmake_file, token)
            if source is not None:
                sources.add(source)
            elif expected is not None:
                unresolved.append((cmake_file, token, expected))
    return sources, unresolved


def parse_configured_sources(root: Path, build_dir: Path | None) -> set[Path]:
    if build_dir is None:
        return set()

    result: set[Path] = set()
    compile_commands = build_dir / "compile_commands.json"
    if compile_commands.exists():
        try:
            entries = json.loads(read_text(compile_commands))
        except json.JSONDecodeError:
            entries = []
        for entry in entries:
            source = Path(entry.get("file", ""))
            if not source.is_absolute():
                source = Path(entry.get("directory", build_dir)) / source
            if source.is_file():
                result.add(source.resolve())

    ninja_file = build_dir / "build.ninja"
    if ninja_file.exists():
        for token in CPP_TOKEN_RE.findall(read_text(ninja_file)):
            raw = Path(token.replace("$ ", " "))
            candidates = [raw] if raw.is_absolute() else [build_dir / raw,
                                                           root / raw]
            for candidate in candidates:
                if candidate.is_file():
                    result.add(candidate.resolve())
                    break
    return result


def audit_sources(root: Path, build_dir: Path | None,
                  result: AuditResult) -> None:
    source_files = set(files_under(root, ("lib", "tools", "runtime"),
                                   {".c", ".cc", ".cpp"}))
    cmake_sources, unresolved = parse_cmake_sources(root)
    configured_sources = parse_configured_sources(root, build_dir)

    for cmake_file, token, expected in unresolved:
        result.add(
            "error", "CMAKE_SOURCE_MISSING", relative(expected, root),
            f"{relative(cmake_file, root)} lists `{token}`, but no source file exists."
        )

    for source in sorted(source_files - cmake_sources):
        path = relative(source, root)
        severity = "error" if path.startswith(("lib/", "tools/")) else "candidate"
        result.add(severity, "SOURCE_NOT_IN_CMAKE", path,
                   "No CMake source list contains this translation unit.")

    if configured_sources:
        expected_active = {
            source for source in source_files
            if relative(source, root).startswith(("lib/", "tools/", "runtime/common/"))
        }
        for source in sorted(expected_active & cmake_sources - configured_sources):
            result.add(
                "candidate", "SOURCE_NOT_IN_CONFIGURED_BUILD",
                relative(source, root),
                "CMake lists this source, but the selected build graph does not compile it."
            )

    result.facts.update({
        "translation_units": len(source_files),
        "cmake_translation_units": len(cmake_sources),
        "configured_translation_units": len(configured_sources),
    })


PASS_ARGUMENT_RE = re.compile(
    r"getArgument\s*\(\s*\)\s*const\s*final\s*\{.*?"
    r"return\s+\"([^\"]+)\"\s*;.*?\}", re.DOTALL)
PASS_CLASS_RE = re.compile(r"(?:struct|class)\s+([A-Za-z0-9_]+Pass)\b")
REGISTER_DECL_RE = re.compile(
    r"\bvoid\s+(register[A-Za-z0-9_]+Pass)\s*\([^;{}]*\)\s*;")
REGISTER_DEF_RE = re.compile(
    r"\bvoid\s+(register[A-Za-z0-9_]+Pass)\s*\([^;{}]*\)\s*\{")
REGISTER_CALL_RE = re.compile(
    r"\b(register[A-Za-z0-9_]+Pass)\s*\([^;{}]*\)\s*;")


def count_references(paths: Iterable[Path], needle: str) -> int:
    return sum(1 for path in paths if needle in read_text(path))


def audit_passes(root: Path, result: AuditResult) -> None:
    implementation_files = files_under(root, ("include", "lib"), CODE_SUFFIXES)
    test_files = files_under(root, ("tests",), TEXT_SUFFIXES)
    doc_files = files_under(root, ("docs", "src"), TEXT_SUFFIXES)
    root_readme = root / "README.md"
    if root_readme.exists():
        doc_files.append(root_readme)

    declarations: set[str] = set()
    definitions: dict[str, Path] = {}
    pass_arguments: list[tuple[str, Path, str | None]] = []
    for path in implementation_files:
        text = read_text(path)
        declarations.update(REGISTER_DECL_RE.findall(text))
        for name in REGISTER_DEF_RE.findall(text):
            if path.name != "Passes.cpp":
                definitions[name] = path
        for match in PASS_ARGUMENT_RE.finditer(text):
            prefix = text[:match.start()]
            classes = PASS_CLASS_RE.findall(prefix)
            pass_class = classes[-1] if classes else None
            pass_arguments.append((match.group(1), path, pass_class))

    registered_calls: set[str] = set()
    for path in implementation_files:
        if path.name == "Passes.cpp":
            registered_calls.update(REGISTER_CALL_RE.findall(read_text(path)))

    for name, path in sorted(definitions.items()):
        if name not in registered_calls:
            result.add(
                "error", "PASS_NOT_REGISTERED", name,
                f"{relative(path, root)} defines this pass registration, but no Passes.cpp calls it."
            )
    for name in sorted(registered_calls - definitions.keys()):
        result.add(
            "error", "PASS_REGISTRATION_MISSING", name,
            "A Passes.cpp entry point calls this function, but no local definition exists."
        )

    seen_arguments: dict[str, Path] = {}
    for argument, path, pass_class in sorted(pass_arguments):
        if argument in seen_arguments:
            result.add(
                "error", "DUPLICATE_PASS_ARGUMENT", argument,
                f"Defined by both {relative(seen_arguments[argument], root)} and {relative(path, root)}."
            )
        seen_arguments[argument] = path

        expected_registration = f"register{pass_class}" if pass_class else None
        if expected_registration and expected_registration not in declarations:
            result.add(
                "error", "PASS_REGISTRATION_UNDECLARED", argument,
                f"{relative(path, root)} has no `{expected_registration}()` declaration."
            )
        registered = bool(expected_registration and
                          expected_registration in registered_calls)
        test_count = count_references(test_files, argument)
        doc_count = count_references(doc_files, argument)
        if test_count == 0 and doc_count == 0:
            result.add(
                "candidate", "PASS_WITHOUT_USAGE", argument,
                "No test or documentation file names this registered pass."
            )
        result.passes.append(
            PassRecord(argument, relative(path, root), expected_registration,
                       registered, test_count, doc_count))

    result.facts["passes"] = len(pass_arguments)


PLANNER_CLASS_RE = re.compile(
    r"class\s+([A-Za-z0-9_]+Planner)\s+(?:final\s+)?:")
PLANNER_REGISTRATION_RE = re.compile(
    r"registerPlanner\s*\(\s*\"([^\"]+)\".*?"
    r"make_unique<([A-Za-z0-9_]+Planner)>", re.DOTALL)


def audit_planners(root: Path, result: AuditResult) -> None:
    planner_root = root / "lib/Dialect/Sculptor/Transforms/planners"
    if not planner_root.exists():
        return

    registry_path = planner_root / "RegisterMappingPlanners.cpp"
    registrations = {
        planner_class: name
        for name, planner_class in PLANNER_REGISTRATION_RE.findall(
            read_text(registry_path))
    }
    implementations: dict[str, Path] = {}
    for header in sorted(planner_root.rglob("*Planner.h")):
        for planner_class in PLANNER_CLASS_RE.findall(read_text(header)):
            implementations[planner_class] = header

    for planner_class, path in sorted(implementations.items()):
        if planner_class not in registrations:
            result.add(
                "error", "PLANNER_NOT_REGISTERED", planner_class,
                f"{relative(path, root)} defines this planner, but the registry omits it."
            )
    for planner_class, name in sorted(registrations.items()):
        if planner_class not in implementations:
            result.add(
                "error", "PLANNER_IMPLEMENTATION_MISSING", name,
                f"The registry names `{planner_class}`, but no planner header defines it."
            )
            continue
        path = implementations[planner_class]
        test_files = files_under(root, ("tests",), TEXT_SUFFIXES)
        doc_files = files_under(root, ("docs", "src"), TEXT_SUFFIXES)
        test_count = count_references(test_files, name)
        doc_count = count_references(doc_files, name)
        if test_count == 0 and doc_count == 0:
            result.add(
                "candidate", "PLANNER_WITHOUT_USAGE", name,
                "No test or documentation file names this registered planner."
            )
        result.planners.append(
            PlannerRecord(name, relative(path, root), planner_class, True,
                          test_count, doc_count))

    result.facts["planners"] = len(registrations)


INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]",
                        re.MULTILINE)


def audit_headers(root: Path, code_files: list[Path],
                  result: AuditResult) -> None:
    headers = files_under(root, ("include", "lib", "tools"),
                          {".h", ".hh", ".hpp"})
    included_names: set[str] = set()
    for path in code_files:
        included_names.update(INCLUDE_RE.findall(read_text(path)))

    unused = 0
    for header in headers:
        if header.is_relative_to(root / "include"):
            include_key = header.relative_to(root / "include").as_posix()
        else:
            include_key = header.name
        consumed = include_key in included_names or any(
            name.endswith("/" + header.name) or name == header.name
            for name in included_names)
        if not consumed:
            unused += 1
            result.add(
                "candidate", "HEADER_WITHOUT_CONSUMER", relative(header, root),
                "No in-repository include directive references this header."
            )
    result.facts.update({"headers": len(headers),
                         "headers_without_consumers": unused})


ATTR_RE = re.compile(
    r"\b(k[A-Za-z0-9_]*AttrName)\s*(?:\(\s*|=\s*)"
    r"\"(sculptor\.[^\"]+)\"\s*\)?\s*;", re.DOTALL)


CALL_BEFORE_IDENTIFIER_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>)?\s*\(")


def attribute_roles(code_files: list[Path], identifier: str) -> tuple[int, int]:
    producer_prefixes = (
        "add", "append", "attach", "build", "create", "emit", "set",
        "write")
    consumer_prefixes = (
        "collect", "decode", "extract", "find", "get", "has", "load",
        "lookup", "parse", "read", "remove", "require", "resolve",
        "validate", "verify")
    both_prefixes = ("clone", "copy", "move", "preserve")
    producer_calls = {"getnamedattr", "getnamedattribute", "namedattribute"}
    producer_matches = 0
    consumer_matches = 0
    identifier_pattern = re.compile(rf"\b{re.escape(identifier)}\b")

    for path in code_files:
        text = read_text(path)
        for occurrence in identifier_pattern.finditer(text):
            prefix = text[max(0, occurrence.start() - 320):occurrence.start()]
            calls = CALL_BEFORE_IDENTIFIER_RE.findall(prefix)
            if not calls:
                continue
            call = calls[-1].lower()
            if call in producer_calls or call.startswith(producer_prefixes):
                producer_matches += 1
            if call not in producer_calls and call.startswith(consumer_prefixes):
                consumer_matches += 1
            if call.startswith(both_prefixes):
                producer_matches += 1
                consumer_matches += 1
    return producer_matches, consumer_matches


def audit_attributes(root: Path, code_files: list[Path],
                     result: AuditResult) -> None:
    combined = "\n".join(read_text(path) for path in code_files)
    definitions: dict[str, tuple[str, Path]] = {}
    for path in code_files:
        for identifier, value in ATTR_RE.findall(read_text(path)):
            definitions[identifier] = (value, path)

    for identifier, (value, path) in sorted(definitions.items()):
        references = len(re.findall(rf"\b{re.escape(identifier)}\b", combined))
        producers, consumers = attribute_roles(code_files, identifier)
        record = AttributeRecord(identifier, value, relative(path, root),
                                 references, producers, consumers)
        result.attributes.append(record)
        if references <= 1:
            result.add(
                "candidate", "ATTRIBUTE_WITHOUT_REFERENCE", identifier,
                f"`{value}` is declared in {relative(path, root)} but has no code reference."
            )
        elif producers == 0 or consumers == 0:
            missing = "producer" if producers == 0 else "consumer"
            result.add(
                "candidate", "ATTRIBUTE_ROLE_UNCLEAR", identifier,
                f"The heuristic scan found no direct {missing} for `{value}`. Review wrapper-based uses before removal."
            )
    result.facts["attribute_constants"] = len(definitions)


def audit_residue(root: Path, result: AuditResult) -> None:
    generated: set[Path] = set()
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if ".git" in rel.parts:
            continue
        if path.is_dir() and path.name == "__pycache__":
            generated.add(path)
    for path in sorted(generated):
        result.add("candidate", "GENERATED_ARTIFACT", relative(path, root),
                   "Generated Python cache data is present in the source tree.")

    empty_directories: list[Path] = []
    for path in root.rglob("*"):
        relative_path = path.relative_to(root)
        if (not path.is_dir() or is_ignored(relative_path) or
                any(part.startswith(".") for part in relative_path.parts)):
            continue
        try:
            if not any(path.iterdir()):
                empty_directories.append(path)
        except OSError:
            continue
    for path in sorted(empty_directories):
        result.add("candidate", "EMPTY_DIRECTORY", relative(path, root),
                   "This source directory is empty.")
    result.facts["empty_directories"] = len(empty_directories)


def run_audit(root: Path, build_dir: Path | None = None) -> AuditResult:
    root = root.resolve()
    selected_build = discover_build_dir(root, build_dir)
    result = AuditResult(root.as_posix(),
                         selected_build.as_posix() if selected_build else None)
    text_files = repository_files(root, TEXT_SUFFIXES)
    code_files = [path for path in text_files if path.suffix in CODE_SUFFIXES]
    result.facts["text_files"] = len(text_files)

    audit_sources(root, selected_build, result)
    audit_passes(root, result)
    audit_planners(root, result)
    audit_headers(root, code_files, result)
    audit_attributes(root, code_files, result)
    audit_residue(root, result)
    result.findings.sort(key=lambda finding: (
        SEVERITY_ORDER[finding.severity], finding.code, finding.subject))
    result.passes.sort(key=lambda record: record.argument)
    result.planners.sort(key=lambda record: record.name)
    result.attributes.sort(key=lambda record: record.identifier)
    return result


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_findings(findings: list[Finding], severity: str) -> list[str]:
    selected = [finding for finding in findings if finding.severity == severity]
    if not selected:
        return ["None.", ""]
    lines = ["| Code | Subject | Detail |", "|---|---|---|"]
    for finding in selected:
        lines.append(
            f"| `{markdown_escape(finding.code)}` | "
            f"`{markdown_escape(finding.subject)}` | "
            f"{markdown_escape(finding.detail)} |")
    lines.append("")
    return lines


def render_markdown(result: AuditResult) -> str:
    counts = Counter(finding.severity for finding in result.findings)
    status = "FAIL" if counts["error"] else "PASS"
    lines = [
        "# Sculptor Maintenance Audit",
        "",
        f"**Status:** {status}",
        "",
        f"**Source root:** `{markdown_escape(result.root)}`",
        "",
        f"**Build directory:** `{markdown_escape(result.build_dir or 'not found')}`",
        "",
        "| Severity | Count |",
        "|---|---:|",
        f"| Structural error | {counts['error']} |",
        f"| Review candidate | {counts['candidate']} |",
        "",
        "Structural errors fail the default command. Review candidates require human inspection.",
        "",
        "## Structural Errors",
        "",
    ]
    lines.extend(render_findings(result.findings, "error"))
    lines.extend(["## Review Candidates", ""])
    lines.extend(render_findings(result.findings, "candidate"))

    lines.extend([
        "## Pass Inventory",
        "",
        "| Pass | Registered | Tests | Documentation | Definition |",
        "|---|:---:|---:|---:|---|",
    ])
    for record in result.passes:
        lines.append(
            f"| `{markdown_escape(record.argument)}` | "
            f"{'yes' if record.registered else 'no'} | "
            f"{record.test_references} | {record.documentation_references} | "
            f"`{markdown_escape(record.definition)}` |")
    lines.append("")

    lines.extend([
        "## Planner Inventory",
        "",
        "| Planner | Registered | Tests | Documentation | Implementation |",
        "|---|:---:|---:|---:|---|",
    ])
    for record in result.planners:
        lines.append(
            f"| `{markdown_escape(record.name)}` | "
            f"{'yes' if record.registered else 'no'} | "
            f"{record.test_references} | {record.documentation_references} | "
            f"`{markdown_escape(record.implementation)}` |")
    lines.append("")

    questionable_attributes = [
        record for record in result.attributes
        if record.references <= 1 or record.producer_matches == 0 or
        record.consumer_matches == 0
    ]
    lines.extend([
        "## Attribute Review",
        "",
        "This scan is heuristic. Wrapper functions can hide producers and consumers.",
        "",
        "| Identifier | Attribute | References | Producers | Consumers |",
        "|---|---|---:|---:|---:|",
    ])
    for record in questionable_attributes:
        lines.append(
            f"| `{markdown_escape(record.identifier)}` | "
            f"`{markdown_escape(record.value)}` | {record.references} | "
            f"{record.producer_matches} | {record.consumer_matches} |")
    if not questionable_attributes:
        lines.append("| _None_ | | | | |")
    lines.append("")

    lines.extend([
        "## Inventory Totals",
        "",
        "| Item | Count |",
        "|---|---:|",
    ])
    for name, value in sorted(result.facts.items()):
        lines.append(f"| `{markdown_escape(name)}` | {markdown_escape(value)} |")
    lines.append("")
    return "\n".join(lines)


def render_json(result: AuditResult) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Sculptor build registration and likely dead code.")
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Sculptor source root. The default is derived from this script.")
    parser.add_argument(
        "--build-dir", type=Path,
        help="Configured CMake build directory. The script auto-detects it when omitted.")
    parser.add_argument("--output", type=Path,
                        help="Write the report to this file instead of stdout.")
    parser.add_argument("--format", choices=("markdown", "json"),
                        default="markdown", help="Report format.")
    parser.add_argument(
        "--fail-on", choices=("error", "candidate", "never"), default="error",
        help="Select which findings produce a nonzero exit status.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.root.is_dir():
        print(f"error: source root does not exist: {args.root}", file=sys.stderr)
        return 2
    if args.build_dir is not None and not args.build_dir.is_dir():
        print(f"error: build directory does not exist: {args.build_dir}",
              file=sys.stderr)
        return 2

    result = run_audit(args.root, args.build_dir)
    report = render_markdown(result) if args.format == "markdown" else render_json(
        result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(report, end="")

    counts = Counter(finding.severity for finding in result.findings)
    if args.fail_on == "error" and counts["error"]:
        return 1
    if args.fail_on == "candidate" and (counts["error"] or counts["candidate"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
