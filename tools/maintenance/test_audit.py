#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import audit


class AuditTest(unittest.TestCase):
    def test_attribute_wrapper_roles_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "Example.cpp"
            source.write_text(
                """
                void produce() {
                  setOptionalI64(kExampleAttrName, value);
                }
                void consume() {
                  verifyOptionalIdentity(op, kExampleAttrName, expected);
                }
                """,
                encoding="utf-8")

            producers, consumers = audit.attribute_roles(
                [source], "kExampleAttrName")

            self.assertEqual(producers, 1)
            self.assertEqual(consumers, 1)

    def test_unlisted_library_source_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "lib/Example.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("void example() {}\n", encoding="utf-8")
            (root / "CMakeLists.txt").write_text("", encoding="utf-8")

            result = audit.run_audit(root)

            self.assertTrue(any(
                finding.code == "SOURCE_NOT_IN_CMAKE" and
                finding.severity == "error" and
                finding.subject == "lib/Example.cpp"
                for finding in result.findings))

    def test_parent_scope_cmake_source_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "lib/mapping/Example.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("void example() {}\n", encoding="utf-8")
            cmake = source.parent / "CMakeLists.txt"
            cmake.write_text(
                "set(EXAMPLE_SOURCES mapping/Example.cpp PARENT_SCOPE)\n",
                encoding="utf-8")

            sources, unresolved = audit.parse_cmake_sources(root)

            self.assertIn(source.resolve(), sources)
            self.assertEqual(unresolved, [])

    def test_registered_pass_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = root / "include/Example.h"
            source = root / "lib/Example.cpp"
            passes = root / "lib/Passes.cpp"
            tests = root / "tests/example.mlir"
            include.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            tests.parent.mkdir(parents=True)
            include.write_text(
                """
                struct ExamplePass {
                  StringRef getArgument() const final {
                    return "sculptor-example";
                  }
                };
                void registerExamplePass();
                """,
                encoding="utf-8")
            source.write_text(
                "void registerExamplePass() { PassRegistration<ExamplePass>(); }\n",
                encoding="utf-8")
            passes.write_text(
                "void registerPasses() { registerExamplePass(); }\n",
                encoding="utf-8")
            tests.write_text(
                "// RUN: sculptor-mlir-opt %s --sculptor-example\n",
                encoding="utf-8")

            result = audit.run_audit(root)

            self.assertEqual(len(result.passes), 1)
            self.assertTrue(result.passes[0].registered)
            self.assertEqual(result.passes[0].test_references, 1)
            self.assertFalse(any(
                finding.code.startswith("PASS_") and
                finding.severity == "error"
                for finding in result.findings))


if __name__ == "__main__":
    unittest.main()
