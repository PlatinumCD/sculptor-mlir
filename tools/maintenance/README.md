# Maintenance Audit

`audit.py` finds structural registration errors and likely dead code.

Run the audit from the Sculptor source root:

```bash
python3 tools/maintenance/audit.py \
  --build-dir ../../build/sculptor-mlir-pivot \
  --output /tmp/sculptor-maintenance-audit.md
```

The command checks these items:

- C++ source registration in CMake and the configured build.
- Pass implementation and pass registration.
- Pass references in tests and documentation.
- Mapping planner implementation and registry entries.
- Planner references in tests and documentation.
- Headers with no in-repository include directive.
- Attribute constants with no clear producer or consumer.
- Empty directories and generated Python cache data.

Structural errors cause a nonzero exit status. Review candidates do not fail the default command.

Use `--fail-on candidate` for a strict audit. Use `--format json` for machine-readable output.

The attribute check is heuristic. Review each attribute candidate before you remove code.
