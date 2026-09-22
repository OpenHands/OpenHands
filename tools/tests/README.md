# tools/ tests

Python tests for the modules the agent-server imports at startup via
`--import-modules`. They are not run by `npm test`; `.github/workflows/ci-script-tests.yml`
runs them with pytest alongside the `.github/scripts/` checkers.

## Running locally

```bash
pip install pytest "openhands-sdk==1.49.3"
python -m pytest tools/tests/ -v
```

`openhands-sdk` is pinned to the same version the launchers install (see
`config/defaults.json`) so the environment resolves the same `mcp` version the
agent-server does. That version is load-bearing for
`test_mcp_legacy_schema_compat.py`: the module under test only installs its
aliases when the installed `mcp` is the strict 1.x reader, so the reader cases
skip under `mcp` 2.x.

## mcp legacy-schema compatibility

`mcp_legacy_schema_compat.py` is the fix for
[OpenHands/OpenHands#17615](https://github.com/OpenHands/OpenHands/issues/17615).
Agent Canvas 1.21.0 pins `fastmcp<4`, which downgrades the transitive `mcp`
package from 2.x to 1.x. The two majors spell several MCP wire fields
differently (`input_schema` vs `inputSchema`, `is_error` vs `isError`), and 1.x
refuses to read what 2.x wrote — so conversations that had an MCP server
attached fail to restore after the upgrade.

The module adds an `AliasChoices` covering both spellings to every `mcp.types`
model field, then recompiles the SDK models that embed them. It runs at import
time and is a no-op under `mcp` 2.x. Remove it once the pinned agent-server reads
both spellings natively.
