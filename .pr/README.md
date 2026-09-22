# PR Artifacts

This directory contains generated PR-only QA artifacts. The PR Artifacts
workflow removes it after approval so these files do not enter the final squash
merge.

## `repro-17615/before-after.png`

Reproduction evidence for #17615, captured against
`openhands-sdk==1.49.3` / `mcp 1.30.0` (the `fastmcp<4` pin). It shows the same
`SystemPromptEvent` payload, persisted with `mcp` 2.x field names, read by the
pinned `mcp` 1.x reader:

- **before** the compatibility module: a pydantic `ValidationError` for the
  missing `inputSchema` field, which is what closes the event WebSocket and
  makes the page loop on `Unable to connect to server`;
- **after** importing `mcp_legacy_schema_compat` (as `--import-modules` does at
  startup): the event loads, `mcp_tool.inputSchema` is populated, and
  round-trip serialization still emits the camelCase spelling.

Regenerate with the commands under "How to Test" in the PR description.
