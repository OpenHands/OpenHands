# PR #15103 local SaaS MCP secret proof

Date: 2026-07-13 UTC

## Refs

- Branch: `codex/preserve-mcp-auth-headers`
- PR head tested: `a3cbe1cfa17bacca4e1e6c6f855085a293d4df5e`
- Current `origin/main` fetched for comparison: `3949e1cc17d9443f1f4ef7d34d428baf065cd919`
- Source comparison: `origin/main` does not contain `openhands/app_server/mcp/mcp_config_adapter.py`; PR head adds the adapter path used by the SDK-native MCP preservation behavior.

## Setup

- Ran `make install-pre-commit-hooks`.
- Ran enterprise dependencies from `enterprise` with Poetry.
- Ran local PostgreSQL and Redis containers on host ports `15432` and `16379`.
- Applied enterprise migrations through `alembic_version=137`.
- Built `frontend/build` so the enterprise SaaS app could start normally.
- Ran the real enterprise app through `saas_server:app` with `OPENHANDS_CONFIG_CLS=server.config.SaaSServerConfig`.
- Used the process sandbox service and the real app conversation/start-task APIs.
- Seeded two disposable users as active members in one disposable org using the production SQL models and API-key auth path.
- Used a local authenticated OpenAI-compatible deterministic LLM server.
- Used a local authenticated FastMCP streamable HTTP server that required both bearer auth and an extra HTTP header.

## Command

```bash
DB_HOST=127.0.0.1 DB_PORT=15432 DB_NAME=openhands_pr15103 \
DB_USER=postgres DB_PASS=<local-db-password> REDIS_HOST=127.0.0.1 REDIS_PORT=16379 \
JWT_SECRET=<local-jwt-secret> OPENHANDS_SUPPRESS_BANNER=1 \
PYTHONPATH=enterprise:. \
poetry -C enterprise run python ../.pr/logs/full_local_saas_mcp_live.py
```

The script prints only structural results and short fingerprints. It does not print raw API keys, bearer tokens, cookies, session keys, or MCP credential values.

## Result

`result: PASS`

Checks that passed:

- Missing LLM credential was rejected by the authenticated local LLM server.
- Wrong MCP bearer credential was rejected by the FastMCP server.
- `GET /api/v1/settings` did not expose raw MCP or LLM secrets.
- `GET /api/v1/users/me` did not expose raw MCP or LLM secrets.
- The peer member in the same org could not see the configured member-private MCP settings.
- SDK-native HTTP MCP auth value was not exposed by the settings API.
- SDK-native HTTP MCP header secret was not exposed by the settings API.
- SDK-native stdio MCP env secrets were not exposed by the settings API.
- A redacted GET followed by an unrelated MCP edit survived app restart.
- A fresh app conversation became ready and finished.
- The deterministic LLM saw the preserved MCP tool in the real tool list.
- The deterministic LLM received the MCP tool result.
- The FastMCP server received authorized traffic with the recovered bearer and header secrets.

Live traffic observed in the successful fresh conversation:

- FastMCP `POST /mcp` and `GET /mcp` with matching bearer/header fingerprints.
- FastMCP `ListToolsRequest`.
- LLM chat completions with tool definitions present.
- FastMCP `CallToolRequest` for the preserved tool.
- Final LLM call with the tool result present.

One unauthorized `POST /mcp` also appeared in the successful conversation from the intentionally unrelated MCP server configured with `auth.strategy=none`; it did not prevent the preserved authenticated server from listing and invoking the tool.

## Durable Files

- Harness: `.pr/logs/full_local_saas_mcp_live.py`
- Evidence: `.pr/logs/full_local_saas_mcp_live_evidence.md`

## Sources

- PASS output from `/tmp/oh-pr15103-full-local-run.log`, sanitized with the same redaction pattern used during the run.
- Product paths exercised by the harness: `enterprise/saas_server.py`, `/api/v1/settings`, `/api/v1/users/me`, `/api/v1/app-conversations`, `/api/v1/app-conversations/start-tasks/search`.
- Issue-specific code paths: `openhands/app_server/mcp/mcp_config_adapter.py`, `openhands/app_server/sandbox/process_sandbox_service.py`, `tests/unit/storage/data_models/test_settings.py`, `tests/unit/app_server/test_process_sandbox_service.py`.
