# PR #15103 final local SaaS/MCP evidence

Date: 2026-07-13 UTC

## Refs

- Pull request: https://github.com/OpenHands/OpenHands/pull/15103
- Linked issue: https://github.com/OpenHands/OpenHands/issues/15226
- Branch: `codex/preserve-mcp-auth-headers`
- GitHub PR head confirmed before rerun: `40b4c931a3f2da7481f1d9811437ad2ada5af115`
- GitHub `refs/pull/15103/head` confirmed before rerun: `40b4c931a3f2da7481f1d9811437ad2ada5af115`
- Current `main` confirmed before rerun: `3949e1cc17d9443f1f4ef7d34d428baf065cd919`
- Product head tested: `40b4c931a3f2da7481f1d9811437ad2ada5af115`
- Current-main baseline tested: `3949e1cc17d9443f1f4ef7d34d428baf065cd919`

## Setup

- Installed pre-commit hooks with `make install-pre-commit-hooks` before editing evidence.
- Created detached worktrees for the exact PR head and exact current main.
- Verified `git status --porcelain=v1` was empty immediately before both harness executions.
- Built frontend production assets in both detached worktrees.
- Ran disposable local PostgreSQL and Redis containers; generated local-only database/JWT credentials were not written to this artifact.
- Applied enterprise migrations through PR-head revision `137` for the PR run and through current-main revision `136` for the main run.
- Ran the real enterprise `saas_server:app` with `SaaSServerConfig`, production API-key auth, process sandbox, two disposable org members, a local authenticated deterministic OpenAI-compatible LLM, and a local authenticated FastMCP streamable-HTTP server.
- In this container, `host.docker.internal` was mapped to loopback so process-sandbox health checks could reach sibling agent-server processes. This was an environment setup change only; the repository checkout stayed clean.

Raw logs remained in `/tmp` only. This artifact records only SHAs, booleans, short fingerprints, and structural outcomes. It omits API keys, bearer tokens, session keys, JWT values, MCP secret values, sandbox URLs, and service ports.

## PR-head result

Result: `PASS`

Clean checkout:

```text
worktree: /tmp/oh-pr15103-head-40b4c93
git status before execution: clean
HEAD: 40b4c931a3f2da7481f1d9811437ad2ada5af115
```

All final-head checks passed:

```json
{
  "api_redaction_no_raw_settings": true,
  "api_redaction_no_raw_users_me": true,
  "fresh_conversation_ready": true,
  "fresh_conversation_terminal_or_running": true,
  "llm_saw_target_tool": true,
  "llm_saw_tool_result": true,
  "mcp_authorized_tool_traffic": true,
  "missing_llm_rejected": true,
  "peer_member_cannot_see_mcp": true,
  "redacted_http_auth_not_exposed": true,
  "redacted_http_header_not_exposed": true,
  "redacted_stdio_env_not_exposed": true,
  "unrelated_mcp_edit_survived_restart": true,
  "wrong_mcp_credentials_rejected": true
}
```

Conversation outcome:

```json
{
  "app_conversation_id_fp": "8a2fe9c7a10a592a",
  "execution_status": "finished",
  "sandbox_status": "RUNNING",
  "start_task_status": "READY"
}
```

Observed local LLM/MCP traffic:

```json
{
  "llm_calls": [
    {"authorized": true, "has_tools": false, "saw_tool_result": false, "target_tool_seen": false, "tool_count": 0},
    {"authorized": true, "has_tools": true, "saw_tool_result": false, "target_tool_seen": true, "tool_count": 23},
    {"authorized": true, "has_tools": true, "saw_tool_result": true, "target_tool_seen": true, "tool_count": 23}
  ],
  "mcp_authorized_events": 5,
  "mcp_unauthorized_events": 1
}
```

Secret fingerprints observed by the harness:

```json
{
  "llm_api_key": "e91095b592a79651",
  "mcp_bearer": "b99a09310a3f2297",
  "mcp_header": "f20071a2f39d732f",
  "stdio_api_key": "2d04508cb5702591",
  "stdio_other": "b12eaba336430d73"
}
```

The one unauthorized MCP event was the intentional wrong/unrelated-server path. The preserved authenticated server still listed and invoked `preserved_auth_probe`, and the deterministic LLM saw the tool result.

## Current-main baseline

Clean checkout:

```text
worktree: /tmp/oh-pr15103-main-3949e1
git status before execution: clean
HEAD: 3949e1cc17d9443f1f4ef7d34d428baf065cd919
```

Same full-app harness shape:

```json
{
  "result": "FAIL",
  "failure": "start task did not finish",
  "last_status": "WAITING_FOR_SANDBOX",
  "agent_server_url_present": false
}
```

This happened before the MCP preservation assertions. It matches the PR-side process-sandbox startup/status fix noted in the PR body and prevents current `main` from completing the full local SaaS conversation path in this container.

MCP-specific early guard on the same clean main checkout:

```json
{
  "adapter_file_present": false,
  "adapter_module_present": false,
  "failure": "missing SDK-native MCP config adapter / secret preservation path",
  "head": "3949e1cc17d9443f1f4ef7d34d428baf065cd919",
  "result": "FAIL",
  "status_before_empty": true
}
```

## Durable files

- Harness: `.pr/logs/full_local_saas_mcp_live.py`
- Evidence: `.pr/logs/full_local_saas_mcp_live_evidence.md`
