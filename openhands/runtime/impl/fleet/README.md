## Fleet Runtime (OpenEnv / Fleet)

Run OpenHands against remote Fleet environments via OpenEnv, using the same **split-plane** model described in the OpenEnv Fleet README ([source](https://raw.githubusercontent.com/fleet-ai/OpenEnv/7c09d5b24a0394f760462a71095bb7721e91933c/src/envs/fleet_env/README.md)):

- **Orchestration (HTTP)**: reset / step / state
- **Agent actions (MCP)**: tools/list + tools/call

### Install

```bash
pip install "openenv-core[fleet]"
```

### Configure

In `config.toml`:

```toml
[core]
runtime = "fleet"

[sandbox]
fleet_api_key = "..."      # or set via env
fleet_env_key = "amazon"   # any Fleet env key

# Optional: export traces (actions/observations) to your API
trace_export_url = "https://your-api.example.com/trace"
# trace_export_api_key = "..."
```

### Run

Run OpenHands normally (GUI/CLI/etc.) using this config. The key change is `runtime = "fleet"`.

### What you should see

On startup, `FleetRuntime` will:

- connect + reset
- `list_tools()` over MCP and log the discovered tool names (e.g. `computer`)

Then your agent (e.g. `CodeActAgent`) can call MCP tools by name. For example, in an OpenEnv Fleet environment that exposes only `computer`, the agent will emit tool calls like:

- `computer({action: "cursor_position"})`

### Traces to an API

If you set `trace_export_url`, OpenHands will stream redacted EventStream events to your endpoint in batches:

- POST body: `{ "events": [ { "session_id": "...", "user_id": null, "event": {...} }, ... ] }`
- Auth: optional `Authorization: Bearer <trace_export_api_key>`

