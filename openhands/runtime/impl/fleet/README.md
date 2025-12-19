## Fleet Runtime (OpenEnv / Fleet)

Run OpenHands against remote Fleet environments via OpenEnv, using the same **split-plane** model described in the OpenEnv Fleet README ([source](https://raw.githubusercontent.com/fleet-ai/OpenEnv/7c09d5b24a0394f760462a71095bb7721e91933c/src/envs/fleet_env/README.md)):

- **Orchestration (HTTP)**: reset / step / state
- **Agent actions (MCP)**: tools/list + tools/call

### Install

```bash
pip install "openenv-core[fleet]"
```

### Configure

Create or update `config.toml`:

```toml
[core]
runtime = "fleet"
default_agent = "CodeActAgent"

[llm]
model = "claude-sonnet-4-20250514"  # or gpt-4o, etc.
api_key = "sk-..."

[sandbox]
fleet_api_key = "fl_..."      # or set FLEET_API_KEY env var
fleet_env_key = "amazon"      # any Fleet env key (e.g., "ubuntu")

# Optional: export traces (actions/observations) to your API
# trace_export_url = "https://your-api.example.com/trace"
# trace_export_api_key = "..."
```

### Run

**CLI mode:**

```bash
python -m openhands.core.main -c config.toml
```

**GUI mode:**

```bash
make run
# then open http://localhost:3000
```

**Programmatic (SDK-style):**

```python
import asyncio
from openhands.core.config import load_config
from openhands.core.setup import create_runtime
from openhands.llm.llm_registry import LLMRegistry

config = load_config("config.toml")
llm_registry = LLMRegistry(config)
runtime = create_runtime(config, llm_registry)

# Connect to Fleet
asyncio.run(runtime.connect())

# Now runtime.available_tools contains the MCP tools from Fleet
print(runtime.available_tools)
```

### What you should see

On startup, `FleetRuntime` will:

1. Connect to Fleet and reset the environment
2. Call `list_tools()` over MCP and log the discovered tool names

Example logs:

```
[FleetRuntime ...] Connecting to Fleet environment: amazon
[FleetRuntime ...] Resetting environment...
[FleetRuntime ...] Discovering tools...
[FleetRuntime ...] Discovered 1 tools: ['computer']
CodeActAgent initialized with 1 tools: ['computer']
```

Then your agent (e.g., `CodeActAgent`) can call MCP tools by name. For example, in an OpenEnv Fleet environment that exposes only `computer`, the agent will emit tool calls like:

```
computer({action: "cursor_position"})
computer({action: "screenshot"})
```

### Traces to an API

Traces are sent to Fleet via **Fleet Sessions** (below).

### Fleet Sessions (Fleet Dashboard Logging)

If you want traces to appear in Fleet's dashboard sessions view, enable Fleet Session export. This logs each **LLM call** (input `history` + `response`) via `fleet.session(...).log(...)`.

In `config.toml`:

```toml
fleet_session_export_enabled = true

# Optional overrides (otherwise uses env vars like FLEET_JOB_ID / FLEET_TASK_KEY / FLEET_INSTANCE_ID)
# fleet_session_export_job_id = "job_..."
# fleet_session_export_task_key = "task_..."
# fleet_session_export_instance_id = "inst_..."
# fleet_session_export_base_url = "https://api.fleetai.com"
# fleet_session_export_model = "anthropic/claude-sonnet-4"
```

Notes:
- Requires `fleet-sdk` to be installed and importable as `fleet`.
- Uses `sandbox.fleet_api_key` (or `FLEET_API_KEY`) to authenticate.

### Browser / Computer tools (injectible)

Fleet environments expose a **unified action space** via MCP tool discovery (e.g. `computer_screenshot`, `mouse_click`, etc.).

OpenHands currently treats these as raw MCP tools (`MCPAction` → `MCPObservation`). The next step is to add a small “browser tool mapping” layer so users can:
- alias/rename tools
- override schemas/descriptions
- define which tool(s) constitute “browser/computer use” for their environment

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: openenv.fleet` | Run `pip install "openenv-core[fleet]"` |
| `ValueError: fleet_api_key is required` | Set `fleet_api_key` in config or `FLEET_API_KEY` env var |
| `ErrorObservation: MCP Tool call failed` | Check Fleet environment supports the tool being called |
