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

If you set `trace_export_url`, OpenHands will stream redacted EventStream events to your endpoint in batches:

- **POST body:** `{ "events": [ { "session_id": "...", "user_id": null, "event": {...} }, ... ] }`
- **Auth:** optional `Authorization: Bearer <trace_export_api_key>`

### LLM Traces (Reasoning Capture)

OpenHands automatically captures LLM response metadata after each call:

| Field | Description |
|-------|-------------|
| `model` | Model name used |
| `prompt_tokens` | Input tokens |
| `completion_tokens` | Output tokens |
| `latency_ms` | Response time |
| `response_content` | Raw text/reasoning from LLM |
| `tool_calls` | List of tools called |
| `cost` | Cost if provided by the API |

These are emitted as `LLMResponseObservation` events and flow through the same trace exporter.

### Image Handling

When MCP tools return images (e.g., `computer_screenshot`), they are automatically parsed and stored in `MCPObservation.images`:

```python
obs = await runtime.call_tool_mcp(MCPAction(tool_name="computer_screenshot", tool_args={}))

if obs.has_images:
    for img in obs.images:
        print(f"Image: {img.mime_type}, {len(img.data)} bytes base64")
        # Use img.to_data_uri() for OpenAI-style image_url
```

The `LLMImageFormatter` can convert images to LLM-specific formats:

```python
from openhands.llm.image_formatter import LLMImageFormatter

# For OpenAI/GPT-4V
content_blocks = LLMImageFormatter.format_for_openai(obs.images)

# For Claude
content_blocks = LLMImageFormatter.format_for_claude(obs.images)

# Auto-detect based on model name
content_blocks = LLMImageFormatter.format_for_litellm(obs.images, model="claude-sonnet-4-20250514")
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: openenv.fleet` | Run `pip install "openenv-core[fleet]"` |
| `ValueError: fleet_api_key is required` | Set `fleet_api_key` in config or `FLEET_API_KEY` env var |
| `ErrorObservation: MCP Tool call failed` | Check Fleet environment supports the tool being called |
