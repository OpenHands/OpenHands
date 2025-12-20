## Fleet Runtime (OpenEnv / Fleet)

Run OpenHands against **remote Fleet environments** via OpenEnv, using the split-plane model:

- **Orchestration (HTTP)**: reset / step / state
- **Agent actions (MCP)**: tools/list + tools/call

### Step-by-step: run OpenHands on Fleet

#### 1) Install dependencies

From your OpenHands environment:

```bash
pip install "openenv-core[fleet]"
```

If you want Fleet dashboard sessions (LLM-call trace logging), also install Fleet SDK:

```bash
pip install fleet-python
```

#### 2) Get credentials

You need:
- **Fleet API key** (`FLEET_API_KEY`) and an **env key** (e.g. `amazon`)
- Your **LLM provider key** (OpenAI/Anthropic/etc.)

You can set Fleet credentials via environment variables:

```bash
export FLEET_API_KEY="fl_..."
```

#### 3) Create `config.toml`

```toml
[core]
runtime = "fleet"
default_agent = "CodeActAgent"

[llm]
# Use a vision-capable model if you want screenshots injected into the prompt
model = "claude-sonnet-4-20250514"
api_key = "sk-..."

[sandbox]
fleet_api_key = "fl_..."      # or use FLEET_API_KEY env var
fleet_env_key = "amazon"      # any Fleet env key (e.g., "ubuntu")

[agent]
# Inject MCP-returned images (e.g., screenshots) into the next LLM prompt when vision is active
enable_mcp_image_injection = true
mcp_max_images_per_observation = 2
# Optional: only inject images from specific MCP tools (use aliased names if configured)
# mcp_image_tool_allowlist = ["screenshot"]

# Optional: Fleet Sessions (Fleet dashboard logging of LLM calls)
fleet_session_export_enabled = true
# Optional overrides (otherwise uses env vars like FLEET_JOB_ID / FLEET_TASK_KEY / FLEET_INSTANCE_ID)
# fleet_session_export_job_id = "job_..."
# fleet_session_export_task_key = "task_..."
# fleet_session_export_instance_id = "inst_..."
# fleet_session_export_base_url = "https://api.fleetai.com"
# fleet_session_export_model = "anthropic/claude-sonnet-4"
```

#### 4) Run OpenHands

**CLI mode:**

```bash
python -m openhands.core.main -c config.toml
```

**GUI mode:**

```bash
make run
# then open http://localhost:3000
```

#### 5) Verify it’s working

On startup you should see logs like:

```
[FleetRuntime ...] Connecting to Fleet environment: amazon
[FleetRuntime ...] Resetting environment...
[FleetRuntime ...] Discovering tools...
[FleetRuntime ...] Discovered N tools: [...]
CodeActAgent initialized with N tools: [...]
```

If your Fleet environment exposes browser/computer tools and you have a vision-capable model, MCP screenshots will be included in the prompt automatically when the MCP tool returns images.

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: openenv.fleet` | Run `pip install "openenv-core[fleet]"` |
| Fleet sessions not logging | Install `fleet-python` and set `fleet_session_export_enabled = true` |
| `ValueError: fleet_api_key is required` | Set `sandbox.fleet_api_key` or `FLEET_API_KEY` |
| `ErrorObservation: MCP Tool call failed` | Check your Fleet env exposes the tool name being called |
