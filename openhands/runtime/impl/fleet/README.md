## Fleet Runtime (OpenEnv + Fleet)

Run OpenHands against **remote Fleet environments** via **OpenEnv**, using the split-plane model:

- **Orchestration (HTTP)**: provision + reset lifecycle (OpenEnv `FleetEnvClient`)
- **Agent actions (MCP)**: tools/list + tools/call

### Step-by-step: run OpenHands on Fleet

#### 1) Install dependencies

##### 1a) Install Poetry (via pipx)

Poetry is the recommended way to manage OpenHands' Python dependencies. Install it with `pipx` (per Poetry docs):

```bash
pipx install poetry
```

If `poetry` isn't found after install, ensure your shell PATH is set:

```bash
pipx ensurepath
```

##### 1b) Install OpenHands (Poetry environment)

From the OpenHands repo root:

```bash
poetry install --with fleet
```

This installs OpenHands plus the optional `fleet` dependency group (`openenv[fleet]` + `fleet-python`).

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
runtime = "openenv" # (alias: "fleet")
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
poetry run python -m openhands.core.main -c config.toml
```

**GUI mode:**

```bash
make run
# then open http://localhost:3000
```

#### 5) Verify it’s working

On startup you should see logs like:

```
[OpenEnvRuntime ...] Connecting to Fleet environment: amazon
[OpenEnvRuntime ...] Resetting environment...
[OpenEnvRuntime ...] Discovered N tools: [...]
CodeActAgent initialized with N tools: [...]
```

If your Fleet environment exposes browser/computer tools and you have a vision-capable model, MCP screenshots will be included in the prompt automatically when the MCP tool returns images.

### Multi-task evals (single Fleet job)

OpenHands itself runs **one task per session**. To evaluate **many tasks under a single Fleet job** (so they show up together in the Fleet dashboard), use the outer runner:

```bash
poetry run python -m openhands.evaluation.fleet_job_runner \
  --config /path/to/config.toml \
  --project-key <FLEET_PROJECT_KEY> \
  --job-name "my-openhands-eval" \
  --max-concurrent 4
```

Or run a fixed set of tasks:

```bash
poetry run python -m openhands.evaluation.fleet_job_runner \
  --config /path/to/config.toml \
  --task-keys task1,task2,task3 \
  --job-name "my-openhands-eval" \
  --max-concurrent 4
```

Concrete example (do **not** include `...`):

```bash
poetry run python -m openhands.evaluation.fleet_job_runner \
  --config fleet.toml \
  --task-keys validate_dissenter_biolabs_deal_stage_and_email,validate_julie_smith_contact_update,validate_deal_modification_and_contact_addition \
  --job-name "test-eval" \
  --max-concurrent 4
```

Notes:
- The runner creates **one Fleet trace job** (`fleet.job_async(...)`) and logs **one session per task** (unique `task_key`).
- Fleet session completion is **explicit** so verifiers can run first.

### Verification (verifiers)

The multi-task runner runs Fleet task verifiers (when present) after OpenHands finishes the task, then marks the Fleet session:
- `complete(verifier_execution_id=...)` on success
- `fail(verifier_execution_id=...)` on failure

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Fleet/OpenEnv deps missing | Run `poetry install --with fleet` |
| Fleet sessions not logging | Install `fleet-python` and set `fleet_session_export_enabled = true` |
| `ValueError: fleet_api_key is required` | Set `sandbox.fleet_api_key` or `FLEET_API_KEY` |
| `ErrorObservation: MCP Tool call failed` | Check your Fleet env exposes the tool name being called |
