# Tool-selection end-to-end validation

Run 2026-09-29 on a native `npm run dev:minimal` stack (no Docker) with the agent-server built from SDK `agent-profile-tool-catalog` @ `a3359d1cd` (`OH_AGENT_SERVER_LOCAL_PATH`), Canvas `feat-profile-tool-catalog`, and the mock LLM from `tests/e2e/mock-llm`. The stack ran twice: once as is, once with `VITE_ENABLE_BROWSER_TOOLS=false` (the agent-server then runs with `OH_ENABLE_BROWSER=false`). No paid model was used. Tools the model saw come from the mock's `GET /admin/requests`.

## Results

| Check | Result |
|---|---|
| `GET /api/tools/catalog` | 20 entries; default set = terminal, file_editor, task_tracker, browser_tool_set, switch_llm |
| Catalog `browser_tool_set.usable` | `true`; `false` with the browser off |
| Create profile with `tools` unset | 201; stored `tools=null`; no `enable_*` switch keys |
| Create profile with `[terminal, glob]` | 201; stored as given |
| GET an unset-tools profile, set `[terminal, glob]`, save under a new name | 201; `[terminal, glob]` (no `switch_llm` re-added) |
| Save a profile carrying `enable_sub_agents` | accepted as deprecated input and folded into `tools` (adds `task_tool_set`); switch not stored. Since SDK `b3eb043`; covered by SDK unit tests, not re-run live |
| Stored v2 profile, `tools=null` + `enable_sub_agents=true` | loads as v3 `[terminal, file_editor, task_tracker, browser_tool_set, task_tool_set, switch_llm]`; re-save writes v3 without switches |
| Stored v2 profile, `[terminal, glob]` + `enable_switch_llm_tool=false` | loads as `[terminal, glob]`; re-save writes v3 without switches |
| Stored v2 profile with string switches (`"false"`/`"true"`) | coerced; `[terminal, switch_llm]` |
| Materialize a draft selecting an unregistered tool | `valid=false`; `unusable_tools=["definitely_not_a_tool"]`; error names the tool |
| Materialize a draft selecting `[terminal, glob]` | `valid=true`; `unusable_tools=[]` |
| Materialize with the browser off, browser selected | browser listed in `unusable_tools` but not an error (the launch leaves it out) |
| `GET /api/settings`, agent schema | no `enable_sub_agents` / `enable_switch_llm_tool` |
| `PATCH agent_settings_diff {enable_sub_agents: true}` | folded into `tools` (adds `task_tool_set`); flags not returned |
| Stored v6 `settings.json` with both switches | loads as `tools=[terminal, file_editor, task_tracker, browser_tool_set, task_tool_set]`; re-save writes v7 without switches |
| API launch, settings with `tools` unset | model tools: terminal, file_editor, task_tracker, the browser tools, finish, think, switch_llm; no browser tools with the browser off |
| API launch, settings `[terminal, browser_tool_set]` | terminal and the browser tools, no switch_llm; only terminal with the browser off |
| API launch, profile `tools=null` | standard set incl. browser and switch_llm; no browser with the browser off |
| API launch, profile `[terminal, glob]` | model tools: terminal, glob (+ finish, think, invoke_skill) |
| API launch, profile `[terminal, task_tool_set]` | model tools include `task` |
| UI: new chat on global settings | launch sends `agent_settings` (schema 7) with no `tools`; the model received the standard set incl. the browser tools and switch_llm |
| UI: profile editor, Choose tools → tick glob + task_tool_set, untick browser + switch_llm, save | stored `[file_editor, task_tool_set, task_tracker, terminal, glob]` |
| UI: reload and reopen that profile | same selection shown; unticking glob and saving removes it |
| UI: settings pages | no sub-agent or LLM-switching toggles anywhere; `/settings/agent` redirects to the profile library |
| `tool-selection-smoke.mjs` | empty, glob and standard profiles all finish with the expected model tools, no duplicates |

The browser-less-host branch of the dry run was not exercised live (this host has a browser); it is covered by SDK unit tests.

## Screenshots

- `01-standard-tools.png`: editing a profile with standard tools.
- `02-choose-tools.png`: a custom selection.

## Reproduce

Start the mock LLM from the SDK checkout:

```sh
uv run python /absolute/path/to/OpenHands/tests/e2e/mock-llm/scripts/mock-llm-server.py --port 18399
```

From the Canvas checkout, start an isolated stack against the SDK branch. On macOS, set `TMUX_TMPDIR` to a short path when the state dir is deep.

```sh
E2E=/private/tmp/tool-selection-e2e
OH_AGENT_SERVER_LOCAL_PATH=/absolute/path/to/software-agent-sdk \
OH_CANVAS_SAFE_STATE_DIR=$E2E/state \
OH_SESSION_API_KEY_PATH=$E2E/api-key.txt \
OH_SECRET_KEY_PATH=$E2E/secret-key.txt \
OH_CANVAS_SAFE_BACKEND_PORT=18300 \
OH_CANVAS_SAFE_VSCODE_PORT=18301 \
VITE_FRONTEND_PORT=3021 \
VITE_WORKING_DIR=$E2E/workspace \
VITE_DO_NOT_TRACK=1 npm run dev:minimal
```

In this run the agent-server's `OH_PERSISTENCE_DIR` was the directory holding the key files; point `E2E_PERSISTENCE_DIR` at whatever yours is. Then:

```sh
E2E_PERSISTENCE_DIR=$E2E python .pr/tool-selection-e2e-api.py
E2E_PERSISTENCE_DIR=$E2E python .pr/tool-selection-e2e-stored.py
E2E_BROWSER_CHANNEL=chrome node .pr/tool-selection-e2e-ui.mjs
TOOL_SELECTION_ISOLATED=1 TOOL_SELECTION_SERVER_URL=http://127.0.0.1:18300 \
  TOOL_SELECTION_MOCK_URL=http://127.0.0.1:18399 TOOL_SELECTION_KEY_FILE=$E2E/api-key.txt \
  TOOL_SELECTION_WORKSPACE=$E2E/workspace node .pr/tool-selection-smoke.mjs
```

Restart the stack with `VITE_ENABLE_BROWSER_TOOLS=false` and run `tool-selection-e2e-api.py` with `E2E_EXPECT_BROWSER=0` for the browser-off checks. Launch payloads must carry `schema_version: 7`; a payload without one is read as the oldest schema and migrated. The Python scripts need `httpx` (the SDK venv has it). `tool-selection-e2e-stored.py` writes v2 profiles and a v6 `settings.json` into the persistence dir and restores `settings.json` afterwards. Use a disposable stack only.
