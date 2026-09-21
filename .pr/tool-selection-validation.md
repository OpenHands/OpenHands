# Tool-selection follow-up verification

Verified 2026-09-21 against Canvas head `448bd1e56415abb0c74400e9143746447c3ccecf` and SDK head `cc0f2493d5f8cf89ceb8768f7a81f942d547d873`.

## Scope and results

This builds on Simon's fixes. Canvas retains the empty/failed standard-response guard and preserves intentional empty selections. Its remaining change initializes a custom selection when both a nonempty standard answer and the catalog arrive, without replacing user edits. Save is disabled while that selection is still uninitialized. The initialization effect precedes the settings-reset effect so a reload's reset wins.

Five real-query-hook cases cover saved/cleared empty selections, an unnamed profile's delayed defaults, edits while pending, and a late catalog after an empty answer. They live separately because the older route suite mocks the hooks globally; this file mocks only services and exercises the actual disabled-to-enabled query transition.

- Before: on the current Canvas head, the unnamed-profile test failed because Save was valid while the custom selection was still unresolved (3 other cases passed).
- After: full Canvas suite passed 724 files / 7,374 tests, with 7 todo. Lint and typecheck passed.
- Live before: Add agent profile, leave name blank, select Choose tools, then enter a name with the mock LLM profile selected. On upstream, all switches remained off even after the mode control unlocked and Save was enabled.
- Live after: the same sequence selected file_editor, task_tracker, terminal, browser_tool_set, and switch_llm. Saved as review-followup-seed and reopened; all five selections persisted. UI checks used the dedicated existing localhost:18501/18500 stack, not the user's instance.
- Fresh SDK smoke: four conversations finished against a newly started stack using SDK cc0f2493 plus the test-only follow-up. Empty, glob-only, standard and legacy SwitchLLMTool selections produced the expected model-facing tools without duplicates. No paid model was used.
- SDK follow-up: only tests/sdk/tool/test_switch_llm.py changes. Tests inspect conversation.agent after initialization and initialize both alias spellings. All 342 tests across the five focused SDK/server files passed; scoped pre-commit checks passed. No SDK production changes are included.

Local evidence logs: /private/tmp/tool-followup-before.log, tool-followup-canvas-full.log, tool-followup-canvas-lint.log, tool-followup-canvas-types.log, tool-followup-sdk-tests.log, tool-followup-sdk-hooks.log and tool-followup-smoke.log. Screenshots: /private/tmp/tool-followup-before.png and tool-followup-after.png (the latter shows the saved profile reopened). These machine-local paths are not accessible to remote reviewers until shared.

```sh
TZ=UTC npm test -- --run --maxWorkers=4
npm run lint
npm run typecheck
uv run pytest tests/sdk/tool/test_switch_llm.py tests/sdk/test_settings.py tests/sdk/profiles/test_agent_profile.py tests/agent_server/test_agent_profile_conv_start.py tests/agent_server/test_conversation_router.py -q --tb=short
uv run pre-commit run --files tests/sdk/tool/test_switch_llm.py
```

Canvas used Node 24. SDK tests used a local pytest isolation plugin that redirects home-path lookups to an empty temporary directory, and an ignored comment-only .env to block inherited tracing configuration. No tests were excluded. Remote CI has not run on these local follow-ups; these numbers are local results, not a claim of PR approval or benchmark validation.

## Reproducible live smoke check

The adjacent `tool-selection-smoke.mjs` is a manual integration check, not a CI test. It creates a uniquely named fake-key LLM profile, four agent profiles, four conversations, and per-conversation workspace directories. It resets the mock server's request history. Use a dedicated disposable stack, never an existing personal or production instance. It leaves fixtures for inspection; stop the stack when finished. No paid model is involved.

Start the mock server from the SDK checkout (with its dependencies installed):

```sh
uv run python /absolute/path/to/OpenHands/tests/e2e/mock-llm/scripts/mock-llm-server.py --port 18504
```

In another terminal, from the Canvas checkout, start an isolated development stack. Set `OH_AGENT_SERVER_LOCAL_PATH` to the SDK PR checkout including the follow-up fixes. Choose unused ports if needed. The launcher creates its state and keys; do not print or commit the keys.

```sh
OH_AGENT_SERVER_LOCAL_PATH=/absolute/path/to/software-agent-sdk \
OH_CANVAS_SAFE_STATE_DIR=/private/tmp/tool-selection-live/state \
OH_SESSION_API_KEY_PATH=/private/tmp/tool-selection-live/api-key.txt \
OH_SECRET_KEY_PATH=/private/tmp/tool-selection-live/secret-key.txt \
OH_CANVAS_SAFE_BACKEND_PORT=18500 \
OH_CANVAS_SAFE_VSCODE_PORT=18502 \
VITE_FRONTEND_PORT=18501 \
VITE_WORKING_DIR=/private/tmp/tool-selection-live/workspace \
VITE_DO_NOT_TRACK=1 npm run dev:minimal
```

Once the backend is ready, from the Canvas checkout:

```sh
TOOL_SELECTION_ISOLATED=1 node .pr/tool-selection-smoke.mjs
```

Optional overrides: `TOOL_SELECTION_SERVER_URL`, `TOOL_SELECTION_MOCK_URL`, `TOOL_SELECTION_KEY_FILE`, and `TOOL_SELECTION_WORKSPACE`. Defaults match the example above; only localhost/127.0.0.1 servers are accepted. The backend and mock must share a host. `TOOL_SELECTION_UI_PROFILE` optionally checks an existing UI-created profile has persisted `tools=[]`, without changing that profile.

The helper checks saved and materialized specs, launches a mock-model conversation for each selection (empty, glob, standard, legacy SwitchLLMTool alias), checks the actual model-facing tool names for duplicates, and requires each conversation to finish. Mandatory finish/think and skill-related invoke_skill are independent of optional tool selection.

For the UI timing check, open `/settings/agents`, add a profile, choose tools **before entering a name**, then enter a name with a valid mock LLM profile selected. Standard tools should become checked once the server responds. Clearing all tools and toggling Standard → Choose must retain the empty selection.


The fresh smoke run overrode the example ports/state with backend 18610, frontend 18611, VS Code 18612, and /private/tmp/tool-followup-live. The mock server stayed on 18504.
