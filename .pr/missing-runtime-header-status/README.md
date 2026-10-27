# Missing runtime: header status and sidebar menu (#18715)

Live evidence from the real Agent Canvas app (local backend, Agent Server 1.56.0,
LLM profile `deepseek-flash`), driven with the repository's `control-openhands`
harness (`.agents/skills/verify-openhands/scripts`). Before and after use the
same setup on the same base commit (3bf5e85): "before" is that commit's sources,
"after" adds the fix.

## Setup (identical for both runs)

```sh
export PATH=$PWD/.agents/skills/verify-openhands/scripts:$PATH
export OH_VERIFY_RUN=$(control-openhands launch --new --print-run --timeout 900)
control-openhands doctor
control-openhands llm preset deepseek && control-openhands onboard --skip
# two empty conversations: sidebar new-thread picker -> "No workspace"
control-openhands browser click 'testid=conversation-panel-new-thread-picker'
control-openhands browser click 'testid=launch-no-workspace' --expect-url '/conversations/[0-9a-f-]+'
# make their runtimes missing: add to $OH_VERIFY_RUN/private/launcher-env.json
#   "OH_CONVERSATION_RUNTIME": "docker", "OH_BIND_HOST": "127.0.0.1"
control-openhands restart && control-openhands doctor
control-openhands api GET '/api/conversations/search?limit=5'
#   -> execution_status: idle, runtime_info: {runtime_status: missing, can_resume: false}
control-openhands browser goto /conversations/<id>
control-openhands browser hover 'testid=server-status-menu-trigger'
control-openhands browser text 'testid=server-status-context-menu'
# sidebar: layouts toggle -> More options -> "Show archived", then the row's ⋮
control-openhands browser text 'testid=context-menu'
control-openhands stop
```

## Observations

| State | Before (3bf5e85) | After (fix) |
| --- | --- | --- |
| Header dot, missing runtime | green dot | gray archive icon, same as the sidebar row |
| Header menu, missing runtime | `Running` / `Stop Runtime` | `Archived`, no Stop or Start action |
| Sidebar row ⋮, missing runtime | `Rename, Edit tags, Stop Conversation, Download conversation data, Delete Conversation` | `Rename, Edit tags, Download conversation data, Delete Conversation` |
| Header menu, available runtime | — | `Running` / `Stop Runtime` (unchanged) |
| Sidebar row ⋮, available runtime | — | still offers `Stop Conversation` (and `Archive Conversation`) |

After the fix, clicking the header status on the missing-runtime conversation
sent no `/interrupt`, `/pause` or resume request
(`control-openhands browser network --filter 'interrupt|resume|pause'` -> 0
requests) and raised no error toast.

## Files

- `before-header-menu-desktop.png`, `before-header-menu-phone.png`,
  `before-sidebar-menu-desktop.png`: base commit, missing runtime.
- `after-header-menu-desktop.png`, `after-header-menu-phone.png`,
  `after-sidebar-menu-desktop.png`, `after-sidebar-menu-phone.png`: fix,
  missing runtime (desktop 1440x1000, phone 390x844).
- `after-available-runtime-header-desktop.png`,
  `after-available-runtime-sidebar-menu-desktop.png`: fix, a conversation whose
  runtime is still available (captured before switching the runtime).
