# PR #18738: real runtime before/after evidence

Verified on 2026-10-23 using the unmodified repository `control-openhands` driver, real Agent Server and real Chrome. No request interception, mocked routes, synthetic runtime responses, model inference, paid provider calls, or user credentials were used.

## Versions and scope

| Item | Baseline | Candidate |
|---|---|---|
| Commit | `9057a57f4a7b5a74e93989090edaffc21e3554ff` | `516598d71807a8d8391553dc5a2fd8db8c2e3f09` |
| Build-input fingerprint | `b57678d84985c93e` | `d99d56b007080ad5` |
| Agent Server / SDK | `1.56.0` | `1.56.0` |
| Automation pin | `1.22.0` | `1.22.0` |
| Node | `24.14.1` | `24.14.1` |
| Chrome | `155.0.8059.85` | `155.0.8059.85` |
| OS | macOS `15.7.5` | macOS `15.7.5` |
| Viewports | 1440 × 1000; 390 × 844 | 1440 × 1000; 390 × 844 |

Each run used a private HOME, state, session key and workspace. Runs executed sequentially and reused the same loopback port block only after the preceding run stopped. `doctor.json` in each folder confirms healthy backends, authenticated API behavior, frontend build identity and absence of page errors in its UI probe.

## Results

| Real state / interaction | Baseline | Candidate |
|---|---|---|
| Available runtime, idle empty conversation | Header Running / Stop Runtime | Unchanged |
| Click Stop Runtime, reopen control conversation | Server Stopped / Start Runtime | Unchanged |
| Separate untouched idle conversation, runtime missing and `can_resume=false` | Header incorrectly Running / Stop Runtime | Archived; neither Stop nor Start |
| Click baseline missing-runtime Stop Runtime | Actual POST `/api/conversations/8368dce0-c10a-4c4f-97e4-8ce970f49054/interrupt` returns **404** | No corresponding action exists |
| Show archived → sidebar conversation menu | Stop Conversation remains available | Stop Conversation absent |
| Phone header for missing runtime | Running / Stop Runtime | Archived; neither Stop nor Start |

Both `idle-missing-metadata.json` and `idle-missing-execution.json` independently record actual API responses: `runtime_status=missing`, `can_resume=false`, `runtime_error=null`, `execution_status=idle`. The candidate fixture has zero events (`after/event-count.json`).

The candidate network capture contains 20 real requests for its target conversation and no `/interrupt` or `/resume` request. This is corroboration of absent controls, not an attempted click on a hidden action. Existing unrelated file/VS Code polling still produces 404 responses for the removed runtime; this change does **not** eliminate all 404s.

## Reproduction

Use the driver's help and the same backend flags on each checkout. Every command must explicitly select its own `--run` directory.

1. Launch with `--new --sdk-version 1.56.0 --automation-version 1.22.0`, using the Chrome executable `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`; run `doctor`, then `onboard --skip` (telemetry off).
2. Arrange a real empty conversation with `api POST /api/conversations --write --data <JSON>`. The payload is:

   ```json
   {
     "agent": {
       "kind": "Agent",
       "llm": {
         "model": "openai/gpt-4o",
         "base_url": "http://127.0.0.1:1/v1",
         "api_key": "qa-not-real",
         "num_retries": 0
       },
       "tools": [],
       "include_default_tools": []
     },
     "workspace": {"working_dir": "<owned-run>/workspace"},
     "autotitle": false
   }
   ```

   There is no initial message. The disposable API-key marker is not a credential; the endpoint is refused loopback. Do not configure a provider or submit a prompt.
3. Open `/conversations/<control-id>`, click `role=button[name="Server Status"]`, capture available status. Click `Stop Runtime`, reopen the conversation, open Server Status and capture Server Stopped / Start Runtime. Click Start Runtime. These interactions control the runtime, without running the agent.
4. Create a **separate untouched empty conversation** with the same payload. Do not pause it: the first control fixture retains `execution_status=paused`, which does not reproduce the reported idle-state bug.
5. In only the owned run's `private/launcher-env.json`, set `OH_CONVERSATION_RUNTIME=docker` and `OH_BIND_HOST=127.0.0.1`, then `restart --timeout 120`. This switches the real backend runtime manager; the saved local conversation has no corresponding Docker runtime. No container or runtime response is fabricated.
6. Run `doctor`; use `api GET /api/conversations/<target-id> --pick runtime_info` and `--pick execution_status`. Reload the target page, click Server Status, capture screenshot and ARIA tree. On baseline click Stop Runtime and record the actual `/interrupt` response with `browser network --filter interrupt`. On candidate record both button counts (zero).
7. Press Escape. Click `testid=conversation-layouts-toggle`, `testid=advanced-options-row`, `testid=toggle-show-archived`, `testid=advanced-options-close`. Hover the target `testid=conversation-card`, then click its `testid=ellipsis-button`. Capture the actual context menu. Hover the card first because the ellipsis is otherwise hidden.
8. Press Escape, set `browser viewport phone`, open Server Status and capture. Stop only the owned run with `control-openhands stop --run <owned-run>`.

The first baseline fixture was also inspected after switching to Docker and correctly identified as paused, so it was not used as the idle bug proof. Baseline returned to local mode to create the untouched idle fixture, then repeated the Docker switch. Candidate used distinct control and idle fixtures from the outset.

## Evidence

- [Before header](before/idle-missing-header.png), [after header](after/idle-missing-header.png)
- [Before Stop error](before/idle-missing-stop-error.png), [actual baseline interrupt requests](before/idle-missing-stop-network.json)
- [Before sidebar menu](before/idle-missing-sidebar-menu.png), [after sidebar menu](after/idle-missing-sidebar-menu.png)
- [Before phone](before/idle-missing-phone.png), [after phone](after/idle-missing-phone.png)
- [Before available](before/available-header.png), [after available](after/available-header.png)
- [Before paused](before/paused-header.png), [after paused](after/paused-header.png)

All included PNGs were visually reviewed by the verification agent and/or the parent reviewer: relevant labels are legible and no credentials are displayed. JSON contains only sanitized driver observations and selected runtime fields. Private run environment, keys, raw backend logs and browser profile are excluded.

## Limits and environment notes

- **Cloud runtime ERROR + idle was not live-tested.** No Cloud ERROR fixture was available. Unit coverage is separate evidence and must not be described as a Cloud live result.
- These are empty local conversations. No active model task, provider inference, or genuine executing-agent stop/resume was tested. “Running” above is the runtime header label for an available runtime.
- This is one manually driven matching before/after reproduction with desktop and phone observations, not a `bug proof` verdict or two independent automated known-bug attempts. No verify skill was changed to add a recipe.
- Node's libuv 1.51 reported low free memory while macOS had reclaimable file-backed/inactive pages. Launch used the documented `--min-free-mb 128` override; only one stack ran at a time.
- The candidate was built with `control-openhands launch --build always`; doctor passed before the evidence run and checked the served build identity.
- Owned runs: `before`, `after` (discarded stamp-less attempt), `after-verified` under `/private/tmp/pr18738-runtime-runs`. All were stopped; private fixture data remains outside the evidence directory.
