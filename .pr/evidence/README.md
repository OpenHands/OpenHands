# Published review artifacts

[Baseline video](before.mp4) · [Fixed video](after.mp4) · [Baseline screenshot](before.png) · [Fixed screenshot](after.png).

The run paths below identify the original local capture artifacts; the links above point to the copies retained for review.

# Real-app draft preservation before/after

**Confirmed in genuine production Canvas builds and real Agent Server persistence/WebSocket integration.** Baseline loses saved unsent B after earlier A's echo and navigation; fixed build preserves B through the exact same sequence.

| | Baseline | Fixed |
|---|---|---|
| Source commit | a96855e5e94cbff253813867137cfc5cbd3b9f38 | f583582ac5caaf7d0018ad0ca2ef910190ff80d7 |
| Checkout | /workspace/openhands-live-base | /workspace/openhands-fix-draft |
| Served build inputs | 8926429f3837eede | eed509f9154b4639 |
| Run | 2026-11-05T1352-f14e3f | 2026-11-05T1359-48e777 |
| Real backend/SDK | 1.59.0 / 1.59.0 | 1.59.0 / 1.59.0 |
| Model profile | qa-draft-loopback | qa-draft-loopback |
| Configured model/base URL | openai/gpt-4o / http://127.0.0.1:9 | same |
| A | genuine UI user send, persisted + echoed by server | same |
| B before releasing server | full B stored | full B stored |
| B storage after A echo | null | full B stored |
| Visible composer before navigation | full B | full B |
| Visible composer after Settings/Back | empty | full B |
| Recording | 23.1s, 1440x1000,10fps | 6.2s,1440x1000,10fps |

All run paths are under `/workspace/openhands-research/live-verification/`.

## Media

Baseline:

- `2026-11-05T1352-f14e3f/evidence/F05.draft-persistence/baseline-earlier-echo-loses-draft.mp4`
- `2026-11-05T1352-f14e3f/evidence/F05.draft-persistence/baseline-earlier-echo-loses-draft.gif`
- `2026-11-05T1352-f14e3f/evidence/F05.draft-persistence/baseline-lost-after-return.png`

Fixed:

- `2026-11-05T1359-48e777/evidence/F05.draft-persistence/fixed-earlier-echo-preserves-draft.mp4`
- `2026-11-05T1359-48e777/evidence/F05.draft-persistence/fixed-earlier-echo-preserves-draft.gif`
- `2026-11-05T1359-48e777/evidence/F05.draft-persistence/fixed-preserved-after-return.png`

Each recording is captured directly by the unchanged verification skill browser daemon's existing `browser record` feature. GIFs are its normal ffmpeg conversions of the genuine frames, not mockups. Both final screenshots and extracted interior video frames (baseline second7, fixed second2) were visually inspected: actual chat messages and composer are readable, no privacy/onboarding/tooltip overlays obscure the feature. The fixed video is shorter because commands ran without a human/model wait between calls; **these durations are not latency measurements**.

The baseline video shows B visibly typed before navigation and an empty textbox after returning. The fixed video shows the same actions and B retained. Server events corroborate A is a real persisted user event; B is never submitted. Metadata and events are saved in each run's evidence directory (`baseline-*` / `fixed-*` JSON), with append-only ledger fail/pass rows.

## Honest limitations

This checks a frontend draft race and genuine user-event persistence/echo, not a successful model answer. There is no approved provider key; the profile uses the skill's nonvalidated dummy-credential setup and **only refused loopback http://127.0.0.1:9**. No paid/model provider calls or model output were used. The agent turn can remain Thinking while its refused endpoint retries; that is unrelated to the draft echo bug. Each turn was explicitly interrupted afterward. A real successful LLM response is unnecessary for this bug, which happens as soon as the server echoes the accepted user event.

Both stacks passed doctor before driving, including pin/minimum1.59.0 and real authenticated API/automation/SPA checks. The only initial warning was the fresh run's default profile lacking a key before replacing it with the loopback QA profile.

No network route interception, synthetic events, app/store injection, custom driver, verification skill changes, state mutation with browser eval, public posting, or deployment was performed. All control uses the existing unchanged CLI.

## Reproduction

The full baseline command sequence is in `/workspace/openhands-research/live-draft-before-evidence.md`. Fixed commands are identical except checkout/run and conversation ID:

- Baseline conversation `e13cf45f-b533-4e5e-80a7-11ef618bccfb`
- Fixed conversation `5a25580a-b898-469d-9435-712df9c53d8d`

In both: set nonvalidated loopback-only profile; onboard skip; create via UI; pause conversation; fill A; start recording; `service pause agent-server`; UI Enter sends A; UI type B; read-only eval waits1200ms and verifies B saved; `service resume agent-server`; wait for actual A user bubble; read draft storage and visible textbox; click Settings; browser Back; read textbox; screenshot; stop recording. `service pause/resume` uses SIGSTOP/SIGCONT on **only owned run processes** to delay/release real request processing. It does not fake a response. The waiting eval is read-only and follows F05's documented draft-save recipe.

## Cleanup

Baseline CLI stop reported browser and launcher stopped and ingress/agentServer/automation/frontend/vscode all closed. Fixed run was also resumed and its model turn interrupted before stopping; `fixed-cleanup.json` preserves final resource status. Evidence remains on disk.

Full command sequence:

# Genuine baseline draft-loss capture

Run: `/workspace/openhands-research/live-verification/2026-11-05T1352-f14e3f`

Canvas baseline: `a96855e5e94cbff253813867137cfc5cbd3b9f38`, served build inputs `8926429f3837eede`, checkout `/workspace/openhands-live-base`. Real local Agent Server and SDK **1.59.0**, healthy doctor before driving. Real Chromium `/usr/bin/chromium`, desktop 1440x1000. No route mocking, synthetic server events, state mutation through browser eval, or verification-harness modifications.

Profile `qa-draft-loopback`: nonvalidated `openai/gpt-4o` with dummy key and explicit Base URL **http://127.0.0.1:9**. No provider account/key or paid calls. This is genuine real-app/backend persistence and echo evidence, **not a model-response or successful-agent-execution demonstration**. A response is unnecessary to trigger the bug; the backend persists and echoes the user's actual message before attempting model generation.

Conversation: `e13cf45f-b533-4e5e-80a7-11ef618bccfb`.

Media under run/evidence/F05.draft-persistence:

- `baseline-earlier-echo-loses-draft.mp4` — 23.1s,1440x1000,10fps; actual user send, B composition, Settings navigation, empty return.
- `baseline-earlier-echo-loses-draft.gif` — same real frames, CLI conversion.
- `baseline-lost-after-return.png` — final unoccluded empty composer with accepted A still present.
- `baseline-server-events.json` — actual persisted backend events, A present, B absent.
- `baseline-stack-status.json` — run version and revision metadata.

A frame at 7 seconds was independently inspected and shows A in the transcript and B visibly in the composer; the final screenshot shows the empty composer after Settings/Back. No masks, mockups, editorial overlays, or fake terminal cards were added.

## Exact supported control sequence

All commands use the baseline checkout's unchanged `control-openhands` CLI with:

```
export PATH="$PWD/.agents/skills/verify-openhands/scripts:$PATH"
export OH_VERIFY_HOME=/workspace/openhands-research/live-verification
export OH_VERIFY_RUN=/workspace/openhands-research/live-verification/2026-11-05T1352-f14e3f
export CONTROL_OPENHANDS_BROWSER=/usr/bin/chromium
```

After successful doctor:

```
QA_DUMMY_KEY=qa-draft-check control-openhands llm set --profile qa-draft-loopback --model openai/gpt-4o --api-key-env QA_DUMMY_KEY --base-url http://127.0.0.1:9 --no-validate
control-openhands onboard --skip
control-openhands conversation start --prompt 'Draft persistence setup: no model response expected.' --timeout 45
control-openhands conversation pause e13cf45f-b533-4e5e-80a7-11ef618bccfb
control-openhands browser fill 'testid=chat-input' 'Earlier submitted message A: check draft persistence.'
control-openhands browser record start --feature F05.draft-persistence --name baseline-earlier-echo-loses-draft --fps 10 --max-seconds 180
control-openhands service pause agent-server
control-openhands browser press Enter --selector 'testid=chat-input'
control-openhands browser type 'testid=chat-input' 'UNSENT follow-up B: retain the rollout checklist, migration notes, and recovery steps.'
control-openhands browser eval "new Promise(r=>setTimeout(()=>r(JSON.parse(localStorage.getItem('conversation-state-e13cf45f-b533-4e5e-80a7-11ef618bccfb')).draftMessage),1200))"
control-openhands service resume agent-server
control-openhands browser wait 'testid=user-message >> has-text=Earlier submitted message A' --timeout 10000
control-openhands browser eval "JSON.parse(localStorage.getItem('conversation-state-e13cf45f-b533-4e5e-80a7-11ef618bccfb')).draftMessage"
control-openhands browser text 'testid=chat-input'
control-openhands browser click 'testid=backend-selector-settings-link' --expect-url '/settings/agents'
control-openhands browser back
control-openhands browser wait 'testid=chat-input' --timeout 10000
control-openhands browser text 'testid=chat-input'
control-openhands browser screenshot --feature F05.draft-persistence --name baseline-lost-after-return
control-openhands browser record stop --gif
```

Observed read-only eval outputs, in order: full B text; then **null** after A echo. `browser text chat-input` still returned full B before navigation, then **empty string** after Back. The pause/resume mechanism sends SIGSTOP/SIGCONT only to this owned run's backend process via the existing harness; it delays genuine message handling rather than replacing any response. Backend resumed before teardown.

Capture used isolated local services; no deployment was performed.
