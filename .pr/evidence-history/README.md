# Persisted conversation search — running-Canvas evidence

Issue: OpenHands/OpenHands#19378. Product code:60009165b3d3ecaebb3b0d90c2cdfc98b502392f; baselinecc99322a1efd0f4be8d6934ae93a4d1d4bc33e05. Canvas1.37.0, published SDK/AgentServer1.60.1, automation1.30.1, Node24.22.1/npm11.12.1 builds, existing Ubuntu WSL Node24.21.0 launcher, Windows Chrome. No model inference or paid call was made.

## Arrange real persisted history

1. Launch this checkout with its unmodified `.agents/skills/verify-openhands/scripts/control-openhands launch --new --no-browser --build auto --port-from 3290 --timeout 240` and run `control-openhands doctor --skip-ui`. The recorded isolated run is2026-11-08T1521-5795c9.
2. Through the existing profile APIs configure a validation LLM profile with model`openai/gpt-4o-mini`, a dummy key and unreachable baseURL`http://127.0.0.1:9`. Configure an OpenHands Agent Profile referring to it, then create a LocalWorkspace conversation without an initial message.
3. Initialize through the real `POST /api/conversations/{id}/events` API with `{ "role":"user", "content":[{"type":"text","text":"History fixture initialization, without running the agent."}], "run":false }`. This lets the SDK create its required SystemPromptEvent before test history is appended.
4. Stop only this isolated run with `control-openhands stop --run <run>`. Use the published Python SDK1.60.1 to append the608 fixture events with the included`seed-history.py`, then `control-openhands restart --run <run>` and `doctor --skip-ui`. The script takes a JSON file containing`runName` and`conversationId`; adapt its run-root path to the local verification directory. It reads/writes only that run's SDK state and workspace. No verifier, feature map or upstream fixture is modified.

The fixture includes an earlier decision, assistant text, a terminal action/result pair and604 ordinary recent turns. The shell command `printf 'older-%s-needle: workspace connection failed\n' tool` is executed during setup; the combined search term occurs only in the saved tool result, beyond the legacy1,000-character summary. These are SDK-prepared test records, not model-generated agent work. Counts also include normal initialization/state events and grow when live messages are sent. A real REST check returned50of615events with both targets absent (`rest-window-check.json`).

## Desktop proof

`desktop.webm` records two searches: `older-tool-needle` in unloaded tool output and `old-decision-needle` in an unloaded turn.

- Open by keyboard, Tab to the field and type. No100-event history read occurs per keystroke.
- Submit: exactly five100-event pages are fetched. The UI reports500examined events and incomplete results, then stops. It fetches no further page without Continue.
- Continue completes the scan in seven pages. Each result has a date and excerpt. Enter opens context using the original page anchor: two page reads for the tool result and one for the decision. Only the nearby events are displayed, with the target and related tool details expanded.
- While the dialog remains open, send one real user event through the authenticated events API with`run:false`. It arrives through the conversation's WebSocket into the still-mounted live chat. Both before/after execution status reads remain`idle`. Return to live reveals the delivered message.
- Apply800ms browser network latency through CDP, cancel while a real history page is pending, and let it finish: the UI stays cancelled at0examined events and no next page is fetched. Browser offline mode then produces the explicit incomplete error. No route or response is intercepted.

`phone.webm` repeats both searches at390×844, opens bounded context and returns to live. It mounts16and13context events respectively, with no horizontal overflow. `live-results.json` contains the native recorder's assertions. Screenshots show the initial window, five-page pause, expanded tool/decision context, phone layout, cancellation, offline error and returned live tail.

## Initial checks (code 60009165)

137focused tests pass across nine suites (one existing todo), including chat integration, Local/Cloud EventService behavior and translation completeness. New cases cover ACP tool projection, full output, visible thoughts, pagination/cursor recovery, bounds, cancellation, stale query/environment results, errors and keyboard access. Service-backed tests exercise Cloud and ACP paths; the running-Canvas proof uses Local/OpenHands.

Final lint/typecheck/format pass (177existing warnings,0errors). Production app and library builds pass. Final complete Windows suite:10,211passed,60failed,45pending,7todo; the60failing names exactly match unchanged-main controls (`full-suite-comparison.json`). This comparison describes the original code 60009165 and its artifact-only head a947ad55. The follow-up below adds two regression cases; no test timeout or test configuration was changed.

## Follow-up: Continue during a cancelled pending request

Code2f0bdda74ab0864f1fa4f122eb7e00eee35fc283 fixes a race found during follow-up review: Cancel followed immediately by Continue could drop the continuation while the earlier transport was still pending. The dialog now retains the pending scan promise, waits for it to settle, and honors that explicit continuation at the saved checkpoint. Cancelling the queued continuation invalidates it.

The regression fails before this change. The current nine focused suites pass139tests (one existing todo); lint/typecheck/format and app/library builds pass. The canonical launcher rebuilt this source, and doctor passes with the same published server/SDK versions. `cancel-continue.webm` records actual HTTP requests with1000ms browser latency, without intercepting responses:

- Cancel a pending page, immediately Continue, then reach500examined events: six requests total, including the discarded page. A further explicit Continue finds the older decision.
- Start again, Cancel, Continue, then Cancel the queued continuation before the old page settles. It remains at0examined events with exactly one page request and no restart.

The recording uses a fresh normally initialized conversation plus the same608SDK-prepared history records, with no model inference. Screenshots and `cancel-continue-results.json` preserve both observations. The original PR body remains unchanged; the PR follow-up comment links current-head CI.
