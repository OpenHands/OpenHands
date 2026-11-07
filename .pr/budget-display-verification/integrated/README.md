# Live budget display verification

Verified on 2026-10-26 with the checkout's `verify-openhands` CLI, its configured Playwright browser, a real local Agent Server, and DeepSeek flash. The canonical cap now appears in Canvas, including live updates, a separate 80–100% warning state, and actual spend above the cap. No requests were intercepted and no browser state, metrics, or legacy caps were injected.

- Canvas source: `90350be6e173d1a28fbd74038fb3a2ca8c650017`, the focused integration on original PR #18533 head `50ae78643fb497b028bdddf6adfd0dea366b5b49`.
- SDK: `209136794c3ece0755af95cc14340a8aaf4bbeca`, local Agent Server 1.58.2.
- Served build fingerprint: `c4139faed8d2b9d7`, unchanged through the run and both restarts. See [final doctor](doctor-final.json).
- Run: `budget-display-integrated`, desktop 1440×1000 and phone 390×844. Source remained frozen; only evidence was written.
- The [exact original PR baseline](../baseline/README.md) uses the same SDK. Two fresh capped conversations had no meter there. The matching first integrated fixture uses cap 0.05, max_iterations 3, autotitle false, and the identical `BASELINE_CAP_OK` prompt.

This is before/after integration evidence, not a formal two-fresh-stack known-bug proof. The proposed F27 budget IDs are PR evidence only, not permanent feature-map entries.

## Results

| Check | Observed result | Evidence |
| --- | --- | --- |
| Unbounded | Actual cost $0.000157776; cap null; no budget meter. | [Receipt](unbounded-observation.json), [desktop](unbounded-desktop.png) |
| Positive cap | First call cost $0.00255006 / $0.05 = 5.10%; foreground fill. | [Receipt](normal-observation.json), [desktop](normal-desktop.png), [phone](normal-phone.png) |
| Live composer update | Without navigation/reload, cost $0.002709036 → $0.002879712; cap $0.05; 5.42% → 5.76%. | [Before](live-before-observation.json), [after](live-after-observation.json), [video](live-visible.mp4), [GIF](live-visible.gif) |
| Warning below cap | $0.00247896 / $0.003 = 82.632%; error fill remains partially full. | [Receipt](warning-observation.json), [desktop](warning-desktop.png), [phone](warning-phone.png) |
| Overshoot | $0.000352608 / $0.0001 = 352.608%; actual amount and percentage retained; bar width 100%. Exactly one think action, then existing MaxBudgetReached. | [Receipt](over-observation.json), [events](over-events.json), [desktop](over-desktop.png), [phone](over-phone.png) |
| Conversation switching | Sidebar capped → unbounded → capped; each shows its own cost and cap. Direct URLs were also used repeatedly during theme checks. | [Unbounded receipt](switch-unbounded-observation.json), [capped receipt](switch-normal-observation.json), [video](switch-readable.mp4) |
| Reload/restart | Same cost $0.002879712 and cap $0.05 after reload and backend restart plus reload. Usage was explicitly reopened. | [Reload](reload-visible-observation.json), [restart](restart-visible-observation.json), [video](recovery-visible.mp4) |
| Real planner child | Created using Planner → Create a plan. A bounded plan-mode composer turn cost $0.000953496 with null cap; parent Usage retained $0.002879712 / $0.05. | [Before](planner-before.json), [after](planner-after.json), [parent UI](planner-parent-observation.json), [video](planner.mp4) |
| Parent after planner | A canonical REST parent message, observed in the visible UI, raised only the parent's cost to $0.003058788 / $0.05 (6.12%); child stayed $0.000953496. | [Before](parent-api-before.json), [after pair](parent-api-after-pair.json), [UI receipt](parent-api-after-observation.json), [video](parent-api-followup.mp4) |

The overshoot is expected: a completed model call is charged before the next step checks the budget. Only the meter geometry is clamped. Legacy per-LLM `max_budget_per_task` remained null in every receipt.

The local historical Canvas head has no usable Code/Plan switch after child creation because that control is gated to Cloud. That UI return path is recorded as **not run**. The parent continuation above was explicitly API-triggered, not proof of a UI mode switch. The [ledger](ledger-report.md) contains ten pass entries and this one not-run entry.

## Rendered theme contrast

Themes were selected through `/settings/app` → Color Theme, then the actual normal and overshoot conversations were reopened. This required no additional LLM calls. The original Neutral theme was restored.

| Theme | Normal fill/track | Error fill/track | Screenshots |
| --- | ---: | ---: | --- |
| OpenHands-DeepSea | 9.36:1 | 3.68:1 | [Normal](theme-0-normal.png), [error](theme-0-over.png) |
| OpenHands-Neutral | 11.01:1 | 4.55:1 | [Normal](theme-1-normal.png), [error](theme-1-over.png) |
| OpenHands-Neo | 11.01:1 | 4.55:1 | [Normal](theme-2-normal.png), [error](theme-2-over.png) |
| Light+ | 13.08:1 | 3.91:1 | [Normal](theme-3-normal.png), [error](theme-3-over.png) |
| Solarized Light | 3.35:1 | 3.48:1 | [Normal](theme-4-normal.png), [error](theme-4-over.png) |

All ten fill/track samples exceed 3:1. [Computed observations](theme-contrasts.json) include actual RGB values, geometry, and semantic role values from the read-only [observer](meter-observer.js). The observer's text contrast field is diagnostic only: the existing Solarized muted budget text measures about 2.61:1 against its surface. This report makes no all-text accessibility claim.

## Bootstrap and reproduction

Use the exact revisions above. Full behavioral recipes are in [BEHAVIORS.md](../BEHAVIORS.md). All commands below target the run returned by launch; private key files stay outside published evidence.

```sh
npm ci --ignore-scripts
export PATH="$PWD/.agents/skills/verify-openhands/scripts:$PATH"
control-openhands launch --new --run-id budget-display-integrated --min-free-mb 1000 --sdk-path /path/to/sdk
# Set BUDGET_RUN to the returned run directory.
control-openhands onboard --skip --run "$BUDGET_RUN"
control-openhands llm preset deepseek --api-key-file /private/path/to/deepseek.key --run "$BUDGET_RUN"
```

The fresh seed agent profile referenced missing LLM profile `default`; `llm preset` correctly left that dangling reference unchanged. We repointed only this run's default agent profile using its supported API:

```sh
control-openhands api GET /api/agent-profiles/default --run "$BUDGET_RUN" > profile-response.json
python3 - <<'PY'
import json
from pathlib import Path
profile = json.loads(Path('profile-response.json').read_text())['body']['profile']
profile['llm_profile_ref'] = 'deepseek-flash'
Path('profile-request.json').write_text(json.dumps(profile))
PY
control-openhands api POST /api/agent-profiles/default --data @profile-request.json --write --run "$BUDGET_RUN"
control-openhands api GET /api/agent-profiles --run "$BUDGET_RUN"
control-openhands fixture folder --name budget-ui --run "$BUDGET_RUN"
```

Use the returned default agent-profile UUID and fixture folder in the create request:

```json
{"agent_profile_id":"<default UUID>","workspace":{"working_dir":"<fixture folder>"},"max_budget_per_run":0.05,"max_iterations":3,"autotitle":false}
```

Create through `api POST /api/conversations --data @request.json --write --run "$BUDGET_RUN"`. Repeat with caps null, 0.003 and 0.0001. Real prompts were sent with `conversation send <id> --prompt ... --wait --run "$BUDGET_RUN"`, except the no-reload live turn and planner turn, which used direct `browser fill testid=chat-input` and `browser click testid=submit-button`.

The matching baseline prompt is:

> Reply exactly BASELINE_CAP_OK. Do not run any tools. Do not inspect files or run commands.

For the parent REST continuation, the supported `POST /api/conversations/<parent>/events` request was:

```json
{"role":"user","content":[{"type":"text","text":"Reply only PARENT_API_OK. Do not run any tools. Do not inspect files or run commands."}],"run":true}
```

Open Usage through the conversation ellipsis → Display Usage and Cost. Before each capture, inspect `browser bbox testid=usage-panel`: `insideViewport` must be true. After phone/desktop changes or reload, use `right-panel-toggle` if necessary and wait for the animation. DOM presence and text alone can describe an offscreen panel. On phone the panel toggle opens the selected Usage panel; selecting an already active Usage item may leave chat visible.

Use `browser record start --feature <ID> --name <name> --fps 4`, drive the real interaction, then `browser record stop --gif`. Actual MP4 durations are 12.75s live, 5.5s switching, 8.75s recovery, 39.25s planner, and 15.5s parent API continuation. Recovery cuts 11.3s while the backend restarts. Other published recordings have no manual cuts; the CLI trims still lead-ins. These are illustrative recordings, not timing benchmarks.

## Health, limits and cleanup

Final doctor passed with no warnings; served source fingerprint was unchanged. No page JavaScript errors occurred. [Browser diagnostics](browser-errors.json) include the known unsupported DeepSeek `/api/llm/balance` 404s, expected missing PLAN.md reads before any plan file exists, and transient connection refusals during the deliberate restarts. [WebSocket lifecycle](websocket-lifecycle.json) records subsequent opened connections. The evidence does not claim automatic recovery without reload, automatic panel restoration, or an offline/online toggle in this integrated run.

The old launcher's free-memory guard initially refused on this Mac despite about 8 GB process-available memory; its supported `--min-free-mb 1000` option allowed launch. No launcher source or dependency pin was changed. Start doctor ran before the real provider setup and therefore reported a missing default API key; final doctor confirms DeepSeek flash.

Recorded conversation spend totaled **$0.007001628**, excluding the tiny profile-validation requests. The two original-PR baseline conversations totaled $0.00500232. Auto-title was disabled. Provider billing was not fetched.

An initial live recording had the Usage panel offscreen after resizing, and the first switch/recovery recordings were unsuitable. They are excluded from the publish manifest; replacements were captured with the panel visibly open and keyframes inspected. Initial selector/observer-expression mistakes were corrected in the temporary verification driver, with no product changes. All final screenshots and representative first/middle/last recording frames were visually reviewed.

The [stop receipt](stop.json) confirms the owned browser and launcher stopped and all five owned ports closed. Only files in `publish-manifest.json` are intended for publication. Private configuration, raw logs, credentials, abandoned captures and the machine-specific helper driver are excluded. Exact DeepSeek and run-session credential bytes were scanned across every proposed published file; see [privacy receipt](privacy-scan.json).

This report was generated by an AI agent (OpenHands) on behalf of the requester.
