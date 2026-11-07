# Replaying the budget display checks

These are PR verification recipes, not additions to the permanent feature map.
They exercise Canvas `90350be6e173d1a28fbd74038fb3a2ca8c650017` with SDK
`209136794c3ece0755af95cc14340a8aaf4bbeca`. The original Canvas PR
`50ae78643fb497b028bdddf6adfd0dea366b5b49` is recorded in
[the baseline report](baseline/README.md).

The cap is `ConversationInfo.max_budget_per_run` in USD. Compare it with the sum
of `stats.usage_to_metrics[*].accumulated_cost`, including across subsequent
runs. An LLM call may overshoot. Display actual cost and percentage; clamp only
the bar width. Do not populate the legacy per-LLM `max_budget_per_task` field.

## Setup and shared observation

Use the checkout's `verify-openhands` CLI with Node 24+, installed dependencies,
a configured real `deepseek-flash` profile, and an isolated run. Keep credentials
in the run's private configuration. Only the run owner controls its browser.
The [live report](integrated/README.md) records this run's bootstrap details,
including repointing a fresh default agent profile to the existing LLM profile.

```sh
export PATH="$PWD/.agents/skills/verify-openhands/scripts:$PATH"
control-openhands launch --new --sdk-path /path/to/sdk-branch --print-run
# Set BUDGET_RUN to the directory printed by launch.
control-openhands doctor --run "$BUDGET_RUN"
control-openhands llm preset deepseek --run "$BUDGET_RUN"
control-openhands onboard --skip --run "$BUDGET_RUN"
control-openhands fixture folder --name budget-ui --run "$BUDGET_RUN"
```

Configure conversations through the supported create API using the configured
agent profile and returned fixture path. This arranges preconditions. Initial
and follow-up proof messages go through the composer, except the explicitly
API-triggered parent continuation described below. Set `autotitle:false` and
`max_iterations:3`. Use caps `0.05` for normal use, `0.003` for the warning
fixture, `0.0001` for overshoot, and omit the cap for unbounded use. Record each
returned ID as `CAPPED`, `WARNING`, `OVERSHOOT`, or `UNBOUNDED`. For example,
`capped.json` contains this payload after replacing the two placeholders:

```json
{
  "agent_profile_id": "<configured profile UUID>",
  "workspace": { "working_dir": "<run workspace>/budget-ui" },
  "max_budget_per_run": 0.05,
  "max_iterations": 3,
  "autotitle": false
}
```

For `UNBOUNDED`, omit `max_budget_per_run` or set it to null. No LLM credentials
belong in these create files; the configured profile supplies them.

```sh
control-openhands api POST /api/conversations --write --data @capped.json --run "$BUDGET_RUN"
control-openhands conversation send "$CAPPED" --prompt 'Reply exactly BASELINE_CAP_OK. Do not run any tools. Do not inspect files or run commands.' --wait --run "$BUDGET_RUN"
control-openhands browser click 'testid=chat-pane-header >> testid=ellipsis-button' --run "$BUDGET_RUN"
control-openhands browser click 'testid=conversation-name-context-menu >> testid=display-cost-button' --run "$BUDGET_RUN"
control-openhands api GET "/api/conversations/$CAPPED" --run "$BUDGET_RUN"
control-openhands browser text 'testid=usage-panel' --run "$BUDGET_RUN"
control-openhands browser eval "$(cat .pr/budget-display-verification/integrated/meter-observer.js)" --run "$BUDGET_RUN"
```

The observer only reads the rendered Usage panel. It distinguishes the dollar
budget row from the context-window meter, returns actual colors and width, and
has no trailing semicolon because `browser eval` expects an expression. Run it
after the width transition settles. A null contrast is inconclusive, not a pass.
At phone width, selecting an already active Usage item can leave chat visible.
Use the actual panel toggle after selecting Usage, then wait for `/panel`:

```sh
control-openhands browser viewport phone --run "$BUDGET_RUN"
# Select Display Usage and Cost from the menu only if Usage is not selected.
control-openhands browser click 'testid=right-panel-toggle' --run "$BUDGET_RUN"
control-openhands browser wait-url '/panel$' --run "$BUDGET_RUN"
control-openhands browser bbox 'testid=usage-panel' --run "$BUDGET_RUN"
```

Skip the toggle if the Usage panel is already visible. Capture the phone view
and inspect its fit; clicking the selected item alone does not prove navigation.

## Cases and assertions

| Proposed check         | Exercise                                                                                  | Assert against the real UI and REST receipt                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `F27.budget-hidden`    | Send a bounded no-tool prompt to `UNBOUNDED`; open desktop Usage.                         | Positive Total Cost remains visible; no budget meter; REST cap is null.                                               |
| `F27.budget-meter`     | Open `CAPPED`, record a second real composer turn, and observe Usage without reloading.   | Cost, configured cap, percentage and bar width agree with REST; cost increases live and the cap stays unchanged.      |
| `F27.budget-warning`   | Send short no-tool turns to `WARNING` until the actual ratio is above 80% and below 100%. | Percentage remains below 100%; the rendered fill uses the error role and has at least 3:1 contrast against its track. |
| `F27.budget-overshoot` | Use the think-first prompt below with `OVERSHOOT`.                                        | Actual cost and percentage exceed the cap/100%; bar stops at 100%; an unfinished run reports `MaxBudgetReached`.      |
| `F27.budget-recovery`  | Reload and restart the same isolated stack; reopen Usage as necessary.                    | The same conversation retains its cap and cumulative cost. Record which recovery required a reload.                   |
| `F27.budget-switch`    | Alternate real sidebar cards for `CAPPED` and `UNBOUNDED`.                                | Each view shows its own cost; the uncapped view has no retained meter and returning to capped restores its cap.       |
| `F27.budget-planner`   | Create a planner through the UI on `CAPPED`, then send a bounded planner turn.            | Parent Usage retains parent cost/cap while the child's separate REST cost changes.                                    |

Record transitions rather than relying on still screenshots. Use direct composer
fill/click for the live update; `conversation send` navigates and can reload the
page, so it does not establish an update without reload.

Before recording, return to desktop, open the capped conversation, and make Usage
visible. Inspect a fresh screenshot: the cost, meter and composer must be in the
viewport together. `browser text`, DOM presence and even a nonzero bounding box
can describe an offscreen panel after a phone-to-desktop resize. Compare the
bounds with the viewport (`insideViewport:true`) and inspect the actual image;
open the selected panel
with `right-panel-toggle` if necessary and wait for its animation to settle.
A recording with the meter offscreen is
not evidence of a live update.

```sh
control-openhands browser viewport desktop --run "$BUDGET_RUN"
control-openhands browser goto "/conversations/$CAPPED" --run "$BUDGET_RUN"
# Open Usage through the menu/toggle as above, then inspect this screenshot.
control-openhands browser bbox 'testid=usage-panel' --run "$BUDGET_RUN"
control-openhands browser screenshot --feature F27.budget-meter --name live-start --run "$BUDGET_RUN"
control-openhands browser record start --feature F27.budget-meter --name live-spend --run "$BUDGET_RUN"
control-openhands browser fill 'testid=chat-input' 'Reply only LIVE_OK. Do not run any tools. Do not inspect files or run commands.' --run "$BUDGET_RUN"
control-openhands browser click 'testid=submit-button' --run "$BUDGET_RUN"
control-openhands conversation wait "$CAPPED" --fresh --run "$BUDGET_RUN"
control-openhands browser text 'testid=usage-panel' --run "$BUDGET_RUN"
control-openhands browser record stop --gif --run "$BUDGET_RUN"
```

Warning costs are not deterministic. Inspect the real ratio after each turn;
never change costs or inject browser state to reach a target. Stop at 20 turns
or $0.10 combined fixture spend. If a call jumps over the warning interval, keep
that overshoot observation and arrange a fresh conversation with a larger cap.
A percentage above 100% does not prove the 80–100% state. Exactly 80% belongs in
the component boundary test, not a claim about exact live billing.

Overshoot prompt:

> First call think with thought "budget probe". Then in a separate later response reply DONE. Do not finish in the first response. Do not use any other tools, inspect files, or run commands.

Inspect `conversation events "$OVERSHOOT" --kinds ActionEvent,ConversationErrorEvent`
to identify the actual tool step and stop reason. If the overshooting call
finishes the task, the task remains finished; do not require an error then.

For the sidebar switching check on this isolated local backend, click the
rendered conversation links, reopen Usage if closed, and read its cost/cap after
each switch. Record the transition with the same visible-panel gate:

```sh
control-openhands browser click "a[href=\"/conversations/$UNBOUNDED?backend=default-local\"]" --expect-url "$UNBOUNDED" --run "$BUDGET_RUN"
# Reopen Usage if needed, then inspect its text and meter observation.
control-openhands browser click "a[href=\"/conversations/$CAPPED?backend=default-local\"]" --expect-url "$CAPPED" --run "$BUDGET_RUN"
```

This records sidebar navigation; direct-URL switching was not independently
verified as another entry point.

For recovery, record the commands and observed final state separately. The
integrated run exercised reload and stack restart followed by reload. It did not
exercise a separate browser offline/online cycle.

```sh
control-openhands browser reload --run "$BUDGET_RUN"
control-openhands restart --run "$BUDGET_RUN"
control-openhands doctor --run "$BUDGET_RUN"
control-openhands browser events --kinds websocket-created,websocket-open,websocket-close,websocket-error --last 100 --run "$BUDGET_RUN"
```

For planning ownership, stay on the capped parent's page and create the child
through Planner. Capture the parent's pre-turn cost. The helper ID comes from
`sub_conversation_ids`; wait for that child, not the parent, after sending:

```sh
control-openhands browser click 'testid=conversation-tab-planner' --run "$BUDGET_RUN"
control-openhands browser click 'role=button[name="Create a plan"]' --run "$BUDGET_RUN"
control-openhands api GET "/api/conversations/$CAPPED" --run "$BUDGET_RUN"
# Set PLANNER to the returned sub_conversation_ids entry.
control-openhands browser click 'testid=conversation-tab-usage' --run "$BUDGET_RUN"
control-openhands browser fill 'testid=chat-input' 'Reply only PLANNER_OK. Do not run any tools. Do not inspect files, create a plan file, or run commands.' --run "$BUDGET_RUN"
control-openhands browser click 'testid=submit-button' --run "$BUDGET_RUN"
control-openhands conversation wait "$PLANNER" --fresh --run "$BUDGET_RUN"
control-openhands api GET "/api/conversations/$PLANNER" --run "$BUDGET_RUN"
control-openhands api GET "/api/conversations/$CAPPED" --run "$BUDGET_RUN"
control-openhands browser text 'testid=usage-panel' --run "$BUDGET_RUN"
```

Record this transition with the same visible-panel gate. The parent meter must
remain paired with the parent's REST cost and cap. Planning-child cost is a
separate conversation cost; it must not replace the numerator beside the parent
cap. The current source falls back to the parent's REST snapshot while the
latest live metrics belong to the child; that snapshot polls every 30 seconds.
This local Canvas revision has no Code/Plan switch to return through after child
creation. To check a subsequent parent event, keep Usage visible and explicitly
start the parent through the canonical API, then observe its live display:

```sh
control-openhands api POST "/api/conversations/$CAPPED/events" --write --data '{"role":"user","content":[{"type":"text","text":"Reply only PARENT_API_OK. Do not run any tools. Do not inspect files or run commands."}],"run":true}' --run "$BUDGET_RUN"
control-openhands conversation wait "$CAPPED" --fresh --run "$BUDGET_RUN"
control-openhands api GET "/api/conversations/$CAPPED" --run "$BUDGET_RUN"
control-openhands api GET "/api/conversations/$PLANNER" --run "$BUDGET_RUN"
control-openhands browser text 'testid=usage-panel' --run "$BUDGET_RUN"
```

Record before/after with the parent meter visible. The parent's cost must update
beside its unchanged cap while the child's cost stays unchanged. This is an
API-triggered continuation with real UI observation, not proof of a local
Code/Plan switching control. No mode or browser state is injected.

Use existing F16 theme controls (`/settings/app` → `color-theme-input`) to select
OpenHands-DeepSea, OpenHands-Neutral, OpenHands-Neo, Light+, and Solarized Light.
Return to the genuine normal and warning conversations after each selection;
read the observer and capture the rendered meter. This needs no additional model
calls. Inspect computed fill-to-track contrast, not only CSS class names.
The observer also reports text contrast diagnostically; that is not a claim
that the preexisting muted-text palette passes in every theme.

## Evidence and promotion

The integrated ledger records **10 passed checks and 1 not-run check**. The
not-run check is the unavailable local Code/Plan UI switch; the separately
API-triggered parent continuation passed. Measured outcomes, commands, media and
limitations are in the
[integrated report](integrated/README.md) and
[generated ledger](integrated/ledger-report.md). The linked receipts establish
what the recorded run did; this recipe alone is not proof that a rerun passed.

After inspection, render `control-openhands evidence report --run "$BUDGET_RUN"`,
stop only this run with `control-openhands stop --run "$BUDGET_RUN"`, and retain
the shutdown receipt. Review every published image and data file for credentials.
No raw browser storage, system prompts, authentication headers or session URLs
belong in the artifacts.

Promotion into the permanent F27 map belongs in a separate verification change
after the product integration lands. Re-drive the recipes then; do not treat
this PR artifact as a new accepted maintenance baseline or formal known-bug proof.

This report was generated by an AI agent (OpenHands) on behalf of the requester.
