# F28 — Live activity view

A single read-only page, `/activity`, that lists every conversation on the active
backend whose agent is actively executing: running or waiting for confirmation.
Each row shows the conversation title, a status chip, the latest action the agent
took, a relative timestamp, the accumulated cost and token count, and — when the
agent fanned out through the `task` tool — the number of running or finished
subagents. The page polls the conversation list and a bounded event tail per
running conversation; it never starts, stops, or otherwise mutates a conversation.

Source: `src/routes/activity.tsx`, `src/components/features/activity/`, `src/hooks/query/use-activity-event-tails.ts`, `src/constants/activity.ts`.

## Sub-features

- `F28.entry`: the sidebar rail shows an **Activity** link to `/activity`; the page header reads "Activity" with the subtitle "Every running agent and subagent at a glance".
- `F28.active-only`: only conversations whose `execution_status` is `RUNNING` or `WAITING_FOR_CONFIRMATION` appear as rows; idle, paused, finished, error and unknown conversations are omitted (a finished conversation disappears from the list on the next poll).
- `F28.row-content`: a row shows the title, a status chip (`Running` / `Waiting`), the latest action title (falling back to the last assistant message, else "Waiting for the next step…"), a relative timestamp, `$<cost>` and `<n> tokens`.
- `F28.needs-attention`: a waiting or error row is flagged with a **Needs attention** chip and a warning border.
- `F28.row-links-into-conversation`: clicking a row opens that conversation (`/conversations/<id>`) and mutates nothing.
- `F28.subagents`: a `task`-tool delegation shows as a "<n> subagents" chip once the conversation's event tail contains a `TaskAction`; it disappears when the action scrolls out with no matching observation.
- `F28.empty`: with no running conversation the page shows "No agents are running" / "Start a conversation and it will appear here while it works." The paged variant ("No running agents on this page" / "Load more to check the older conversations.") shows when the loaded page holds no active agent but more pages exist, and a **Load more** button stays available whenever more pages exist.
- `F28.error`: a failed conversation-list request shows "Couldn't load running agents" with a **Retry** button that refetches.
- `F28.backend-unavailable`: with no active backend the page shows "Activity is unavailable" / "Connect to an agent server to see your running agents." with a **Retry** button.
- `F28.home-pin`: pinning Activity as the home route (`sidebar-pin-home-toggle-activity`) makes `/` redirect to `/activity`.

## How to get to it (user POV)

- The sidebar rail: the **Activity** link (`sidebar-activity-link`) opens `/activity` from any page.
- Direct URL: enter `/activity`.
- Pin as home: the pin action on the sidebar link stores `/activity` as the home route, so the next visit to `/` redirects there.
- A row drills into its conversation: click a row to open `/conversations/<id>`.

## Driving it with control-openhands

Preconditions:

- Baseline state (launched, doctored, `onboard --skip` done) and `control-openhands llm preset deepseek` (deepseek-flash active).
- Have one running conversation to populate the list: `control-openhands conversation start --prompt "Run exactly one terminal command: sleep 45 && echo qa-f28. Then reply with only its output."` (no `--wait`) prints the id and leaves it executing.
- `F28.error` needs a second, fresh `control-openhands launch --new --build never` with no `llm preset` (export its run dir as `OH_VERIFY_RUN`, then `control-openhands stop` it).

- **Entry and header (`F28.entry`).** Run `control-openhands browser goto /` then `control-openhands browser click 'testid=sidebar-activity-link' --expect-url '/activity'`. `control-openhands browser url` ends `/activity`, `control-openhands browser text 'role=heading[level=1]'` is `Activity`, and `control-openhands browser text 'role=heading[level=1] >> xpath=following-sibling::p[1]'` is `Every running agent and subagent at a glance`. Screenshot with `control-openhands browser screenshot --feature F28.entry --name list`.
- **Row content and drill-down (`F28.active-only`, `F28.row-content`, `F28.row-links-into-conversation`).** With the `sleep 45` conversation running, `control-openhands browser wait 'testid=activity-row' --timeout 30000`, then `control-openhands browser count 'testid=activity-row'` is `1` and `control-openhands browser text 'testid=activity-row'` contains the title (model-written), a status chip (`Running`), a `$`-prefixed cost and `tokens`. `control-openhands browser text 'testid=activity-status'` is `Running`. Click the row: `control-openhands browser click 'testid=activity-row' --expect-url '/conversations/'`, then `control-openhands browser back` and `control-openhands browser wait 'testid=activity-row'` to return.
- **Active-only filtering (`F28.active-only`).** Let the conversation finish (`control-openhands conversation wait <id> --timeout 120`), then `control-openhands browser goto /activity` and `control-openhands browser reload`: after the list poll, `control-openhands browser count 'testid=activity-row'` is `0` and `control-openhands browser count 'testid=activity-empty'` is `1`.
- **Needs attention (`F28.needs-attention`).** Arrange with confirmation mode on (see F06's confirmation recipe), start a conversation that reaches confirmation (`control-openhands conversation start --prompt "Run the terminal command 'echo qa-f28-confirm' and reply with its output."`), then on `/activity`: `control-openhands browser wait 'testid=activity-row' --timeout 30000`, `control-openhands browser text 'testid=activity-status'` is `Waiting` and `control-openhands browser count 'testid=activity-attention'` is `1`. This sub-feature is currently `blocked` in a fresh run because it needs the F06 arrange.
- **Subagents (`F28.subagents`).** Start a conversation whose prompt asks the agent to delegate through `task` (`control-openhands conversation start --prompt "Use the task tool to delegate a one-line subagent task, then reply DONE."`), then on `/activity` `control-openhands browser wait 'testid=activity-subagents' --timeout 30000` and `control-openhands browser text 'testid=activity-subagents'` matches `<n> subagent(s)`. This sub-feature is `blocked` in a fresh run: the delegated `TaskAction` must sit inside the polled event tail, which a single fresh run does not reliably produce.
- **Empty state (`F28.empty`).** With no running conversation, `control-openhands browser goto /activity` shows `control-openhands browser text 'testid=activity-empty'` reading `No agents are running Start a conversation and it will appear here while it works.`; `control-openhands browser count 'testid=activity-row'` is `0`. The paged variant needs more than one page of conversations (a fresh run has one), so it is `blocked` today; when reachable, the text reads `No running agents on this page Load more to check the older conversations.` and the **Load more** button stays.
- **Home pin (`F28.home-pin`).** Run `control-openhands browser goto /`, `control-openhands browser click 'testid=sidebar-pin-home-toggle-activity'`, then `control-openhands browser click 'testid=sidebar-activity-link' --expect-url '/activity'` and `control-openhands browser goto /`: `control-openhands browser wait-url '/activity'` (the index redirects). Clear the pin again (`control-openhands browser storage --clear` or unpin from the sidebar) before the other recipes.
- **Error state (`F28.error`, `blocked`).** On the no-LLM run, stop the agent server (`control-openhands service stop agent-server`) so the conversation-list request fails, `control-openhands browser goto /activity` and `control-openhands browser wait 'testid=activity-error' --timeout 20000`: the text reads `Couldn't load running agents` with a `Retry` button; `control-openhands restart` and `control-openhands browser click 'testid=activity-error >> role=button[name="Retry"]'` clears it.
- **Backend unavailable (`F28.backend-unavailable`, `blocked`).** Needs a run with no active backend configured, which a launched local stack does not provide; when reachable, `/activity` shows `Activity is unavailable Connect to an agent server to see your running agents.` with a `Retry` button.

## Gotchas

- The list is sorted by update time, not by status, so a page can hold no running agent while later pages do. A zero-row page is not proof that no agent runs anywhere: the empty state distinguishes an exhausted list from an empty fetched page, and **Load more** stays whenever more pages exist.
- The latest action and the subagent count come from a bounded per-conversation event tail, then are carried across polls only for unresolved `task` actions; a delegated task whose action scrolled out of the window with no matching observation can stop being counted, so verify the chip while the action is recent.
- `F28.needs-attention`, `F28.subagents`, `F28.error`, `F28.backend-unavailable` and the paged empty variant are `blocked` in a fresh single-run session: they each need a multi-conversation or fault-injected state the baseline recipe does not create. Record them in the evidence ledger as `blocked`, never `pass`.
- `/activity` is one of the pinnable home routes; a pinned `/activity` makes a later `browser goto /` leave the page, so clear the pin before recipes that expect the default home.
- A row click navigates and removes the list from the DOM; `browser back` then needs a `browser wait 'testid=activity-row'` before counting, because the list remounts a moment after the URL changes.
