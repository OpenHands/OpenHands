# OpenHands Agent Canvas feature map

This directory is the maintained source for verifying the user-facing behavior
of Agent Canvas. Read this index before driving the app, then open the matching
feature file and run its recipe with `control-openhands`. The map is not a claim
that every row passes today; the run's evidence ledger is.

## Baseline preconditions

- `control-openhands` is on `PATH` (see [the skill](../../SKILL.md)).
- `control-openhands launch` started this checkout; `control-openhands doctor`
  is `ok` for the run in `$OH_VERIFY_RUN` (or the `current` symlink).
- `control-openhands onboard --skip` has answered telemetry consent and closed the
  onboarding modal, unless the recipe tests onboarding itself.
- Model-backed recipes: `control-openhands llm preset deepseek` saved
  `deepseek-flash` (active) and `deepseek-pro` from `DEEPSEEK_API_KEY` (or
  `--api-key-file PATH`). States that exist only while no LLM is configured
  (the home banner, onboarding's LLM step) need a fresh `launch --new` and must
  run before the preset. When the key fails validation (an empty balance, a
  rate limit), `llm preset deepseek --no-validate` still saves the profiles:
  recipes that only need a conversation to exist can run on, and every
  recipe that reads a model reply is `blocked`, never a pass.
- The browser is at the desktop viewport (1440×1000) unless a recipe says
  otherwise; `control-openhands browser viewport phone` is 390×844.
- Never drive an instance that this verification run did not start.

## Driving conventions

- Start every recipe from the baseline state unless its preconditions say
  otherwise, and return to it afterwards (delete fixtures, re-activate
  `deepseek-flash`, restore toggles).
- Selectors: prefer `testid=...` scoped with ` >> ` to the owning form, row or
  dialog, then `role=...[name="..."]`, then `label=`/`text=`. Discover handles
  with `control-openhands browser testids` and `browser snapshot`.
- Treat commands as literal. Fixture names start with `QA_` or `qa-` so cleanup
  and assertions never collide with real data.
- UI navigation and direct URL entry are different entry points; record which
  one a check used. After a navigating click, wait for the destination:
  `browser click '<sel>' --expect-url '<regex>'`; the plain click returns the old URL.
- Many settings switches are a hidden `<input>` inside a `<label>`: click
  `'testid=<switch> >> xpath=ancestor::label'` (language-independent; the label
  text also works in English) and read `.checked` with `browser eval`.
- Verbs that are easy to miss: `browser mouse-click X Y` (backdrops and
  overlays), `click --modifiers Control,Meta` (open in new tab, multi-select),
  `click --hover-first` (hover-driven toggles), `click --expect-new-url`,
  `browser tooltip <sel>`, `browser wait <sel> --state hidden|visible|detached`
  (use it after toggles: animations make an immediate `count` or `visible`
  lie), `browser media --clear` (forget earlier sounds before a negative sound
  check). `browser eval` takes one expression; wrap statements in an IIFE.
- Input that is not a click: `browser upload-via <trigger> <file>` answers the
  real file chooser; `browser drop-files <sel> <file>` and `browser paste <sel>
  --file F --text T` deliver drag-and-drop and paste events (target an element
  inside the drop zone: events bubble up); `browser drag` reorders draggable
  rows or moves a grip `--by DX,DY`; `browser choose <combobox> <label>` picks
  an autocomplete option; `browser clipboard` reads what a Copy button wrote.
- Short-lived feedback: `browser toasts --history` lists every toast since the
  page loaded; `click --observe SEL` records transient labels. An error toast
  is neither a page nor an HTTP error, so `browser errors` misses it: read the
  toast history after every mutation, backend add or switch.
- Hover-driven controls open on pointer enter only: `browser hover` on the
  element the pointer is already over does nothing. Hover something else first
  (`browser hover 'testid=command-menu-trigger'`), or use `click --hover-first`.
- Agent-side proof: `conversation events <id> --grep TEXT [--from-start]`
  searches whole events (the system prompt's skills and tools, tool
  arguments); rows show activated skills. After sending a follow-up message,
  `conversation wait <id> --fresh` ignores the previous terminal status.
  `fixture skill` writes a personal or project `SKILL.md`; `api GET ...
  --pick a.b.c` reads one field.
- An already-open page does not refetch settings written by `llm` or `api
  --write`: `browser reload` before asserting UI state after an arrange step.
- Settings and panels scroll inside a container: `browser scroll '<sel>' --by 600`
  scrolls the right one. `--full-page` cannot help there, because the page
  itself is only one viewport tall: take one screenshot per scroll position.
- `conversation start` leaves the browser on the conversation page, which also
  has a `testid=submit-button`. `browser goto` the next page explicitly, or a
  later `wait --state detached` passes vacuously on the wrong page.
- `click --expect-url` passes at once when the URL already matches (one
  conversation to another); use `--expect-new-url` there.
- Select rows by a prompt word or an id prefix, never by a model-written title:
  titles differ between runs. Only the 20 most recently updated conversations
  load, so a recipe that creates more must not rely on older rows after a reload.
- Click composite cards by their title or a detail element: the centre can be
  a region that stops propagation, and the click then does nothing.
- Dialogs can match twice (nested `role=dialog`): scope by test id or add
  `>> nth=0`. `browser testids <scope>` misses portaled parts of a dialog or
  menu; run it unscoped with `--filter`.
- Panels opened by a click mount after a moment: `browser wait` before
  counting. A disabled rich composer is `contenteditable=false`, so `fill`
  fails there; check `browser attr ... contenteditable`.
- Error toasts can arrive after the UI retries (about 1.5 s): read them with
  `browser toasts --history`.
- Pop-ups opened with `window.open` (sign-in, docs links) appear a moment after
  the click: `browser wait-tab '<url regex>'`, then `browser tab <i>`.
- Drag: `browser drag <sel> <target-sel>` moves the pointer in steps; if a list
  still does not reorder, drag its handle with `--by DX,DY --steps 20`.
- An MCP server to test with: `control-openhands fixture mcp-server` writes a
  dependency-free stdio server with one tool, `qa_echo`.
- A conversation in a folder or repo: `control-openhands fixture git-repo --name
  qa-repo [--remote https://github.com/qa-example/qa-repo.git]`, then
  `control-openhands conversation start --workspace qa-repo --prompt ...`. It
  drives Open Workspace and the folder browser, which has no path field
  (`workspace open qa-repo` does only the picking). `--remote` only sets
  `origin`, which is enough for the repo/branch links and Pull/Push chips.
- Sidebar and row navigation can append `?backend=<id>`: end `--expect-url`
  regexes with `(\?|$)`, not `$`.
- Some confirmation buttons render without the `data-testid` their source
  passes (`BrandButton` takes `testId`, not `data-testid`). When
  `browser testids 'role=dialog'` lists nothing, use
  `'role=dialog >> role=button[name="Confirm"]'`.
- `browser errors --clear` and `browser network --clear` print the list, then
  empty it: run them before the action, then read again after it. `errors`
  lists failures only; prove that a request happened (or did not) with
  `network`. Page errors (uncaught exceptions) are failures; console warnings
  such as missing translations are reported, not failures.
- `api GET ... --pick a.b.c` paths start inside the response body (a leading
  `body.` is accepted).
- In a sandbox, requests to external hosts (fonts, model catalogs) can fail;
  judge the errors sweep by its `app:` rows, and `browser errors --app-only
  --no-warnings` hides translation warnings.
- Downloads are saved as `<run>/private/downloads/<ms>-<suggested name>`;
  claims about their content need `browser downloads --last 1 --inspect`.
- `onboard --skip` after `browser reset` skips only the modal: consent is
  stored on the backend, so it is not asked again.
- The Agent Server runs with `HOME=<run>/private/home` (personal skills,
  plugins, `~/.agents`) and keeps catalog caches under `<run>/private/cache`.
  Fixtures that belong there (`fixture skill`) write into the run, never into
  the operator's home.
- Each port is its own origin: `browser goto` needs `--allow-external` for any
  other localhost port, and that origin starts at first run again.
- Two stacks (a second backend, backend switching): launch the second one with
  `export OH_VERIFY_RUN_2=$(OH_VERIFY_RUN= control-openhands launch --new
  --no-browser --print-run)` and address it with `--run "$OH_VERIFY_RUN_2"`
  (before or after the command). Add it in the UI as a backend; activating a
  backend asks for its own telemetry consent, which `onboard --skip` answers.
  While its services are stopped the browser logs CORS errors for that origin;
  they come from the dead upstream, not from the app.
- At phone width the desktop sidebar stays in the DOM, hidden, so sidebar test
  ids match twice: scope them, `'testid=sidebar-mobile-drawer >> ...'`.
- Card toggles that swap their icon on hover (plugins, skills, pickers) can
  swallow a plain click: use `click --hover-first` and assert `aria-checked`
  plus the API state.
- An open autocomplete listbox closes when another command touches the page;
  `browser choose` opens, filters and picks in one step.
- Side effects outside the app are valid second views: `git -C <path> log`
  on a `fixture git-remote` after a push or Git Sync, a downloaded file with
  `downloads --inspect`. Mark such shell commands as read-only checks.
- Negative tests (a 409, a 422, a stopped service) add expected HTTP errors:
  run `browser errors --clear` after them so the family's sweep only shows
  surprises, and name the expected ones in the evidence row.
- Cleanup across families: deleting a fixture through the UI belongs to the
  family that maps deletion; elsewhere `api DELETE ... --write` is fine
  (arrange, not proof).
- When a label depends on the UI language or a toast may already be gone,
  assert the persisted state after `browser reload` instead of `wait-text`.
- Pass `--timeout 5000` to waits for elements that may legitimately not appear;
  the 30 s default adds up quickly.
- Conversations created as fixtures may stay when no recipe in the family covers
  deleting them; they vanish with the run's private state.
- Rows with generated ids: select by prefix plus the fixture name,
  `'[data-testid^="automation-card-"] >> has-text=QA_Pong'`. Portal menus without
  test ids: scope by role and a unique item, `'role=menu >> has-text=Delete'`.
- Restore per-viewer state too (view mode, pinned home route, collapsed panels
  live in localStorage); some controls disable themselves once their data is
  gone, so restore before deleting fixtures.
- `control-openhands api ... --write`, `llm` and `fixture` arrange preconditions.
  They are never the proof step for the feature under test.
- Accessible-name matching is a case-insensitive substring match:
  `role=heading[name="Activity"]` also matches **Activity Log**. Add `[exact]`
  (`role=heading[name="Activity"][exact]`) when a shorter name is a prefix of
  another. Use plain `text=` substrings inside ` >> ` chains; a regex
  `text=/.../` segment there does not match.
- Give every asserted value its selector (`browser text 'role=heading[level=1]'`,
  not "the h1 is X"): pages differ in landmarks, and the automation detail page
  has no `main`, so a habitual `'role=main >> ...'` times out there.
- Time-sensitive steps (a key press inside a 2 s busy window, a click while a
  request is pending) go in the same shell line as the action that opens the
  window: `browser click A && browser press Space`. A separate tool call is
  often too late.
- `browser network --last N` returns its rows under `recent`.
- Backend versions come from launch flags only (`--sdk-version`, `--sdk-ref`,
  `--sdk-path` and the `--automation-*` equivalents); `OH_AGENT_SERVER_*` in
  your shell is not forwarded. An Agent Server below the UI's minimum
  (`config/defaults.json` `compatibility.minimumAgentServer`) never gets past
  the backend-compatibility gate, so it cannot reach capability-missing states
  inside a page.

## Proof and skip reporting

- Capture the user action and the resulting state, not only the final screen.
- Mutations need a read-only second view: `browser reload`, reopening the item,
  or `control-openhands api GET ...`.
- Visual checks: `browser screenshot --feature <ID> --name <label>` at the stated
  viewport, plus `browser bbox` for overflow/geometry claims.
- After each family, `control-openhands browser errors --app-only` must not show
  new page errors; count and report them even when the UI recovered. Record the
  sweep under the sub-feature whose page produced the errors (or the family's
  page-level ID); `evidence add` warns about IDs that are not in the map.
- Exports and downloads: `browser downloads --last 1 --inspect [--contains TEXT]`
  shows a text file's head or a zip's entry names.
- Record every check with `control-openhands evidence add --feature <ID> --result
  pass|fail|blocked|not-run`, one row per sub-feature ID a recipe bullet proves
  (a bullet that covers two IDs gets two rows). A skipped entry point is never
  verified through a different one. Blocked rows name the missing prerequisite
  and the attempted path.
- Resuming after `stop`: `control-openhands restart` brings the stack back with
  its state, ports and ledger, and `browser start` opens a fresh page at
  `about:blank`; `browser goto` the page you were on before asserting.

## Feature entry contract

Each feature file starts with an H1 title, a short paragraph describing the
user-visible behavior and a `Source:` line. It then uses exactly four H2 sections
in this order (`control-openhands map check` enforces it):

1. `Sub-features`: one bullet per stable ID (`` `Fnn.slug` ``) and behavior.
2. `How to get to it (user POV)`: every user entry point.
3. `Driving it with control-openhands`: starts with `Preconditions:`, then labeled
   bullets pairing each user action with exact commands and the observable result.
4. `Gotchas`: traps that waste or invalidate a run, and linked known issues.

How to write and prove new entries: [../mapping.md](../mapping.md).

## Features

| ID | Family | What it covers | Entry points | Needs | Sub-features |
|---|---|---|---|---|---|
| F01 | [First run, onboarding and sign-in](F01-first-run-and-sign-in.md) | telemetry consent, onboarding modal, public-mode backend step and API-key screen, route error page | any URL on a fresh browser profile; `launch --public` | fresh run or `browser reset`; LLM for say-hello | 21 |
| F02 | [App shell, sidebar and command menu](F02-app-shell.md) | rail and phone drawer, collapse, home pinning, getting-started checklist, command menu, toasts | every page; `Control+k` / `Meta+k` | baseline; LLM for checklist progress | 25 |
| F03 | [Home and starting work](F03-home.md) | home composer, workspace picker, plugin picker, recommended automations rail, no-LLM banner | `/`, **New Chat**, command menu | runs before and after `llm preset`; fixture repo | 39 |
| F04 | [Conversation list and folders](F04-conversation-list.md) | sidebar list, View presets, workspace folders, tags, rename/pin/archive/delete, load more | sidebar on every page | several conversations; fixture repos | 40 |
| F05 | [Composer, slash commands and plan mode](F05-composer.md) | send, drafts, attachments, model pill, slash commands, `/goal`, plan mode, dictation | home and conversation composer | LLM | 23 |
| F06 | [Agent activity](F06-agent-activity.md) | messages, tool events, thinking, empty state, confirmation mode, stop/resume, errors, branching | `/conversations/<id>` | LLM | 33 |
| F07 | [Conversation page, header and menu](F07-conversation-page.md) | title rename, status menu, ⋯ menu (skills, hooks, tools, export, download, cost, stop, delete), git bar, overview | sidebar row, URL, `/panel` | LLM; fixture repo; Cloud items blocked | 25 |
| F08 | [Workspace drawer: files and changes](F08-workspace-files-and-changes.md) | Files tab, Commits and uncommitted changes, diff viewer, file links in chat | panel toggle, file links | LLM; fixture repo | 33 |
| F09 | [Settings shell and navigation](F09-settings-shell.md) | settings navigation, phone hub, deep links, update card, backend note | gear, command menu, URLs | baseline; deep links before `llm preset` | 24 |
| F10 | [LLM profiles](F10-llm-profiles.md) | list, add, edit, rename, duplicate, delete, default, advanced fields, validation | `/settings/llm` | fresh run; DeepSeek key | 31 |
| F11 | [Provider connections](F11-provider-connections.md) | add, edit, rotate key, delete connections; profiles from a connection | `/settings/llm` | fresh run; DeepSeek key | 21 |
| F12 | [Model router](F12-model-router.md) | meta-profiles, templates, run on first message, routing effect | `/settings/meta-llm` | LLM profiles | 25 |
| F13 | [Agent profiles](F13-agent-profiles.md) | OpenHands and ACP profiles, editor, name rules, active profile, MCP and secret scoping | `/settings/agents` | fresh run; LLM | 20 |
| F14 | [Secrets](F14-secrets.md) | list, add, edit, rename, delete, agent access | `/settings/secrets` | LLM for agent access | 11 |
| F15 | [Condenser, agent context and verification](F15-agent-behavior-settings.md) | condenser fields, agent context, confirmation mode and critic, their effect on runs | `/settings/condenser`, `/agent-context`, `/verification` | LLM for the effects | 23 |
| F16 | [Application settings](F16-application-settings.md) | language, theme, analytics, sound, checklist, title model, voice input, git identity | `/settings/app` | LLM for title and git checks | 19 |
| F17 | [Customize hub and MCP servers](F17-mcp-servers.md) | catalog, install, test, edit, delete, custom servers, agent use | **Customize**, `/mcp` | `uvx`/`npx`; LLM | 26 |
| F18 | [Skills catalog](F18-skills.md) | facets, search, enable/disable, add skill, Use skill, personal and project skills | `/skills` | `fixture skill`; LLM | 19 |
| F19 | [Plugins and plugin launch](F19-plugins.md) | catalog, install, update, uninstall, enable, launch deep links | `/plugins`, `/launch` | LLM | 25 |
| F20 | [Canvas apps](F20-canvas-apps.md) | install from path or git, enable, update, uninstall, extension pages | `/apps` | none | 23 |
| F21 | [Automations dashboard and actions](F21-automations-dashboard.md) | cards and list, filters, sort, Run now, enable, export, import, delete, pin | `/automations` | LLM; automation service | 34 |
| F22 | [Creating automations](F22-automation-creation.md) | templates, setup dialog, custom automations, import | `/automations/templates`, `/automations/new/<id>` | LLM | 28 |
| F23 | [Automation detail, runs and editing](F23-automation-detail.md) | detail page, runs, logs, edit dialog, triggers, debug | `/automations/<id>` | LLM; runs | 39 |
| F24 | [Automation Git Sync](F24-git-sync.md) | configure, sync cycles, encryption, status | `/automations/git-sync` | `fixture git-remote` | 26 |
| F25 | [Backends, Cloud and sharing](F25-backends-and-cloud.md) | add, edit, remove and switch backends, per-backend consent, Cloud login, shared pages | backend selector | a second stack; Cloud account (blocked) | 27 |
| F26 | [Launcher modes, Docker, desktop and library](F26-runtime-variants.md) | launcher flags, partial stacks, LAN bind, Docker, Electron, embeddable library | a terminal | Docker/Electron where available | 27 |
| F27 | [Workspace tools](F27-workspace-tools.md) | terminal, browser, planner, task list, usage, `canvas_ui_control` | drawer tabs | LLM | 29 |

27 families, 716 sub-features. `control-openhands map ids` lists every ID with its file.

### Neighbouring families

Several pages are shared. Each behavior has one owner; the others reference its ID instead of re-mapping it.

- Conversation page: the composer is F05, what the agent produces is F06, the header and its menus are F07, the drawer is F08 (files, changes) and F27 (tools).
- Starting work: Home is F03; the recommended-automation launcher is shared by F01 (onboarding), F03 (rail) and F22 (templates), and owned by F22.
- LLM settings: profiles are F10, provider connections F11 (same page), routers F12, agent profiles F13.
- Automations: dashboard F21, creation F22, detail and runs F23, Git Sync F24.
- Customize: MCP F17, skills F18, plugins F19, apps F20; the hub and its navigation are in F17.
- Backends: adding and switching backends is F25; launcher flags that create them are F26.

## Not mapped

Everything else a user can reach is mapped. These are left out on purpose:

- `src/components/features/context-menu`: a shared menu primitive, not a feature of its own. Its behavior is checked through the menus that use it (F04, F07, F21, F27).
- OpenHands Cloud behavior (Cloud login and device flow, organizations, sharing, task URLs, sandbox pause): needs a Cloud account. The recipes are written up to that point and recorded as `blocked` (for example `F07.cloud-only` and the Cloud rows of F25).
- Locked-to-Cloud deployments: `scripts/static-server.mjs --lock-to-cloud` exists, but `bin/agent-canvas.mjs` does not forward it, so `control-openhands launch` cannot start that mode.
- Page-local load errors (for example the LLM profiles or apps list failing while the rest of the backend works) need fault injection; stopping a service with `service stop` replaces the whole app with the backend-unavailable screen instead. The pages' empty and error copy is mapped where reachable.
- Real microphone dictation, native file dialogs outside the browser, and Electron window chrome beyond what F26 drives.
