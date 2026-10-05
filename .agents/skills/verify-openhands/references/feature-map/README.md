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
  run before the preset.
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
  lie). `browser eval` takes one expression; wrap statements in an IIFE.
- Input that is not a click: `browser upload-via <trigger> <file>` answers the
  real file chooser; `browser drop-files <sel> <file>` and `browser paste <sel>
  --file F --text T` deliver drag-and-drop and paste events (target an element
  inside the drop zone: events bubble up); `browser drag` reorders draggable
  rows or moves a grip `--by DX,DY`; `browser choose <combobox> <label>` picks
  an autocomplete option; `browser clipboard` reads what a Copy button wrote.
- Short-lived feedback: `browser toasts --history` lists every toast since the
  page loaded; `click --observe SEL` records transient labels.
- Agent-side proof: `conversation events <id> --grep TEXT [--from-start]`
  searches whole events (the system prompt's skills and tools, tool
  arguments); rows show activated skills. After sending a follow-up message,
  `conversation wait <id> --fresh` ignores the previous terminal status.
  `fixture skill` writes a personal or project `SKILL.md`; `api GET ...
  --pick a.b.c` reads one field.
- An already-open page does not refetch settings written by `llm` or `api
  --write`: `browser reload` before asserting UI state after an arrange step.
- Settings and panels scroll inside a container: `browser scroll '<sel>' --by 600`
  scrolls the right one, and `--full-page` screenshots capture only the
  viewport there, so take one screenshot per scroll position.
- `conversation start` leaves the browser on the conversation page, which also
  has a `testid=submit-button`. `browser goto` the next page explicitly.
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
  pass|fail|blocked|not-run`. A skipped entry point is never verified through a
  different one. Blocked rows name the missing prerequisite and the attempted path.

## Feature entry contract

Each feature file starts with an H1 title, one paragraph describing the
user-visible behavior and a `Source:` line. It then uses exactly four H2 sections
in this order (`control-openhands map check` enforces it):

1. `Sub-features`: one bullet per stable ID (`` `Fnn.slug` ``) and behavior.
2. `How to get to it (user POV)`: every user entry point.
3. `Driving it with control-openhands`: starts with `Preconditions:`, then labeled
   bullets pairing each user action with exact commands and the observable result.
4. `Gotchas`: traps that waste or invalidate a run, and linked known issues.

How to write and prove new entries: [../mapping.md](../mapping.md).

## Features

<!-- Families table: filled from `control-openhands map ids`. -->
