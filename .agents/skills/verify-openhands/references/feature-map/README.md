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
- Many settings switches are a hidden `<input>` inside a `<label>`: click the
  label text (or the visible track) and read `.checked` with `browser eval`.
- Settings and panels scroll inside a container: `browser scroll '<sel>' --by 600`
  scrolls the right one, and `--full-page` screenshots capture only the
  viewport there, so take one screenshot per scroll position.
- `conversation start` leaves the browser on the conversation page, which also
  has a `testid=submit-button`. `browser goto` the next page explicitly.
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
  new page errors; count and report them even when the UI recovered.
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
