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
  `deepseek-flash` (active) and `deepseek-pro` from `DEEPSEEK_API_KEY`.
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
  one a check used.
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
