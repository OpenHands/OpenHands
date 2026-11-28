# Outcome store: related PRs in OpenHands/automation

State read on 2026-11-25.

| PR | Title | State |
| --- | --- | --- |
| [#770](https://github.com/OpenHands/automation/pull/770) | feat: store automation outcome values for analytics | merged |
| [#793](https://github.com/OpenHands/automation/pull/793) | feat: add factory-validated outcome parsers | merged |
| [#794](https://github.com/OpenHands/automation/pull/794) | feat: wire factory outcomes into automation runs | merged |
| [#807](https://github.com/OpenHands/automation/pull/807) | refactor: store factory outcome keys as objects | merged |
| [#810](https://github.com/OpenHands/automation/pull/810) | chore: remove the flat outcome-keys read shim | merged |
| [#809](https://github.com/OpenHands/automation/pull/809) | refactor: take outcome parsers off the plugin-spec path | open |
| [#811](https://github.com/OpenHands/automation/pull/811) | fix: make the outcome read routes' until bound exclusive | open |
| [#812](https://github.com/OpenHands/automation/pull/812) | feat: reserve the platform. outcome key prefix | open |

## What the cards read

- `outcome_router.py.txt`: `GET /factories/{id}/outcomes/values` (raw) and `.../outcomes/aggregate`.
- `outcome_aggregation.py.txt`: aggregate operations `sum count avg min max count_true count_false`.
- `outcome_schemas.py.txt`: key definitions (a `type` per key today), value types, limits.
- `outcome_values.py.txt`: how a submission becomes stored rows.
- `042_add_automation_outcome_values.py.txt`: the table shape (no dimension columns).
- `F20-outcomes.md`: the repo's own feature notes for outcomes.
