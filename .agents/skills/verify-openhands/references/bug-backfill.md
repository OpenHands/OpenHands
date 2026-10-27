# Accounting for existing bugs

[OpenHands/OpenHands#18309](https://github.com/OpenHands/OpenHands/issues/18309)
backfills bugs that predate the map's maintenance baseline. Freeze the current
`main` SHA and the open `bug` issue inventory separately: a Git revision fixes
the product under test; the inventory timestamp fixes which issues were open.
Do not move the maintenance baseline during this backfill.

The inventory includes every open Canvas `bug`, excluding pull requests, and
SDK or automation bugs with a concrete mapped Canvas consumer. Read the report
and the relevant source before assigning a family. Bugs in a removed legacy
product, CI, documentation, unused SDK client paths or build/repository tooling
without a current Canvas consumer are outside this map, with an explicit
reason. User-visible installation and startup failures belong to the runtime
map even when their cause is packaging or build output. Missing credentials,
OS or fixture data are verification prerequisites, not reasons to exclude a
Canvas bug.

`ready-for-dev` means the issue is sufficiently specified and accepted for
implementation. Use it to prioritize live checks. It was introduced recently,
so older unmarked reports with usable reproduction steps are also candidates.
Do not add or remove readiness, priority, `bug` or the existing `reproduced`
label as a side effect of map bookkeeping.

## Three outcomes

| Outcome | Required record | GitHub label | Color |
| --- | --- | --- | --- |
| Mapped | Existing note, a live `Known failure` beside the unchanged expected result, or a `Known issue` Gotchas entry naming the concrete prerequisite when no recipe can exercise it. | `feature-map:mapped` | Blue `#1D76DB` |
| Not reproduced | A live run on the frozen target, and an issue comment with the revision, backend versions, entry point, exact steps and observed result. | `feature-map:not-reproduced` | Amber `#D4A72C` |
| Out of scope | A report row explaining why the issue has no current mapped Canvas feature. | `feature-map:out-of-scope` | Gray `#6E7781` |

These are accounting outcomes. A mapped issue is not necessarily reproduced;
its note distinguishes a reproduced failure from a reported failure blocked by
a prerequisite. Not reproduced does not mean fixed, and out of scope does not
dismiss the bug. Record verification separately: a `Known issue` with an
unavailable prerequisite is **BLOCKED**, with that prerequisite named; it is not
a `Known failure`. Reserve `Known failure` for an observed live reproduction
and preserve its date. An available scenario awaiting a run is `not-run`, not
blocked merely because it has not been attempted. Text carries this meaning
even when label colors cannot be seen. Each processed issue has exactly one outcome label in repositories where
these labels have been authorized. An unprocessed issue has none; keep it in
the pending section of the ledger instead of guessing a completed outcome.

## Evidence and labels

The map note or report is the record; GitHub labels are a searchable mirror.
Create labels or publish comments only when authorized by the requester. Before
adding a label, validate its supporting note or run, then remove only any of the
other two outcome labels on that same issue. Preserve all unrelated labels.
Cross-repository notes spell out the repository in their issue links.

A successful build, source fix or unit test cannot establish not reproduced.
Run the original user-visible scenario through `control-openhands`, and capture
its read-only result as the recipe requires. If the reproduction can be driven,
run it; do not turn a time limit into a missing-prerequisite claim. When no
recipe can drive a reported scenario, record the exact missing account, model,
OS, old persisted data or reproduction detail in the Gotchas note. Preserve
reproduction dates for unresolved notes already in the map. The map describes
current application behavior: once a fix applies at the current Canvas and
pinned backend versions, remove the resolved failure note and obsolete
reproduction recipe. Keep earlier observations in the dated backfill ledger,
not in historical sections of the feature map.

For a not-reproduced comment, introduce the agent and requester as instructed,
then include the frozen revision, actual backend versions, run ID, user entry
point, commands, expected behavior and observation. Leave closure to maintainers.
Do not publish keys, live session URLs, raw traces or unreviewed screenshots.

## Pass ledger

Keep one row per `(repository, issue number)` with outcome, owning family and
feature IDs, note or comment link, verification status, exact missing
prerequisite when blocked, and run/evidence details
when exercised. Report counts per repository and distinguish existing notes,
new reproductions and blocked known issues within the mapped outcome. Include
all pending rows, even if another shard owns them. The pass is complete only
when every frozen-inventory issue has exactly one supported outcome, every
not-reproduced comment has been published, and `control-openhands map check`
reports `ok`. Follow [report.md](report.md) for run identity, coverage and
teardown, and [daily.md](daily.md#several-agents) for whole-family sharding.

Daily and weekly passes query open bugs without a supported map or ledger
outcome, and re-triage reopened or newly bug-labeled issues even if an older
snapshot recorded an outcome. A historical ledger does not by itself cover a
changed scenario. Do not use issue creation time or `ready-for-dev` as a
completeness filter.

Re-check a not-reproduced result when its scenario, relevant source or backend
changes; a result applies to its recorded revision and configuration. Re-check
out-of-scope classification if the feature becomes supported. After a mapped
issue closes, confirm the fix applies at the current Canvas and pinned backend
versions and re-drive its scenario where possible. Remove resolved bug notes
and obsolete recipes from the map; retain the current expected behavior.
Record verification limits and earlier observations in the dated ledger. Keep
label reconciliation in that same pass.

The completed [2026-10-12 backfill](backfills/2026-10-12-open-bugs.md) records
its frozen inventory, per-issue outcomes, published comments and retained run
evidence. The [2026-10-16 main refresh](backfills/2026-10-16-main-refresh.md)
records the later inventory and reconciles closed reports with current source,
pinned backend versions and fresh observations. Read the latest reconciliation
before treating a historical result as current. Fixture tests use the installed SDK Python, which supplies `httpx`:
`python -m unittest discover -s .agents/skills/verify-openhands/scripts/fixtures -p 'test_*.py'`.

## Refresh after main changes

Keep the original snapshot and its evidence dated. Freeze the new main target
and issue inventory separately, inherit main's accepted maintenance baseline,
and add a reconciliation ledger. Re-read every affected family, including
notes imported by main and closed issues absent from the original snapshot.
A merged SDK fix may be newer than Canvas's released pin: a closed issue is
not enough to retire a supported-version failure. Conversely, when main has
fixed or replaced a flow, describe the current behavior and remove obsolete
failure notes and reproduction steps. Keep superseded observations only in
the dated backfill ledger, outside the current feature map.

For each closed report distinguish current live observation, accepted source
change awaiting the original scenario, and a fix unavailable at the tested
pin. A source review, closure or passing unit test is not a verified runtime
fix. Retain the concrete account, model, platform or fixture prerequisite for
scenarios that cannot be driven. Refresh changed not-reproduced scenarios and
append the new revision, pins and steps to their existing issue comments; keep
the earlier result intact. Reconcile only the three outcome labels after the
current supporting record exists, leaving unrelated labels unchanged.
