# Upstream migration of the October 17 maintenance pass

This record is dated 2026-10-18. It describes migration and validation, not a new
weekly live audit.

Source: [enyst/playground#129](https://github.com/enyst/playground/pull/129),
branch `codex/weekly-verify-2026-10-17`, head
`6aeca3d013802ba58922675f09ec7bd4e27f24db`, tracked by
[enyst/playground#128](https://github.com/enyst/playground/issues/128).
All ten source commits were cherry-picked with original authors and `-x`
provenance onto upstream main `852132d38dbc15e03d6825bbacbeb971c72575be`.

The upstream cleanup issue is
[#18424](https://github.com/OpenHands/OpenHands/issues/18424), which already
carries `ready-for-dev`. The source umbrella tracker is retained rather than
duplicated. Cleanup alternatives [#18486](https://github.com/OpenHands/OpenHands/pull/18486)
and [#18503](https://github.com/OpenHands/OpenHands/pull/18503), and map maintenance
[#18590](https://github.com/OpenHands/OpenHands/pull/18590), were open and unmerged
when checked. Their overlap is disclosed; it does not establish that the changes
are present on main.

Migration adjustments:

- Preserve the newer [#18611](https://github.com/OpenHands/OpenHands/pull/18611)
  critic guidance: use an available authorized OpenHands/proxy profile and reuse
  its LLM key; a separate critic key is not inherently required. Retain the
  source's additive Local rendering, bounded nudge and task-result observations,
  and main's #18541 entry.
- Point maintained recipes to upstream reports #18601, #18613, #18615 and #18614
  for fork sources #124, #125, #126 and #127, respectively. Historical audit
  records and actual evidence URLs retain their original provenance.
- Remove one trailing blank line from the retained controller test log. No
  functional script or fixture behavior was changed during migration.

Fresh migration checks: 73 controller tests across two files passed; changed
JavaScript syntax and Python parsing passed; the map retains 28 families and 779
IDs with zero map problems, 31 routes and 29 component directories with zero
unmapped entries, and 1,110 cited test IDs with zero unresolved references. The
fixture archive contains only executable `backend.py`, matches the readable
source, and matches its manifest SHA256. These checks do not replace live proof.

The retained [running-Canvas fixture screenshot](https://github.com/enyst/playground/blob/c461066866f4dcffec9e845995b57e9ba25ffb89/.agents/skills/verify-openhands/references/backfills/evidence/2026-10-17/final/checkedin-app-backend-fixture.png)
and [controller live receipt](https://github.com/enyst/playground/blob/c461066866f4dcffec9e845995b57e9ba25ffb89/.agents/skills/verify-openhands/references/backfills/evidence/2026-10-17/final/controller-live-proof.json)
belong to the original Linux audit at frozen target
`61a3bd93e642a0ff4def73b29496a73245628209`. They are not new Mac captures. The
receipt distinguishes successful HTTP and direct WebSocket observations from
the known SDK #5648 proxied-WebSocket failure. The original blocked/partial
verdict, historical publication-policy failures and accepted baseline remain
historical evidence, not current CI claims. This migration does not rerun all
779 behaviors or advance the accepted baseline.
