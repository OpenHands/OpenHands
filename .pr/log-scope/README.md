# Live Canvas: automation logs retain conversation scope

[Before/after recording](logs.gif), captured 2026-09-13 around15:26 UTC, opens the same completed watchdog run in two isolated Canvas frontends against the unchanged live Agent Server and Automation services.

Before, opening Logs issues an unscoped `/api/bash/bash_events/search` request and receives400: Docker requires a conversation identity. A later conversation lookup allows a scoped retry to succeed, but the user sees the routing-error toast. After, every request is conversation-scoped and returns200; the log opens with no routing error. [Observed paths/statuses](observations.json).

The displayed log is a real factory action: watchdog run `7a447b5a-58b1-4c4a-ba27-85365ac1baed` automatically merged Airbnb PR46 as `aee04f1ba9af10eaeed6f9fae9c1e5139932ecef`. The capture reads its existing log; it does not trigger another merge or run.

The fix passes the already-known owning conversation ID through the existing BashService and SDK client options and includes that ID in the query cache key. No new transport or runtime endpoint is added. A missing owner cannot issue an unscoped request and uses the existing missing-conversation message. Twenty-seven targeted service/hook tests pass, plus type checking and focused lint; the final missing-owner message assertion was rerun separately.

## Controlled versions

Both frontends use the same Docker Agent Server19104 and Automation19105 with the same definition/run. Baseline assets are retained from Canvas negative-control build `204c1845b9643ce4f190bbd6d20a5d50a117fafa`; fixed assets are integration `4cd1513bbe74abd2c1f9e58a77cf64375aafe001`, containing scoped-log fix `8216ce50f`. The final missing-owner message refinement does not affect this recording's known-owner case.

The baseline also contains the previously documented profile/plugin/static-asset negative controls. Those paths are not exercised here: this scenario only reads an existing automation run, with no profile creation, catalog setup, or asset replacement. Fixed integration additionally includes independent onboarding17211, also not exercised. Both use the same SDK TypeScript package integration `ac6d12b0b9d76f1cc38a6eb1ea51cd92e34a0bf2`. This is a composed preview comparison, not a claim of compatibility with an unreleased client's published predecessor.

Frontends ran on9110/9111 and are stopped. Original PNGs remain under `factory-state/evidence/log-scope`. The earlier exploratory pair accidentally reused a refreshed baseline build and is excluded; only the final400→200 comparison is published. Four genuine screenshots are displayed for three seconds each.
