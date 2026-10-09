# Automation consent recovery — 9 October 2026

Verified executable source: `3900865bbfcb80ee7a399bc04a6a115ab4d5babd` (#16959).
Baseline: `95dcdcdf12bd78f02c52b80cccf2f5f4f35242a4` (main).
Both used isolated production `bin/agent-canvas.mjs` stacks, real Agent Server/SDK
1.53.0, the same pinned Automation service, Node 25.9.0, installed Google Chrome,
English UI and 1440×1000 viewports. Doctor verified stack health before the outage.
No API response was mocked and no model was configured or executed.

## Result

**The real startup race reproduces on main and recovers with this PR.**

Both browsers mounted while only their owned Automation service was stopped,
with consent denied. Both sent the actual POST to
`/api/automation/v1/telemetry/consent` and received HTTP 502. The service was then
restored with `control-openhands restart`, retaining the browser page, keys and
state. Neither browser was reloaded after its initial failed sync, and effective
consent remained denied.

- Main: one failed POST at 19:55:17.691 UTC, no subsequent attempt by the final
  observation at 19:56:23 UTC, although Automation was healthy again.
- PR: requests at 19:53:33.406, 34.425, 36.429, 40.433, 48.436 and 19:54:04.445 UTC.
  The final request received HTTP 200. Two restart-window requests have no HTTP
  response recorded; they are not counted as acknowledgements. No additional
  consent POST appeared through the final observation after acknowledgement.
- Original observed payload on every request:
  `{"consent_granted":false,"frontend_distinct_id":null}`.
- Both real Automation health reads returned HTTP 200 with `status=ok`.
- External analytics requests: zero in both browsers. The builds use
  `VITE_DO_NOT_TRACK=1`, so capture stays disabled and no real actor ID is used.

Original safe observations: [main request timeline](main-final-network.json),
[PR request timeline](head-final-network.json), [main health](main-healthy.json),
[PR health](head-healthy.json), [main analytics check](main-analytics-network.json),
[PR analytics check](head-analytics-network.json).

## Reproduction commands

Start fresh isolated runs on the two exact revisions. Do not use someone else's
stack or browser. Keep `OH_VERIFY_RUN` set to each returned run directory.

```sh
export PATH="$PWD/.agents/skills/verify-openhands/scripts:$PATH"
export OH_VERIFY_RUN=$(control-openhands launch --new --print-run)
control-openhands doctor
control-openhands onboard --skip
control-openhands browser goto /settings/app
control-openhands service stop automation
control-openhands browser network --clear
control-openhands browser record start --feature F16.analytics --name startup
control-openhands browser reload
control-openhands browser network --filter /api/automation/v1/telemetry/consent --bodies
control-openhands browser record pause
control-openhands restart
control-openhands api GET /api/automation/health --pick status
# Leave the page mounted and consent unchanged; allow the capped backoff to run.
control-openhands browser network --filter /api/automation/v1/telemetry/consent --bodies
control-openhands browser record resume
control-openhands browser record stop --gif
control-openhands browser network --filter 'posthog|z\.openhands\.dev'
control-openhands stop --purge-private
```

The baseline fixture's visible analytics preference was restored to off before
its final recorded attempt, matching the draft. Hard opt-out kept effective
consent denied in both. No consent mutation, page reload or UI doctor probe was
used between the failed sync and the recorded recovery result; a probe would
mount another app and contaminate the original page's request timeline.

## Reviewed media and limits

[Main recording](main-startup-consent.mp4), [PR recording](head-startup-consent.mp4),
[main UI](main-consent-ui.png), [PR UI](head-consent-ui.png). GIFs preview the same
recordings. These background requests do not change the form's visible controls;
the real request timestamps and statuses establish retry ordering and recovery.
Long service-restoration waits were cut from the videos, as shown above.

This proves denied-consent startup recovery and stop-after-acknowledgement against
real services. It does not claim live Cloud consent, analytics emission, identity
reset, or privacy-clear actor revocation; the existing regression suite covers
those supported state transitions. No automation was created or dispatched.
All owned stacks were stopped, owned ports closed, and private state purged while
review evidence remained available. Raw logs, storage and keys are not published.

Verification performed by an AI agent for Engel Nyst (@enyst).
