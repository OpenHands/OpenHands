# Issue 19974 evidence

All captures were produced from isolated `control-openhands launch --new` runs with the same workflow: open `/` in a clean browser profile, observe the first-run state, then confirm telemetry preferences and observe onboarding.

## Baseline on current main

Run: `/tmp/openhands-verify/2026-11-22T2250-6fc78a`  
Build: `c5e47c4b4dab` / Agent Server `1.63.1`

- Visual: [`before-first-run-overlap.png`](before-first-run-overlap.png)
- Semantic: [`before-first-run-overlap.aria.txt`](before-first-run-overlap.aria.txt)
- Result: telemetry and onboarding are both present as dialogs.

## Fixed branch

Grant run: `/tmp/openhands-verify/2026-11-22T2301-358319`  
Deny run: `/tmp/openhands-verify/2026-11-22T2302-21fd07`  
Build: dirty branch build `e19aba878c21b3e2-dirty` / Agent Server `1.63.1`

- First load visual: [`after-first-run-telemetry-only.png`](after-first-run-telemetry-only.png)
- First load semantic: [`after-first-run-telemetry-only.aria.txt`](after-first-run-telemetry-only.aria.txt)
- Grant choice visual: [`after-onboarding-after-grant.png`](after-onboarding-after-grant.png)
- Grant choice semantic: [`after-onboarding-after-grant.aria.txt`](after-onboarding-after-grant.aria.txt)
- Deny choice visual: [`after-onboarding-after-deny.png`](after-onboarding-after-deny.png)
- Deny choice semantic: [`after-onboarding-after-deny.aria.txt`](after-onboarding-after-deny.aria.txt)
- Existing-user visual: [`after-existing-user-no-modals.png`](after-existing-user-no-modals.png)
- Existing-user semantic: [`after-existing-user-no-modals.aria.txt`](after-existing-user-no-modals.aria.txt)
- Existing-user after reload visual: [`after-existing-user-after-reload.png`](after-existing-user-after-reload.png)
- Existing-user after reload semantic: [`after-existing-user-after-reload.aria.txt`](after-existing-user-after-reload.aria.txt)

The fixed first load exposes one dialog (`Help improve OpenHands`). Confirming either consent choice advances to one onboarding dialog (`Welcome to OpenHands`). After onboarding is skipped, both the immediate existing-user state and a real reload have no modal dialog.
