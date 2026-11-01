# PR #17585 review smoke evidence — 2026-10-18

Codex, reviewing on Engel’s behalf.

The screenshots show the genuine running Canvas settings experience at source commit `f06b4e5ddec035ad00eb6cc31bf8585335fd7fed` (merged current main plus the wrapper/import cleanup). They are smoke evidence for a behavior-preserving refactor, **not a before/after bug reproduction**.

- `desktop-settings-smoke.png`: Application settings and desktop navigation at 1440 × 1000.
- `phone-settings-hub-smoke.png`: real Settings hub at 390 × 844.
- `phone-application-smoke.png`: Application settings after clicking its link in the real phone hub at 390 × 844.

All three images were inspected for readability and sensitive content. Consent/onboarding overlays were dismissed using the CLI; no DOM injection, mocked backend or model response was used. No model was configured or called.

## What this does and does not demonstrate

The live settings shell and phone navigation render, and the Application link reaches its real page without a page or app error. The phone hub has no horizontal overflow. Both before-drive and after-drive `doctor` checks pass, including the served-build identity check.

Current `SettingsLayout` mounts `SettingsDesktopSidebar` and `SettingsCompactNavigation`. It does not mount the legacy `SettingsNavigation` / `SettingsMobileDrawer` compatibility components containing the inlined header/divider (the real DOM contains zero `settings-navbar` legacy nodes). Screenshots of the product therefore cannot demonstrate those dormant compatibility branches; the existing component tests cover them. These captures make no such claim.

## Reproduction and environment

Runtime: macOS, Node 24.19.0, Canvas 1.27.1, Agent Server/SDK 1.55.1, automation 1.20.1, configured Playwright Chromium 153.0.8010.12. Real isolated local backend with private run state; no shared service was restarted or modified.

The app and library builds had already passed with identical source content before the source merge commit. The verifier reused the app build (`--build never`) and its generated `verify-revision.txt` marker recorded build identity `4d0dd4a347778e43`. The source commit and complete build-input match were checked by `doctor`.

From the recorded commit, after building the app:

```sh
export PATH="$PWD/.agents/skills/verify-openhands/scripts:$PATH"
export OH_VERIFY_HOME="$(mktemp -d)"
export OH_VERIFY_RUN=$(control-openhands launch --new --print-run)
control-openhands doctor
control-openhands onboard --skip
control-openhands browser viewport desktop
control-openhands browser goto /settings/app
control-openhands browser wait 'testid=app-settings-screen'
control-openhands browser screenshot --feature F09.settings-shell --name desktop-settings-smoke
control-openhands browser viewport phone
control-openhands browser goto /settings
control-openhands browser wait 'testid=settings-mobile-hub'
control-openhands browser screenshot --feature F09.settings-shell --name phone-settings-hub-smoke
control-openhands browser click 'testid=settings-mobile-hub >> role=link[name="Application"][exact]' --expect-url '/settings/app$'
control-openhands browser wait 'testid=app-settings-screen'
control-openhands browser screenshot --feature F09.settings-shell --name phone-application-smoke
control-openhands doctor
control-openhands stop --purge-private
```

Cleanup succeeded: the owned launcher/browser stopped, all five owned ports closed, private state was purged, and all three evidence files survived. `summary.json` records sanitized checks and image hashes.
