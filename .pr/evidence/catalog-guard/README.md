# Catalog guard validation, 2026-10-18

I’m Codex, reviewing and maintaining this PR on Engel’s behalf.

Product commit: `fd815080295a11b5d9baea175a574f72ec94e37b`, after merging main `78beca15f1` and resolving the catalog-test conflict. Production app build passed. The only production delta from main remains the two nullish prerequisite guards.

## Live scope

An isolated real Canvas stack used Agent Server 1.55.1, automation 1.20.1, extensions 0.30.0, Node 25.9.0, and Playwright Chromium 153.0.8010.12 on macOS. No LLM API key was configured, and no conversation, model request, or automation run was created. Telemetry consent was declined and onboarding skipped. The stack passed `doctor` after startup completed.

Commands, using this checkout's `control-openhands` CLI and the isolated run:

1. `launch --new --run-id codex-pr16278-20261018 --port-from 35400 --no-browser`
2. `doctor`; `browser start`; `onboard --skip`
3. `browser goto /automations/templates`
4. `browser count '[data-testid^="recommended-automation-card-"]'` returned **23**.
5. `browser screenshot --feature F22.templates-page --name current-catalog-desktop` at **1440×1000**.
6. `browser viewport phone`; `browser screenshot --feature F22.phone --name current-catalog-phone` at **390×844**.
7. The section's phone bounding-box inspection reported **no horizontal page overflow**. `browser errors --app-only --no-warnings` reported **0 page errors and 0 app errors**.

`desktop.png` and `phone.png` are unchanged captures of that real current catalog. They do **not** show a malformed catalog or a before/after crash reproduction. All 23 entries in the pinned package have `requires.integrations`; the historical issue cites extensions 0.11.0, while the original PR already pinned 0.13.0.

## Defensive regression coverage

The component suite temporarily removes `requires` or replaces it with `{}` on one catalog entry, then renders the real `RecommendedAutomationsSection`. Each case asserts that the malformed card remains selectable without integration pills and another valid card remains visible. The catalog object is restored in `finally`.

Both component cases were first run with the two guards removed and failed inside `getIntegrationIds` with the expected missing-prerequisite errors. Restoring the guards produced **79 passing tests across four suites**: catalog helpers, recommended automations, the recommendation rail, and manifest prerequisites. Typecheck and focused ESLint passed. These are injected malformed-data regressions; no production fault is claimed.

The preexisting helper cases cover both `getIntegrationIds` and `getRequiredIntegrationIds`. Current icon, impact-statement, and launch-prompt tests were retained during conflict resolution. The prior review's `SetupEntry` observation concerns a separate input contract and does not warrant expanding this patch.
