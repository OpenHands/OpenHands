# OSS-15337 frontend availability evidence

Baseline: 47241dc090a84b1eee561cbfc3b846bc120a25eb, production build input ID 3869caed09473fba-dirty (only non-production tests/helper were dirty at baseline build).
Fixed production build input ID: 5b0a601239a15775-dirty. Canvas 1.38.0, agent-server/SDK 1.61.0, automation 1.31.0. Chromium; desktop 1440x1000, phone 390x844.

This is real running Canvas with a real private Agent Server and automation service, plus an explicitly synthetic deployment-policy facade. It is NOT a real OHE/Replicated deployment test. No Playwright route interception, fake HTML or LLM calls. The facade serves existing web-client config with self_hosted and enable_agent_profiles_page false/true. Profiles and automation capabilities are forwarded from the private services. Only the absent-automation-capability control removes the two profile feature names. Cloud identity/workspace envelopes are diagnostic fixtures. The backend is named “Diagnostic policy control (not OHE)”.

| Policy | Automation profiles capability | Before selector | After selector |
|---|---|---|---|
| disabled | present | visible (bug) | absent |
| enabled | present | visible | visible, default profile selectable |
| enabled | absent | not recorded | absent |

Disabled form reaches review. Phone dialog inside viewport, no horizontal overflow. No automation created/dispatched. The single real seeded profile is intentionally still selectable when policy permits it; profile-count UX is not changed.

Reproduce with the checked-in verify-openhands CLI and fixture:
1. Launch baseline/fixed checkout with `control-openhands launch --new`, doctor, browser start, onboard --skip.
2. `node .pr/oss-15337/profile-availability.mjs RUN [PORT]` on a loopback-only free port.
3. Arrange a diagnostic cloud backend in this owned browser only, with dummy `diagnostic-placeholder` key and facade host, using backend-registry storage keys.
4. Set RUN/private/profile-policy.json to {"enabled":false} or {"enabled":true}; full navigate `/automations/new/news-digest` between states to clear query caches. For missing runtime support use {"enabled":true,"automationProfiles":false}.
5. `browser count testid=automation-agent-profile`, screenshot; enabled: choose default. Disabled: Continue → setup-review. Repeat phone.

Affected suites: 151 passing tests across template dialog, setup panel, composer and shared availability hook. New disabled-policy regression fails on baseline (expected absent, actual visible). Build, typegen+tsc and scoped ESLint pass. Test mocks control only external configuration/service boundaries; production availability and form paths execute.

## Independent-review timing follow-up

Two additional regressions failed before the follow-up: pending blur preflight retained old eligibility, and reviewed values remained confirmable after policy changed. The dialog now cancels queued blur checks on eligibility changes, reads current effective values in delayed validation, discards stale Continue results, and binds confirmation to the exact reviewed values. Eligibility changes send reviewed forms back for validation. 153 affected tests pass; final template suite 28/28, typecheck/build passed. Disabled and enabled-options captures refreshed from the rebuilt private `reviewed2` run; other captures describe unchanged visual behavior from the earlier run. Configuration facade remains explicitly synthetic. No actual OHE policy refresh or automation execution is claimed.
