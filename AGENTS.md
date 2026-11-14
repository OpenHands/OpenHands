# Repository Notes

## General

- This repository is the OpenHands Agent Canvas React/TypeScript frontend.

## i18n workflow
- After editing `src/i18n/translation.json`, run `npm run make-i18n` to
  regenerate `src/i18n/declaration.ts` (gitignored, auto-generated) and
  `scripts/check-translation-completeness.cjs` to confirm full locale coverage.
- Removing an i18n key: delete the whole locale block from translation.json,
  then `make-i18n`, then remove any test-mock entries (tests inline a
  `translations` map in the `useTranslation` vi.mock).

## Commit messages
- The persistent shell can garble long multi-line `commit -m` bodies (echo
  loop corrupts the stored message). Use `git commit -F <file>` with a
  written temp file instead of `-m` for any non-trivial message.

- Primary verification commands are `npm run lint`, `npm test`, `npm run build`, and `npm run build:lib`.
- Direct dependencies and dev dependencies are exact-pinned. Use the committed `package-lock.json` with `npm ci`, and update `package.json` and `package-lock.json` together through npm.
- Public skills come from `@openhands/extensions`; project-specific contributor guidance lives under `.agents/skills/`.
- Default working-directory behavior must reuse `DEFAULT_WORKING_DIR` from `src/api/agent-server-config.ts` rather than hardcoding `/workspace/project`.
- All pull requests must comply with [`.agents/skills/custom-codereview-guide.md`](.agents/skills/custom-codereview-guide.md), in addition to the general contribution requirements and CI checks.

## Repository Ownership

Before adding code, confirm that the change belongs in this repository.

| Repository | Owns | Add code there when… |
|---|---|---|
| [`OpenHands/OpenHands`](https://github.com/OpenHands/OpenHands) | Agent Canvas UI, frontend state, backend selection, frontend service integration, and local-stack orchestration. | Changing UI, frontend state, or how Canvas consumes an existing API. |
| [`OpenHands/software-agent-sdk`](https://github.com/OpenHands/software-agent-sdk) | Python SDK, Agent Server, agents/tools, conversations, events, canonical REST/WebSocket API, and `clients/typescript/`. | Adding or changing backend behavior, endpoints, wire contracts, or typed client access. |
| [`OpenHands/extensions`](https://github.com/OpenHands/extensions) | Reusable public skills, automations, plugins, and integrations. | Adding or editing reusable extension content rather than Canvas behavior. |
| [`OpenHands/automation`](https://github.com/OpenHands/automation) | Automation definitions, scheduling, webhooks, run history, and dispatch. | Changing when automations run or their scheduling/webhook lifecycle. |

The normal dependency direction is Agent Server contract → TypeScript client → Agent Canvas. Do not reimplement Agent Server endpoints or contracts in Canvas.

## Shared Frontend Change Checklist

Before changing a shared adapter, conversation builder, setting, state selector, or presentation helper:

- Raise `compatibility.minimumAgentServer` in `config/defaults.json` when Canvas begins requiring a new Agent Server endpoint, field, schema, or behavior.
- Enumerate affected consumers across Local and Cloud backends, OpenHands and ACP agents, standard/planning/delegated conversations, and standalone and embedded Canvas. Test every path with distinct control or data flow without testing an irrelevant Cartesian product.
- Give durable frontend state one named owner and one obvious writer. Handle existing-state hydration or migration and all supported create, update, delete, default, and active-item transitions. Leave destructive transitions in a deterministic valid state or an explicit empty state handled by the UI.

## PR Description Human Check

The `HUMAN:` section in PR descriptions is reserved for human contributors only. AI agents must not add to, edit, move, or remove it. If validation fails because it is missing or empty, stop and ask the human user to update it in their own words. If a human already updated it, report the exact validator error rather than editing it.

## Functionality-Specific Skills

Detailed contributor knowledge is split into skills so it loads only for relevant work. Invoke every applicable skill before changing that area; cross-cutting changes may require several skills.

| Skill | Use for |
|---|---|
| [`telemetry-analytics`](.agents/skills/telemetry-analytics/SKILL.md) | PostHog, telemetry consent/identity, typed tracking events, Cloud funnel observability, and onboarding instrumentation. |
| [`e2e-testing`](.agents/skills/e2e-testing/SKILL.md) | Live or mock-LLM Playwright suites, Docker E2E, E2E CI/reporting, artifacts, and failure diagnosis. |
| [`frontend-api-contracts`](.agents/skills/frontend-api-contracts/SKILL.md) | `src/api`, typed Agent Server calls, Cloud/runtime transport, compatibility, backends, settings, secrets, auth, and conversation contracts. |
| [`local-stack-runtime`](.agents/skills/local-stack-runtime/SKILL.md) | Dev launchers, runtime-services metadata, ingress, automation startup, process shutdown, centralized versions, and Docker. |
| [`desktop-electron`](.agents/skills/desktop-electron/SKILL.md) | Electron startup, branding, packaging, bundled Node/uv, universal macOS builds, and desktop CI. |
| [`frontend-development`](.agents/skills/frontend-development/SKILL.md) | React/UI work, i18n, named identifiers, MSW mock mode, lazy loading, bundle performance, and feature-specific UI invariants. |
| [`pr-design-doc`](.agents/skills/pr-design-doc/SKILL.md) | Writing the PR design document stored in the PR body. |
| [`release`](.agents/skills/release.md) | Cutting and verifying an `@openhands/agent-canvas` release. |
| [`verify-openhands`](.agents/skills/verify-openhands/SKILL.md) | Driving the real app like a user with `control-openhands` and checking behavior against the existing feature map. |

The detailed rules live in each skill's `references/guide.md`; do not copy them back into this file. Update the owning contributor skill when its invariant changes, except for the independent verification skills described below.

### Verification Skill Independence

Implementor agents must use `.agents/skills/verify-*/` as read-only verification tools. Do not update their feature maps, expected results, known-bug recipes, drivers, or fixtures while implementing a product change. This separation applies even when an issue's acceptance criteria ask for verification-skill updates. Report a mismatch or missing coverage instead of changing the specification to match the implementation.

Verification-skill maintenance belongs in a separate PR and a separate agent context. For a product fix, make that follow-up after the fix lands. A verify-skill PR may include only verify-skill paths, `AGENTS.md` files, and root `.pr/` evidence. Product source, tests, CI, and other docs must stay separate.

## Testing Baseline

Create TDD tests for behavioral changes. Keep tests focused on user behavior and real code paths:

- Use Arrange, Act, Assert structure and clear test data.
- Avoid duplicate cases and duplicate assertions.
- Mock an underlying service rather than the hook that consumes it.
- Extend an existing test file when it is a natural home; create a new file only when necessary.
- Avoid brittle presentation-only assertions. Test functional CSS contracts directly when they are behavior.
- Do not mirror literal source, fixture, translation, or class-string definitions in tests. Assert consumer behavior or an actual build/runtime contract instead, and do not export internals solely to make them testable. Preserve coverage for routing, accessibility, async behavior, behavioral contracts, and functional CSS/build contracts.
- Use the minimum number of cases that fully cover the intended behavior and edge cases.

Use the `e2e-testing` skill for suite selection and E2E-specific requirements.

## Releases

- Follow `.agents/skills/release.md` for release automation. Never hand-edit a release-please branch.

## DigitalOcean Managed Agents

Before changing `scripts/mars-*` / `scripts/tunnel-*`, `src/api/mars/`, the Managed Agents UI (`src/components/features/backends/managed-agents/`) or the MARS hooks, read `src/api/mars/AGENTS.md`.

## Dependency release-age policy

- Wait at least seven days after publication before adding or updating a
  third-party npm or PyPI version, including transitive lockfile changes.
  A weekly update schedule alone does not enforce this minimum age.
- Keep the npm Dependabot `cooldown.default-days: 7`. This repository has
  no Python Dependabot entry or uv project/lock. Apply the PyPI age rule to
  version changes in Docker or launcher install commands as well.
  Existing named first-party exceptions do not exempt their third-party
  dependencies. Do not add broad exemptions or disable the cutoff to make a
  resolver succeed.
- Verify the exact version's publication/upload time from npm or PyPI metadata
  rather than the package's creation date.
  If age cannot be established, report the missing evidence rather than guess.
- Security updates can bypass Dependabot cooldown. An urgent exception needs
  explicit maintainer approval recorded in the PR with the package, version,
  advisory/reason, and publication time. Agents must not grant themselves an
  exception. Keep overrides narrow and remove temporary exceptions afterward.
- Repository resolver settings do not protect downstream `pip install` or
  `uv tool install` consumers, and do not retroactively validate existing locks.
  Use the committed lock for reproducible installs; review newly resolved
  artifacts before committing a lock update.

This repository pins npm 11.12.1 in `package.json` (`packageManager` and Volta);
use `npm --version` to confirm the active tool. The previous npm 10.5.0 pin
silently ignored `--min-release-age`. For manual npm dependency resolution, run
`npm install --min-release-age=7` (days), or verify registry timestamps before
committing the result. The option requires npm >= 11.10. `npm ci` replays the
lock and is not proof that newly locked versions meet the age rule.
