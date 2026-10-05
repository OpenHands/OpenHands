---
name: custom-codereview-guide
description: Repository-specific triage and review rules for the OpenHands Agent Canvas frontend.
triggers:
  - /codereview
---

# OpenHands Agent Canvas Code Review Guidelines

This guide supplements the public `code-review` skill with rules specific to
`OpenHands/OpenHands`, the Agent Canvas frontend. Read `AGENTS.md` first; it is
the detailed source of truth for current architecture and test conventions.

## Repository and Product Scope

Confirm scope before detailed inspection. This repository owns the Agent Canvas
product: its UI, product behavior, and Canvas-specific integration of existing
SDK and automation capabilities.

Out of scope here, because another repository owns it:

- reusable agent-server, runtime, SDK, and client contracts
  (`OpenHands/software-agent-sdk`);
- generic automation scheduling, state, dispatch, and profile machinery
  (`OpenHands/automation`); and
- reusable extensions, skills, plugins, and automation bundles
  (`OpenHands/extensions`).

Cross-repository work is acceptable when the PR contains only the Canvas-owned
integration and depends on public interfaces from the owning repository. When a
change belongs in one of the repositories above, say so and ask a maintainer to
confirm before reviewing the rest.

## Triage: Ownership and Scope

Consult this guide during issue triage for ownership, supported behavior, and
acceptance criteria. Implementation and merge checks apply to a PR, not an
unwritten fix: do not require new regression tests, PR artifacts, or before/after
fix evidence to make an issue actionable.

Trace the failing boundary before assigning an owner. Inspect
`OpenHands/software-agent-sdk` for events, tools, ACP, profiles, conversation
lifecycle, and persistence; inspect `OpenHands/automation` for form contracts,
run state, and pagination. Check producers and consumers at actually installed
or supported versions using package locks, runtime versions, and
`config/defaults.json`, not just upstream main. Cite the version and source when
ownership or compatibility depends on it.

Prefer a fix in the owning provider over a downstream Canvas workaround. Separate
Canvas integration from provider work and state release/dependency constraints.
Distinguish introduced regressions from preexisting, independent upstream issues;
do not silently expand scope to fix them. Recommend material follow-up work, but
do not automatically edit other repositories or create tickets without
authorization. Use answers already in the issue or linked discussion rather than
asking the reporter to repeat them.

## Triage: Behavioral Acceptance and Readiness

Acceptance criteria describe observable outcomes, affected modes, and meaningful
failure or recovery cases—not a required implementation or PR-artifact checklist.
Keep the verification plan separate. Bugs need expected vs. actual behavior,
environment/version, and a reproducible scenario; features need the desired
outcome and scope, not evidence of an unimplemented feature.

Match issue evidence to behavior: actual Canvas captures for visual defects;
commands, logs, API output, or a focused reproducer/test for nonvisual defects.
Consider evidence already linked in comments; do not demand screenshots of logs
or relocation of an existing capture just to satisfy a template. State remaining
uncertainty without calling inferred or defensive handling a verified production
reproduction.

Readiness is not PR approval. Follow the current label policy in
`.github/workflows/issue-readiness-check.yml`: actors with `write`, `maintain`, or
`admin` permission may grant `ready-for-dev`, with no blanket bot exception.
Do not override a failing required check or invent a bot-only authorization rule.

## Implementation and Merge Review

The remaining checkpoints apply to implemented changes and their PR evidence.

## Review Sequence and Decision

Review the current PR head in this order:

1. Read the linked issue, its acceptance criteria, and unresolved review threads.
2. Confirm that the change belongs in this repository and follows the dependency
   direction below.
3. Apply every relevant blocking checkpoint in this guide.
4. Inspect tests and production-facing evidence for the behavior changed.

Submit exactly one review:

- **APPROVE** when every applicable checkpoint passes and there are no material
  correctness, security, compatibility, architecture, or evidence gaps.
- **COMMENT** when a material gap remains. State the concrete consequence, the
  unmet checkpoint or acceptance criterion, and the smallest viable correction.
- Never use **REQUEST_CHANGES**. A human maintainer owns the blocking decision.

For changes that can affect agent or benchmark behavior—prompts, tool selection,
conversation payloads, terminal behavior, planning, memory, or evaluation
paths—state the eval risk in the review. Missing optional eval evidence is not a
material code finding: when the current head otherwise passes review, APPROVE so
the automation can request a human maintainer to choose the appropriate
lightweight evaluation. Use COMMENT when an acceptance criterion or required
check calls for specific eval evidence and that evidence is missing or failing,
or when available results show a regression.

Keep the code verdict separate from merge readiness. If a required check fails,
the branch is unmergeable, or required evidence is missing, use **COMMENT** and
state the gate even when the code looks sound. The summary, final verdict/footer,
and submitted review state must agree; never end an unresolved material finding
or merge gate with `APPROVED`. Optional evaluation follow-up remains nonblocking
as described above.

Include a compact checklist for every linked acceptance criterion. Meeting the
checklist is necessary but does not replace review for regressions, security, or
maintainability.

## Repository Ownership and Dependency Direction

| Repository                     | Owns                                                                                                            |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `OpenHands/OpenHands`          | Agent Canvas UI, frontend state, backend selection, frontend service integration, and local-stack orchestration |
| `OpenHands/software-agent-sdk` | Agent Server, SDK, canonical server API, and browser-compatible client in `clients/typescript/`                 |
| `OpenHands/extensions`         | Reusable skills, plugins, and integrations                                                                      |
| `OpenHands/automation`         | Scheduling, webhooks, run history, and automation dispatch                                                      |

The normal dependency direction is Agent Server contract → TypeScript client →
Canvas. Submit **COMMENT** for raw endpoint reimplementations, Canvas-local copies
of server contracts, or behavior implemented in the wrong repository.

## Blocking Checkpoints

### Agent Server and Cloud API access

Apply this checkpoint when a change calls or models an Agent Server, Cloud, or
runtime-sandbox API.

- `src/api/no-direct-agent-server-calls.test.ts` is the executable source of
  truth. Agent Server access must use `@openhands/typescript-client` with options
  from `src/api/agent-server-client-options.ts`.
- Cloud **App-API** requests must use `callCloudProxy` (now a direct browser
  call, since the SaaS permits CORS for API-key-authenticated requests). Per-
  conversation **runtime-sandbox** requests are not proxied: they must call the
  conversation's runtime URL directly via the typed client
  (`ConversationClient` / `BashClient` / `RemoteWorkspace` / `FileClient`, or a
  typed wrapper built via `getAgentServerHttpClientOptions`) with its session
  API key -- the same path local mode uses. Do not route runtime calls through
  `callCloudProxy` with `hostOverride`: the `/api/cloud-proxy` envelope it relied
  on was removed from the agent-server (software-agent-sdk #3326) and 405s on
  current backends.
- Treat changes to the guard's allowlist as architecture changes. Do not copy the
  allowlist into this guide.

Submit **COMMENT** if the PR adds raw `fetch`, `axios`, shared `openHands`, or
low-level HTTP client access to an Agent Server or cloud endpoint.

### Agent Server compatibility

Apply this checkpoint when Canvas begins relying on a new Agent Server endpoint,
field, schema, or behavior. Canvas and the Agent Server are independently
versioned.

Require an increase to `minimumAgentServer` to the first compatible released version.

Before demanding or removing a compatibility fallback, inspect the SDK producer
at the supported release floor and affected deployed versions. Show which payload
or behavior those versions can actually produce; an old fixture or hypothetical
legacy field alone does not establish a supported compatibility requirement.

Verify the compatibility boundary. Adding a TypeScript-client method does not
make older Agent Servers support it. Submit **COMMENT** if a supported backend
can reach the new code and fail because the required server behavior is absent.

### Affected Canvas modes

Apply this checkpoint when a change touches a shared adapter, conversation
builder, setting, state selector, or presentation helper.

Enumerate the affected consumers across these dimensions:

- Local and Cloud backends;
- OpenHands and ACP agents;
- standard, planning, and delegated conversations; and
- standalone and embedded Canvas.

Verify every affected path. Do not require the full Cartesian product when
control or data flow proves a dimension is isolated. Submit **COMMENT** when an
affected variant can take a distinct path but the implementation or evidence
covers only the default.

### Event wire contracts

Apply this checkpoint when a change reads, extends, or renders Agent Server
events.

The SDK event model is the wire authority, the TypeScript client mirrors it, and
Canvas consumes the published client type. A contract change must land in this
order:

1. SDK model/schema and serialization coverage.
2. TypeScript-client mirror derived from the SDK payload.
3. Published client release.
4. Canvas consumption and rendering or telemetry coverage.

Canvas-only presentation state belongs in a separate view model keyed by event
identity. Submit **COMMENT** for Canvas-local wire redeclarations, partial
intersections, module augmentation, or presentation fields added to wire types.

### Durable state and transitions

Apply this checkpoint when a change reads or writes a backend setting, profile,
conversation cache entry, or persisted browser value.

- Give the value one named owner and one obvious writer. Do not mirror an
  authoritative store into component-local state.
- When a stored shape or selection rule changes, verify existing-state hydration
  or migration as well as create, update, delete, default, and active-selection
  transitions that the feature supports.
- A destructive transition must leave a deterministic valid state or an explicit
  empty state that the UI handles.

Submit **COMMENT** if existing users can lose state, a delete or reset can leave
an invalid selection, or multiple writers can race or overwrite one another.

Telemetry has stricter named owners:

- `src/services/telemetry.ts` exclusively owns the Canvas PostHog client.
- React events use typed functions from `src/hooks/use-tracking.ts`.
- Consent rendering uses the telemetry consent external store;
  `setTelemetryConsent` is the only consent controller.
- A business milestone has one canonical capture.

### Docker user and permission workarounds

When a PR recommends a Docker `--user` or `HOME` workaround, validate the whole
supported execution path under that UID/GID: home directory, configuration and
cache directories, temporary files, and subprocess/browser startup. A writable
mounted persistence directory alone does not prove that the recommended mode
works. Check ownership and permissions outside the mount as well.

## Design Review

Prefer named hooks, services, stores, and feature modules over shared-root
branches or switches. Keep necessary exceptions in a narrow allowlist beside an
executable guard. Do not add layers that only rename or forward arguments, or
split a cohesive file because it is long.

`useEffect` is for synchronization with an external system, not derived render
data, user actions, lazy initialization, store mirroring, or ordering repairs.
Subscriptions, browser APIs, timers, and network synchronization remain valid
when cleanup and dependencies are explicit.

## Design Context for Deep PRs

A diff shows each changed line, not the design. Expect durable design context
when a reviewer cannot judge a PR from the diff in a couple of minutes, for
example:

- a new or changed Agent Server, Cloud, or automation integration, event
  contract, or persisted-state shape;
- a new feature module or subsystem, a cross-cutting refactor, or a migration;
- a behavior change in backend selection, conversation flow, or settings and
  profile transitions; or
- a large change whose intent cannot be reconstructed from the diff, even if no
  single hunk is complex.

Do not ask for design context on trivial, generated, or self-explanatory
changes: a typo, a one-line guard, a dependency bump, a copy or style tweak, or a
small localized fix. Size alone does not make a PR deep.

Adequate design context states:

- **Intent:** the problem and why this approach;
- **Before and after:** the important behavior, UI flow, or API shape on each
  side;
- **Compatibility and risk:** affected Canvas modes, stored state, and supported
  backends, and what can break; and
- **Code references:** links to the real code at a commit SHA.

Put it in the PR description, or link a `.pr/` design doc that covers it from
the `pr-design-doc` skill. Count a `.pr/` doc only when the link is pinned to a
commit SHA, not the branch name. Approving a same-repository PR runs the
`PR Artifacts` cleanup, which removes `.pr/` from the branch, so a branch link
stops resolving before a human maintainer reads it, while a SHA link keeps
working.

Scale the response to the risk assessment:

- **Deep and 🔴 HIGH risk without adequate context:** submit **COMMENT** and ask
  for the write-up or a SHA-pinned doc.
- **Deep and 🟡 MEDIUM risk:** ask for it when the change is hard to reconstruct
  from the diff; a small, self-evident change does not need it.
- **🟢 LOW risk:** never withhold approval for missing design context.

Design context is a review aid, not a merge gate by itself. It does not excuse a
correctness, security, compatibility, or architecture defect.

## Dependencies and Releases

- Direct dependencies are exact-pinned. Update `package.json` and
  `package-lock.json` together through npm.
- Treat dependency exemptions, git pins, and security overrides as policy
  changes. `__tests__/package-library.test.ts` is the executable source of truth.
- Scrutinize newly published third-party versions for supply-chain risk.
  First-party OpenHands packages are exempt from a waiting period, not from
  contract and release-order review.
- Package version changes belong in explicit release PRs and must match release
  workflow expectations.

## Testing and Production Evidence

- Choose evidence by changed behavior, not by a frontend filename alone.
  Nonvisual logic and test-only changes can use reproducible commands, actual
  logs/output, and tests exercising real code paths; docs-only changes need no
  runtime tests or media.
- Validate rendered behavior in the actual Canvas app. Token, class-name, or
  source-string assertions do not establish computed styling, isolation, layout,
  or readable controls; they cannot replace a required real-app capture.
- Require evidence proportional to the behavior changed. UI changes need a
  screenshot or video from the real app; CLI, API, and script changes need the
  exact runtime command and observed result.
- Authentic evidence comes from the real Agent Canvas app, browser, OS dialog,
  generated or downloaded artifact, or actual terminal/runtime output. It must
  include enough surrounding context and reproduction steps to establish what
  produced it. Mockups, Figma or design images, diagrams, manually recreated
  terminal output, synthetic before/after cards, and other illustrative graphics
  do not prove that the code ran.
- "Real app" evidence must exercise the production integration path with a real
  backend and, when the behavior depends on model execution, a real LLM. A
  mock-LLM or mocked backend run is E2E regression coverage, not live evidence.
  The artifact and description must identify the backend and model used so the
  reviewer can distinguish live evidence from a mock fixture.
- Runtime and user-visible bug fixes require the same production-facing setup
  before and after the change. The base or released version must reproduce the
  bug; the PR head must show the corrected behavior. If the claimed state cannot
  be produced through the real product, require the issue and PR to say so and
  describe the change as defensive handling rather than a reproduced production
  bug.
- Match the artifact type to the behavior. A screenshot can prove a static render
  state, but not duration, ordering, disappearance, refresh, navigation, or any
  other temporal behavior. Those changes require a video that visibly shows the
  trigger, the relevant transition, and the final corrected state. A still image
  of a supposedly stuck or stale state does not establish how long it persisted.
- Visual evidence must leave the changed behavior and relevant controls readable.
  Dismiss privacy, consent, onboarding, cookie, tooltip, and other overlays before
  capture; an obscured target is not evidence even if the underlying app is real.
- When unsure what genuine product evidence looks like, compare it with these
  official OpenHands documentation captures: [first-time setup](https://github.com/OpenHands/docs/blob/2409e927af05a40acc27190383a424615fc0f807/openhands/static/img/agent-canvas-setup-step-1.png),
  [LLM profiles manager](https://github.com/OpenHands/docs/blob/2409e927af05a40acc27190383a424615fc0f807/openhands/static/img/agent-canvas-llm-profiles-manager.png),
  and [`/model` interaction](https://github.com/OpenHands/docs/blob/2409e927af05a40acc27190383a424615fc0f807/openhands/static/img/model-command-agent-canvas.png).
- Lifecycle fixes must also verify resulting process or resource state, such as
  the parent exit code and remaining child services or listening ports.
- Tests must exercise real logic and observable state. A claimed regression test
  must reach the target behavior and fail when that behavior regresses; mocks
  that only prove another mock was called are insufficient.
- Do not duplicate library behavior or add brittle presentation-only snapshots.

Tests, mock-LLM runs, and mocked-backend runs are regression proof, not a
substitute for live evidence when the changed behavior requires it. Submit
**COMMENT** when production-facing evidence is required but absent or ambiguous,
and name the exact capture or verification still needed before approval.

Check the current PR validator as a separate merge gate:
`.github/scripts/check_pr_description.py` requires media for frontend-code PRs,
not for the bug label alone. Nonvisual bug fixes may provide reproduction commands
and observed before/after results in the required Summary and How to Test
sections. The checker verifies presence; reviewers must assess the substance.
The frontend path-based gate can still cover nonvisual frontend logic: report
that distinction rather than fabricating media or relabeling a PR. Do not apply
PR requirements to triage.

Follow the test routing in `AGENTS.md`. Mock-LLM, Docker mock-LLM, and live
LLM-backed E2E suites run after changes reach `main`, not from PR labels. For
risky pre-merge changes, recommend manually dispatching the relevant workflow
against the PR branch. Never broaden secret exposure for convenience.

## Final Context and Comment Discipline

Before submitting, compare the review summary and every finding with the current
PR title, head commit, changed-file manifest, linked issues, acceptance criteria,
and existing review threads. If a finding describes files or behavior outside
that context, stop and re-read the PR.

Do not comment on formatting handled by tooling, minor style, praise, optional
unrelated refactors, extra tests for straightforward data/config changes, or
temporary `.pr/` artifacts. Do not manufacture feedback to avoid approval.

For every finding, trace the call or data flow far enough to show the concrete
user-visible or architectural consequence. Prefer one root-cause comment over
several symptoms, and suggest the smallest viable correction.
