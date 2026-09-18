---
name: custom-codereview-guide
description: Repository-specific review rules for the OpenHands Agent Canvas frontend.
triggers:
  - /codereview
---

# OpenHands Agent Canvas Code Review Guidelines

This guide supplements the public `code-review` skill with rules specific to
`OpenHands/OpenHands`, the Agent Canvas frontend. Read `AGENTS.md` first; it is
the detailed source of truth for current architecture and test conventions.

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

Do not approve changes that can affect agent or benchmark behavior—prompts, tool
selection, conversation payloads, terminal behavior, planning, memory, or
evaluation paths—without human review and appropriate lightweight evals.

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
- Cloud and runtime-sandbox requests must use `callCloudProxy`; runtime requests
  must provide the correct `hostOverride` and authentication mode.
- Treat changes to the guard's allowlist as architecture changes. Do not copy the
  allowlist into this guide.

Submit **COMMENT** if the PR adds raw `fetch`, `axios`, shared `openHands`, or
low-level HTTP client access to an Agent Server endpoint.

### Agent Server compatibility

Apply this checkpoint when Canvas begins relying on a new Agent Server endpoint,
field, schema, or behavior. Canvas and the Agent Server are independently
versioned.

Require an increase to `minimumAgentServer` to the first compatible released version.

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

## Design Review

Prefer named hooks, services, stores, and feature modules over shared-root
branches or switches. Keep necessary exceptions in a narrow allowlist beside an
executable guard. Do not add layers that only rename or forward arguments, or
split a cohesive file because it is long.

`useEffect` is for synchronization with an external system, not derived render
data, user actions, lazy initialization, store mirroring, or ordering repairs.
Subscriptions, browser APIs, timers, and network synchronization remain valid
when cleanup and dependencies are explicit.

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

- Require evidence proportional to the behavior changed. UI changes need a
  screenshot or video from the real app; CLI, API, and script changes need the
  exact runtime command and observed result.
- Runtime and user-visible bug fixes require the same production-facing setup
  before and after the change. The base or released version must reproduce the
  bug; the PR head must show the corrected behavior.
- Lifecycle fixes must also verify resulting process or resource state, such as
  the parent exit code and remaining child services or listening ports.
- Tests must exercise real logic and observable state. A claimed regression test
  must reach the target behavior and fail when that behavior regresses; mocks
  that only prove another mock was called are insufficient.
- Do not duplicate library behavior or add brittle presentation-only snapshots.

Tests are regression proof, not a substitute for required live evidence. Submit
**COMMENT** when production-facing evidence is required but absent, and name the
exact verification still needed.

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
