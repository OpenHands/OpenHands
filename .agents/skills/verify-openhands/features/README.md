# OpenHands feature verification map

Use this index before driving the app. This is a maintained map of user behavior,
not a claim that every row has passed. Seeded from main `a07364828c8f202e7745c6bce3dcef3915ae7ac1`
and the September 2026 live audit; recheck source and runtime at each target SHA.

## Baseline and driving contract

Follow [verify-openhands](../SKILL.md) and its [runtime recipe](../references/runtime.md).
Use disposable backend state, workspace and browser contexts. A real authenticated
stack is required; only LLM/integration checks require their additional credentials.
Run doctor before the first action and after unexpected failures. Keep evidence
outside cleanup targets and outside served directories.

In recipe snippets, `page` is a Playwright page authenticated to this run and `base`
is its recorded UI URL. `getByTestId('x')` means the current `data-testid="x"`
handle. Scope repeated handles to the named form/row/dialog. Never copy route
interception from a mock-LLM spec into a live proof. UI navigation and direct-link
navigation are distinct entry points; record which was exercised.

Every feature file contains four sections: sub-features with stable IDs, user
entry points, a precondition/action/observable-result recipe, and gotchas. Source
anchors locate the implementation for drift checks; they are not a replacement
for using the feature. Keep IDs stable when labels/routes move.

## Feature families

| ID | User-facing family | Routes / non-route entry points | Preconditions beyond the local stack |
|---|---|---|---|
| F01 | [Entry, onboarding and backends](entry-and-backends.md) | `/`, `/conversations`; command menu, workspace picker, backend selector | Second owned backend for switching; authorized provider for onboarding completion |
| F02 | [Conversations](conversations.md) | `/conversations/:conversationId`; sidebar, composer, conversation menus | Real authorized LLM or ACP; disposable workspace |
| F03 | [Workspace and panels](workspace-and-panels.md) | Conversation Files/Changes/Terminal/Browser/Planner/Tasks/Usage; `/conversations/:conversationId/panel` | Real file/tool events; optional VS Code/browser tool |
| F04 | [LLM and agent profiles](profiles.md) | `/settings/llm`, `/settings/agent`, `/settings/agents`; composer selectors | Authorized provider/ACP for saved/executing profiles |
| F05 | [Settings and secrets](settings-and-secrets.md) | `/settings`, `/settings/app`, `/settings/condenser`, `/settings/agent-context`, `/settings/verification`, `/settings/secrets` | Dummy values suffice for safe persistence checks |
| F06 | [MCP integrations](mcp.md) | `/mcp`; catalog, install/custom editor, connection test | Approved local package for Time; test accounts for external integrations |
| F07 | [Skills and plugins](skills-and-plugins.md) | `/customize`, `/skills`, `/plugins`, `/launch`; conversation Skills/Hooks/Tools | Reviewed fixture plugin/skill; LLM for execution |
| F08 | [Canvas apps](canvas-apps.md) | `/apps`, `/extensions/:extensionName/*` | Backend supporting extensions; reviewed bundled fixture |
| F09 | [Automations](automations.md) | `/automations`, `/automations/templates`, `/automations/new/:automationId`, `/automations/:automationId`; pinned home cards | Real automation service; authorized bounded dispatch for execution |
| F10 | [Automation Git Sync](git-sync.md) | `/automations/git-sync` | Explicitly approved disposable remote, branch and credentials |
| F11 | [Cloud and sharing](cloud-and-sharing.md) | Cloud backend/org selector, `/oauth/device/verify`, `/shared/conversations/:conversationId` | Authorized test account, organization, sandbox and sharing entitlement |
| F12 | [Runtime and embedding variants](runtime-and-embedding.md) | CLI/npm, Docker, Electron; provider/root library embedding | Platform/install-specific runner; separate host app for library checks |

This covers the registered route families in [src/routes.ts](../../../../src/routes.ts)
plus non-route interactions. Sweep that registry, menus, exported library surface
and merged changes for additions; the map is deliberately not a generated file list.
Use [the testing matrix](../../../../docs/TESTING_MATRIX.md) for the installer × OS ×
agent matrix, not this web pass as a substitute for native certification.

## Evidence and coverage

Use [the report contract](../references/report.md). Record every required check and
entry point as pass/fail/blocked/not-run, with artifacts and actual prerequisites.
One smoke check does not pass a family; one local backend does not pass Cloud.
Include family and check denominators. Record excluded platform variants explicitly.

Historical repro candidates: [#17562](https://github.com/OpenHands/OpenHands/issues/17562)
(dialog width), [#17563](https://github.com/OpenHands/OpenHands/issues/17563) (edit height),
[#17564](https://github.com/OpenHands/OpenHands/issues/17564) (dashboard header),
[#17565](https://github.com/OpenHands/OpenHands/issues/17565) (plugin search),
[#17566](https://github.com/OpenHands/OpenHands/issues/17566) (terminal reload), and
[#17567](https://github.com/OpenHands/OpenHands/issues/17567) (diff/Planner lifecycle).
Check current issue status and reproduce; these are not permanent pass exemptions.
