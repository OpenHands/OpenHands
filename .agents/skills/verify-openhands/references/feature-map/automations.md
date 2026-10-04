# F09 — Automation discovery, configuration and execution

Configure a task and follow a real dispatch through its conversation and activity.
Source: [automation routes](../../../../src/routes.ts),
[automation UI](../../../../src/components/features/automations),
[existing lifecycle recipe](../../../../tests/e2e/mock-llm/automations/mock-llm-automation.spec.ts).

## Sub-features

- `F09.dashboard`: grid/list, filters/empty state, action header and pinned home cards.
- `F09.templates`: browse template, configure prerequisites, create/import/help states.
- `F09.configuration`: detail/edit/delete confirmation and persistence.
- `F09.run`: manual dispatch, status transitions, completed output and conversation link.
- `F09.activity`: populated activity/logs; bounded failure/cancellation where supported.
- `F09.schedule`: schedule/webhook enablement only in an explicitly authorized sandbox.

## How to get to it (user POV)

Open Automate (`/automations`), templates (`/automations/templates`), template setup
(`/automations/new/:automationId`) and an existing detail page
(`/automations/:automationId`). Also follow an actual pinned home automation card.

## Driving it with Playwright

Preconditions: healthy real automation service. Use a dedicated disabled QA record;
manual execution additionally needs an authorized model and bounded harmless task.
External-notification templates and schedules require separate permission.

- **Dashboard:** inspect grid/list and real filters; try no-match and clear states.
  At 390px check heading/description remain readable beside actions and controls.
- **Create/configure:** use a reviewed no-external-side-effect template; follow its
  setup and actual creation flow. Verify the resulting detail/configuration after
  reload. Record any helper conversation/model prerequisite rather than bypassing
  it through an API and claiming UI creation passed.
- **Edit:** open Edit on the QA record, change a harmless field, save and reload;
  inspect header/close/footer and internal scrolling at 390×844. Cancel a deletion
  before confirming cleanup after the run.
- **Dispatch:** with permission, manually run the bounded QA task. Verify real status
  progression to `run-status-icon-completed`, actual output/side effect, populated
  activity and the linked conversation. Inspect cancellation/error feedback with a
  safe fixture, without enabling recurring dispatch or posting to external services.
- **Cleanup:** keep schedules disabled, remove the run-owned definition when safe,
  and check no scheduled or in-flight fixture remains.

## Gotchas

An empty activity screen is not proof of run history. A definition visible in the
UI is not successful dispatch; UI success must agree with actual run status/output.
Recheck [#17563](https://github.com/OpenHands/OpenHands/issues/17563) and
[#17564](https://github.com/OpenHands/OpenHands/issues/17564). Scheduling/dispatch
belongs to `OpenHands/automation`; execution belongs to the Agent Server/SDK.
