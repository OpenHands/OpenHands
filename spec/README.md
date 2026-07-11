# Formal Specification

This directory contains executable formal models written in
[Quint](https://quint-lang.org/). The first model covers the V1 App Server's
asynchronous app-conversation start task, including its post-`READY`
pending-message tail.

The Agent and Agent Server have moved to
[`OpenHands/software-agent-sdk`](https://github.com/OpenHands/software-agent-sdk),
and Agent Canvas has moved to
[`OpenHands/agent-canvas`](https://github.com/OpenHands/agent-canvas). This model
deliberately stops at this repository's App Server orchestration boundary and
abstracts the external sandbox and Agent Server responses.

## Files

| File                                       | Purpose                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------ |
| `app_conversation_start.qnt`               | Desired start-task state machine, actions, and aggregate safety property |
| `app_conversation_start_tests.qnt`         | Six deterministic desired-contract scenarios                             |
| `app_conversation_start_current.qnt`       | Analysis model for two current implementation failure paths              |
| `app_conversation_start_current_tests.qnt` | Two deterministic counterexample witnesses                               |
| `QUINT_PLAN.md`                            | Scope, bounds, and proposed next milestones                              |
| `CHANGELOG.md`                             | Pending discrepancies and newest-first spec history                      |
| `MUTATION_REPORT.md`                       | Targeted semantic-mutation evidence                                      |

The desired model and current-code analysis are separate on purpose. A passing
current-code witness means the undesirable state is reachable in the analysis;
it does not mean the desired contract is satisfied.

## Modeled contract

The documented status progression is:

```text
WORKING → WAITING_FOR_SANDBOX → PREPARING_REPOSITORY
→ RUNNING_SETUP_SCRIPT → SETTING_UP_GIT_HOOKS → SETTING_UP_SKILLS
→ STARTING_CONVERSATION → READY
```

`ERROR` may be published from any nonterminal phase.

- **ACST-001 — Ordered, complete readiness.** `READY` is published only after a
  running sandbox, Agent Server URL, live conversation, app metadata, callbacks,
  and the task's conversation link all exist.
- **ACST-002 — Clean failure.** A failure before `READY` publishes `ERROR` with
  diagnostic detail and leaves no unlinked conversation artifacts.
- **ACST-003 — Ancillary pending work.** Pending-message delivery may succeed or
  fail after `READY`, but cannot rewrite the terminal start result.
- **ACST-004 — Terminal absorption.** Once `READY` or `ERROR` is published, the
  start-task status cannot change.
- **ACST-005 — Task isolation.** A transition for one detached start task cannot
  mutate another task.

The finite model uses two task identities. That is sufficient to explore every
per-task phase and cross-task isolation without modeling UUID contents, SQL
rows, HTTP payloads, repository data, or agent behavior.

## Findings

The current-code model records two open discrepancies in
[`CHANGELOG.md`](./CHANGELOG.md):

1. `READY` is yielded before pending-message processing. An exception in that
   ancillary tail reaches the surrounding catch and publishes `ERROR` after
   callers were told that `READY` is terminal.
2. Agent Server conversation creation and app metadata persistence occur before
   callback persistence and the task's `app_conversation_id` link. A callback
   failure can therefore publish `ERROR` while leaving unlinked artifacts.

These are formal-analysis findings. This change does not modify application
behavior or claim that either path has been observed in production.

## Authoring references

- [Ray Myers's Quint authoring prompt](https://gist.github.com/raymyers/7066fb7ebef80df48d48516f3314d663): state records, pure transitions, thin actions, bounded identities, and deterministic scenarios.
- [OpenHands Runtime API PR #487](https://github.com/OpenHands/runtime-api/pull/487): spec-local README, plan, changelog, witnesses, and CI simulation.

The suite pins `@informalsystems/quint` **0.32.0**, type-checks every `.qnt`
file, and runs every filename-matched `*_tests.qnt` module in CI.

## Run locally

Type-check every executable file:

```bash
for spec_file in spec/*.qnt; do
  npx --yes @informalsystems/quint@0.32.0 typecheck "$spec_file"
done
```

Run all deterministic scenarios and current-code witnesses:

```bash
for test_file in spec/*_tests.qnt; do
  main_module="${test_file##*/}"
  main_module="${main_module%.qnt}"
  npx --yes @informalsystems/quint@0.32.0 test "$test_file" \
    --main="$main_module" --match='.*Test'
done
```

Explore randomized desired-model traces:

```bash
npx --yes @informalsystems/quint@0.32.0 run \
  spec/app_conversation_start.qnt \
  --main=app_conversation_start \
  --invariant=allSafetyProperties \
  --max-steps=30 --max-samples=1000
```

Ask the current-code model to find a counterexample:

```bash
npx --yes @informalsystems/quint@0.32.0 run \
  spec/app_conversation_start_current.qnt \
  --main=app_conversation_start_current \
  --init=init --step=currentStep \
  --invariant=allSafetyProperties \
  --max-steps=30 --max-samples=1000
# Expected: "Invariant violated" — an undesirable path is reachable.
```

Run bounded verification with Apalache when Java is available:

```bash
npx --yes @informalsystems/quint@0.32.0 verify \
  spec/app_conversation_start.qnt \
  --main=app_conversation_start \
  --invariant=allSafetyProperties \
  --max-steps=12
```

Random simulation can find counterexamples but is not a proof. Bounded
`verify` exhaustively explores this finite model only up to the supplied step
bound.

## Implementation correspondence

- Public terminal/status contract:
  `openhands/app_server/app_conversation/app_conversation_service.py`
- Status enum and task fields:
  `openhands/app_server/app_conversation/app_conversation_models.py`
- HTTP handoff to the detached consumer:
  `openhands/app_server/app_conversation/app_conversation_router.py`
- Setup phase ordering:
  `openhands/app_server/app_conversation/app_conversation_service_base.py`
- Concrete generator, artifact ordering, and pending-message tail:
  `openhands/app_server/app_conversation/live_status_app_conversation_service.py`
- Per-yield persistence:
  `openhands/app_server/app_conversation/sql_app_conversation_start_task_service.py`
- Existing behavioral tests:
  `tests/unit/app_server/test_live_status_app_conversation_service.py` and
  `tests/unit/app_server/test_sql_app_conversation_start_task_service.py`

Quint abstracts external service payloads, timing distributions, process
scheduling, database implementation details, repository contents, LLM behavior,
and pending-message delivery policy. Those remain the responsibility of Python
unit, integration, and end-to-end tests.
