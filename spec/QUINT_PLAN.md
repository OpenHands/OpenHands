# Quint specification plan: OpenHands App Server

## System boundary

The App Server coordinates slow, asynchronous app-conversation startup across
its database, sandbox service, and an external Agent Server. The formal target
is the App Server's published task protocol and artifact ownership—not the
external Agent, runtime implementation, React UI, or network payload formats.

## Milestone 1 — App-conversation start lifecycle (implemented)

“Implemented” means the desired and current-code models are executable, not
that the application satisfies every modeled property. Current discrepancies
are tracked in [`CHANGELOG.md`](./CHANGELOG.md).

- [x] Model all documented start-task statuses and ordered success progression.
- [x] Allow `ERROR` from every nonterminal phase.
- [x] Require `READY` to include sandbox readiness, Agent Server URL, created
      conversation, metadata, callbacks, and the task link.
- [x] Keep `READY` and `ERROR` terminal after publication.
- [x] Treat post-`READY` pending-message work as ancillary.
- [x] Require failure compensation for unlinked conversation artifacts.
- [x] Explore two task identities to check transition isolation.
- [x] Encode the current `READY → ERROR` and orphan-artifact witnesses
      separately.
- [x] Check desired invariants through randomized simulation.

Finite bounds:

- Tasks: `task-a`, `task-b`
- Statuses: one internal `NotSubmitted` state plus the nine public statuses
- Pending outcomes: none, queued, delivered, failed
- Artifacts: Boolean existence/linkage abstractions

These bounds cover every relevant lifecycle phase, both terminal results, the
post-ready tail, partial artifact creation, and cross-task isolation.

## Milestone 2 — Detached-worker durability and recovery (proposed)

- [ ] Model the HTTP route returning after the first `WORKING` yield.
- [ ] Distinguish a live in-process consumer from a durable worker lease.
- [ ] Model process death and database-save failure between yields.
- [ ] Define whether nonterminal tasks must be reconciled to `ERROR` or resumed.
- [ ] Add a liveness property under explicit fairness/recovery assumptions.

Relevant implementation:

- `openhands/app_server/app_conversation/app_conversation_router.py`
- `openhands/app_server/app_conversation/live_status_app_conversation_service.py`
- `openhands/app_server/app_conversation/sql_app_conversation_start_task_service.py`

## Milestone 3 — Pending-message delivery semantics (proposed)

- [ ] Decide whether pending delivery is at-most-once or at-least-once.
- [ ] Model ordered delivery of multiple messages.
- [ ] Model per-message HTTP failure and database deletion failure.
- [ ] Require the implementation, tests, and docstring to agree on retry/loss
      behavior before adding a reliability invariant.

Relevant implementation:

- `LiveStatusAppConversationService._process_pending_messages()`
- `tests/unit/app_server/test_pending_message_service.py`
- `tests/unit/app_server/test_pending_message_router.py`

## Milestone 4 — Remote sandbox session capabilities (proposed)

- [ ] Model session-key rotation across start, pause, resume, and delete.
- [ ] Require all non-running sandbox statuses to reject secret access.
- [ ] Compose sandbox ownership, path binding, and user dual-auth checks.
- [ ] Model rollback-resistant key revocation on transient delete failure.

Relevant implementation:

- `openhands/app_server/sandbox/remote_sandbox_service.py`
- `openhands/app_server/sandbox/session_auth.py`
- `openhands/app_server/sandbox/sandbox_router.py`
- `openhands/app_server/user/user_router.py`

## Scope exclusions

- Agent and Agent Server internals owned by `OpenHands/software-agent-sdk`
- Agent Canvas behavior owned by `OpenHands/agent-canvas`
- React rendering, accessibility, and client polling implementation
- SQLAlchemy transaction mechanics and concrete UUID serialization
- Repository cloning contents, setup-script commands, and skill contents
- LLM behavior, tool execution, and sandbox resource management
- Probability and latency distributions

Those concerns are either owned elsewhere or better covered by unit,
integration, and end-to-end tests.
