# Spec Changelog

This is the research log for the executable Quint specifications. It records
pending implementation discrepancies first, followed by dated model changes and
fixes verified by the specifications. Product release notes continue to live in
GitHub Releases.

When a pending item is resolved, remove it from `Pending` and add a new dated
entry above the older entries. Link the implementation change and the
Quint/Python evidence that verifies the resolution.

## Pending

### ACST/F1 — Pending-message failure can rewrite `READY` to `ERROR`

The service contract tells callers to stop iterating at `READY` or `ERROR`.
The concrete generator publishes and persists `READY`, then processes pending
messages inside the same surrounding `try`. If re-keying, loading, or deleting
pending messages raises, the broad catch changes the already-linked task to
`ERROR` and publishes it again:

```text
... → STARTING_CONVERSATION → READY
READY → pending-message database failure → ERROR
```

Executable witness:
[`readyThenPendingFailureCanPublishErrorTest`](./app_conversation_start_current_tests.qnt).

Implementation source:
`LiveStatusAppConversationService._start_app_conversation()` in
`openhands/app_server/app_conversation/live_status_app_conversation_service.py`.

Resolution requires pending-message failures to leave the start task `READY`,
a Python regression test that consumes beyond the `READY` yield, and the
current-code witness becoming unreachable.

### ACST/F2 — Callback failure can leave an unlinked live conversation

Agent Server conversation creation and app metadata persistence occur before
event callbacks are saved and before `task.app_conversation_id` is populated.
If a callback save raises, the catch publishes `ERROR` without compensating for
the already-created conversation or metadata:

```text
STARTING_CONVERSATION
→ Agent Server conversation created
→ app metadata saved
→ callback save fails
→ ERROR with no task conversation link
```

Executable witness:
[`callbackFailureCanLeaveOrphanedConversationTest`](./app_conversation_start_current_tests.qnt).

Implementation source:
`LiveStatusAppConversationService._start_app_conversation()` in
`openhands/app_server/app_conversation/live_status_app_conversation_service.py`.

Resolution requires atomic linkage or compensating cleanup, a Python
failure-injection regression test, and the current-code witness becoming
unreachable.

### Quint milestones 2–4 not yet modeled

The following behaviors are tracked in [`QUINT_PLAN.md`](./QUINT_PLAN.md) but do
not yet have executable coverage:

- Detached-worker durability and nonterminal-task recovery
- Pending-message retry/deletion policy
- Remote sandbox session-key lifecycle and secret authorization

---

## 2026-07-11 — App-conversation start model added; two discrepancies found

Added the bounded desired lifecycle, six deterministic contract scenarios, two
current-code counterexample witnesses, the `allSafetyProperties` simulation
invariant, and targeted [mutation evidence](./MUTATION_REPORT.md). The desired
scenarios passed while the current-code analysis exposed ACST/F1 and ACST/F2.
No application fix is verified by this entry.
