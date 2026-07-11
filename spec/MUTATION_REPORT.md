# Quint scenario mutation report

Date: 2026-07-11
Quint: `@informalsystems/quint@0.32.0`

The repository does not currently provide an automated Quint mutation-testing
tool. Five targeted semantic mutants were applied to temporary copies of
`app_conversation_start.qnt`, and the desired-contract scenario suite was run
against each copy.

| Mutant | Behavioral change                                          | Killing scenario                         | Result |
| ------ | ---------------------------------------------------------- | ---------------------------------------- | ------ |
| M1     | Sandbox assignment skips `WAITING_FOR_SANDBOX`             | `orderedStartReachesCompleteReadyTest`   | Killed |
| M2     | `READY` is published without a task conversation link      | `orderedStartReachesCompleteReadyTest`   | Killed |
| M3     | Pending-message failure changes the task to `ERROR`        | `pendingFailureKeepsReadyTerminalTest`   | Killed |
| M4     | The ordinary failure transition remains enabled at `READY` | `readyRejectsStartFailureTransitionTest` | Killed |
| M5     | A transition writes the same task state to both task IDs   | `taskTransitionsAreIsolatedTest`         | Killed |

Mutation score: **5/5 killed (100%)**.

This is focused semantic mutation analysis, not an exhaustive mutation pass.
The selected mutations cover ordered publication, readiness completeness,
terminal absorption, ancillary failure handling, and cross-task isolation.
