// -*- mode: Bluespec; -*-

module app_conversation_start_tests {
  import app_conversation_start.* from "./app_conversation_start"

  /// ACST-001 — The published lifecycle follows the documented status order
  /// and READY contains every identifier and artifact callers rely on.
  run orderedStartReachesCompleteReadyTest =
    init
      .then(submitTaskAction(TASK_A))
      .expect(taskA.status == Working)
      .then(assignSandboxAction(TASK_A))
      .expect(taskA.status == WaitingForSandbox)
      .then(sandboxReadyAction(TASK_A))
      .expect(taskA.status == PreparingRepository)
      .then(repositoryPreparedAction(TASK_A))
      .expect(taskA.status == RunningSetupScript)
      .then(setupScriptFinishedAction(TASK_A))
      .expect(taskA.status == SettingUpGitHooks)
      .then(gitHooksFinishedAction(TASK_A))
      .expect(taskA.status == SettingUpSkills)
      .then(skillsFinishedAction(TASK_A))
      .expect(taskA.status == StartingConversation)
      .then(completeConversationStartAction(TASK_A))
      .expect(and {
        taskA.status == Ready,
        taskA.sandboxAssigned,
        taskA.sandboxRunning,
        taskA.agentUrlSet,
        taskA.agentConversationCreated,
        taskA.metadataSaved,
        taskA.callbacksSaved,
        taskA.conversationLinked,
        taskA.readyPublished,
        not(taskA.errorPublished),
        allSafetyProperties,
      })

  /// ACST-002 — An exception before READY publishes one clean ERROR result;
  /// no live conversation or metadata is left without a task link.
  run preterminalFailurePublishesCleanErrorTest =
    init
      .then(submitTaskAction(TASK_A))
      .then(assignSandboxAction(TASK_A))
      .then(sandboxReadyAction(TASK_A))
      .then(repositoryPreparedAction(TASK_A))
      .then(setupScriptFinishedAction(TASK_A))
      .then(gitHooksFinishedAction(TASK_A))
      .then(skillsFinishedAction(TASK_A))
      .then(failStartAction(TASK_A))
      .expect(and {
        taskA.status == Error,
        taskA.detailSet,
        taskA.errorPublished,
        not(taskA.readyPublished),
        not(taskA.backgroundAlive),
        not(taskA.agentConversationCreated),
        not(taskA.metadataSaved),
        not(taskA.callbacksSaved),
        not(taskA.conversationLinked),
        allSafetyProperties,
      })

  /// ACST-003 — Pending-message work is ancillary. A failure after READY may
  /// be recorded, but it cannot rewrite the already-published terminal status.
  run pendingFailureKeepsReadyTerminalTest =
    init
      .then(submitTaskAction(TASK_A))
      .then(assignSandboxAction(TASK_A))
      .then(sandboxReadyAction(TASK_A))
      .then(repositoryPreparedAction(TASK_A))
      .then(setupScriptFinishedAction(TASK_A))
      .then(gitHooksFinishedAction(TASK_A))
      .then(skillsFinishedAction(TASK_A))
      .then(completeConversationStartAction(TASK_A))
      .then(queuePendingMessagesAction(TASK_A))
      .then(failPendingMessagesAction(TASK_A))
      .expect(and {
        taskA.status == Ready,
        taskA.pendingOutcome == PendingFailed,
        taskA.readyPublished,
        not(taskA.errorPublished),
        not(taskA.backgroundAlive),
        allSafetyProperties,
      })

  /// ACST-004 — READY is absorbing for the start-task status machine.
  run readyRejectsStartFailureTransitionTest =
    init
      .then(submitTaskAction(TASK_A))
      .then(assignSandboxAction(TASK_A))
      .then(sandboxReadyAction(TASK_A))
      .then(repositoryPreparedAction(TASK_A))
      .then(setupScriptFinishedAction(TASK_A))
      .then(gitHooksFinishedAction(TASK_A))
      .then(skillsFinishedAction(TASK_A))
      .then(completeConversationStartAction(TASK_A))
      .then(failStartAction(TASK_A))
      .fail()

  /// ACST-004 — ERROR is also absorbing.
  run errorRejectsFurtherProgressTest =
    init
      .then(submitTaskAction(TASK_A))
      .then(assignSandboxAction(TASK_A))
      .then(failStartAction(TASK_A))
      .then(sandboxReadyAction(TASK_A))
      .fail()

  /// ACST-005 — Updating one detached start task cannot mutate another task.
  run taskTransitionsAreIsolatedTest =
    init
      .then(submitTaskAction(TASK_A))
      .then(submitTaskAction(TASK_B))
      .then(assignSandboxAction(TASK_A))
      .expect(and {
        taskA.status == WaitingForSandbox,
        taskA.sandboxAssigned,
        taskB.status == Working,
        not(taskB.sandboxAssigned),
        taskB.backgroundAlive,
        allSafetyProperties,
      })
}
