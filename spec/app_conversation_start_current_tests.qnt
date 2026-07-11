// -*- mode: Bluespec; -*-

module app_conversation_start_current_tests {
  import app_conversation_start.* from "./app_conversation_start"
  import app_conversation_start_current.* from "./app_conversation_start_current"

  /// ACST/F1 — The implementation yields READY before pending-message work,
  /// then the outer catch can publish ERROR if that ancillary work raises.
  run readyThenPendingFailureCanPublishErrorTest =
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
      .then(pendingFailureLikeCurrentAction(TASK_A))
      .expect(and {
        taskA.status == Error,
        taskA.readyPublished,
        taskA.errorPublished,
        taskA.conversationLinked,
        taskA.pendingOutcome == PendingFailed,
        not(terminalPublicationIsStable(taskA)),
        not(linkedConversationIsReady(taskA)),
        not(allSafetyProperties),
      })

  /// ACST/F2 — Agent Server creation and metadata persistence happen before
  /// callback persistence and before the task is linked at READY. A callback
  /// failure reaches ERROR without compensation for those artifacts.
  run callbackFailureCanLeaveOrphanedConversationTest =
    init
      .then(submitTaskAction(TASK_A))
      .then(assignSandboxAction(TASK_A))
      .then(sandboxReadyAction(TASK_A))
      .then(repositoryPreparedAction(TASK_A))
      .then(setupScriptFinishedAction(TASK_A))
      .then(gitHooksFinishedAction(TASK_A))
      .then(skillsFinishedAction(TASK_A))
      .then(createAgentConversationLikeCurrentAction(TASK_A))
      .then(saveMetadataLikeCurrentAction(TASK_A))
      .then(callbackFailureLikeCurrentAction(TASK_A))
      .expect(and {
        taskA.status == Error,
        taskA.agentConversationCreated,
        taskA.metadataSaved,
        not(taskA.callbacksSaved),
        not(taskA.conversationLinked),
        taskA.errorPublished,
        not(noOrphanedConversationArtifacts(taskA)),
        not(errorIsClean(taskA)),
        not(allSafetyProperties),
      })
}
