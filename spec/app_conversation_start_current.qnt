// -*- mode: Bluespec; -*-

/// Analysis model for two failure paths in
/// LiveStatusAppConversationService._start_app_conversation().
///
/// These transitions intentionally reproduce current implementation ordering.
/// Their deterministic tests are counterexample witnesses, not desired behavior.
module app_conversation_start_current {
  import app_conversation_start.* from "./app_conversation_start"

  /// ACST/F1 — READY is yielded at lines 604–607, pending messages are then
  /// processed inside the same try block, and the broad catch at lines 618–622
  /// rewrites the task to ERROR if that ancillary work raises.
  pure def purePendingFailureLikeCurrent(
    state: State,
    id: TaskId
  ): TransitionResult =
    val task = state.tasks.get(id)
    if (task.status != Ready or task.pendingOutcome != PendingQueued) {
      unchanged(state)
    } else {
      withTask(state, id, {
        ...task,
        status: Error,
        errorPublished: true,
        detailSet: true,
        backgroundAlive: false,
        pendingOutcome: PendingFailed,
      })
    }

  /// ACST/F2 — The Agent Server conversation is created before app metadata,
  /// callbacks, and the READY task link are committed.
  pure def pureCreateAgentConversationLikeCurrent(
    state: State,
    id: TaskId
  ): TransitionResult =
    val task = state.tasks.get(id)
    if (task.status != StartingConversation or task.agentConversationCreated) {
      unchanged(state)
    } else {
      withTask(state, id, { ...task, agentConversationCreated: true })
    }

  pure def pureSaveMetadataLikeCurrent(
    state: State,
    id: TaskId
  ): TransitionResult =
    val task = state.tasks.get(id)
    if (
      task.status != StartingConversation
      or not(task.agentConversationCreated)
      or task.metadataSaved
    ) {
      unchanged(state)
    } else {
      withTask(state, id, { ...task, metadataSaved: true })
    }

  /// A callback save failure reaches the broad catch without deleting the
  /// already-created Agent Server conversation or its app metadata.
  pure def pureCallbackFailureLikeCurrent(
    state: State,
    id: TaskId
  ): TransitionResult =
    val task = state.tasks.get(id)
    if (
      task.status != StartingConversation
      or not(task.agentConversationCreated)
      or not(task.metadataSaved)
      or task.callbacksSaved
    ) {
      unchanged(state)
    } else {
      withTask(state, id, {
        ...task,
        status: Error,
        errorPublished: true,
        detailSet: true,
        backgroundAlive: false,
      })
    }

  action pendingFailureLikeCurrentAction(id: TaskId): bool =
    applyTransition(purePendingFailureLikeCurrent(currentState, id))

  action createAgentConversationLikeCurrentAction(id: TaskId): bool =
    applyTransition(pureCreateAgentConversationLikeCurrent(currentState, id))

  action saveMetadataLikeCurrentAction(id: TaskId): bool =
    applyTransition(pureSaveMetadataLikeCurrent(currentState, id))

  action callbackFailureLikeCurrentAction(id: TaskId): bool =
    applyTransition(pureCallbackFailureLikeCurrent(currentState, id))

  action currentStep = any {
    step,

    nondet id = TASK_IDS.oneOf()
    any {
      pendingFailureLikeCurrentAction(id),
      createAgentConversationLikeCurrentAction(id),
      saveMetadataLikeCurrentAction(id),
      callbackFailureLikeCurrentAction(id),
    }
  }
}
