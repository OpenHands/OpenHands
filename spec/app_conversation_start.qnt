// -*- mode: Bluespec; -*-

/// Desired contract for the V1 app-conversation start-task lifecycle.
///
/// The model follows the async generator documented by
/// AppConversationService.start_app_conversation(). Two task identities are
/// sufficient to explore lifecycle ordering and cross-task isolation. External
/// sandbox, database, and Agent Server calls are abstracted to successful phase
/// transitions or a failure transition.
module app_conversation_start {
  // ───────────────────────────────────────────────────────────────────────────
  // 1. TYPES
  // ───────────────────────────────────────────────────────────────────────────

  type TaskId = str

  type StartStatus =
    | NotSubmitted
    | Working
    | WaitingForSandbox
    | PreparingRepository
    | RunningSetupScript
    | SettingUpGitHooks
    | SettingUpSkills
    | StartingConversation
    | Ready
    | Error

  type PendingOutcome =
    | NoPendingWork
    | PendingQueued
    | PendingDelivered
    | PendingFailed

  type StartTask = {
    status: StartStatus,
    sandboxAssigned: bool,
    sandboxRunning: bool,
    agentUrlSet: bool,
    agentConversationCreated: bool,
    metadataSaved: bool,
    callbacksSaved: bool,
    conversationLinked: bool,
    readyPublished: bool,
    errorPublished: bool,
    detailSet: bool,
    backgroundAlive: bool,
    pendingOutcome: PendingOutcome,
  }

  type State = {
    tasks: TaskId -> StartTask,
  }

  type TransitionResult = {
    success: bool,
    newState: State,
  }

  // ───────────────────────────────────────────────────────────────────────────
  // 2. CONSTANTS
  // ───────────────────────────────────────────────────────────────────────────

  pure val TASK_A: TaskId = "task-a"
  pure val TASK_B: TaskId = "task-b"
  pure val TASK_IDS: Set[TaskId] = Set(TASK_A, TASK_B)

  pure val EMPTY_TASK: StartTask = {
    status: NotSubmitted,
    sandboxAssigned: false,
    sandboxRunning: false,
    agentUrlSet: false,
    agentConversationCreated: false,
    metadataSaved: false,
    callbacksSaved: false,
    conversationLinked: false,
    readyPublished: false,
    errorPublished: false,
    detailSet: false,
    backgroundAlive: false,
    pendingOutcome: NoPendingWork,
  }

  pure val INITIAL_TASKS: TaskId -> StartTask = Map(
    TASK_A -> EMPTY_TASK,
    TASK_B -> EMPTY_TASK,
  )

  // ───────────────────────────────────────────────────────────────────────────
  // 3. PURE TRANSITIONS
  // ───────────────────────────────────────────────────────────────────────────

  pure def unchanged(state: State): TransitionResult =
    { success: false, newState: state }

  pure def withTask(
    state: State,
    id: TaskId,
    task: StartTask
  ): TransitionResult =
    if (not(TASK_IDS.contains(id))) {
      unchanged(state)
    } else {
      { success: true, newState: {
        ...state,
        tasks: state.tasks.set(id, task),
      } }
    }

  pure def isInProgress(status: StartStatus): bool = match status {
    | Working => true
    | WaitingForSandbox => true
    | PreparingRepository => true
    | RunningSetupScript => true
    | SettingUpGitHooks => true
    | SettingUpSkills => true
    | StartingConversation => true
    | _ => false
  }

  /// impl: live_status_app_conversation_service.py::_start_app_conversation()
  /// creates and first yields a WORKING task.
  pure def pureSubmitTask(state: State, id: TaskId): TransitionResult =
    val task = state.tasks.get(id)
    if (task.status != NotSubmitted) {
      unchanged(state)
    } else {
      withTask(state, id, {
        ...EMPTY_TASK,
        status: Working,
        backgroundAlive: true,
      })
    }

  /// impl: _wait_for_sandbox_start() assigns the sandbox before yielding
  /// WAITING_FOR_SANDBOX.
  pure def pureAssignSandbox(state: State, id: TaskId): TransitionResult =
    val task = state.tasks.get(id)
    if (task.status != Working) {
      unchanged(state)
    } else {
      withTask(state, id, {
        ...task,
        status: WaitingForSandbox,
        sandboxAssigned: true,
      })
    }

  /// impl: wait_for_sandbox_running() completes before run_setup_scripts()
  /// yields PREPARING_REPOSITORY.
  pure def pureSandboxReady(state: State, id: TaskId): TransitionResult =
    val task = state.tasks.get(id)
    if (task.status != WaitingForSandbox or not(task.sandboxAssigned)) {
      unchanged(state)
    } else {
      withTask(state, id, {
        ...task,
        status: PreparingRepository,
        sandboxRunning: true,
      })
    }

  /// impl: app_conversation_service_base.py::run_setup_scripts().
  pure def pureRepositoryPrepared(state: State, id: TaskId): TransitionResult =
    val task = state.tasks.get(id)
    if (task.status != PreparingRepository) {
      unchanged(state)
    } else {
      withTask(state, id, { ...task, status: RunningSetupScript })
    }

  /// impl: app_conversation_service_base.py::run_setup_scripts().
  pure def pureSetupScriptFinished(state: State, id: TaskId): TransitionResult =
    val task = state.tasks.get(id)
    if (task.status != RunningSetupScript) {
      unchanged(state)
    } else {
      withTask(state, id, { ...task, status: SettingUpGitHooks })
    }

  /// impl: app_conversation_service_base.py::run_setup_scripts().
  pure def pureGitHooksFinished(state: State, id: TaskId): TransitionResult =
    val task = state.tasks.get(id)
    if (task.status != SettingUpGitHooks) {
      unchanged(state)
    } else {
      withTask(state, id, { ...task, status: SettingUpSkills })
    }

  /// impl: LiveStatusAppConversationService sets agent_server_url when it
  /// yields STARTING_CONVERSATION.
  pure def pureSkillsFinished(state: State, id: TaskId): TransitionResult =
    val task = state.tasks.get(id)
    if (task.status != SettingUpSkills) {
      unchanged(state)
    } else {
      withTask(state, id, {
        ...task,
        status: StartingConversation,
        agentUrlSet: true,
      })
    }

  /// ACST-001 — The externally visible READY publication is atomic: the Agent
  /// Server conversation, app metadata, callbacks, and task link all exist.
  pure def pureCompleteConversationStart(
    state: State,
    id: TaskId
  ): TransitionResult =
    val task = state.tasks.get(id)
    if (
      task.status != StartingConversation
      or not(task.sandboxAssigned)
      or not(task.sandboxRunning)
      or not(task.agentUrlSet)
    ) {
      unchanged(state)
    } else {
      withTask(state, id, {
        ...task,
        status: Ready,
        agentConversationCreated: true,
        metadataSaved: true,
        callbacksSaved: true,
        conversationLinked: true,
        readyPublished: true,
      })
    }

  /// ACST-002/004 — ERROR may be published from any nonterminal phase. The
  /// desired contract assumes compensation removes any partially-created
  /// conversation artifacts before the terminal error becomes observable.
  pure def pureFailStart(state: State, id: TaskId): TransitionResult =
    val task = state.tasks.get(id)
    if (not(isInProgress(task.status))) {
      unchanged(state)
    } else {
      withTask(state, id, {
        ...task,
        status: Error,
        agentConversationCreated: false,
        metadataSaved: false,
        callbacksSaved: false,
        conversationLinked: false,
        errorPublished: true,
        detailSet: true,
        backgroundAlive: false,
        pendingOutcome: NoPendingWork,
      })
    }

  /// Pending messages are modeled as ancillary work after READY because the
  /// implementation yields READY before entering _process_pending_messages().
  pure def pureQueuePendingMessages(
    state: State,
    id: TaskId
  ): TransitionResult =
    val task = state.tasks.get(id)
    if (
      task.status != Ready
      or not(task.backgroundAlive)
      or task.pendingOutcome != NoPendingWork
    ) {
      unchanged(state)
    } else {
      withTask(state, id, { ...task, pendingOutcome: PendingQueued })
    }

  pure def pureDeliverPendingMessages(
    state: State,
    id: TaskId
  ): TransitionResult =
    val task = state.tasks.get(id)
    if (task.status != Ready or task.pendingOutcome != PendingQueued) {
      unchanged(state)
    } else {
      withTask(state, id, {
        ...task,
        pendingOutcome: PendingDelivered,
        backgroundAlive: false,
      })
    }

  /// ACST-003 — An ancillary pending-message failure is recorded without
  /// changing the already-published READY start-task result.
  pure def pureFailPendingMessages(
    state: State,
    id: TaskId
  ): TransitionResult =
    val task = state.tasks.get(id)
    if (task.status != Ready or task.pendingOutcome != PendingQueued) {
      unchanged(state)
    } else {
      withTask(state, id, {
        ...task,
        pendingOutcome: PendingFailed,
        backgroundAlive: false,
      })
    }

  pure def pureFinishWithoutPending(
    state: State,
    id: TaskId
  ): TransitionResult =
    val task = state.tasks.get(id)
    if (
      task.status != Ready
      or task.pendingOutcome != NoPendingWork
      or not(task.backgroundAlive)
    ) {
      unchanged(state)
    } else {
      withTask(state, id, { ...task, backgroundAlive: false })
    }

  // ───────────────────────────────────────────────────────────────────────────
  // 4. STATE VARIABLES
  // ───────────────────────────────────────────────────────────────────────────

  var tasks: TaskId -> StartTask

  val currentState: State = { tasks: tasks }
  val taskA: StartTask = tasks.get(TASK_A)
  val taskB: StartTask = tasks.get(TASK_B)

  // ───────────────────────────────────────────────────────────────────────────
  // 5. SAFETY PROPERTIES
  // ───────────────────────────────────────────────────────────────────────────

  pure def hasConversationArtifacts(task: StartTask): bool = or {
    task.agentConversationCreated,
    task.metadataSaved,
    task.callbacksSaved,
  }

  /// ACST-001 — READY exposes a fully usable, linked conversation.
  pure def readyIsComplete(task: StartTask): bool =
    task.status != Ready or and {
      task.sandboxAssigned,
      task.sandboxRunning,
      task.agentUrlSet,
      task.agentConversationCreated,
      task.metadataSaved,
      task.callbacksSaved,
      task.conversationLinked,
      task.readyPublished,
      not(task.errorPublished),
    }

  /// ACST-004 — Once a terminal status has been published, it remains the
  /// task's status for every later ancillary transition.
  pure def terminalPublicationIsStable(task: StartTask): bool = and {
    not(task.readyPublished) or task.status == Ready,
    not(task.errorPublished) or task.status == Error,
    not(task.readyPublished and task.errorPublished),
  }

  pure def linkedConversationIsReady(task: StartTask): bool =
    not(task.conversationLinked) or task.status == Ready

  /// ACST-002 — Artifacts may be transiently unlinked only while the internal
  /// Agent Server creation phase is still running, never after ERROR.
  pure def noOrphanedConversationArtifacts(task: StartTask): bool =
    not(hasConversationArtifacts(task))
      or task.conversationLinked
      or task.status == StartingConversation

  pure def errorIsClean(task: StartTask): bool =
    task.status != Error or and {
      task.detailSet,
      task.errorPublished,
      not(task.readyPublished),
      not(task.backgroundAlive),
      not(task.agentConversationCreated),
      not(task.metadataSaved),
      not(task.callbacksSaved),
      not(task.conversationLinked),
    }

  pure def pendingWorkBelongsToReady(task: StartTask): bool =
    task.pendingOutcome == NoPendingWork or and {
      task.status == Ready,
      task.readyPublished,
      task.conversationLinked,
    }

  pure def unsubmittedTaskIsEmpty(task: StartTask): bool =
    task.status != NotSubmitted or task == EMPTY_TASK

  pure def taskSafety(task: StartTask): bool = and {
    readyIsComplete(task),
    terminalPublicationIsStable(task),
    linkedConversationIsReady(task),
    noOrphanedConversationArtifacts(task),
    errorIsClean(task),
    pendingWorkBelongsToReady(task),
    unsubmittedTaskIsEmpty(task),
  }

  val allSafetyProperties: bool =
    TASK_IDS.forall(id => taskSafety(tasks.get(id)))

  temporal safetyAlwaysHolds = always(allSafetyProperties)

  // ───────────────────────────────────────────────────────────────────────────
  // 6. THIN ACTIONS
  // ───────────────────────────────────────────────────────────────────────────

  action applyTransition(result: TransitionResult): bool = all {
    result.success,
    tasks' = result.newState.tasks,
  }

  action submitTaskAction(id: TaskId): bool =
    applyTransition(pureSubmitTask(currentState, id))

  action assignSandboxAction(id: TaskId): bool =
    applyTransition(pureAssignSandbox(currentState, id))

  action sandboxReadyAction(id: TaskId): bool =
    applyTransition(pureSandboxReady(currentState, id))

  action repositoryPreparedAction(id: TaskId): bool =
    applyTransition(pureRepositoryPrepared(currentState, id))

  action setupScriptFinishedAction(id: TaskId): bool =
    applyTransition(pureSetupScriptFinished(currentState, id))

  action gitHooksFinishedAction(id: TaskId): bool =
    applyTransition(pureGitHooksFinished(currentState, id))

  action skillsFinishedAction(id: TaskId): bool =
    applyTransition(pureSkillsFinished(currentState, id))

  action completeConversationStartAction(id: TaskId): bool =
    applyTransition(pureCompleteConversationStart(currentState, id))

  action failStartAction(id: TaskId): bool =
    applyTransition(pureFailStart(currentState, id))

  action queuePendingMessagesAction(id: TaskId): bool =
    applyTransition(pureQueuePendingMessages(currentState, id))

  action deliverPendingMessagesAction(id: TaskId): bool =
    applyTransition(pureDeliverPendingMessages(currentState, id))

  action failPendingMessagesAction(id: TaskId): bool =
    applyTransition(pureFailPendingMessages(currentState, id))

  action finishWithoutPendingAction(id: TaskId): bool =
    applyTransition(pureFinishWithoutPending(currentState, id))

  // ───────────────────────────────────────────────────────────────────────────
  // 7. INITIALIZATION AND EXPLORATION
  // ───────────────────────────────────────────────────────────────────────────

  action init = all {
    tasks' = INITIAL_TASKS,
  }

  action step = any {
    nondet id = TASK_IDS.oneOf()
    any {
      submitTaskAction(id),
      assignSandboxAction(id),
      sandboxReadyAction(id),
      repositoryPreparedAction(id),
      setupScriptFinishedAction(id),
      gitHooksFinishedAction(id),
      skillsFinishedAction(id),
      completeConversationStartAction(id),
      failStartAction(id),
      queuePendingMessagesAction(id),
      deliverPendingMessagesAction(id),
      failPendingMessagesAction(id),
      finishWithoutPendingAction(id),
    }
  }
}
