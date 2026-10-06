# Live Activity View Specs

---

### LAV-001: Only actively executing agents are listed

- [x] The activity view shall list a conversation only while its
      `execution_status` is `RUNNING` or `WAITING_FOR_CONFIRMATION`.
- [x] Idle, paused, finished, error, and unknown conversations shall be omitted
      from the live list and remain reachable through the conversation list.
- [x] The view shall read only conversations belonging to the active backend.

### LAV-002: A row conveys status, current step, and spend

- [x] Each row shall show the conversation title, a status chip, the latest
      activity, a relative timestamp, accumulated cost, and total tokens.
- [x] The latest activity shall be the newest action/tool call rendered through
      the shared action-title descriptor, falling back to the last assistant
      message, and a defined placeholder when neither exists.
- [x] A row shall be flagged "needs attention" when the status requires user
      action: waiting for confirmation, error, or stuck.
- [x] A row shall link into its conversation and perform no mutation.

### LAV-003: Subagent fan-out is derived from the event stream

- [x] A `TaskAction` shall open a delegation keyed by its tool-call id, labelled
      with the requested subagent type.
- [x] The matching `TaskObservation` (paired by `action_id`) shall close the
      delegation as completed, or as errored when it reports an error.
- [x] A delegation without a matching observation shall remain "running".
- [x] The row shall show the number of delegations.

### LAV-004: Data is bounded and read-only

- [x] The view shall poll the conversation list and a bounded per-conversation
      event tail; it shall not open a WebSocket per conversation.
- [x] A conversation without a resolved `conversation_url` shall render with an
      empty tail rather than a failing request.
- [x] The view shall never start, stop, pause, or otherwise mutate a
      conversation.

### LAV-005: The view is reachable and has defined states

- [x] The view shall be reachable from the sidebar navigation at `/activity`.
- [x] Loading, empty, error, and backend-unavailable states shall each render a
      defined message, with a retry affordance for the error and
      backend-unavailable states.
