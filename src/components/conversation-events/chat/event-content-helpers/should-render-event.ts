import { MessageEvent, OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isObservationEvent,
  isMessageEvent,
  isAgentErrorEvent,
  isConversationStateUpdateEvent,
  isGoalConversationStateUpdateEvent,
  isHookExecutionEvent,
  isACPToolCallEvent,
  isStreamingDeltaEvent,
} from "#/types/agent-server/type-guards";

// Prefixes of the SDK goal-loop re-prompts (FOLLOWUP_PROMPT / RESUME_PROMPT in
// openhands.sdk .../conversation/goal/prompts.py). The goal loop injects these
// as `user` messages each round to steer the agent, and FOLLOWUP_PROMPT embeds
// the judge's verdict — which the goal banner already surfaces. The persisted
// event carries no marker distinguishing them from real user input, so we match
// the prompt text to keep this machinery out of the chat. Brittle by design;
// keep in sync with the SDK prompts (the durable fix is a persisted goal-loop
// flag on the event).
const GOAL_REPROMPT_PREFIXES = [
  "The goal is NOT yet complete (audit iteration",
  "Resuming a goal that was paused or interrupted.",
];

const isGoalLoopReprompt = (event: MessageEvent): boolean => {
  if (event.llm_message?.role !== "user") return false;
  const content = event.llm_message.content;
  const text = Array.isArray(content)
    ? content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\n")
    : "";
  return GOAL_REPROMPT_PREFIXES.some((prefix) => text.startsWith(prefix));
};

export const shouldRenderEvent = (event: OpenHandsEvent) => {
  if (isConversationStateUpdateEvent(event)) {
    // A finished `/goal` loop renders inline so it settles into the
    // conversation; the live (active) banner is shown separately by
    // GoalStatusBanner, and all other state updates (and the in-progress goal
    // events) stay hidden.
    return isGoalConversationStateUpdateEvent(event) && !event.value.active;
  }

  // Render action events (with filtering)
  if (isActionEvent(event)) {
    // For V1, action is an object with kind property
    const actionType = event.action.kind;

    if (!actionType) {
      return false;
    }

    // Hide user commands from the chat interface
    if (actionType === "ExecuteBashAction" && event.source === "user") {
      return false;
    }

    // Hide PlanningFileEditorAction - handled separately with PlanPreview component
    if (actionType === "PlanningFileEditorAction") {
      return false;
    }

    // The model switch tool reuses the same inline model message UI as
    // `/model <profile>` once the observation arrives.
    if (actionType === "SwitchLLMAction") {
      return false;
    }

    return true;
  }

  // Render observation events
  if (isObservationEvent(event)) {
    // Successful model switches are rendered through ModelMessages so they
    // look identical to `/model <profile>` confirmations. Failed switches
    // still render as observations so the error remains visible in chat.
    if (
      event.observation.kind === "SwitchLLMObservation" &&
      !event.observation.is_error
    ) {
      return false;
    }

    return true;
  }

  // Render message events (user and assistant messages), except the goal loop's
  // injected re-prompts — the judge feedback they carry is shown in the goal
  // banner, so otherwise they leak into the chat as fake user turns.
  if (isMessageEvent(event)) {
    return !isGoalLoopReprompt(event);
  }

  // Render agent error events
  if (isAgentErrorEvent(event)) {
    return true;
  }

  // Render hook execution events
  if (isHookExecutionEvent(event)) {
    return true;
  }

  // Render ACP sub-agent tool call events at every lifecycle stage. The SDK
  // now persists exactly two events per ``tool_call_id`` — one early
  // ``started`` event (``pending`` / ``in_progress``) and one terminal
  // (``completed`` / ``failed``) event — the action->observation pair for a
  // tool call. The ``started`` event renders the card as "running" (no check
  // mark; see ``getACPToolCallResult``) and ``handleEventForUI`` replaces it
  // in place by ``tool_call_id`` once the terminal event arrives, mirroring
  // how an ObservationEvent supersedes its ActionEvent. The old terminal-only
  // gate existed because the source fanned out one cumulative-output frame per
  // ``ToolCallProgress``, which flashed half-formed cards mid-stream; that
  // fan-out is gone, so the running card is now a single clean event.
  if (isACPToolCallEvent(event)) {
    return true;
  }

  if (isStreamingDeltaEvent(event)) {
    return event.content !== null || event.reasoning_content !== null;
  }

  // Don't render any other event types (system events, etc.)
  return false;
};

export const hasUserEvent = (events: OpenHandsEvent[]) =>
  events.some((event) => event.source === "user");
