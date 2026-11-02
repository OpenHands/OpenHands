import { ActionEvent, OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isACPToolCallEvent,
  isMessageEvent,
  isStreamingDeltaEvent,
  isHookExecutionEvent,
  isObservationEvent,
  isPlanningFileEditorObservationEvent,
  isUserRejectObservation,
} from "#/types/agent-server/type-guards";
import { isMarkdownFileEditorEvent } from "#/components/features/chat/tool-visualizers/primitives/markdown-file-preview";
import type { SessionActivity } from "#/stores/session-activity-store";
import { parseMessageFromEvent } from "./event-content-helpers/parse-message-from-event";
import {
  getReasoningContent,
  splitInlineThink,
  getThoughtSourceAction,
} from "./event-thought-helpers";

/** Minimum run-length before consecutive actions get folded into a single
 *  collapsible group. Even pairs are folded so the chat scroll stays compact
 *  when the agent fires off back-to-back tool calls. */
export const EVENT_GROUP_MIN_SIZE = 2;

/**
 * Returns true if the given event is one of the action / observation cards
 * that we want to fold into an `EventGroup` when several appear in a row.
 *
 * Successful hook runs fold into the surrounding run; blocked or failed hooks
 * stay visible on their own. Events that have their own dedicated rendering
 * (FinishAction, AgentError, MessageEvent, PlanPreview, markdown
 * file artifacts, TaskTracker) are treated as group breakers.
 */
export const isGroupableEvent = (
  event: OpenHandsEvent,
  correspondingAction?: ActionEvent,
): boolean => {
  if (isActionEvent(event)) {
    const { kind } = event.action;
    if (kind === "FinishAction") {
      return false;
    }
    // Keep markdown *create* artifact cards outside collapsed groups so their
    // clipped preview is visible without expanding a parent summary.
    if (isMarkdownFileEditorEvent(event, correspondingAction)) {
      return false;
    }
    return true;
  }

  if (isObservationEvent(event)) {
    if (isPlanningFileEditorObservationEvent(event)) {
      return false;
    }
    if (isMarkdownFileEditorEvent(event, correspondingAction)) {
      return false;
    }
    if (event.observation.kind === "TaskTrackerObservation") {
      return false;
    }
    return true;
  }

  if (isACPToolCallEvent(event)) return event.status !== "failed";

  if (isHookExecutionEvent(event)) {
    return event.success && !event.blocked;
  }

  // A rejected action stays wherever the action itself was: in its run, or
  // on its own when the action is a group breaker.
  if (isUserRejectObservation(event)) {
    return correspondingAction ? isGroupableEvent(correspondingAction) : true;
  }

  return false;
};

export type RenderedItem =
  | { kind: "single"; event: OpenHandsEvent; index: number }
  | { kind: "thought"; action: ActionEvent; index: number }
  | {
      kind: "group";
      events: OpenHandsEvent[];
      startIndex: number;
      session?: SessionActivity;
      reasoning?: { id: string; text: string; index: number }[];
    };

/**
 * Walk a list of UI events and bucket consecutive groupable events into
 * `group` items. Anything that breaks the run, or runs shorter than
 * `EVENT_GROUP_MIN_SIZE`, is emitted as `single` items so they keep rendering
 * the way they always have.
 *
 * Whenever a groupable event carries an agent thought (either an
 * `ActionEvent.thought` or the corresponding action of an
 * `ObservationEvent`), the thought is hoisted out as its own `thought`
 * `RenderedItem`, the current run is flushed, and a new run is started with
 * the event itself. This keeps reasoning text in the main message stream
 * instead of buried inside a collapsed action group.
 *
 * `allEvents` should be the full event history so observations can find
 * their matching action; if omitted, it defaults to `events`.
 *
 * `trailingMinSize` applies to the run at the very end of `events`. While the
 * agent is working, pass 1 so the stage it is on is a group from its first
 * step and does not jump from a lone row into a group when a second arrives.
 */
export const groupEvents = (
  events: OpenHandsEvent[],
  minSize: number = EVENT_GROUP_MIN_SIZE,
  allEvents: OpenHandsEvent[] = events,
  trailingMinSize: number = minSize,
  sessions: readonly SessionActivity[] = [],
): RenderedItem[] => {
  if (minSize < 1 || trailingMinSize < 1) {
    throw new Error("minSize must be at least 1");
  }

  const items: RenderedItem[] = [];
  const emittedThoughtActionIds = new Set<string>();
  let run: Omit<Extract<RenderedItem, { kind: "group" }>, "kind"> | null = null;
  const startRun = (index: number) => {
    if (!run) run = { events: [], startIndex: index };
    return run;
  };
  const insertions = new Map<number, SessionActivity[]>();
  for (const session of sessions) {
    let index: number;
    if (session.kind === "initialize") {
      const userIndex = events.findIndex((event) => event.source === "user");
      index = userIndex < 0 ? 0 : userIndex + 1;
    } else {
      const anchor = session.afterEventId
        ? events.findIndex((event) => event.id === session.afterEventId)
        : -1;
      index =
        anchor >= 0
          ? anchor + 1
          : events.findIndex(
              (event) => Date.parse(event.timestamp ?? "") >= session.startedAt,
            );
      if (index < 0) index = events.length;
      if (events[index]?.source === "user") index += 1;
    }
    insertions.set(index, [...(insertions.get(index) ?? []), session]);
  }

  const flushRun = (size = minSize) => {
    if (!run) return;
    if (run.session || run.reasoning?.length || run.events.length >= size) {
      items.push({
        kind: "group",
        ...run,
      });
    } else {
      run.events.forEach((event, offset) => {
        items.push({ kind: "single", event, index: run!.startIndex + offset });
      });
    }
    run = null;
  };

  const insertSessions = (index: number) => {
    for (const session of insertions.get(index) ?? []) {
      flushRun();
      startRun(index).session = session;
    }
  };

  events.forEach((event, index) => {
    insertSessions(index);
    if (
      (isMessageEvent(event) && event.source === "agent") ||
      isStreamingDeltaEvent(event)
    ) {
      const streaming = isStreamingDeltaEvent(event);
      const { reasoning: inline, message } = splitInlineThink(
        streaming ? (event.content ?? "") : parseMessageFromEvent(event),
        { streaming },
      );
      const reasoning = [
        streaming
          ? event.reasoning_content
          : getReasoningContent(event.llm_message),
        inline,
      ]
        .filter(Boolean)
        .join("\n\n");
      if (reasoning) {
        const current = startRun(index);
        current.reasoning = [
          ...(current.reasoning ?? []),
          { id: event.id, text: reasoning, index },
        ];
      }
      const hasVisiblePayload =
        isMessageEvent(event) &&
        (event.critic_result != null ||
          (Array.isArray(event.llm_message.content) &&
            event.llm_message.content.some(
              (content) =>
                content.type === "image" && content.image_urls.length > 0,
            )));
      if (!message.trim() && !hasVisiblePayload && (streaming || reasoning))
        return;
    }
    const correspondingAction =
      isObservationEvent(event) || isUserRejectObservation(event)
        ? allEvents.find(
            (candidate): candidate is ActionEvent =>
              isActionEvent(candidate) && candidate.id === event.action_id,
          )
        : undefined;
    if (isGroupableEvent(event, correspondingAction)) {
      const reasoningAction = isActionEvent(event)
        ? event
        : correspondingAction;
      const reasoning = reasoningAction
        ? getReasoningContent(reasoningAction)
        : "";
      if (reasoning && reasoningAction) {
        const current = startRun(index);
        if (
          !current.reasoning?.some((entry) => entry.id === reasoningAction.id)
        ) {
          current.reasoning = [
            ...(current.reasoning ?? []),
            { id: reasoningAction.id, text: reasoning, index },
          ];
        }
      }
      const thoughtAction = getThoughtSourceAction(event, allEvents);
      if (thoughtAction && !emittedThoughtActionIds.has(thoughtAction.id)) {
        flushRun();
        emittedThoughtActionIds.add(thoughtAction.id);
        items.push({ kind: "thought", action: thoughtAction, index });
      }
      if (!run) run = { events: [], startIndex: index };
      run.events.push(event);
    } else {
      flushRun();
      items.push({ kind: "single", event, index });
    }
  });

  insertSessions(events.length);
  flushRun(trailingMinSize);
  return items;
};
