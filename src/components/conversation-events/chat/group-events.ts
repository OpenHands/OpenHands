import { ActionEvent, OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isObservationEvent,
  isPlanningFileEditorObservationEvent,
} from "#/types/agent-server/type-guards";
import { getThoughtSourceAction } from "./event-thought-helpers";

/** Minimum run-length before consecutive actions get folded into a single
 *  collapsible group. Even pairs are folded so the chat scroll stays compact
 *  when the agent fires off back-to-back tool calls. */
export const EVENT_GROUP_MIN_SIZE = 2;

/**
 * Returns true if the given event is one of the action / observation cards
 * that we want to fold into an `EventGroup` when several appear in a row.
 *
 * Events that have their own dedicated rendering (FinishAction, ThinkAction,
 * HookExecution, AgentError, MessageEvent, PlanPreview, TaskTracker) are
 * treated as group breakers.
 */
export const isGroupableEvent = (event: OpenHandsEvent): boolean => {
  if (isActionEvent(event)) {
    const { kind } = event.action;
    if (kind === "FinishAction" || kind === "ThinkAction") {
      return false;
    }
    return true;
  }

  if (isObservationEvent(event)) {
    if (isPlanningFileEditorObservationEvent(event)) {
      return false;
    }
    if (event.observation.kind === "TaskTrackerObservation") {
      return false;
    }
    return true;
  }

  return false;
};

export type RenderedItem =
  | { kind: "single"; event: OpenHandsEvent; index: number }
  | { kind: "thought"; action: ActionEvent; index: number }
  | { kind: "group"; events: OpenHandsEvent[]; startIndex: number };

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
 */
export const groupEvents = (
  events: OpenHandsEvent[],
  minSize: number = EVENT_GROUP_MIN_SIZE,
  allEvents: OpenHandsEvent[] = events,
): RenderedItem[] => {
  if (minSize < 1) {
    throw new Error("minSize must be at least 1");
  }

  const items: RenderedItem[] = [];
  const emittedThoughtActionIds = new Set<string>();
  let run: { events: OpenHandsEvent[]; startIndex: number } | null = null;

  const flushRun = () => {
    if (!run) return;
    if (run.events.length >= minSize) {
      items.push({
        kind: "group",
        events: run.events,
        startIndex: run.startIndex,
      });
    } else {
      run.events.forEach((event, offset) => {
        items.push({ kind: "single", event, index: run!.startIndex + offset });
      });
    }
    run = null;
  };

  events.forEach((event, index) => {
    if (isGroupableEvent(event)) {
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

  flushRun();
  return items;
};
