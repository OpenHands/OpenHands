import { isAskUserAnswerEvent } from "@openhands/typescript-client";
import type { OHEvent } from "#/stores/use-event-store";
import type { MessageEvent } from "#/types/agent-server/core";
import {
  isMessageEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";

/**
 * Opening of the user message the server saves after an answer picked on a
 * decision card (`ANSWER_HEADER_PREFIX` in the SDK's event/ask_user.py).
 */
export const ASK_USER_ANSWER_PREFIX = "[Answer to your question";

export const isAskUserAnswerMessage = (event: MessageEvent): boolean => {
  const { content } = event.llm_message;
  const first = Array.isArray(content)
    ? content.find((c) => c.type === "text")
    : undefined;
  return !!first && first.text.startsWith(ASK_USER_ANSWER_PREFIX);
};

const isUserOwnMessage = (event: OHEvent): event is MessageEvent =>
  isMessageEvent(event) && event.source === "user" && !event.sender;

export interface DecisionClosure {
  /** What closed the question: a delivered answer, a chat reply, or a newer question. */
  closing: OHEvent;
  /** For a delivered answer, the generated message that carried it. */
  delivery?: MessageEvent;
}

/**
 * How each `ask_user` question ended, keyed by its observation id, read the
 * way the server reads it: the user's own next message closes a question, and
 * a newer question replaces it. An answer event counts only when its generated
 * message follows it, so an answer whose message was lost in a crash stays
 * retryable, and an ordinary reply after it reads as an answer in the chat.
 */
export function indexDecisionClosures(
  events: readonly OHEvent[],
): Map<string, DecisionClosure> {
  const closures = new Map<string, DecisionClosure>();
  const next = new Map<boolean, DecisionClosure>();
  // Projects: the user answers a worker's question on the coordinator's
  // card, and the server delivers the answer to the worker with the
  // coordinator as its `sender`. That delivery settles the answer too, as
  // the Agent Server's `find_pending_ask` reads it; on its own it closes
  // nothing.
  const relayed = new Map<boolean, MessageEvent>();
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    const planner = !!event.isFromPlanningAgent;
    if ("kind" in event && isAskUserAnswerEvent(event)) {
      const later = next.get(planner);
      const delivery = relayed.get(planner);
      if (delivery) {
        next.set(planner, { closing: event, delivery });
        relayed.delete(planner);
      } else if (
        later &&
        isMessageEvent(later.closing) &&
        isAskUserAnswerMessage(later.closing)
      ) {
        next.set(planner, { closing: event, delivery: later.closing });
      }
      continue;
    }
    const isQuestion =
      isObservationEvent(event) &&
      event.observation.kind === "AskUserObservation" &&
      !event.observation.is_error;
    if (isQuestion) {
      const later = next.get(planner);
      if (later) closures.set(event.id, later);
    }
    if (isQuestion || isUserOwnMessage(event)) {
      next.set(planner, { closing: event });
      relayed.delete(planner);
    } else if (
      isMessageEvent(event) &&
      event.source === "user" &&
      isAskUserAnswerMessage(event)
    ) {
      relayed.set(planner, event);
    }
  }
  return closures;
}

/** The answer events that reached the agent, for exports. */
export function deliveredAnswerIds(events: readonly OHEvent[]): Set<string> {
  const ids = new Set<string>();
  for (const { closing } of indexDecisionClosures(events).values()) {
    if ("kind" in closing && isAskUserAnswerEvent(closing) && closing.id) {
      ids.add(closing.id);
    }
  }
  return ids;
}
