import { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isACPToolCallEvent,
  isObservationEvent,
  isStreamingDeltaEvent,
} from "#/types/agent-server/type-guards";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";
import type {
  DeltaFrame,
  ItemStartedFrame,
} from "#/types/agent-server/session-frames";

/**
 * An event as held by the UI list. `seq` is the log index a `durable` frame
 * carried; slots and transient events have none, which is what makes "has a
 * seq" mean "is resumable".
 */
export type UIEvent = OpenHandsEvent & {
  isFromPlanningAgent?: boolean;
  seq?: number;
};

export interface StreamingSlotMeta {
  isFromPlanningAgent?: boolean;
}

const attemptOf = (slot: StreamingDeltaEvent): number => slot.attempt ?? 1;

const findSlotIndex = (uiEvents: UIEvent[], itemId: string): number =>
  uiEvents.findIndex(
    (event) => isStreamingDeltaEvent(event) && event.id === itemId,
  );

/**
 * Where a slot anchored after `anchorSeq` belongs: before the first event the
 * server sequenced after it. This is what keeps a user message that lands
 * mid-stream *below* the bubble instead of splitting it (#15433).
 */
const slotInsertIndex = (
  uiEvents: UIEvent[],
  anchorSeq: number | null | undefined,
): number => {
  if (anchorSeq === null || anchorSeq === undefined) {
    return uiEvents.length;
  }
  const index = uiEvents.findIndex(
    (event) => event.seq !== undefined && event.seq > anchorSeq,
  );
  return index === -1 ? uiEvents.length : index;
};

/**
 * Open (or re-open, for a higher attempt) the slot an `item_started` frame
 * announces. A lower attempt is a late frame from a stream already superseded
 * and is ignored.
 */
export const openStreamingSlot = (
  frame: ItemStartedFrame,
  uiEvents: UIEvent[],
  meta: StreamingSlotMeta = {},
): UIEvent[] => {
  const attempt = frame.attempt ?? 1;
  const existingIndex = findSlotIndex(uiEvents, frame.item_id);

  if (existingIndex !== -1) {
    const slot = uiEvents[existingIndex] as StreamingDeltaEvent;
    if (attemptOf(slot) > attempt) {
      return uiEvents;
    }
    const next = [...uiEvents];
    next[existingIndex] = {
      ...slot,
      attempt,
      content: null,
      reasoning_content: null,
    };
    return next;
  }

  const insertIndex = slotInsertIndex(uiEvents, frame.anchor_seq);
  // Borrow the anchor's timestamp so the store's timestamp sort (which fires
  // whenever an out-of-order event arrives) cannot drag the slot away from it.
  const anchor = uiEvents[insertIndex - 1];
  const slot: StreamingDeltaEvent & StreamingSlotMeta = {
    kind: "StreamingDeltaEvent",
    id: frame.item_id,
    source: "agent",
    timestamp:
      anchor && "timestamp" in anchor && anchor.timestamp
        ? anchor.timestamp
        : new Date().toISOString(),
    content: null,
    reasoning_content: null,
    attempt,
    ...meta,
  };

  const next = [...uiEvents];
  next.splice(insertIndex, 0, slot);
  return next;
};

/**
 * Append a batch of `delta` frames to their slots.
 *
 * A delta with no open slot is dropped: after a reconnect the slot was
 * discarded on purpose, and conjuring one back would recreate the orphan
 * bubble the slot model exists to prevent. The text is not lost — the durable
 * message carries it.
 */
export const appendStreamingDeltas = (
  frames: DeltaFrame[],
  uiEvents: UIEvent[],
): UIEvent[] => {
  let next = uiEvents;
  let copied = false;

  for (const frame of frames) {
    const index = findSlotIndex(next, frame.item_id);
    if (index === -1) {
      continue;
    }
    const slot = next[index] as StreamingDeltaEvent;
    const attempt = frame.attempt ?? 1;
    if (attempt < attemptOf(slot)) {
      continue;
    }
    // A higher attempt re-streams the item, so the old tail is superseded.
    const superseded = attempt > attemptOf(slot);
    const content = superseded ? "" : (slot.content ?? "");
    const reasoning = superseded ? "" : (slot.reasoning_content ?? "");
    const isReasoning = frame.kind === "reasoning";

    if (!copied) {
      next = [...next];
      copied = true;
    }
    next[index] = {
      ...slot,
      attempt,
      content: (isReasoning ? content : content + frame.content) || null,
      reasoning_content:
        (isReasoning ? reasoning + frame.content : reasoning) || null,
    };
  }

  return next;
};

/**
 * Drop the slot for `itemId`. Used by `item_aborted`: the stream ended without
 * a durable event, so nothing is coming to supersede the provisional text.
 */
export const abortStreamingSlot = (
  itemId: string,
  uiEvents: UIEvent[],
): UIEvent[] => {
  const index = findSlotIndex(uiEvents, itemId);
  if (index === -1) {
    return uiEvents;
  }
  return uiEvents.filter((_, position) => position !== index);
};

/**
 * Drop every open slot for one socket. Called on connect and disconnect:
 * progress frames are never replayed, so an open slot cannot survive the gap,
 * and the message it was standing in for arrives on the durable cursor.
 */
export const clearStreamingSlots = (
  uiEvents: UIEvent[],
  isFromPlanningAgent = false,
): UIEvent[] => {
  const next = uiEvents.filter(
    (event) =>
      !isStreamingDeltaEvent(event) ||
      Boolean((event as UIEvent).isFromPlanningAgent) !== isFromPlanningAgent,
  );
  return next.length === uiEvents.length ? uiEvents : next;
};

/**
 * Handles adding an event to the UI events array
 * Replaces actions with observations when they arrive (so UI shows observation instead of action)
 * Exception: ThinkAction is NOT replaced because the thought content is in the action, not in the observation
 *
 * ACPToolCallEvent merge: the SDK emits two events per ``tool_call_id`` — an
 * early ``started`` event (``pending`` / ``in_progress``) and one terminal
 * (completed / failed) event, the action->observation pair for a tool call.
 * Replace the started entry in place with the terminal one so a single card
 * updates from running to its result, exactly like an observation superseding
 * its action below.
 */
export const handleEventForUI = (
  event: UIEvent,
  uiEvents: UIEvent[],
): UIEvent[] => {
  const newUiEvents = [...uiEvents];

  // The durable event *is* the item: the SDK mints its id when the stream
  // opens, so one equality test retires the slot — and replacing in place
  // keeps the finished message where the streamed text already was.
  const eventId = "id" in event ? event.id : undefined;
  if (eventId !== undefined) {
    const slotIndex = findSlotIndex(newUiEvents, eventId);
    if (slotIndex !== -1) {
      newUiEvents[slotIndex] = event;
      return newUiEvents;
    }
  }

  if (isACPToolCallEvent(event)) {
    const existingIndex = newUiEvents.findIndex(
      (uiEvent) =>
        isACPToolCallEvent(uiEvent) &&
        uiEvent.tool_call_id === event.tool_call_id,
    );
    if (existingIndex !== -1) {
      newUiEvents[existingIndex] = event;
    } else {
      newUiEvents.push(event);
    }
    return newUiEvents;
  }

  if (isObservationEvent(event)) {
    // Don't add ThinkObservation at all - we keep the ThinkAction instead
    // The thought content is in the action, not the observation
    if (event.observation.kind === "ThinkObservation") {
      return newUiEvents;
    }

    // Don't add FinishObservation at all - we keep the FinishAction instead
    // Both contain the same message content, so we only need to display one
    // This also prevents duplicate messages when events arrive out of order due to React batching
    if (event.observation.kind === "FinishObservation") {
      return newUiEvents;
    }

    // Find and replace the corresponding action from uiEvents
    const actionIndex = newUiEvents.findIndex(
      (uiEvent) => uiEvent.id === event.action_id,
    );
    if (actionIndex !== -1) {
      newUiEvents[actionIndex] = event;
    } else {
      // Action not found in uiEvents, just add the observation
      newUiEvents.push(event);
    }
  } else {
    // For non-observation events, just add them to uiEvents
    newUiEvents.push(event);
  }

  return newUiEvents;
};
