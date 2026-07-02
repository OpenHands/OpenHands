import { MessageEvent, OpenHandsEvent } from "#/types/v1/core";
import { StreamingDeltaEvent } from "#/types/v1/core/events/streaming-delta-event";
import {
  isACPToolCallEvent,
  isActionEvent,
  isConversationStateUpdateEvent,
  isMessageEvent,
  isObservationEvent,
  isStreamingDeltaEvent,
  isUserMessageEvent,
} from "#/types/v1/type-guards";

/**
 * Concatenate two streaming deltas into one. Token chunks join directly with no
 * separator. Keeps the *existing* delta's identity (id/timestamp) so the
 * rendered bubble has a stable React key as it grows.
 */
export const mergeStreamingDeltaEvent = (
  incoming: StreamingDeltaEvent,
  existing: StreamingDeltaEvent,
): StreamingDeltaEvent => ({
  ...existing,
  content: `${existing.content ?? ""}${incoming.content ?? ""}` || null,
  reasoning_content:
    `${existing.reasoning_content ?? ""}${incoming.reasoning_content ?? ""}` ||
    null,
});

const appendContentToStreamingDeltaEvent = (
  existing: StreamingDeltaEvent,
  content: string,
): StreamingDeltaEvent => ({
  ...existing,
  content: `${existing.content ?? ""}${content}` || null,
});

const findLastUserMessageIndex = (events: OpenHandsEvent[]): number => {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (isUserMessageEvent(events[index])) {
      return index;
    }
  }
  return -1;
};

// Join text blocks WITHOUT a separator: streaming deltas concatenate content
// tokens directly with no separator between LLM content blocks, so using "\n"
// here would cause startsWith/findTextSegmentsInOrder to miss when reconciling
// a multi-block MessageEvent against the already-rendered streaming delta.
const getAgentMessageText = (event: MessageEvent): string =>
  event.llm_message.content
    .filter((content) => content.type === "text")
    .map((content) => content.text)
    .join("");

const getFinalAgentText = (event: OpenHandsEvent): string | null => {
  if (isActionEvent(event) && event.action.kind === "FinishAction") {
    return event.action.message;
  }

  if (isMessageEvent(event) && event.llm_message.role === "assistant") {
    return getAgentMessageText(event);
  }

  return null;
};

const findTextSegmentsInOrder = (
  text: string,
  segments: string[],
): { matched: boolean; lastMatchEnd: number } => {
  let searchStart = 0;
  let lastMatchEnd = 0;

  for (const segment of segments) {
    const index = text.indexOf(segment, searchStart);
    if (index === -1) {
      return { matched: false, lastMatchEnd };
    }
    lastMatchEnd = index + segment.length;
    searchStart = lastMatchEnd;
  }

  return { matched: true, lastMatchEnd };
};

/**
 * When the final agent message (a ``FinishAction`` or an assistant
 * ``MessageEvent``) arrives, fold it into the streaming deltas that produced it
 * instead of appending a duplicate bubble. The last content-bearing delta
 * (extended with any text that only arrived in the final event) becomes the
 * canonical rendered message for the turn.
 *
 * Returns the updated uiEvents (with the final event intentionally NOT appended)
 * when there are content-bearing deltas for the current turn that reconcile
 * against the final text; otherwise returns ``null`` so the caller appends the
 * final event normally (e.g. non-streamed responses, or reasoning-only deltas).
 */
const finalizeStreamingDeltasInPlace = (
  finalEvent: OpenHandsEvent,
  uiEvents: OpenHandsEvent[],
): OpenHandsEvent[] | null => {
  const lastUserMessageIndex = findLastUserMessageIndex(uiEvents);
  const currentTurnStreamingDeltaIndexes = uiEvents
    .map((uiEvent, index) => ({ uiEvent, index }))
    .filter(
      ({ uiEvent, index }) =>
        index > lastUserMessageIndex && isStreamingDeltaEvent(uiEvent),
    )
    .map(({ index }) => index);

  if (currentTurnStreamingDeltaIndexes.length === 0) {
    return null;
  }

  const finalText = getFinalAgentText(finalEvent);
  // Only the regular `content` field participates in reconciliation.
  // Reasoning-only deltas (those that carry only `reasoning_content`) produce
  // an empty streamingSegments list, causing the function to return null so the
  // finalEvent is appended normally. This is intentional: reasoning content
  // renders in its own block and never overlaps with the assistant's regular
  // message text.
  const contentStreamingDeltas = currentTurnStreamingDeltaIndexes
    .map((index) => ({ event: uiEvents[index], index }))
    .filter(
      (item): item is { event: StreamingDeltaEvent; index: number } =>
        isStreamingDeltaEvent(item.event) &&
        (item.event.content?.length ?? 0) > 0,
    );
  const streamingSegments = contentStreamingDeltas.map(
    ({ event }) => event.content ?? "",
  );

  if (!finalText || streamingSegments.length === 0) {
    return null;
  }

  const nextUiEvents = [...uiEvents];
  const streamedText = streamingSegments.join("");
  let unstreamedSuffix = "";

  if (finalText.startsWith(streamedText)) {
    unstreamedSuffix = finalText.slice(streamedText.length);
  } else {
    const match = findTextSegmentsInOrder(finalText, streamingSegments);
    if (!match.matched) {
      return null;
    }
    unstreamedSuffix = finalText.slice(match.lastMatchEnd);
  }

  const lastDeltaIndex = contentStreamingDeltas.at(-1)?.index;
  const lastDelta =
    lastDeltaIndex === undefined ? undefined : nextUiEvents[lastDeltaIndex];
  if (
    unstreamedSuffix &&
    lastDeltaIndex !== undefined &&
    lastDelta &&
    isStreamingDeltaEvent(lastDelta)
  ) {
    nextUiEvents[lastDeltaIndex] = appendContentToStreamingDeltaEvent(
      lastDelta,
      unstreamedSuffix,
    );
  }

  // Intentionally return nextUiEvents WITHOUT appending finalEvent. The last
  // content-bearing streaming delta (possibly extended with unstreamedSuffix
  // above) is the canonical final rendered bubble for this turn. Appending
  // finalEvent here would display the assistant message twice.
  return nextUiEvents;
};

/**
 * Handles adding an event to the UI events array
 * Replaces actions with observations when they arrive (so UI shows observation instead of action)
 * Exception: ThinkAction is NOT replaced because the thought content is in the action, not in the observation
 *
 * StreamingDeltaEvent: consecutive deltas merge in place into a single growing
 * bubble; when the turn's final agent message arrives it is reconciled into that
 * bubble (see `finalizeStreamingDeltasInPlace`) rather than appended.
 *
 * ACPToolCallEvent dedup: multiple events share a ``tool_call_id`` as an ACP
 * tool call progresses (in_progress → completed / failed). Collapse them to
 * the latest state at the original position so the card updates in place.
 */
export const handleEventForUI = (
  event: OpenHandsEvent,
  uiEvents: OpenHandsEvent[],
): OpenHandsEvent[] => {
  const newUiEvents = [...uiEvents];

  if (isStreamingDeltaEvent(event)) {
    // Drop empty boundary deltas (e.g. after a tool call) that carry no text.
    if (event.content === null && event.reasoning_content === null) {
      return newUiEvents;
    }

    // Merge into the most recent streaming delta, treating interleaved non-chat
    // events as transparent: the agent server emits periodic
    // ConversationStateUpdateEvents (metrics) mid-stream, and those land in
    // uiEvents between deltas. Stopping the search at the very last event would
    // fragment one streamed response into several bubbles. A genuine rendered
    // event (message/action/observation) does end the run, so subsequent deltas
    // correctly start a fresh bubble.
    let mergeIndex = -1;
    for (let i = newUiEvents.length - 1; i >= 0; i -= 1) {
      const candidate = newUiEvents[i];
      if (isStreamingDeltaEvent(candidate)) {
        mergeIndex = i;
        break;
      }
      // Skip transparent metrics updates; stop at any genuine rendered event.
      if (!isConversationStateUpdateEvent(candidate)) {
        break;
      }
    }

    if (mergeIndex !== -1) {
      newUiEvents[mergeIndex] = mergeStreamingDeltaEvent(
        event,
        newUiEvents[mergeIndex] as StreamingDeltaEvent,
      );
      return newUiEvents;
    }

    newUiEvents.push(event);
    return newUiEvents;
  }

  // The turn's final agent text supersedes the streaming deltas that produced
  // it (when present), so reconcile before the generic handling below.
  if (
    (isActionEvent(event) && event.action.kind === "FinishAction") ||
    (isMessageEvent(event) && event.llm_message.role === "assistant")
  ) {
    const finalizedUiEvents = finalizeStreamingDeltasInPlace(
      event,
      newUiEvents,
    );
    if (finalizedUiEvents) {
      return finalizedUiEvents;
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
