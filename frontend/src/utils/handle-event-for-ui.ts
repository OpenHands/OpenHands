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

/**
 * Newest timestamp among the durable (non-delta, non-metrics) agent/environment
 * events already rendered. Streaming deltas older than this are stale: their
 * content is already represented by durable events (the WebSocket history
 * replay does not include deltas, while the REST history does, so on a reload
 * the REST deltas can arrive AFTER the turn's final message was rendered).
 */
const findNewestDurableTimestamp = (
  events: OpenHandsEvent[],
): string | null => {
  let newest: string | null = null;
  for (const uiEvent of events) {
    const isPreviewOrMetrics =
      isStreamingDeltaEvent(uiEvent) || isConversationStateUpdateEvent(uiEvent);
    const isDurableSource =
      uiEvent.source === "agent" || uiEvent.source === "environment";
    if (!isPreviewOrMetrics && isDurableSource) {
      const ts = uiEvent.timestamp;
      if (typeof ts === "string" && (newest === null || ts > newest)) {
        newest = ts;
      }
    }
  }
  return newest;
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
  let matched = false;
  let lastMatchEnd = 0;

  // The SDK strips the finalized text, so it may lack trailing whitespace the
  // model streamed - tolerate that by also trying the trailing-trimmed
  // streamed text (mirrors agent-canvas #1552).
  for (const candidate of [streamedText, streamedText.trimEnd()]) {
    if (candidate && finalText.startsWith(candidate)) {
      matched = true;
      lastMatchEnd = candidate.length;
      break;
    }
  }
  if (!matched) {
    const lastIndex = streamingSegments.length - 1;
    const searchSegments = streamingSegments.map((segment, index) =>
      index === lastIndex ? segment.trimEnd() : segment,
    );
    const match = findTextSegmentsInOrder(finalText, searchSegments);
    matched = match.matched;
    lastMatchEnd = match.lastMatchEnd;
  }
  if (!matched) {
    // The streamed preview never reconciles (e.g. it contains earlier
    // steps' text, or chunks were reordered in delivery). The durable final
    // message is canonical: drop the preview deltas and render it once.
    const removeSet = new Set(contentStreamingDeltas.map(({ index }) => index));
    const cleaned = uiEvents.filter((_, index) => !removeSet.has(index));
    cleaned.push(finalEvent);
    return cleaned;
  }
  const unstreamedSuffix = finalText.slice(lastMatchEnd);

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
 * A tool-call ``ActionEvent`` carries the agent's ``thought`` - the same text
 * that was just streamed as deltas. The action card renders that thought
 * itself, so once the action arrives, the streamed preview deltas for the turn
 * are redundant: leaving them in place displays every step's reasoning twice.
 *
 * When the current turn's content-bearing deltas reconcile (in order) against
 * the action's thought text, return uiEvents with those deltas removed so the
 * caller can append the action as the single rendered copy. Returns ``null``
 * (no change) when there is nothing to reconcile or the text doesn't match.
 */
const supersedeStreamingDeltasForAction = (
  actionEvent: OpenHandsEvent,
  uiEvents: OpenHandsEvent[],
): OpenHandsEvent[] | null => {
  if (!isActionEvent(actionEvent)) {
    return null;
  }
  const lastUserMessageIndex = findLastUserMessageIndex(uiEvents);
  const currentTurnDeltaIndexes = uiEvents
    .map((uiEvent, index) => ({ uiEvent, index }))
    .filter(
      ({ uiEvent, index }) =>
        index > lastUserMessageIndex && isStreamingDeltaEvent(uiEvent),
    )
    .map(({ index }) => index);

  if (currentTurnDeltaIndexes.length === 0) {
    return null;
  }

  // The action renders its own thought, so the streamed preview text is
  // redundant once the durable action arrives. Text-matching is deliberately
  // NOT required: chunk reordering between the delta stream and event
  // persistence would otherwise leak stray preview bubbles that later merge
  // with the next stream. For many models the delta is the sole carrier of
  // reasoning_content, though - when the action does not render reasoning of
  // its own, keep the delta as a reasoning-only bubble (content cleared)
  // instead of dropping it (mirrors agent-canvas #1552).
  const actionRendersReasoning =
    Boolean(actionEvent.reasoning_content?.trim()) ||
    (actionEvent.thinking_blocks?.length ?? 0) > 0;
  const stripSet = new Set(currentTurnDeltaIndexes);
  const nextUiEvents: OpenHandsEvent[] = [];
  uiEvents.forEach((uiEvent, index) => {
    if (!stripSet.has(index) || !isStreamingDeltaEvent(uiEvent)) {
      nextUiEvents.push(uiEvent);
      return;
    }
    if (!actionRendersReasoning && uiEvent.reasoning_content) {
      nextUiEvents.push({ ...uiEvent, content: null });
    }
  });
  return nextUiEvents;
};

/**
 * Handles adding an event to the UI events array
 * Replaces actions with observations when they arrive (so UI shows observation instead of action)
 * Exception: ThinkAction is NOT replaced because the thought content is in the action, not in the observation
 *
 * StreamingDeltaEvent: consecutive deltas merge in place into a single growing
 * bubble; when the turn's final agent message arrives it is reconciled into that
 * bubble (see `finalizeStreamingDeltasInPlace`) rather than appended, and a
 * tool-call action removes the deltas that streamed its thought (see
 * `supersedeStreamingDeltasForAction`). Deltas that arrive out of order (older
 * than an already-rendered durable event) are dropped as stale.
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

    // Drop stale deltas. Deltas are a live-preview mechanism; one that is
    // older than the newest durable event arrived late (e.g. the REST history
    // response racing the WebSocket replay, which omits deltas). Its content
    // is already represented by durable events, so rendering it would fragment
    // and duplicate the finished message.
    const newestDurableTimestamp = findNewestDurableTimestamp(newUiEvents);
    if (
      newestDurableTimestamp !== null &&
      typeof event.timestamp === "string" &&
      event.timestamp < newestDurableTimestamp
    ) {
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

  // A tool-call action renders its own thought; remove the preview deltas that
  // streamed it, then fall through so the action is appended normally below.
  if (isActionEvent(event) && event.action.kind !== "FinishAction") {
    const superseded = supersedeStreamingDeltasForAction(event, newUiEvents);
    if (superseded) {
      superseded.push(event);
      return superseded;
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
