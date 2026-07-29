import {
  ActionEvent,
  ImageContent,
  MessageEvent,
  OpenHandsEvent,
  TextContent,
} from "#/types/agent-server/core";
import {
  isACPToolCallEvent,
  isActionEvent,
  isMessageEvent,
  isObservationEvent,
  isStreamingDeltaEvent,
} from "#/types/agent-server/type-guards";
import { StreamingDeltaEvent } from "#/types/agent-server/core/events/streaming-delta-event";
import {
  getReasoningContent,
  splitInlineThink,
} from "#/components/conversation-events/chat/event-thought-helpers";

// A user-message echo can arrive before one or both agent streams finish. Keep
// that message at the visual tail until every stream that crossed it finalizes.
// The non-enumerable symbol keeps this UI-only state off the server event shape.
const DEFERRED_USER_MESSAGE_SENDERS = Symbol("deferred-user-message-senders");

type DeferredUserMessageEvent = MessageEvent & {
  [DEFERRED_USER_MESSAGE_SENDERS]: ReadonlySet<boolean>;
};

const isDeferredUserMessage = (
  event: OpenHandsEvent,
): event is DeferredUserMessageEvent =>
  isMessageEvent(event) &&
  event.source === "user" &&
  Boolean(
    (
      event as MessageEvent & {
        [DEFERRED_USER_MESSAGE_SENDERS]?: ReadonlySet<boolean>;
      }
    )[DEFERRED_USER_MESSAGE_SENDERS]?.size,
  );

const getStreamingSender = (
  event: OpenHandsEvent & { isFromPlanningAgent?: boolean },
): boolean => Boolean(event.isFromPlanningAgent);

const deferUserMessageForSenders = (
  event: MessageEvent,
  senders: ReadonlySet<boolean>,
): MessageEvent =>
  Object.defineProperty({ ...event }, DEFERRED_USER_MESSAGE_SENDERS, {
    value: senders,
  });

const deferUserMessage = (
  event: MessageEvent,
  streamingEvent: StreamingDeltaEvent,
): MessageEvent =>
  deferUserMessageForSenders(
    event,
    new Set([getStreamingSender(streamingEvent)]),
  );

const addDeferredUserMessageSender = (
  event: OpenHandsEvent,
  streamingEvent: StreamingDeltaEvent,
): OpenHandsEvent => {
  if (!isDeferredUserMessage(event)) {
    return event;
  }

  const senders = new Set(event[DEFERRED_USER_MESSAGE_SENDERS]);
  senders.add(getStreamingSender(streamingEvent));
  return deferUserMessageForSenders(event, senders);
};

const releaseDeferredUserMessageSender = (
  event: OpenHandsEvent,
  finalEvent: OpenHandsEvent,
): OpenHandsEvent => {
  if (!isDeferredUserMessage(event)) {
    return event;
  }

  const senders = new Set(event[DEFERRED_USER_MESSAGE_SENDERS]);
  senders.delete(getStreamingSender(finalEvent));
  return senders.size > 0
    ? deferUserMessageForSenders(event, senders)
    : { ...event };
};

const releaseDeferredUserMessagesForSender = (
  events: OpenHandsEvent[],
  finalEvent: OpenHandsEvent,
): OpenHandsEvent[] =>
  events.map((event) => releaseDeferredUserMessageSender(event, finalEvent));

const findEventBeforeDeferredUserMessages = (
  events: OpenHandsEvent[],
): { event: OpenHandsEvent; index: number } | null => {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (!isDeferredUserMessage(event)) {
      return { event, index };
    }
  }
  return null;
};

const insertBeforeDeferredUserMessages = (
  events: OpenHandsEvent[],
  event: OpenHandsEvent,
): OpenHandsEvent[] => {
  let insertionIndex = events.length;
  while (
    insertionIndex > 0 &&
    isDeferredUserMessage(events[insertionIndex - 1])
  ) {
    insertionIndex -= 1;
  }

  if (insertionIndex === events.length) {
    return [...events, event];
  }

  const deferredMessages = events
    .slice(insertionIndex)
    .map((deferredEvent) =>
      isStreamingDeltaEvent(event)
        ? addDeferredUserMessageSender(deferredEvent, event)
        : deferredEvent,
    );
  return [...events.slice(0, insertionIndex), event, ...deferredMessages];
};

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

// Deltas only merge into one bubble when they share a sender; the planning flag
// is the only discriminator (every delta has `source: "agent"`). Prevents a
// planning-agent delta concatenating onto a main-agent one, misattributed (#1656).
export const isSameStreamingSender = (
  a: OpenHandsEvent & { isFromPlanningAgent?: boolean },
  b: OpenHandsEvent & { isFromPlanningAgent?: boolean },
): boolean => Boolean(a.isFromPlanningAgent) === Boolean(b.isFromPlanningAgent);

const findLastUserMessageIndex = (events: OpenHandsEvent[]): number => {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (
      isMessageEvent(event) &&
      event.source === "user" &&
      !isDeferredUserMessage(event)
    ) {
      return index;
    }
  }
  return -1;
};

// Join text blocks WITHOUT a separator: streaming deltas concatenate content
// tokens directly with no separator between LLM content blocks, so using "\n"
// here would cause startsWith/findTextSegmentsInOrder to miss when reconciling
// a multi-block message/thought against the already-rendered streaming delta.
const joinTextBlocks = (blocks: (TextContent | ImageContent)[]): string =>
  blocks
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("");

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

// Content-bearing streaming deltas of the current turn (after the last user
// message) that share the final event's sender. Reasoning-only deltas are
// excluded: reasoning renders in its own collapsed bubble and never overlaps
// the message text being reconciled. Sender scoping matters because the main
// and planning sockets share this event store — without it, one agent's final
// event would strip the other agent's still-live streamed deltas (#1656).
const getCurrentTurnContentDeltas = (
  uiEvents: OpenHandsEvent[],
  finalEvent: OpenHandsEvent,
): { event: StreamingDeltaEvent; index: number }[] => {
  const lastUserMessageIndex = findLastUserMessageIndex(uiEvents);
  return uiEvents
    .map((event, index) => ({ event, index }))
    .filter(
      (item): item is { event: StreamingDeltaEvent; index: number } =>
        item.index > lastUserMessageIndex &&
        isStreamingDeltaEvent(item.event) &&
        (item.event.content?.length ?? 0) > 0 &&
        isSameStreamingSender(finalEvent, item.event),
    );
};

// The current step's streaming delta(s): the trailing run at the end of
// `uiEvents`. Earlier steps' deltas are separated by their observations, so
// this never folds an earlier step's delta into the current one.
const getTrailingDeltas = (
  uiEvents: OpenHandsEvent[],
  selects: (event: StreamingDeltaEvent) => boolean,
): { event: StreamingDeltaEvent; index: number }[] => {
  const deltas: { event: StreamingDeltaEvent; index: number }[] = [];
  for (let index = uiEvents.length - 1; index >= 0; index -= 1) {
    const event = uiEvents[index];
    if (isDeferredUserMessage(event)) {
      continue;
    }
    if (!isStreamingDeltaEvent(event)) {
      break;
    }
    if (selects(event)) {
      deltas.unshift({ event, index });
    }
  }
  return deltas;
};

const getTrailingContentDeltas = (uiEvents: OpenHandsEvent[]) =>
  getTrailingDeltas(uiEvents, (event) => (event.content?.length ?? 0) > 0);

// Sender-scoped: the main and planning sockets share this event store, so a
// main-agent action must not strip the planning agent's live reasoning (#1656).
const getTrailingReasoningDeltas = (
  uiEvents: OpenHandsEvent[],
  finalEvent: OpenHandsEvent,
) =>
  getTrailingDeltas(
    uiEvents,
    (event) =>
      Boolean(event.reasoning_content) &&
      isSameStreamingSender(finalEvent, event),
  );

// Strip the streamed content deltas, keeping a delta only when it carries
// reasoning the replacement itself won't render (many models stream reasoning
// solely through the delta). Shared by the finalize and intermediate-action
// reconciliation paths.
const supersedeStreamingContent = (
  uiEvents: OpenHandsEvent[],
  contentDeltas: { event: StreamingDeltaEvent; index: number }[],
  replacementRendersReasoning: boolean,
): OpenHandsEvent[] => {
  const indexesToStrip = new Set(contentDeltas.map(({ index }) => index));
  const nextUiEvents: OpenHandsEvent[] = [];
  uiEvents.forEach((event, index) => {
    if (!indexesToStrip.has(index) || !isStreamingDeltaEvent(event)) {
      nextUiEvents.push(event);
      return;
    }

    // Keep the delta only to render reasoning the replacement itself lacks.
    if (!replacementRendersReasoning && event.reasoning_content) {
      nextUiEvents.push({ ...event, content: null });
    }
  });
  return nextUiEvents;
};

// Whether the streamed `segments` (in order) reconcile against `targetText`.
// `lastMatchEnd` is the offset past the matched text, so callers can recover
// any not-yet-streamed suffix. The SDK strips the finalized text, so tolerate
// leading and trailing whitespace that appeared only in the stream.
const matchStreamedSegments = (
  targetText: string,
  segments: string[],
): { matched: boolean; lastMatchEnd: number } => {
  const streamedText = segments.join("");
  const candidates = new Set([
    streamedText,
    streamedText.trimEnd(),
    streamedText.trimStart(),
    streamedText.trim(),
  ]);
  for (const candidate of candidates) {
    if (candidate && targetText.startsWith(candidate)) {
      return { matched: true, lastMatchEnd: candidate.length };
    }
  }
  // Segments may be interleaved with not-yet-streamed text; locate them in
  // order, trimming the last segment's trailing whitespace.
  const lastIndex = segments.length - 1;
  const searchSegments = segments.map((segment, index) =>
    index === lastIndex ? segment.trimEnd() : segment,
  );
  return findTextSegmentsInOrder(targetText, searchSegments);
};

// Whether the finalized event renders its own reasoning: an ActionEvent via
// reasoning_content/thinking_blocks, an agent MessageEvent via an inline
// <think> block in its content. Decides if a replaced delta's reasoning must
// be preserved separately.
const eventRendersReasoning = (event: OpenHandsEvent): boolean => {
  if (isActionEvent(event)) {
    return getReasoningContent(event).trim().length > 0;
  }

  if (isMessageEvent(event) && event.source === "agent") {
    return (
      splitInlineThink(joinTextBlocks(event.llm_message.content)).reasoning
        .length > 0
    );
  }

  return false;
};

// The final MessageEvent/FinishAction is authoritative for the turn's text. Drop
// the provisional streamed deltas and render the canonical final event instead,
// so the message is rendered exactly once (never holey or duplicated) and its
// metadata — critic_result, activated_microagents — renders too. Stream-only
// reasoning is preserved. Returns null when there is no streamed content to
// reconcile, leaving the caller to append the final event normally.
const finalizeStreamingDeltasInPlace = (
  finalEvent: OpenHandsEvent,
  uiEvents: OpenHandsEvent[],
): OpenHandsEvent[] | null => {
  const contentStreamingDeltas = getCurrentTurnContentDeltas(
    uiEvents,
    finalEvent,
  );
  if (contentStreamingDeltas.length === 0) {
    return null;
  }

  const nextUiEvents = supersedeStreamingContent(
    uiEvents,
    contentStreamingDeltas,
    eventRendersReasoning(finalEvent),
  );
  return releaseDeferredUserMessagesForSender(
    insertBeforeDeferredUserMessages(nextUiEvents, finalEvent),
    finalEvent,
  );
};

/**
 * Reconcile the current turn's streaming delta when an intermediate
 * (tool-calling) `ActionEvent` arrives. With `stream=true` the step's
 * pre-tool-call text is streamed as delta `content`, then the action's
 * `thought` repeats it and the chat hoists that into its own message (see
 * `group-events.ts`), so the text would render twice (issue #1534).
 *
 * The action must stay because it owns the tool call. The streamed text is
 * cleared from the delta, and the delta is kept only to carry reasoning the
 * action itself lacks (for many models the delta is the sole reasoning
 * carrier), otherwise dropped.
 *
 * Only the current step's trailing delta run is considered. Returns the updated
 * array, or `null` when there is nothing to reconcile.
 */
const supersedeStreamedThoughtWithAction = (
  action: ActionEvent,
  uiEvents: OpenHandsEvent[],
): OpenHandsEvent[] | null => {
  const thoughtText = joinTextBlocks(action.thought);
  if (!thoughtText) {
    return null;
  }

  const contentDeltas = getTrailingContentDeltas(uiEvents);
  if (contentDeltas.length === 0) {
    return null;
  }

  const streamingSegments = contentDeltas.map(
    ({ event }) => event.content ?? "",
  );

  // Only strip when the streamed text is the action's rendered thought.
  if (!matchStreamedSegments(thoughtText, streamingSegments).matched) {
    return null;
  }

  // Keeping the delta's reasoning would duplicate the action's own "Thinking".
  return supersedeStreamingContent(
    uiEvents,
    contentDeltas,
    getReasoningContent(action).trim().length > 0,
  );
};

// Drop the current step's streamed reasoning when the action renders that
// reasoning itself, so "Thinking" renders once. Unlike
// `supersedeStreamedThoughtWithAction` this does not require the streamed text
// to match the thought, which models streaming reasoning-only deltas never do.
const supersedeStreamedReasoningWithAction = (
  action: ActionEvent,
  uiEvents: OpenHandsEvent[],
): OpenHandsEvent[] | null => {
  if (getReasoningContent(action).trim().length === 0) {
    return null;
  }

  const reasoningDeltas = getTrailingReasoningDeltas(uiEvents, action);
  if (reasoningDeltas.length === 0) {
    return null;
  }

  const indexesToStrip = new Set(reasoningDeltas.map(({ index }) => index));
  const nextUiEvents: OpenHandsEvent[] = [];
  uiEvents.forEach((event, index) => {
    if (!indexesToStrip.has(index) || !isStreamingDeltaEvent(event)) {
      nextUiEvents.push(event);
      return;
    }

    // Keep the delta only for streamed text the action itself lacks.
    if (event.content) {
      nextUiEvents.push({ ...event, reasoning_content: null });
    }
  });
  return nextUiEvents;
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
  event: OpenHandsEvent,
  uiEvents: OpenHandsEvent[],
): OpenHandsEvent[] => {
  const newUiEvents = [...uiEvents];

  if (isStreamingDeltaEvent(event)) {
    if (event.content === null && event.reasoning_content === null) {
      return newUiEvents;
    }

    const activeTail = findEventBeforeDeferredUserMessages(newUiEvents);
    if (
      activeTail &&
      isStreamingDeltaEvent(activeTail.event) &&
      isSameStreamingSender(event, activeTail.event)
    ) {
      newUiEvents[activeTail.index] = mergeStreamingDeltaEvent(
        event,
        activeTail.event,
      );
      return newUiEvents;
    }

    return insertBeforeDeferredUserMessages(newUiEvents, event);
  }

  const isFinalAgentEvent =
    (isActionEvent(event) && event.action.kind === "FinishAction") ||
    (isMessageEvent(event) && event.source === "agent");

  if (isFinalAgentEvent) {
    const finalizedUiEvents = finalizeStreamingDeltasInPlace(
      event,
      newUiEvents,
    );
    if (finalizedUiEvents) {
      return finalizedUiEvents;
    }
  }

  // Intermediate tool-calling action whose thought was streamed: clear the
  // duplicated text from the delta (issue #1534). ThinkAction is excluded — its
  // thought renders through its own collapsible, not a hoisted thought.
  if (
    isActionEvent(event) &&
    event.action.kind !== "FinishAction" &&
    event.action.kind !== "ThinkAction"
  ) {
    const reconciledUiEvents =
      supersedeStreamedThoughtWithAction(event, newUiEvents) ??
      supersedeStreamedReasoningWithAction(event, newUiEvents);
    if (reconciledUiEvents) {
      return insertBeforeDeferredUserMessages(reconciledUiEvents, event);
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
      return insertBeforeDeferredUserMessages(newUiEvents, event);
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
      return insertBeforeDeferredUserMessages(newUiEvents, event);
    }
  } else {
    const activeTail = findEventBeforeDeferredUserMessages(newUiEvents);
    if (
      isMessageEvent(event) &&
      event.source === "user" &&
      activeTail &&
      isStreamingDeltaEvent(activeTail.event)
    ) {
      newUiEvents.push(deferUserMessage(event, activeTail.event));
    } else {
      const nextUiEvents = insertBeforeDeferredUserMessages(newUiEvents, event);
      return isFinalAgentEvent
        ? releaseDeferredUserMessagesForSender(nextUiEvents, event)
        : nextUiEvents;
    }
  }

  return newUiEvents;
};
