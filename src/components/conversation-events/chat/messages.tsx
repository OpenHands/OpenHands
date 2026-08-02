import React from "react";
import { OpenHandsEvent } from "#/types/agent-server/core";
import { EventMessage } from "./event-message";
import { usePlanPreviewEvents } from "./hooks/use-plan-preview-events";
import { groupEvents } from "./group-events";
import { EventGroup } from "./event-message-components/event-group";
import { ThoughtEventMessage } from "./event-message-components/thought-event-message";
import { useModelStore } from "#/stores/model-store";
import { ModelMessages } from "#/components/features/chat/model-messages";
import { SlashCommandMessages } from "#/components/features/chat/slash-command-messages";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import { resolveSlashCommandOutputPlacements } from "./slash-command-output-placement";
// TODO: Implement microagent functionality for V1 when APIs support V1 event IDs
// import { AgentState } from "#/types/agent-state";
// import MemoryIcon from "#/icons/memory_icon.svg?react";

interface MessagesProps {
  messages: OpenHandsEvent[]; // UI events (actions replaced by observations)
  allEvents: OpenHandsEvent[]; // Full event history (for action lookup)
}

const getLastEventId = (events: OpenHandsEvent[]) => events.at(-1)?.id;
const getLastEvent = (events: OpenHandsEvent[]) => events.at(-1);

export const Messages: React.FC<MessagesProps> = React.memo(
  ({ messages, allEvents }) => {
    const { conversationId } = useOptionalConversationId();
    // Get the set of event IDs that should render PlanPreview
    // This ensures only one preview per user message "phase"
    const planPreviewEventIds = usePlanPreviewEvents(allEvents);

    // Set of event ids that have a /model entry anchored to them — used to
    // avoid mounting <ModelMessages> for every event (the component would
    // otherwise early-return null).
    const modelEntries = useModelStore((s) =>
      conversationId ? s.entriesByConversation[conversationId] : undefined,
    );
    const modelAnchorIds = React.useMemo(() => {
      if (!modelEntries || modelEntries.length === 0) return null;
      const ids = new Set<string>();
      for (const entry of modelEntries) {
        if (entry.anchorEventId !== null) ids.add(entry.anchorEventId);
      }
      return ids.size > 0 ? ids : null;
    }, [modelEntries]);
    const slashCommandEntries = useSlashCommandOutputStore((state) =>
      conversationId ? state.entriesByScope[conversationId] : undefined,
    );
    const slashCommandPlacements = React.useMemo(
      () =>
        resolveSlashCommandOutputPlacements(
          slashCommandEntries ?? [],
          allEvents,
          messages,
        ),
      [slashCommandEntries, allEvents, messages],
    );

    const maybeRenderModelMessages = (eventId: string | number | undefined) => {
      if (!modelAnchorIds || eventId === undefined) return null;
      const key = String(eventId);
      if (!modelAnchorIds.has(key)) return null;
      return (
        <ModelMessages conversationId={conversationId} anchorEventId={key} />
      );
    };
    const renderSlashCommandMessages = (
      entries: NonNullable<typeof slashCommandEntries>,
    ) => {
      if (entries.length === 0) return null;
      return (
        <SlashCommandMessages
          outputScopeId={conversationId}
          outputs={entries}
        />
      );
    };

    // Fold consecutive action/observation events into collapsible groups so a
    // long sequence of tool calls doesn't dominate the chat scroll. Items that
    // can't be grouped (or that fall in a short run) are still rendered one by
    // one, identically to before. Agent thoughts attached to an action are
    // hoisted out as their own rendered item so they always show up in the
    // message pane and a thought between actions starts a fresh group.
    const renderedItems = React.useMemo(
      () =>
        groupEvents(
          messages,
          undefined,
          allEvents,
          slashCommandPlacements.breakBeforeEventIds,
        ),
      [messages, allEvents, slashCommandPlacements.breakBeforeEventIds],
    );

    const getRenderedItemStartEventId = (
      item: (typeof renderedItems)[number],
    ) =>
      String(
        item.kind === "single"
          ? item.event.id
          : item.kind === "thought"
            ? item.action.id
            : item.events[0]?.id,
      );

    const firstRenderedItemIndexByEventId = new Map<string, number>();
    renderedItems.forEach((item, index) => {
      const eventId = getRenderedItemStartEventId(item);
      if (!firstRenderedItemIndexByEventId.has(eventId)) {
        firstRenderedItemIndexByEventId.set(eventId, index);
      }
    });

    const renderEventMessage = (
      event: OpenHandsEvent,
      index: number,
      suppressThought: boolean,
    ) => (
      <EventMessage
        key={event.id}
        event={event}
        messages={allEvents}
        isLastMessage={messages.length - 1 === index}
        isInLast10Actions={messages.length - 1 - index < 10}
        planPreviewEventIds={planPreviewEventIds}
        suppressThought={suppressThought}
      />
    );

    return (
      <>
        {renderedItems.map((item, itemIndex) => {
          const itemStartEventId = getRenderedItemStartEventId(item);
          const slashCommandOutputBeforeItem = renderSlashCommandMessages(
            firstRenderedItemIndexByEventId.get(itemStartEventId) === itemIndex
              ? (slashCommandPlacements.entriesBeforeEvent.get(
                  itemStartEventId,
                ) ?? [])
              : [],
          );

          if (item.kind === "single") {
            return (
              <React.Fragment key={`single-${item.event.id}`}>
                {slashCommandOutputBeforeItem}
                {/* Thoughts for singles are also hoisted as their own
                    "thought" item, so suppress the inline render to avoid
                    duplication. */}
                {renderEventMessage(item.event, item.index, true)}
                {maybeRenderModelMessages(item.event.id)}
              </React.Fragment>
            );
          }

          if (item.kind === "thought") {
            return (
              <React.Fragment key={`thought-${item.action.id}`}>
                {slashCommandOutputBeforeItem}
                <ThoughtEventMessage event={item.action} />
                {maybeRenderModelMessages(item.action.id)}
              </React.Fragment>
            );
          }

          // A group is "finalized" once another rendered item appears after
          // it, signalling the agent has moved on. While the group is still
          // the live tail, it keeps showing the latest action title as its
          // prominent summary.
          const isFinalized = itemIndex < renderedItems.length - 1;
          const groupKey = item.events[0]?.id ?? `group-${item.startIndex}`;
          return (
            <React.Fragment key={`group-${groupKey}`}>
              {slashCommandOutputBeforeItem}
              <EventGroup
                events={item.events}
                allEvents={allEvents}
                isFinalized={isFinalized}
              >
                {item.events.map((event, offset) =>
                  renderEventMessage(event, item.startIndex + offset, true),
                )}
              </EventGroup>
              {item.events.map((event) => (
                <React.Fragment key={`client-command-${event.id}`}>
                  {maybeRenderModelMessages(event.id)}
                </React.Fragment>
              ))}
            </React.Fragment>
          );
        })}
        {renderSlashCommandMessages(slashCommandPlacements.tailEntries)}
        {renderSlashCommandMessages(
          slashCommandPlacements.unresolvedActiveEntries,
        )}
      </>
    );
  },
  (prevProps, nextProps) =>
    prevProps.messages.length === nextProps.messages.length &&
    prevProps.allEvents.length === nextProps.allEvents.length &&
    getLastEventId(prevProps.messages) === getLastEventId(nextProps.messages) &&
    getLastEventId(prevProps.allEvents) ===
      getLastEventId(nextProps.allEvents) &&
    getLastEvent(prevProps.messages) === getLastEvent(nextProps.messages) &&
    getLastEvent(prevProps.allEvents) === getLastEvent(nextProps.allEvents),
);

Messages.displayName = "Messages";
