import { useSessionActivities } from "#/hooks/use-session-activities";
import { useSessionActivityStore } from "#/stores/session-activity-store";
import { SessionActivityDetails } from "./event-message-components/session-activity";
import {
  CollapsibleThinking,
  ThinkingContent,
} from "./event-message-components/collapsible-thinking";
import React from "react";
import { useTranslation } from "react-i18next";
import { LoaderCircle } from "lucide-react";
import { OpenHandsEvent } from "#/types/agent-server/core";
import { I18nKey } from "#/i18n/declaration";
import { useAgentState, usePlanningAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import {
  isActionEvent,
  isObservationEvent,
  isStreamingDeltaEvent,
  isUserRejectObservation,
} from "#/types/agent-server/type-guards";
import { EventMessage } from "./event-message";
import { usePlanPreviewEvents } from "./hooks/use-plan-preview-events";
import { groupEvents } from "./group-events";
import { buildActionById } from "./event-thought-helpers";
import { getSingleToolEvent } from "./event-message-components/event-group-summary";
import { EventGroup } from "./event-message-components/event-group";
import { ThoughtEventMessage } from "./event-message-components/thought-event-message";
import { getStepKey, StepDisclosureContext } from "./step-disclosure-context";
import { useModelStore } from "#/stores/model-store";
import { ModelMessages } from "#/components/features/chat/model-messages";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { ConversationConfirmationButtons } from "#/components/shared/buttons/conversation-confirmation-buttons";
import type { SecretRequest } from "#/stores/secret-request-store";
import {
  SecretRequestCard,
  usePendingSecretRequests,
} from "#/components/features/conversation/secret-request-card";

interface MessagesProps {
  messages: OpenHandsEvent[]; // UI events (actions replaced by observations)
  allEvents: OpenHandsEvent[]; // Full event history (for action lookup)
  hasOlderEvents?: boolean;
  pendingMessages?: React.ReactNode;
  pendingSessionId?: string;
}

const getLastEventId = (events: OpenHandsEvent[]) => events.at(-1)?.id;
const getLastEvent = (events: OpenHandsEvent[]) => events.at(-1);

export const Messages: React.FC<MessagesProps> = React.memo(
  ({
    messages,
    allEvents,
    hasOlderEvents = false,
    pendingMessages,
    pendingSessionId,
  }) => {
    const { conversationId } = useOptionalConversationId();
    const { key: sessionKey, sessions } = useSessionActivities(conversationId);
    const visibleSessions = React.useMemo(
      () =>
        sessions.filter(
          (session) =>
            session.phase === "pending" ||
            session.phase === "connecting" ||
            !hasOlderEvents ||
            (session.kind === "resume" &&
              (messages.some((event) => event.id === session.afterEventId) ||
                session.startedAt >= Date.parse(messages[0]?.timestamp ?? ""))),
        ),
      [sessions, hasOlderEvents, messages],
    );
    // Get the set of event IDs that should render PlanPreview
    // This ensures only one preview per user message "phase"
    const planPreviewEventIds = usePlanPreviewEvents(allEvents);

    // Which rows and groups the user has opened or closed, keyed by tool call.
    // Rows remount when a running action is replaced by its observation or a
    // lone step becomes part of a group, so their state lives here instead.
    const [openRows, setOpenRows] = React.useState<
      ReadonlyMap<string, boolean>
    >(() => new Map());
    React.useEffect(() => setOpenRows(new Map()), [conversationId]);
    const setRowOpen = React.useCallback((key: string, open: boolean) => {
      setOpenRows((prev) => new Map(prev).set(key, open));
    }, []);

    // EventMessage used to receive the complete allEvents array and the plan
    // Set, so every append changed every historical item's shallow props.
    // Derive the two event-specific values once and keep unchanged wrappers
    // eligible for React.memo without removing allEvents from grouping logic.
    const actionById = React.useMemo(
      () => buildActionById(allEvents),
      [allEvents],
    );

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

    const maybeRenderModelMessages = (eventId: string | number | undefined) => {
      if (!modelAnchorIds || eventId === undefined) return null;
      const key = String(eventId);
      if (!modelAnchorIds.has(key)) return null;
      return (
        <ModelMessages conversationId={conversationId} anchorEventId={key} />
      );
    };

    // Pending `request_secret` forms sit right after their tool call, so what
    // the agent writes next lands below them. A call that is not loaded falls
    // back to the end of the chat.
    const pendingSecretRequests = usePendingSecretRequests();
    const secretRequests = React.useMemo(() => {
      const unanchored = new Map(
        pendingSecretRequests.map((request) => [request.toolCallId, request]),
      );
      const byAnchorId = new Map<string, SecretRequest>();
      for (const event of messages) {
        if (unanchored.size === 0) break;
        if (!isActionEvent(event) && !isObservationEvent(event)) continue;
        const request = unanchored.get(event.tool_call_id);
        if (!request) continue;
        unanchored.delete(event.tool_call_id);
        byAnchorId.set(event.id, request);
      }
      return { byAnchorId, unanchored: [...unanchored.values()] };
    }, [pendingSecretRequests, messages]);

    const maybeRenderSecretRequest = (eventId: string | undefined) => {
      const request = eventId && secretRequests.byAnchorId.get(eventId);
      return request ? <SecretRequestCard request={request} /> : null;
    };

    // One activity group lasts until visible agent text or an actionable event.
    const { t } = useTranslation("openhands");
    const { curAgentState } = useAgentState();
    const { isPlanningAgentRunning } = usePlanningAgentState();
    const isAgentRunning =
      curAgentState === AgentState.RUNNING || isPlanningAgentRunning;
    const renderedItems = React.useMemo(
      () => groupEvents(messages, 1, allEvents, 1, visibleSessions, actionById),
      [messages, allEvents, visibleSessions, actionById],
    );
    const eventIndexes = React.useMemo(
      () => new Map(messages.map((event, index) => [event, index])),
      [messages],
    );
    const groupedReasoningIds = new Set(
      renderedItems.flatMap((item) =>
        item.kind === "group"
          ? (item.reasoning ?? []).map((reasoning) => reasoning.id)
          : [],
      ),
    );
    const showThinkingLine =
      isAgentRunning &&
      renderedItems.at(-1)?.kind !== "group" &&
      !(
        messages.length && isStreamingDeltaEvent(messages[messages.length - 1])
      );

    const renderEventMessage = (
      event: OpenHandsEvent,
      index: number,
      suppressThought: boolean,
      detailsOnly = false,
    ) => {
      const stepKey = getStepKey(event);
      return (
        <StepDisclosureContext.Provider
          key={stepKey}
          value={{
            open: openRows.get(stepKey),
            detailsOnly,
            setOpen: (open) => setRowOpen(stepKey, open),
          }}
        >
          <EventMessage
            event={event}
            correspondingAction={
              (isObservationEvent(event) || isUserRejectObservation(event)) &&
              event.action_id
                ? (actionById.get(event.action_id) ?? null)
                : null
            }
            isLastMessage={messages.length - 1 === index}
            isInLast10Actions={messages.length - 1 - index < 10}
            showPlanPreview={
              event.id ? planPreviewEventIds.has(event.id) : false
            }
            suppressThought={suppressThought}
            suppressReasoning={groupedReasoningIds.has(
              isObservationEvent(event) || isUserRejectObservation(event)
                ? event.action_id
                : (event.id ?? ""),
            )}
          />
        </StepDisclosureContext.Provider>
      );
    };

    return (
      <>
        {renderedItems.map((item, itemIndex) => {
          if (item.kind === "single") {
            return (
              <React.Fragment key={`single-${getStepKey(item.event)}`}>
                {/* Thoughts for singles are also hoisted as their own
                    "thought" item, so suppress the inline render to avoid
                    duplication. */}
                {renderEventMessage(item.event, item.index, true)}
                {maybeRenderModelMessages(item.event.id)}
                {maybeRenderSecretRequest(item.event.id)}
              </React.Fragment>
            );
          }

          if (item.kind === "thought") {
            return (
              <React.Fragment key={`thought-${item.action.id}`}>
                <ThoughtEventMessage event={item.action} />
                {maybeRenderModelMessages(item.action.id)}
              </React.Fragment>
            );
          }

          const singleTool = getSingleToolEvent(item.events);
          const showSingleToolDetails =
            item.events.length === 1 &&
            !item.session &&
            !item.reasoning?.length;
          const groupKey =
            item.session?.id ??
            (item.reasoning?.[0] && item.reasoning[0].index <= item.startIndex
              ? item.reasoning[0].id
              : item.events[0]
                ? isObservationEvent(item.events[0]) ||
                  isUserRejectObservation(item.events[0])
                  ? item.events[0].action_id
                  : item.events[0].id
                : `group-${item.startIndex}`);
          // A group that forms around a step the user opened starts open, so
          // that step stays visible.
          const groupOpen =
            item.session?.open ??
            openRows.get(`group:${groupKey}`) ??
            item.events.some((event) => openRows.get(getStepKey(event)));
          const Reasoning =
            item.events.length === 0 && !item.session
              ? ThinkingContent
              : CollapsibleThinking;
          return (
            <React.Fragment key={`group-${groupKey}`}>
              {pendingSessionId &&
                item.session?.id === pendingSessionId &&
                pendingMessages}
              <EventGroup
                events={item.events}
                session={item.session}
                hasReasoning={!!item.reasoning?.length}
                allEvents={allEvents}
                actionById={actionById}
                isLive={
                  isAgentRunning && itemIndex === renderedItems.length - 1
                }
                open={groupOpen}
                onOpenChange={(open) => {
                  if (item.session)
                    useSessionActivityStore
                      .getState()
                      .setOpen(sessionKey, item.session.id, open);
                  else setRowOpen(`group:${groupKey}`, open);
                }}
              >
                {item.session && (
                  <SessionActivityDetails
                    session={item.session}
                    nested={item.events.length > 0}
                  />
                )}
                {[
                  ...item.events.map((event) => ({
                    index: eventIndexes.get(event)!,
                    order: 1,
                    node: renderEventMessage(
                      event,
                      eventIndexes.get(event)!,
                      true,
                      event === singleTool && showSingleToolDetails,
                    ),
                  })),
                  ...(item.reasoning ?? []).map((reasoning) => ({
                    index: reasoning.index,
                    order: 0,
                    node: (
                      <Reasoning
                        key={`reasoning-${reasoning.id}`}
                        content={reasoning.text}
                      />
                    ),
                  })),
                ]
                  .sort((a, b) => a.index - b.index || a.order - b.order)
                  .map((entry) => entry.node)}
              </EventGroup>
              {item.events.map((event) => (
                <React.Fragment key={`model-${event.id}`}>
                  {maybeRenderModelMessages(event.id)}
                  {maybeRenderSecretRequest(event.id)}
                </React.Fragment>
              ))}
            </React.Fragment>
          );
        })}
        {!(
          pendingSessionId &&
          renderedItems.some(
            (item) =>
              item.kind === "group" && item.session?.id === pendingSessionId,
          )
        ) && pendingMessages}
        {showThinkingLine && (
          <div
            className="mt-2 flex items-center gap-2 py-1 text-sm text-muted"
            data-testid="live-thinking"
            role="status"
          >
            <span>{t(I18nKey.ACTION_MESSAGE$THINK)}</span>
            <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
          </div>
        )}
        {secretRequests.unanchored.map((request) => (
          <SecretRequestCard key={request.toolCallId} request={request} />
        ))}
        <ConversationConfirmationButtons />
      </>
    );
  },
  (prevProps, nextProps) =>
    prevProps.pendingMessages === nextProps.pendingMessages &&
    prevProps.pendingSessionId === nextProps.pendingSessionId &&
    prevProps.hasOlderEvents === nextProps.hasOlderEvents &&
    prevProps.messages.length === nextProps.messages.length &&
    prevProps.allEvents.length === nextProps.allEvents.length &&
    getLastEventId(prevProps.messages) === getLastEventId(nextProps.messages) &&
    getLastEventId(prevProps.allEvents) ===
      getLastEventId(nextProps.allEvents) &&
    getLastEvent(prevProps.messages) === getLastEvent(nextProps.messages) &&
    getLastEvent(prevProps.allEvents) === getLastEvent(nextProps.allEvents),
);

Messages.displayName = "Messages";
