import { useParams } from "react-router";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { useEventStore } from "#/stores/use-event-store";
import { latestExecutionStatusFromEvents } from "#/utils/status";
import { getAgentStateEmoji } from "#/utils/agent-state-emoji";

const APP_TITLE = "OpenHands";

export const useAppTitle = () => {
  const { conversationId } = useParams<{ conversationId: string }>();
  const { data: conversation } = useUserConversation(conversationId ?? null);
  const liveExecutionStatus = useConversationStateStore((state) =>
    conversationId
      ? (state.executionStatusByConversation[conversationId] ?? null)
      : null,
  );
  // See use-agent-state.ts: the REST-preloaded event tail can carry the run's
  // `running` conversation-state update before any live WebSocket update does,
  // so read it between the live store and the REST conversation fallback.
  const preloadedExecutionStatus = useEventStore((state) =>
    conversationId && state.loadedConversationId === conversationId
      ? latestExecutionStatusFromEvents(state.events)
      : null,
  );

  const conversationTitle = conversation?.title;
  const baseTitle =
    conversationId && conversationTitle
      ? `${conversationTitle} | ${APP_TITLE}`
      : APP_TITLE;

  if (!conversationId) {
    return baseTitle;
  }

  const executionStatus =
    liveExecutionStatus ??
    preloadedExecutionStatus ??
    conversation?.execution_status ??
    null;
  const emoji = getAgentStateEmoji(executionStatus);

  return emoji ? `${emoji} ${baseTitle}` : baseTitle;
};
