import { useMatch, useParams } from "react-router";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { getAgentStateEmoji } from "#/utils/agent-state-emoji";
import {
  selectCanvasExtensionPage,
  useCanvasExtensionsRuntime,
} from "#/components/features/canvas-extensions/canvas-extensions-runtime";

const APP_TITLE = "OpenHands";

export const useAppTitle = () => {
  const appRoute = useMatch("/extensions/:extensionName/*");
  const { pages } = useCanvasExtensionsRuntime();
  const appPage = selectCanvasExtensionPage(
    pages,
    appRoute?.params.extensionName ?? "",
    appRoute?.params["*"] ?? "",
  );
  const { conversationId } = useParams<{ conversationId: string }>();
  const { data: conversation } = useUserConversation(conversationId ?? null);
  const liveExecutionStatus = useConversationStateStore((state) =>
    conversationId
      ? (state.executionStatusByConversation[conversationId] ?? null)
      : null,
  );

  const conversationTitle = conversation?.title;
  const baseTitle =
    conversationId && conversationTitle
      ? `${conversationTitle} | ${APP_TITLE}`
      : APP_TITLE;

  if (!conversationId) {
    return appPage ? `${appPage.contribution.title} | ${APP_TITLE}` : baseTitle;
  }

  const executionStatus =
    liveExecutionStatus ?? conversation?.execution_status ?? null;
  const emoji = getAgentStateEmoji(executionStatus);

  return emoji ? `${emoji} ${baseTitle}` : baseTitle;
};
