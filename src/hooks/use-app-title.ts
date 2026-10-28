import { useParams } from "react-router";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { getAgentStateEmoji } from "#/utils/agent-state-emoji";
import { useCanvasExtensionsRuntime } from "#/components/features/canvas-extensions/canvas-extensions-runtime";

const APP_TITLE = "OpenHands";
const APP_EXTENSION_SUFFIX = " - OpenHands App";

export const useAppTitle = () => {
  const params = useParams<{
    conversationId?: string;
    extensionName?: string;
    "*"?: string;
  }>();
  const conversationId = params.conversationId;
  const extensionName = params.extensionName;
  const routePath = params["*"] ?? "";
  const { pages } = useCanvasExtensionsRuntime();

  const { data: conversation } = useUserConversation(conversationId ?? null);
  const liveExecutionStatus = useConversationStateStore((state) =>
    conversationId
      ? (state.executionStatusByConversation[conversationId] ?? null)
      : null,
  );

  if (conversationId) {
    const conversationTitle = conversation?.title;
    const baseTitle = conversationTitle
      ? `${conversationTitle} | ${APP_TITLE}`
      : APP_TITLE;

    const executionStatus =
      liveExecutionStatus ?? conversation?.execution_status ?? null;
    const emoji = getAgentStateEmoji(executionStatus);

    return emoji ? `${emoji} ${baseTitle}` : baseTitle;
  }

  if (extensionName) {
    const matchedExtensionPage = pages
      .filter(
        (candidate) =>
          candidate.extension.enabled &&
          candidate.extension.name === extensionName,
      )
      .sort(
        (left, right) =>
          right.contribution.path.length - left.contribution.path.length,
      )
      .find(
        (candidate) =>
          routePath === candidate.contribution.path ||
          routePath.startsWith(`${candidate.contribution.path}/`),
      );

    if (matchedExtensionPage?.contribution.title) {
      return `${matchedExtensionPage.contribution.title}${APP_EXTENSION_SUFFIX}`;
    }
  }

  return APP_TITLE;
};
