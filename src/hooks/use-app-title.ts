import { useMatches, useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { getAgentStateEmoji } from "#/utils/agent-state-emoji";
import {
  selectCanvasExtensionPage,
  useCanvasExtensionsRuntime,
} from "#/components/features/canvas-extensions/canvas-extensions-runtime";
import {
  getInterfaceCopy,
  hasAutomationInterface,
} from "#/manifests/automation-interface";
import type { RouteTitleHandle } from "#/routes/title-handle";

const APP_TITLE = "OpenHands";

// @spec AWI-001 — Localized matched-route titles with stable conversation status
export const useAppTitle = () => {
  const matches = useMatches();
  const { t } = useTranslation("openhands");
  const { pages } = useCanvasExtensionsRuntime();
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
    const match = [...matches].reverse().find(({ handle }) => {
      const titleHandle = handle as RouteTitleHandle | undefined;
      return titleHandle?.titleKey || titleHandle?.titleSource;
    });
    const handle = match?.handle as RouteTitleHandle | undefined;
    let sectionTitle: string | undefined;
    if (handle?.titleSource === "automation") {
      sectionTitle = hasAutomationInterface()
        ? getInterfaceCopy().sidebarLabel
        : undefined;
    } else if (handle?.titleSource === "extension") {
      sectionTitle = selectCanvasExtensionPage(
        pages,
        match?.params.extensionName ?? "",
        match?.params["*"] ?? "",
      )?.contribution.title;
    } else if (handle?.titleKey) {
      sectionTitle = t(handle.titleKey);
    }
    return sectionTitle ? `${sectionTitle} | ${APP_TITLE}` : APP_TITLE;
  }

  const executionStatus =
    liveExecutionStatus ?? conversation?.execution_status ?? null;
  const emoji = getAgentStateEmoji(executionStatus);

  return emoji ? `${emoji} ${baseTitle}` : baseTitle;
};
