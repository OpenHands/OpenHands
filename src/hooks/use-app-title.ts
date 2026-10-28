import { matchPath, useLocation, useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { getAgentStateEmoji } from "#/utils/agent-state-emoji";

const APP_TITLE = "OpenHands";

const SECTION_TITLES = [
  { path: "/", key: I18nKey.BUTTON$HOME },
  { path: "/conversations", key: I18nKey.SIDEBAR$CONVERSATIONS },
  { path: "/automations/*", key: I18nKey.FEATURED_AUTOMATIONS$SECTION_TITLE },
  { path: "/settings/*", key: I18nKey.SETTINGS$TITLE },
];

// @spec AWI-001 — Localized section titles with stable conversation status
export const useAppTitle = () => {
  const { pathname } = useLocation();
  const { t } = useTranslation("openhands");
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
    const section = SECTION_TITLES.find(({ path }) =>
      matchPath(path, pathname),
    );
    return section ? `${t(section.key)} | ${APP_TITLE}` : APP_TITLE;
  }

  const executionStatus =
    liveExecutionStatus ?? conversation?.execution_status ?? null;
  const emoji = getAgentStateEmoji(executionStatus);

  return emoji ? `${emoji} ${baseTitle}` : baseTitle;
};
