import { useNavigation } from "#/context/navigation-context";
import { useConversationStateStore } from "#/stores/conversation-state-store";
import { useTranslation } from "react-i18next";
import RuckusMark from "#/assets/branding/ruckus-mark.svg?react";
import { NavigationLink } from "#/components/shared/navigation-link";
import { I18nKey } from "#/i18n/declaration";

export function RuckusBrand() {
  const { t } = useTranslation("openhands");
  const { conversationId } = useNavigation();
  const isRunning = useConversationStateStore((state) =>
    conversationId
      ? state.executionStatusByConversation[conversationId] === "running"
      : false,
  );
  return (
    <NavigationLink
      to="/conversations"
      className="ruckus-brand"
      data-running={isRunning || undefined}
      aria-label={t(I18nKey.BRANDING$OPENHANDS_LOGO)}
    >
      <RuckusMark aria-hidden="true" />
      <span>{t(I18nKey.BRANDING$OPENHANDS)}</span>
    </NavigationLink>
  );
}
