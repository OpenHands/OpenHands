import { useTranslation } from "react-i18next";
import { ConversationTabEmptyState } from "#/components/features/conversation/conversation-tab-empty-state";
import { I18nKey } from "#/i18n/declaration";
import { SquareChevronRight } from "lucide-react";

export function EmptyTerminalMessage() {
  const { t } = useTranslation("openhands");

  return (
    <ConversationTabEmptyState
      className="h-full"
      icon={
        <SquareChevronRight aria-hidden strokeWidth={2} className="size-full" />
      }
    >
      {t(I18nKey.TERMINAL$NO_OUTPUT)}
    </ConversationTabEmptyState>
  );
}
