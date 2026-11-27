import { MousePointerClick } from "lucide-react";
import { useTranslation } from "react-i18next";
import { ConversationTabEmptyState } from "#/components/features/conversation/conversation-tab-empty-state";
import { I18nKey } from "#/i18n/declaration";

export function NoFileSelectedMessage() {
  const { t } = useTranslation("openhands");

  return (
    <ConversationTabEmptyState
      className="h-full"
      icon={
        <MousePointerClick aria-hidden strokeWidth={2} className="size-full" />
      }
    >
      {t(I18nKey.FILES$NO_FILE_SELECTED)}
    </ConversationTabEmptyState>
  );
}
