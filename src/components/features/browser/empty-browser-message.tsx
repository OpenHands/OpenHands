import { useTranslation } from "react-i18next";
import { IoIosGlobe } from "react-icons/io";
import { I18nKey } from "#/i18n/declaration";
import { ConversationTabEmptyState } from "#/components/features/conversation/conversation-tab-empty-state";

export function EmptyBrowserMessage() {
  const { t } = useTranslation("openhands");

  return (
    <ConversationTabEmptyState icon={<IoIosGlobe />}>
      {t(I18nKey.BROWSER$NO_PAGE_LOADED)}
    </ConversationTabEmptyState>
  );
}
