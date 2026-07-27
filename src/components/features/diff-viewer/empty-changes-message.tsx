import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ConversationTabEmptyState } from "#/components/features/conversation/conversation-tab-empty-state";
import { DiffDrawerIcon } from "./diff-drawer-icon";

export function EmptyChangesMessage() {
  const { t } = useTranslation("openhands");

  return (
    <ConversationTabEmptyState icon={<DiffDrawerIcon />}>
      {t(I18nKey.DIFF_VIEWER$NO_CHANGES)}
    </ConversationTabEmptyState>
  );
}
