import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

interface ConversationListErrorStateProps {
  onRetry: () => void;
}

export function ConversationListErrorState({
  onRetry,
}: ConversationListErrorStateProps) {
  const { t } = useTranslation();

  return (
    <div
      data-testid="conversation-list-error-state"
      className="flex flex-col items-center justify-center gap-3 h-full px-4 py-6"
    >
      <p className="text-sm text-neutral-400 text-center">
        {t(I18nKey.CONVERSATION$FAILED_TO_LOAD_CONVERSATIONS)}
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="text-xs text-white underline hover:text-neutral-300"
      >
        {t(I18nKey.CONVERSATION$RETRY)}
      </button>
    </div>
  );
}
