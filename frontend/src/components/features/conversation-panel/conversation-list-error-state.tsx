import { RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface ConversationListErrorStateProps {
  className?: string;
  isRetrying?: boolean;
  onRetry: () => void;
}

export function ConversationListErrorState({
  className,
  isRetrying = false,
  onRetry,
}: ConversationListErrorStateProps) {
  const { t } = useTranslation();

  return (
    <div
      role="alert"
      data-testid="conversation-list-error"
      className={cn(
        "flex h-full flex-col items-center justify-center gap-3 px-6 text-center",
        className,
      )}
    >
      <p className="text-sm font-medium text-neutral-200">
        {t(I18nKey.ERROR$GENERIC)}
      </p>
      <button
        type="button"
        onClick={onRetry}
        disabled={isRetrying}
        aria-label={t(I18nKey.BUTTON$REFRESH)}
        className="inline-flex items-center gap-2 rounded-md border border-neutral-600 px-3 py-2 text-xs font-medium text-neutral-100 transition-colors hover:bg-neutral-700 disabled:cursor-not-allowed disabled:opacity-60"
      >
        <RefreshCw size={14} className={isRetrying ? "animate-spin" : ""} />
        <span>
          {isRetrying ? t(I18nKey.HOME$LOADING) : t(I18nKey.BUTTON$REFRESH)}
        </span>
      </button>
    </div>
  );
}
