import { useTranslation } from "react-i18next";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { I18nKey } from "#/i18n/declaration";
import type { AppConversationStartTaskStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { getTaskStatusI18nKey } from "#/utils/status";

interface Props {
  status?: AppConversationStartTaskStatus;
  error?: string | null;
  onEdit?: () => void;
  editLabel?: string;
  onRetry?: () => void;
}

/** Show server-reported progress without guessing completed or skipped steps. */
export function ConversationStartupProgress({
  status,
  error,
  onEdit,
  editLabel,
  onRetry,
}: Props) {
  const { t } = useTranslation("openhands");
  return (
    <div
      data-testid="conversation-startup-progress"
      className="my-4 rounded-xl border border-border p-4 text-sm text-text-secondary"
    >
      {error ? (
        <div className="flex flex-col items-start gap-3">
          <p
            role="alert"
            className="whitespace-pre-wrap break-words text-status-error"
          >
            {error}
          </p>
          {onEdit && (
            <button
              type="button"
              className="button-base rounded-md border border-border px-3 py-1.5"
              onClick={onEdit}
            >
              {editLabel ?? t(I18nKey.BUTTON$EDIT)}
            </button>
          )}
          {onRetry && (
            <button
              type="button"
              className="button-base rounded-md border border-border px-3 py-1.5"
              onClick={onRetry}
            >
              {t(I18nKey.CHAT_INTERFACE$MESSAGE_RETRY)}
            </button>
          )}
        </div>
      ) : (
        <div role="status" className="flex items-center gap-3">
          <LoadingSpinner size="small" />
          <span>
            {t(
              status
                ? getTaskStatusI18nKey(status)
                : I18nKey.HOME$CREATING_CONVERSATION,
            )}
          </span>
        </div>
      )}
    </div>
  );
}
