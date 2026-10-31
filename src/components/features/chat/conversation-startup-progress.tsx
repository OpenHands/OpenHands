import { useTranslation } from "react-i18next";
import { LoaderCircle } from "lucide-react";
import { GenericEventMessage } from "./generic-event-message";
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
  const isRunning = !error && status !== "READY" && status !== "ERROR";
  return (
    <div data-testid="conversation-startup-progress">
      <GenericEventMessage
        title={
          error ? (
            <span
              role="alert"
              className="whitespace-pre-wrap break-words text-status-error"
            >
              {error}
            </span>
          ) : (
            <span role="status" aria-busy={isRunning}>
              {t(
                status
                  ? getTaskStatusI18nKey(status)
                  : I18nKey.HOME$CREATING_CONVERSATION,
              )}
            </span>
          )
        }
        details=""
        titleTrailing={
          isRunning ? (
            <LoaderCircle
              aria-hidden="true"
              className="h-4 w-4 ml-2 inline animate-spin text-muted"
            />
          ) : undefined
        }
      />
      {error && (onEdit || onRetry) && (
        <div className="flex flex-wrap items-center gap-2 pb-2">
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
      )}
    </div>
  );
}
