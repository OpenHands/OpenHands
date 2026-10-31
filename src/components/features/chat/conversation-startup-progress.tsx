import { useState } from "react";
import { useTranslation } from "react-i18next";
import { LoaderCircle } from "lucide-react";
import { GenericEventMessage } from "./generic-event-message";
import { CollapsibleEventGroup } from "./collapsible-event-group";
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
  const label = status
    ? getTaskStatusI18nKey(status)
    : I18nKey.HOME$CREATING_CONVERSATION;
  const [observedSteps, setObservedSteps] = useState([label]);
  if (!observedSteps.includes(label)) {
    setObservedSteps([...observedSteps, label]);
  }
  return (
    <div data-testid="conversation-startup-progress">
      <CollapsibleEventGroup
        testId="startup-progress-group"
        title={
          <span role="status" aria-busy={isRunning}>
            {t(error ? I18nKey.COMMON$ERROR : label)}
          </span>
        }
        expandLabel={t(I18nKey.BUTTON$EXPAND)}
        collapseLabel={t(I18nKey.BUTTON$COLLAPSE)}
        trailing={
          isRunning ? (
            <LoaderCircle
              aria-hidden="true"
              className="h-4 w-4 ml-2 inline animate-spin text-muted"
            />
          ) : undefined
        }
      >
        {observedSteps.map((step) => (
          <GenericEventMessage key={step} title={t(step)} details="" />
        ))}
      </CollapsibleEventGroup>
      {error && (
        <p
          role="alert"
          className="whitespace-pre-wrap break-words text-sm text-status-error"
        >
          {error}
        </p>
      )}
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
