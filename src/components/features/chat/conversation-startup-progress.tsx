import { useId, useState } from "react";
import { useTranslation } from "react-i18next";
import { LoaderCircle } from "lucide-react";
import { DisclosureCaret } from "#/components/shared/disclosure-caret";
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
  const [expanded, setExpanded] = useState(false);
  const contentId = useId();
  const buttonId = `${contentId}-toggle`;
  const expandable = observedSteps.length > 1;
  const title = (
    <span role="status" aria-busy={isRunning} className="truncate">
      {t(error ? I18nKey.COMMON$ERROR : label)}
    </span>
  );
  const spinner = isRunning ? (
    <LoaderCircle
      aria-hidden="true"
      className="h-3.5 w-3.5 shrink-0 animate-spin"
    />
  ) : null;
  return (
    <div data-testid="conversation-startup-progress">
      <div
        className="my-1 w-full py-1 text-sm text-muted"
        data-testid="startup-progress-group"
      >
        {expandable ? (
          <button
            id={buttonId}
            type="button"
            onClick={() => setExpanded((previous) => !previous)}
            aria-controls={contentId}
            aria-expanded={expanded}
            aria-label={t(
              expanded ? I18nKey.BUTTON$COLLAPSE : I18nKey.BUTTON$EXPAND,
            )}
            className="flex max-w-full items-center gap-2 text-left cursor-pointer"
          >
            {title}
            <DisclosureCaret open={expanded} />
            {spinner}
          </button>
        ) : (
          <div className="flex items-center gap-2">
            {title}
            {spinner}
          </div>
        )}
        {expandable && expanded && (
          <div
            id={contentId}
            role="region"
            aria-labelledby={buttonId}
            className="mt-1.5 flex flex-col divide-y divide-border-subtle/50 rounded-lg border border-border-subtle/50"
          >
            {observedSteps.map((step) => (
              <div key={step} className="px-3 py-2">
                {t(step)}
              </div>
            ))}
          </div>
        )}
      </div>
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
