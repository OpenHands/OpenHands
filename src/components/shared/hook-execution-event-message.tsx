import React from "react";
import { useTranslation } from "react-i18next";
import { isHookExecutionEvent } from "#/types/agent-server/type-guards";
import { OpenHandsEvent } from "#/types/agent-server/core";
import { GenericEventMessage } from "#/components/features/chat/generic-event-message";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";

interface HookExecutionEventMessageProps {
  event: OpenHandsEvent;
}

function formatHookCommand(command: string): string {
  // Truncate long commands for display
  if (command.length > 80) {
    return `${command.slice(0, 77)}...`;
  }
  return command;
}

// Successful hooks are the common case and get no badge.
function getStatusText(blocked: boolean, success: boolean): string | null {
  if (blocked) return "blocked";
  if (success) return null;
  return "failed";
}

function getStatusClassName(blocked: boolean): string {
  if (blocked) return "bg-feedback-warning/10 text-feedback-warning";
  return "bg-feedback-error/10 text-feedback-error";
}

export function HookExecutionEventMessage({
  event,
}: HookExecutionEventMessageProps) {
  const { t } = useTranslation("openhands");

  if (!isHookExecutionEvent(event)) {
    return null;
  }

  const statusText = getStatusText(event.blocked, event.success);
  const statusClassName = getStatusClassName(event.blocked);

  // Determine the overall success indicator for GenericEventMessage.
  // When blocked, suppress the success indicator entirely — the amber "blocked"
  // badge in the title is the authoritative status signal.
  const getSuccessStatus = (): "success" | "error" | undefined => {
    if (event.blocked) return undefined;
    return event.success ? "success" : "error";
  };
  const successStatus = getSuccessStatus();

  const title = (
    <span>
      {t(I18nKey.HOOK$HOOK_LABEL)}: {event.hook_event_type}
      {event.tool_name && (
        <span className="text-muted ml-2">({event.tool_name})</span>
      )}
      {statusText && (
        <span
          className={cn("ml-2 px-1 py-0.5 rounded text-xs", statusClassName)}
        >
          {statusText}
        </span>
      )}
    </span>
  );

  const details = (
    <div className="flex flex-col gap-2 text-muted">
      <div>
        <span className="text-text-subtle">{t(I18nKey.HOOK$COMMAND)}:</span>{" "}
        <code className="text-xs bg-surface px-1 py-0.5 rounded">
          {formatHookCommand(event.hook_command)}
        </code>
      </div>

      {event.exit_code !== null && (
        <div>
          <span className="text-text-subtle">{t(I18nKey.HOOK$EXIT_CODE)}:</span>{" "}
          {event.exit_code}
        </div>
      )}

      {event.blocked && event.reason && (
        <div className="text-feedback-warning">
          <span className="text-text-subtle">
            {t(I18nKey.HOOK$BLOCKED_REASON)}:
          </span>{" "}
          {event.reason}
        </div>
      )}

      {event.additional_context && (
        <div>
          <span className="text-text-subtle">{t(I18nKey.HOOK$CONTEXT)}:</span>{" "}
          {event.additional_context}
        </div>
      )}

      {event.error && (
        <div className="text-feedback-error">
          <span className="text-text-subtle">{t(I18nKey.HOOK$ERROR)}:</span>{" "}
          {event.error}
        </div>
      )}

      {event.stdout && (
        <div>
          <span className="text-text-subtle">{t(I18nKey.HOOK$OUTPUT)}:</span>
          <pre className="text-xs bg-surface p-2 rounded mt-1 overflow-x-auto max-h-40 overflow-y-auto">
            {event.stdout}
          </pre>
        </div>
      )}

      {event.stderr && (
        <div>
          <span className="text-text-subtle">{t(I18nKey.HOOK$STDERR)}:</span>
          <pre className="text-xs bg-surface p-2 rounded mt-1 overflow-x-auto max-h-40 overflow-y-auto text-feedback-warning">
            {event.stderr}
          </pre>
        </div>
      )}
    </div>
  );

  return (
    <GenericEventMessage
      title={title}
      details={details}
      success={successStatus}
    />
  );
}
