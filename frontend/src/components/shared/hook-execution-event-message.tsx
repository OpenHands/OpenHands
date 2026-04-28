import React from "react";
import { useTranslation } from "react-i18next";
import { isHookExecutionEvent } from "#/types/v1/type-guards";
import { OpenHandsEvent } from "#/types/v1/core";
import { GenericEventMessage } from "#/components/features/chat/generic-event-message";

interface HookExecutionEventMessageProps {
  event: OpenHandsEvent;
}

function getHookIcon(hookType: string, blocked: boolean): string {
  if (blocked) {
    return "🚫";
  }

  switch (hookType) {
    case "PreToolUse":
      return "⏳";
    case "PostToolUse":
      return "✅";
    case "UserPromptSubmit":
      return "📝";
    case "SessionStart":
      return "🚀";
    case "SessionEnd":
      return "🏁";
    case "Stop":
      return "⏹️";
    default:
      return "🔗";
  }
}

function formatHookCommand(command: string): string {
  // Truncate long commands for display
  if (command.length > 80) {
    return `${command.slice(0, 77)}...`;
  }
  return command;
}

function getStatusText(blocked: boolean, success: boolean): string {
  if (blocked) return "blocked";
  if (success) return "ok";
  return "failed";
}

function getStatusClassName(blocked: boolean, success: boolean): string {
  if (blocked) return "bg-amber-900/50 text-amber-300";
  if (success) return "bg-green-900/50 text-green-300";
  return "bg-red-900/50 text-red-300";
}

export function HookExecutionEventMessage({
  event,
}: HookExecutionEventMessageProps) {
  const { t } = useTranslation();

  if (!isHookExecutionEvent(event)) {
    return null;
  }

  const icon = getHookIcon(event.hook_event_type, event.blocked);
  const statusText = getStatusText(event.blocked, event.success);
  const statusClassName = getStatusClassName(event.blocked, event.success);

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
      {icon} {t("HOOK$HOOK_LABEL")}: {event.hook_event_type}
      {event.tool_name && (
        <span className="text-neutral-400 ml-2">({event.tool_name})</span>
      )}
      <span className={`ml-2 px-1 py-0.5 rounded text-xs ${statusClassName}`}>
        {statusText}
      </span>
    </span>
  );

  const details = (
    <div className="flex flex-col gap-2 text-neutral-400">
      <div>
        <span className="text-neutral-500">{t("HOOK$COMMAND")}:</span>{" "}
        <code className="text-xs bg-neutral-800 px-1 py-0.5 rounded">
          {formatHookCommand(event.hook_command)}
        </code>
      </div>

      {event.exit_code !== null && (
        <div>
          <span className="text-neutral-500">{t("HOOK$EXIT_CODE")}:</span>{" "}
          {event.exit_code}
        </div>
      )}

      {event.blocked && event.reason && (
        <div className="text-amber-400">
          <span className="text-neutral-500">{t("HOOK$BLOCKED_REASON")}:</span>{" "}
          {event.reason}
        </div>
      )}

      {event.additional_context && (
        <div>
          <span className="text-neutral-500">{t("HOOK$CONTEXT")}:</span>{" "}
          {event.additional_context}
        </div>
      )}

      {event.error && (
        <div className="text-red-400">
          <span className="text-neutral-500">{t("HOOK$ERROR")}:</span>{" "}
          {event.error}
        </div>
      )}

      {event.stdout && (
        <div>
          <span className="text-neutral-500">{t("HOOK$OUTPUT")}:</span>
          <pre className="text-xs bg-neutral-800 p-2 rounded mt-1 overflow-x-auto max-h-40 overflow-y-auto">
            {event.stdout}
          </pre>
        </div>
      )}

      {event.stderr && (
        <div>
          <span className="text-neutral-500">{t("HOOK$STDERR")}:</span>
          <pre className="text-xs bg-neutral-800 p-2 rounded mt-1 overflow-x-auto max-h-40 overflow-y-auto text-amber-300">
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
