import { useTranslation } from "react-i18next";
import type { AppConversationStartTaskStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { cn } from "#/utils/utils";
import { getTaskStatusI18nKey } from "#/utils/status";

interface StartTaskStatusBadgeProps {
  taskStatus: AppConversationStartTaskStatus;
}

export function StartTaskStatusBadge({
  taskStatus,
}: StartTaskStatusBadgeProps) {
  const { t } = useTranslation("openhands");

  // Don't show badge for WORKING status (most common, clutters UI)
  if (taskStatus === "WORKING") {
    return null;
  }

  // Localized status label — getTaskStatusI18nKey maps every status (including
  // the terminal READY/ERROR states) to its localized key.
  const getStatusLabel = () => t(getTaskStatusI18nKey(taskStatus));

  // Get status color
  const getStatusStyle = () => {
    switch (taskStatus) {
      case "READY":
        return "bg-green-500/10 text-green-400 border-green-500/20";
      case "ERROR":
        return "bg-red-500/10 text-red-400 border-red-500/20";
      default:
        return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20";
    }
  };

  return (
    <span
      className={cn(
        "text-xs font-medium px-2 py-0.5 rounded border flex-shrink-0",
        getStatusStyle(),
      )}
    >
      {getStatusLabel()}
    </span>
  );
}
