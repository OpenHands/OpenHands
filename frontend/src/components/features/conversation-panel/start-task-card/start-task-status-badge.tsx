import type { V1AppConversationStartTaskStatus } from "#/api/conversation-service/v1-conversation-service.types";
import { cn } from "#/utils/utils";

interface StartTaskStatusBadgeProps {
  taskStatus: V1AppConversationStartTaskStatus;
}

export function StartTaskStatusBadge({
  taskStatus,
}: StartTaskStatusBadgeProps) {
  // Don't show badge for WORKING status (most common, clutters UI)
  if (taskStatus === "WORKING") {
    return null;
  }

  // Format status for display
  const formatStatus = (status: string) =>
    status
      .toLowerCase()
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());

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
      {formatStatus(taskStatus)}
    </span>
  );
}
