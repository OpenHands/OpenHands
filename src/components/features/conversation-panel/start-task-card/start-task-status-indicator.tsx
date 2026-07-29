import type { AppConversationStartTaskStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { cn } from "#/utils/utils";

interface StartTaskStatusIndicatorProps {
  taskStatus: AppConversationStartTaskStatus;
}

export function StartTaskStatusIndicator({
  taskStatus,
}: StartTaskStatusIndicatorProps) {
  const getStatusColor = () => {
    switch (taskStatus) {
      case "READY":
        return "bg-green-500";
      case "ERROR":
        return "bg-red-500";
      case "WORKING":
      case "WAITING_FOR_SANDBOX":
      case "PREPARING_REPOSITORY":
      case "RUNNING_SETUP_SCRIPT":
      case "SETTING_UP_GIT_HOOKS":
      case "SETTING_UP_SKILLS":
      case "STARTING_CONVERSATION":
        return "bg-yellow-500 animate-pulse";
      default:
        return "bg-[var(--oh-interactive-selected)]";
    }
  };

  return (
    <div
      className={cn("w-2 h-2 rounded-full flex-shrink-0", getStatusColor())}
      aria-label={`Task status: ${taskStatus}`}
    />
  );
}
