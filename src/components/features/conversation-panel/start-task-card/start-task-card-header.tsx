import type { AppConversationStartTaskStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { StartTaskStatusIndicator } from "./start-task-status-indicator";
import { StartTaskStatusBadge } from "./start-task-status-badge";

interface StartTaskCardHeaderProps {
  title: string;
  taskStatus: AppConversationStartTaskStatus;
}

export function StartTaskCardHeader({
  title,
  taskStatus,
}: StartTaskCardHeaderProps) {
  return (
    <div className="flex items-center gap-2 flex-1 min-w-0 overflow-hidden mr-2">
      {/* Status Indicator */}
      <div className="flex items-center">
        <StartTaskStatusIndicator taskStatus={taskStatus} />
      </div>

      {/* Title */}
      <h3 className="text-sm font-medium text-content-2 truncate flex-1">
        {title}
      </h3>

      {/* Status Badge */}
      <StartTaskStatusBadge taskStatus={taskStatus} />
    </div>
  );
}
