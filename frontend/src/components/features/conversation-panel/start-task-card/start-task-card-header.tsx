import type { V1AppConversationStartTaskStatus } from "#/api/conversation-service/v1-conversation-service.types";
import { ConversationVersionBadge } from "../conversation-card/conversation-version-badge";
import { StartTaskStatusIndicator } from "./start-task-status-indicator";
import { StartTaskStatusBadge } from "./start-task-status-badge";

interface StartTaskCardHeaderProps {
  title: string;
  taskStatus: V1AppConversationStartTaskStatus;
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

      {/* Version Badge - V1 tasks are always V1 */}
      <ConversationVersionBadge version="V1" />

      {/* Title */}
      <h3 className="text-sm font-medium text-neutral-100 truncate flex-1">
        {title}
      </h3>

      {/* Status Badge */}
      <StartTaskStatusBadge taskStatus={taskStatus} />
    </div>
  );
}
