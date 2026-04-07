import { V1SandboxStatus } from "#/api/sandbox-service/sandbox-service.types";
import { ConversationStatus } from "#/types/conversation-status";
import { ConversationCardTitle } from "./conversation-card-title";
import { ConversationStatusIndicator } from "../../home/recent-conversations/conversation-status-indicator";

interface ConversationCardHeaderProps {
  title: string;
  titleMode: "view" | "edit";
  onTitleSave: (title: string) => void;
  sandboxStatus?: V1SandboxStatus;
}

// Map V1SandboxStatus to ConversationStatus for the indicator
const mapSandboxStatusToConversationStatus = (status?: V1SandboxStatus): ConversationStatus => {
  if (status === "PAUSED") return "PAUSED";
  if (status === "RUNNING") return "RUNNING";
  return "STOPPED";
};

export function ConversationCardHeader({
  title,
  titleMode,
  onTitleSave,
  sandboxStatus,
}: ConversationCardHeaderProps) {
  const isConversationArchived = sandboxStatus === "STOPPED";
  const conversationStatus = mapSandboxStatusToConversationStatus(sandboxStatus);

  return (
    <div className="flex items-center gap-2 flex-1 min-w-0 overflow-hidden mr-2">
      {/* Status Indicator - map sandbox status to indicator */}
      {sandboxStatus && (
        <div className="flex items-center">
          <ConversationStatusIndicator
            conversationStatus={conversationStatus}
          />
        </div>
      )}
      <ConversationCardTitle
        title={title}
        titleMode={titleMode}
        onSave={onTitleSave}
        isConversationArchived={isConversationArchived}
      />
    </div>
  );
}
