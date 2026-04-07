import { V1SandboxStatus } from "#/api/sandbox-service/sandbox-service.types";
import { ConversationCardTitle } from "./conversation-card-title";
import { SandboxStatusIndicator } from "../../home/recent-conversations/sandbox-status-indicator";

interface ConversationCardHeaderProps {
  title: string;
  titleMode: "view" | "edit";
  onTitleSave: (title: string) => void;
  sandboxStatus?: V1SandboxStatus;
}

export function ConversationCardHeader({
  title,
  titleMode,
  onTitleSave,
  sandboxStatus,
}: ConversationCardHeaderProps) {
  const isConversationArchived =
    sandboxStatus === "STOPPED" || sandboxStatus === "MISSING";

  return (
    <div className="flex items-center gap-2 flex-1 min-w-0 overflow-hidden mr-2">
      {/* Status Indicator - use V1 sandbox status directly */}
      {sandboxStatus && (
        <div className="flex items-center">
          <SandboxStatusIndicator sandboxStatus={sandboxStatus} />
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
