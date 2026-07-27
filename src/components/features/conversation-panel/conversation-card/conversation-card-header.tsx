import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { isArchivedSandboxStatus } from "#/utils/conversation-archive-status";
import { ConversationCardTitle } from "./conversation-card-title";
import { ConversationStatusDot } from "../conversation-status-dot";

interface ConversationCardHeaderProps {
  title: string;
  titleMode: "view" | "edit";
  onTitleSave: (title: string) => void;
  executionStatus?: ExecutionStatus | null;
  sandboxStatus?: SandboxStatus | null;
  searchQuery?: string;
}

export function ConversationCardHeader({
  title,
  titleMode,
  onTitleSave,
  executionStatus,
  sandboxStatus,
  searchQuery,
}: ConversationCardHeaderProps) {
  const isArchived = isArchivedSandboxStatus(sandboxStatus);
  return (
    <div className="flex items-center gap-2 flex-1 min-w-0 overflow-hidden">
      {executionStatus !== undefined && (
        <div className="flex w-[18px] shrink-0 items-center justify-center">
          <ConversationStatusDot
            executionStatus={executionStatus}
            sandboxStatus={sandboxStatus}
            showTooltip={false}
          />
        </div>
      )}
      <ConversationCardTitle
        title={title}
        titleMode={titleMode}
        onSave={onTitleSave}
        isConversationArchived={isArchived}
        searchQuery={searchQuery}
      />
    </div>
  );
}
