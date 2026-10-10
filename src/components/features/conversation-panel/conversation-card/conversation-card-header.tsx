import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { isMissingSandboxStatus } from "#/utils/conversation-archive-status";
import { ConversationCardTitle } from "./conversation-card-title";
import { ConversationStatusDot } from "../conversation-status-dot";

interface ConversationCardHeaderProps {
  title: string;
  titleMode: "view" | "edit";
  onTitleSave: (title: string) => void;
  executionStatus?: ExecutionStatus | null;
  sandboxStatus?: SandboxStatus | null;
  /**
   * The row's effective archived state (explicit archive or missing runtime).
   * Dims the title and grays the status dot so both agree with the "Archived"
   * chip. Defaults to the runtime-derived case for callers that only know the
   * sandbox status.
   */
  isArchived?: boolean;
}

export function ConversationCardHeader({
  title,
  titleMode,
  onTitleSave,
  executionStatus,
  sandboxStatus,
  isArchived,
}: ConversationCardHeaderProps) {
  const archived = isArchived ?? isMissingSandboxStatus(sandboxStatus);
  return (
    <div className="flex items-center gap-2 flex-1 min-w-0 overflow-hidden">
      {executionStatus !== undefined && (
        <div className="flex w-4.5 shrink-0 items-center justify-center">
          <ConversationStatusDot
            executionStatus={executionStatus}
            sandboxStatus={sandboxStatus}
            isArchived={archived}
            showTooltip={false}
          />
        </div>
      )}
      <ConversationCardTitle
        title={title}
        titleMode={titleMode}
        onSave={onTitleSave}
        isConversationArchived={archived}
      />
    </div>
  );
}
