import { useTranslation } from "react-i18next";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { isMissingSandboxStatus } from "#/utils/conversation-archive-status";
import { ConversationCardTitle } from "./conversation-card-title";
import { ConversationStatusDot } from "../conversation-status-dot";
import { I18nKey } from "#/i18n/declaration";

interface ConversationCardHeaderProps {
  title: string;
  titlePending?: boolean;
  titleMode: "view" | "edit";
  onTitleSave: (title: string) => void;
  executionStatus?: ExecutionStatus | null;
  isStarting?: boolean;
  sandboxStatus?: SandboxStatus | null;
  /**
   * The row's effective archived state (explicit archive or missing runtime).
   * Dims the title and grays the status dot so both agree with the "Archived"
   * chip. Defaults to the runtime-derived case for callers that only know the
   * sandbox status.
   */
  isArchived?: boolean;
  isUnread?: boolean;
}

export function ConversationCardHeader({
  title,
  titlePending,
  titleMode,
  onTitleSave,
  executionStatus,
  isStarting,
  sandboxStatus,
  isArchived,
  isUnread,
}: ConversationCardHeaderProps) {
  const { t } = useTranslation("openhands");
  const archived = isArchived ?? isMissingSandboxStatus(sandboxStatus);
  return (
    <div className="flex items-center gap-2 flex-1 min-w-0 overflow-hidden">
      {isUnread && (
        <span
          data-testid="conversation-unread-dot"
          className="size-2 rounded-full bg-primary shrink-0"
          aria-label={t(I18nKey.CONVERSATION$UNREAD_INDICATOR)}
        />
      )}
      {(executionStatus !== undefined || isStarting) && (
        <div className="flex w-4.5 shrink-0 items-center justify-center">
          <ConversationStatusDot
            executionStatus={executionStatus}
            isStarting={isStarting}
            sandboxStatus={sandboxStatus}
            isArchived={archived}
            showTooltip={false}
          />
        </div>
      )}
      <ConversationCardTitle
        title={title}
        titlePending={titlePending}
        titleMode={titleMode}
        onSave={onTitleSave}
        isConversationArchived={archived}
      />
    </div>
  );
}
