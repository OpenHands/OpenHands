import { ConversationStatus } from "#/types/conversation-status";
import { ConversationCardTitle } from "./conversation-card-title";
import { ConversationStatusIndicator } from "../../home/recent-conversations/conversation-status-indicator";
import { ConversationStatusBadges } from "./conversation-status-badges";
import { ConversationVersionBadge } from "./conversation-version-badge";

interface ConversationCardHeaderProps {
  title: string;
  titleMode: "view" | "edit";
  onTitleSave: (title: string) => void;
  conversationStatus?: ConversationStatus;
  conversationVersion?: "V0" | "V1";
}

export function ConversationCardHeader({
  title,
  titleMode,
  onTitleSave,
  conversationStatus,
  conversationVersion,
}: ConversationCardHeaderProps) {
  const isConversationArchived = conversationStatus === "ARCHIVED";

  return (
    <div className="flex items-center gap-2 flex-1 min-w-0 overflow-hidden mr-2">
      {/* Status Indicator */}
      {conversationStatus && (
        <div className="flex items-center">
          <ConversationStatusIndicator
            conversationStatus={conversationStatus}
          />
        </div>
      )}
      {/* Version Badge */}
      <ConversationVersionBadge
        version={conversationVersion}
        isConversationArchived={isConversationArchived}
      />
      <ConversationCardTitle
        title={title}
        titleMode={titleMode}
        onSave={onTitleSave}
        isConversationArchived={isConversationArchived}
      />
      {/* Status Badges */}
      {conversationStatus && (
        <ConversationStatusBadges conversationStatus={conversationStatus} />
      )}
    </div>
  );
}
