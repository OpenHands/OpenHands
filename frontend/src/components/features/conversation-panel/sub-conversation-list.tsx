import { NavLink } from "react-router";
import { useSubConversations } from "#/hooks/query/use-sub-conversations";
import { ConversationCard } from "./conversation-card/conversation-card";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ConversationStatus } from "#/types/conversation-status";
import { V1SandboxStatus } from "#/api/sandbox-service/sandbox-service.types";
import { Provider } from "#/types/settings";

function mapSandboxStatusToConversationStatus(
  sandboxStatus: V1SandboxStatus,
): ConversationStatus {
  switch (sandboxStatus) {
    case "RUNNING":
      return "RUNNING";
    case "STARTING":
      return "STARTING";
    case "PAUSED":
    case "STOPPED":
      return "STOPPED";
    case "MISSING":
      return "ARCHIVED";
    default:
      return "STOPPED";
  }
}

interface SubConversationListProps {
  subConversationIds: string[];
  onClose: () => void;
  onDelete: (conversationId: string, title: string) => void;
  onStop: (conversationId: string, version?: "V0" | "V1") => void;
  onChangeTitle: (conversationId: string, newTitle: string) => void;
  openContextMenuId: string | null;
  onContextMenuToggle: (id: string | null) => void;
}

export function SubConversationList({
  subConversationIds,
  onClose,
  onDelete,
  onStop,
  onChangeTitle,
  openContextMenuId,
  onContextMenuToggle,
}: SubConversationListProps) {
  const { data: subConversations, isLoading } =
    useSubConversations(subConversationIds);

  if (isLoading) {
    return (
      <div className="ml-4 border-l-2 border-neutral-600 py-2 pl-2">
        <LoadingSpinner size="small" />
      </div>
    );
  }

  const validConversations = (subConversations ?? []).filter(Boolean);

  if (validConversations.length === 0) {
    return null;
  }

  return (
    <div className="ml-4 border-l-2 border-neutral-600">
      {validConversations.map((conv) => {
        if (!conv) return null;
        const status = mapSandboxStatusToConversationStatus(
          conv.sandbox_status,
        );
        return (
          <NavLink
            key={conv.id}
            to={`/conversations/${conv.id}`}
            onClick={onClose}
          >
            <ConversationCard
              onDelete={() => onDelete(conv.id, conv.title ?? "Untitled")}
              onStop={() => onStop(conv.id, "V1")}
              onChangeTitle={(title) => onChangeTitle(conv.id, title)}
              title={conv.title ?? "Sub-conversation"}
              selectedRepository={{
                selected_repository: conv.selected_repository,
                selected_branch: conv.selected_branch,
                git_provider: conv.git_provider as Provider,
              }}
              lastUpdatedAt={conv.updated_at}
              createdAt={conv.created_at}
              conversationStatus={status}
              conversationId={conv.id}
              conversationVersion="V1"
              trigger={conv.trigger ?? undefined}
              contextMenuOpen={openContextMenuId === conv.id}
              onContextMenuToggle={(isOpen) =>
                onContextMenuToggle(isOpen ? conv.id : null)
              }
            />
          </NavLink>
        );
      })}
    </div>
  );
}
