import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationId } from "#/hooks/use-conversation-id";

function workspaceBasename(path: string | null | undefined): string | null {
  if (!path) {
    return null;
  }
  return path.replace(/\/+$/, "").split("/").pop() || path;
}

/** Workspace label for the overview panel (git stats come from other hooks). */
export function useConversationOverviewStats() {
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();

  const storedMetadata = conversationId
    ? getStoredConversationMetadata(conversationId)
    : null;
  const workspacePath =
    storedMetadata?.selected_workspace ??
    conversation?.selected_workspace ??
    null;

  return {
    workspaceName: workspaceBasename(workspacePath),
  };
}
