import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { isArchivedSandboxStatus } from "#/utils/conversation-archive-status";

export function useIsArchivedConversation() {
  const { data: conversation } = useActiveConversation();
  return isArchivedSandboxStatus(conversation?.sandbox_status);
}
