import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";

/**
 * Returns whether the user explicitly attached a "source" to the active
 * conversation — i.e. picked a repository on the home page *or* picked a
 * local workspace folder. From the Files tab's point of view those two
 * cases are equivalent: both mean there's an existing working tree the
 * user came in to inspect, so the diff view is the more useful default.
 *
 * We deliberately do *not* probe the filesystem here — the agent-server
 * can initialise conversations without an explicit selection in generated git
 * worktrees, so a positive `git status` does not by itself imply the user
 * attached a real source. That's why we read the explicit selection signals
 * (`selected_repository` on the conversation, `selected_workspace` from the
 * conversation-metadata store) instead.
 */
export function useHasAttachedSource(): {
  hasAttachedSource: boolean;
  isLoading: boolean;
} {
  const { data: conversation, isLoading } = useActiveConversation();
  const storedMetadata = conversation?.id
    ? getStoredConversationMetadata(conversation.id)
    : null;
  return {
    hasAttachedSource:
      !!conversation?.selected_repository ||
      !!storedMetadata?.selected_workspace,
    isLoading,
  };
}
