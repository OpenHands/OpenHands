import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { isRuntimeUnavailableSandboxStatus } from "#/utils/conversation-archive-status";

/**
 * Whether the active conversation's runtime is unavailable, so runtime-backed
 * affordances (chat input, git/panel toggles) are read-only. Covers both a
 * missing (non-resumable) sandbox and an errored one.
 *
 * This is the read-only gate, not the archive predicate: use
 * `isEffectivelyArchivedConversation` for list visibility, the "Archived"
 * chip, and the archive/unarchive direction.
 */
export function useIsArchivedConversation() {
  const { data: conversation } = useActiveConversation();
  return isRuntimeUnavailableSandboxStatus(conversation?.sandbox_status);
}
