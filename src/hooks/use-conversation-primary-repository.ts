import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useLocalGitInfo } from "#/hooks/query/use-local-git-info";
import { useTaskPolling } from "#/hooks/query/use-task-polling";
import type { Provider } from "#/types/settings";

/**
 * Resolves the conversation's primary connected git repository using the
 * same priority as the conversation overview git section:
 * conversation metadata → task polling → local workspace probe.
 */
export function useConversationPrimaryRepository() {
  const { data: conversation } = useActiveConversation();
  const { repositoryInfo } = useTaskPolling();
  const { data: localGitInfo } = useLocalGitInfo();

  const conversationRepository =
    conversation?.selected_repository || repositoryInfo?.selectedRepository;
  const conversationProvider = (conversation?.git_provider ||
    repositoryInfo?.gitProvider) as Provider | undefined;
  const conversationBranch =
    conversation?.selected_branch || repositoryInfo?.selectedBranch;

  const repository = conversationRepository || localGitInfo?.repository || null;
  const provider = (conversationProvider ||
    localGitInfo?.provider ||
    null) as Provider | null;
  const branch = conversationBranch || localGitInfo?.branch || null;
  // Only the workspace remote knows which instance a repository lives on.
  const remoteHost =
    localGitInfo?.host &&
    localGitInfo.repository === repository &&
    localGitInfo.provider === provider
      ? localGitInfo.host
      : null;

  return {
    repository,
    provider,
    branch,
    remoteHost,
    isConnected: Boolean(repository && provider),
  };
}
