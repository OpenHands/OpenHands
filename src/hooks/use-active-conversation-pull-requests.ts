import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationPullRequests } from "#/hooks/query/use-conversation-pull-requests";
import { useLocalGitInfo } from "#/hooks/query/use-local-git-info";
import { useConversationPrimaryRepository } from "#/hooks/use-conversation-primary-repository";

// `git rev-parse --abbrev-ref HEAD` prints this for a detached HEAD.
const DETACHED_HEAD = "HEAD";

/**
 * PRs that belong to the open conversation: the ones it tracks, else the
 * ones opened from the branch its workspace is on. The workspace's checked-out
 * branch is preferred over the branch picked at start, since the agent
 * usually works on a branch it created.
 */
export function useActiveConversationPullRequests(enabled = true) {
  const { data: conversation } = useActiveConversation();
  const { data: localGitInfo } = useLocalGitInfo();
  const { repository, provider, branch } = useConversationPrimaryRepository();

  const checkedOutBranch =
    localGitInfo?.repository === repository &&
    localGitInfo?.branch &&
    localGitInfo.branch !== DETACHED_HEAD
      ? localGitInfo.branch
      : null;

  return useConversationPullRequests({
    repository: enabled ? repository : null,
    provider: enabled ? provider : null,
    branch: checkedOutBranch ?? branch,
    prNumbers: conversation?.pr_number,
  });
}
