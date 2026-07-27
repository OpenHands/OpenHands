import { useQuery } from "@tanstack/react-query";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";

/**
 * Probes whether the conversation's working-directory git repository has
 * at least one commit reachable from HEAD.
 *
 * Local-only by design: we deliberately avoid driving
 * `/api/bash/execute_bash_command` from the frontend on cloud backends.
 * On cloud, `hasCommits` stays `null` and the Files tab keeps its
 * optimistic diff-view default — fine in practice since cloud
 * conversations almost always have an attached repo with commits.
 *
 * Used by the Files tab to decide whether the diff view is a sensible
 * default: an attached source (repo or local workspace) with no commits
 * — e.g. a brand-new empty GitHub repo, a freshly `git init`-ed
 * workspace, or a plain non-git workspace — has no diff base to compare
 * against, so the file viewer is a better landing experience.
 *
 * Returns `hasCommits: null` while the probe is in-flight so callers can
 * distinguish "still loading" from a definitive "no commits".
 */
export function useHasGitCommits(options?: { enabled?: boolean }): {
  hasCommits: boolean | null;
  isLoading: boolean;
} {
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();
  const { backend } = useActiveBackend();
  const isLocalBackend = backend.kind === "local";

  const conversationId = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const workingDir = conversation?.workspace?.working_dir?.trim();

  const enabled =
    (options?.enabled ?? true) &&
    isLocalBackend &&
    runtimeIsReady &&
    !!conversationId &&
    !!workingDir;

  const query = useQuery<boolean>({
    queryKey: [
      "has-git-commits",
      conversationId,
      conversationUrl,
      sessionApiKey,
      workingDir,
    ],
    queryFn: async () => {
      // `git rev-parse --verify HEAD` exits 0 iff HEAD resolves to a
      // real commit. Returns non-zero in three cases that all collapse
      // to "no diff base, show files view":
      //   - unborn branch (`git init` with no commits)
      //   - not a git repository at all (plain workspace directory)
      //   - other git error
      // Callers gate this hook on the user having attached a source
      // (see `useHasAttachedSource`) so we don't shell out for the
      // unattached-conversation case where the answer is moot.
      const result = await AgentServerRuntimeService.executeCommand(
        conversationUrl,
        sessionApiKey,
        "git rev-parse --verify HEAD",
        workingDir,
        10,
      );
      return result.exit_code === 0;
    },
    enabled,
    retry: false,
    staleTime: 60_000,
    gcTime: 5 * 60_000,
    meta: { disableToast: true },
  });

  return {
    hasCommits: query.data ?? null,
    isLoading: query.isLoading,
  };
}
