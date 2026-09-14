import React from "react";
import { useQuery } from "@tanstack/react-query";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { getGitPath } from "#/utils/get-git-path";

/**
 * The diff view should reflect the *current* git state. Every fetch (initial,
 * auto-invalidate after a commit/push event, or the periodic poll below)
 * returns the authoritative, complete ordered list from the server, so we hand
 * that straight to the UI. Historically this hook accumulated deltas into an
 * `orderedChanges` buffer whose closure-dependent merge could pin stale rows
 * after a commit — that's the "diff still shows after commit" bug. The server
 * already returns the whole picture, so no buffer is needed.
 *
 * A short `refetchInterval` also closes the gap where the one-shot invalidate
 * from the commit event can fire *before* the commit lands on disk: the next
 * poll picks it up, so the diff clears on its own instead of waiting for a
 * manual refresh.
 */
export const useUnifiedGetGitChanges = () => {
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();

  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const selectedRepository = conversation?.selected_repository;
  const workingDir = conversation?.workspace?.working_dir?.trim();

  const gitPath = React.useMemo(
    () => getGitPath(selectedRepository, workingDir),
    [selectedRepository, workingDir],
  );

  const result = useQuery({
    queryKey: [
      "file_changes",
      conversationId,
      conversationUrl,
      sessionApiKey,
      gitPath,
    ],
    queryFn: async () => {
      if (!conversationId) throw new Error("No conversation ID");

      return AgentServerGitService.getGitChanges(
        conversationId,
        conversationUrl,
        sessionApiKey,
        gitPath,
      );
    },
    retry: false,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
    refetchOnMount: "always",
    // Keep the diff in sync after a commit/push lands. The commit event's
    // invalidate can fire before the backend finishes committing; this poll
    // guarantees the view converges to the true state shortly after.
    refetchInterval: 5000,
    enabled: runtimeIsReady && !!conversationId,
    meta: {
      disableToast: true,
    },
  });

  // Return a stable, flat shape (not the full UseQueryResult union) so
  // consumers and test mocks can rely on these fields. `data` is always an
  // array: `undefined` becomes `[]`.
  return {
    data: result.data ?? [],
    isLoading: result.isLoading,
    isFetching: result.isFetching,
    isSuccess: result.isSuccess,
    isError: result.isError,
    error: result.error,
    refetch: result.refetch,
  };
};
