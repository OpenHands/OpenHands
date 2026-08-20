import { useMemo } from "react";
import { useQueries } from "@tanstack/react-query";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useUnifiedGetGitChanges } from "#/hooks/query/use-unified-get-git-changes";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { getGitPath } from "#/utils/get-git-path";
import {
  countGitChangeDiffStats,
  sumGitDiffLineStats,
} from "#/utils/git-diff-stats";

export function useConversationOverviewGitDiffStats() {
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady();
  const {
    data: gitChanges,
    isLoading: isLoadingChanges,
    isSuccess: isChangesSuccess,
    isError: isChangesError,
  } = useUnifiedGetGitChanges();

  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const gitPath = useMemo(
    () =>
      getGitPath(
        conversation?.selected_repository,
        conversation?.workspace?.working_dir?.trim(),
      ),
    [conversation?.selected_repository, conversation?.workspace?.working_dir],
  );

  const diffCandidates = useMemo(
    () => (gitChanges ?? []).filter((change) => change.status !== "D"),
    [gitChanges],
  );

  const diffQueries = useQueries({
    queries: diffCandidates.map((change) => ({
      queryKey: [
        "conversation_overview_file_diff",
        conversationId,
        conversationUrl,
        sessionApiKey,
        gitPath,
        change.path,
      ],
      queryFn: async () =>
        AgentServerGitService.getGitChangeDiff(
          conversationId!,
          conversationUrl,
          sessionApiKey,
          `${gitPath}/${change.path}`,
        ),
      enabled:
        runtimeIsReady && !!conversationId && isChangesSuccess && !!change.path,
      staleTime: 1000 * 60 * 5,
      meta: {
        disableToast: true,
      },
    })),
  });

  const isLoadingDiffs = diffQueries.some((query) => query.isLoading);
  const totals = sumGitDiffLineStats(
    diffQueries.flatMap((query) =>
      query.data ? [countGitChangeDiffStats(query.data)] : [],
    ),
  );

  return {
    additions: totals.additions,
    deletions: totals.deletions,
    changeCount: gitChanges?.length ?? 0,
    isLoading: isLoadingChanges || isLoadingDiffs,
    isError: isChangesError,
  };
}
