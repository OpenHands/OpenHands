import React from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import V1GitService from "#/api/git-service/v1-git-service.api";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { getGitPath } from "#/utils/get-git-path";
import type { GitChange } from "#/api/open-hands.types";

/**
 * Unified hook to get git changes for both legacy (V0) and V1 conversations
 * - V0: Uses the legacy GitService.getGitChanges API endpoint
 * - V1: Uses the V1GitService.getGitChanges API endpoint with runtime URL
 */
export const useUnifiedGetGitChanges = () => {
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const [orderedChanges, setOrderedChanges] = React.useState<GitChange[]>([]);
  const previousDataRef = React.useRef<GitChange[] | null>(null);
  const queryClient = useQueryClient();
  const runtimeIsReady = useRuntimeIsReady();

  const isV1Conversation = conversation?.conversation_version === "V1";
  const conversationUrl = conversation?.url;
  const sessionApiKey = conversation?.session_api_key;
  const selectedRepository = conversation?.selected_repository;

  // Calculate git path based on selected repository
  const gitPath = React.useMemo(
    () => getGitPath(selectedRepository),
    [selectedRepository],
  );

  const result = useQuery({
    queryKey: [
      "file_changes",
      conversationId,
      isV1Conversation,
      conversationUrl,
      gitPath,
    ],
    queryFn: async () => {
      if (!conversationId) throw new Error("No conversation ID");

      // V1: Use the V1 API endpoint with runtime URL
      if (isV1Conversation) {
        return V1GitService.getGitChanges(
          conversationUrl,
          sessionApiKey,
          gitPath,
        );
      }

      // V0 (Legacy): Use the legacy API endpoint
      return GitService.getGitChanges(conversationId);
    },
    retry: false,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
    enabled: runtimeIsReady && !!conversationId,
    meta: {
      disableToast: true,
    },
  });

  React.useEffect(() => {
    if (!result.isFetching && result.isSuccess && result.data) {
      const currentData = result.data;

      const hasDataChanged =
        JSON.stringify(currentData) !== JSON.stringify(previousDataRef.current);

      if (hasDataChanged) {
        previousDataRef.current = currentData;
        setOrderedChanges(
          Array.isArray(currentData) ? currentData : [currentData],
        );
      }
    }
  }, [result.isFetching, result.isSuccess, result.data]);

  const customRefetch = React.useCallback(async () => {
    previousDataRef.current = null;
    await queryClient.invalidateQueries({
      queryKey: [
        "file_changes",
        conversationId,
        isV1Conversation,
        conversationUrl,
        gitPath,
      ],
    });
    return result.refetch();
  }, [
    queryClient,
    conversationId,
    isV1Conversation,
    conversationUrl,
    gitPath,
    result.refetch,
  ]);

  return {
    data: orderedChanges,
    isLoading: result.isLoading,
    isSuccess: result.isSuccess,
    isError: result.isError,
    error: result.error,
    refetch: customRefetch,
  };
};
