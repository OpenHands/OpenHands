import React from "react";
import { useQuery } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import V1GitService from "#/api/git-service/v1-git-service.api";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { getGitPath } from "#/utils/get-git-path";
import type { GitChange } from "#/api/open-hands.types";

/**
 * Check if the URL is a local development URL (relative path without host)
 * In local development, we should use V0 endpoints even for V1 conversations
 */
function isLocalDevelopment(conversationUrl: string | null | undefined): boolean {
  // If conversationUrl is a relative path (starts with /api), it's local development
  // In this case, the backend doesn't have a separate agent-server
  return !conversationUrl || conversationUrl.startsWith('/api');
}

/**
 * Unified hook to get git changes for both legacy (V0) and V1 conversations
 * - V0: Uses the legacy GitService.getGitChanges API endpoint
 * - V1: Uses the V1GitService.getGitChangeDiff API endpoint with runtime URL
 *   (but falls back to V0 for local development where there's no separate agent-server)
 */
export const useUnifiedGetGitChanges = () => {
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const [orderedChanges, setOrderedChanges] = React.useState<GitChange[]>([]);
  const previousDataRef = React.useRef<GitChange[] | null>(null);
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
      // But for local development (where conversationUrl is relative), use V0 endpoints
      // because there's no separate agent-server in local mode
      if (isV1Conversation && !isLocalDevelopment(conversationUrl)) {
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

  // Latest changes should be on top
  React.useEffect(() => {
    if (!result.isFetching && result.isSuccess && result.data) {
      const currentData = result.data;

      // If this is new data (not the same reference as before)
      if (currentData !== previousDataRef.current) {
        previousDataRef.current = currentData;

        // Figure out new items by comparing with what we already have
        if (Array.isArray(currentData)) {
          const currentIds = new Set(currentData.map((item) => item.path));
          const existingIds = new Set(orderedChanges.map((item) => item.path));

          // Filter out items that already exist in orderedChanges
          const newItems = currentData.filter(
            (item) => !existingIds.has(item.path),
          );

          // Filter out items that no longer exist in the API response
          const existingItems = orderedChanges.filter((item) =>
            currentIds.has(item.path),
          );

          // Add new items to the beginning
          setOrderedChanges([...newItems, ...existingItems]);
        } else {
          // If not an array, just use the data directly
          setOrderedChanges([currentData]);
        }
      }
    }
  }, [result.isFetching, result.isSuccess, result.data]);

  return {
    data: orderedChanges,
    isLoading: result.isLoading,
    isSuccess: result.isSuccess,
    isError: result.isError,
    error: result.error,
    refetch: result.refetch,
  };
};
