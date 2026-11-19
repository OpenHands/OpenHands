import React from "react";
import { useQuery } from "@tanstack/react-query";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { getGitPath } from "#/utils/get-git-path";
import type { GitChange } from "#/api/open-hands.types";

export const useUnifiedGetGitChanges = () => {
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const [orderedChanges, setOrderedChanges] = React.useState<GitChange[]>([]);
  const previousDataRef = React.useRef<GitChange[] | null>(null);
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
          // The API can return duplicate entries for the same path (e.g.
          // staged and unstaged entries for one file); keep only the first
          // occurrence so the diff list shows one entry per changed file
          const seenPaths = new Set<string>();
          const uniqueCurrentData = currentData.filter((item) => {
            if (seenPaths.has(item.path)) {
              return false;
            }
            seenPaths.add(item.path);
            return true;
          });

          const currentIds = new Set(
            uniqueCurrentData.map((item) => item.path),
          );

          // Merge against the latest stored changes (functional update) so
          // entries already processed in previous renders are never re-added
          setOrderedChanges((previousChanges) => {
            const existingIds = new Set(
              previousChanges.map((item) => item.path),
            );

            // Filter out items that already exist in orderedChanges
            const newItems = uniqueCurrentData.filter(
              (item) => !existingIds.has(item.path),
            );

            // Filter out items that no longer exist in the API response
            const existingItems = previousChanges.filter((item) =>
              currentIds.has(item.path),
            );

            // Add new items to the beginning
            return [...newItems, ...existingItems];
          });
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
    isFetching: result.isFetching,
    isSuccess: result.isSuccess,
    isError: result.isError,
    error: result.error,
    refetch: result.refetch,
  };
};
