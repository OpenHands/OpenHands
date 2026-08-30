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

  // Identity of the git surface this hook is observing. When it changes
  // (conversation switch, different repo/workdir), path ordering from the
  // previous workspace must not leak into the next one (#16949).
  const queryIdentity = React.useMemo(
    () => [conversationId, conversationUrl, sessionApiKey, gitPath].join("|"),
    [conversationId, conversationUrl, sessionApiKey, gitPath],
  );

  // Ordering is preserved as PATH IDENTITIES only; the objects rendered always
  // come from the latest response. The previous implementation mirrored stale
  // `GitChange` objects, so a retained path whose status changed (e.g. M -> D)
  // kept showing the old status until its cache entry expired (#16949).
  const [ordering, setOrdering] = React.useState<{
    identity: string;
    paths: string[];
  }>({ identity: queryIdentity, paths: [] });

  React.useEffect(() => {
    if (result.isFetching || !result.isSuccess || !result.data) return;
    const currentPaths = result.data.map((item) => item.path);
    setOrdering((prev) => {
      const currentPathSet = new Set(currentPaths);
      // Conversation/git identity changed: rebuild ordering from the new
      // response instead of reusing the previous workspace's path order.
      if (prev.identity !== queryIdentity) {
        return { identity: queryIdentity, paths: currentPaths };
      }
      // Drop removed paths, keep the established relative order, and prepend
      // newly-seen paths so the latest changes stay on top.
      const retained = prev.paths.filter((path) => currentPathSet.has(path));
      const prevPathSet = new Set(prev.paths);
      const newPaths = currentPaths.filter((path) => !prevPathSet.has(path));
      return { identity: prev.identity, paths: [...newPaths, ...retained] };
    });
  }, [result.isFetching, result.isSuccess, result.data, queryIdentity]);

  // Project the latest response objects in the established path order. Always
  // derive synchronously from the current response so a refetch never renders
  // a stale object (issue #16949).
  const orderedChanges = React.useMemo<GitChange[]>(() => {
    const current = result.data ?? [];
    if (ordering.paths.length === 0) return current;
    const byPath = new Map(current.map((item) => [item.path, item]));
    const ordered = ordering.paths
      .map((path) => byPath.get(path))
      .filter((change): change is GitChange => change !== undefined);
    // Append any path the response reports that we haven't ordered yet (e.g.
    // the very first data arriving before the ordering effect has run).
    const seen = new Set(ordered.map((item) => item.path));
    for (const item of current) {
      if (!seen.has(item.path)) ordered.push(item);
    }
    return ordered;
  }, [result.data, ordering.paths]);

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
