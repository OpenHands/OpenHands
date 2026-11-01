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
  const [order, setOrder] = React.useState<{
    identity: string;
    paths: string[];
  }>({ identity: "", paths: [] });
  const runtimeIsReady = useRuntimeIsReady();

  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  const selectedRepository = conversation?.selected_repository;
  const workingDir = conversation?.workspace?.working_dir?.trim();

  const gitPath = React.useMemo(
    () => getGitPath(selectedRepository, workingDir),
    [selectedRepository, workingDir],
  );

  const queryKey = [
    "file_changes",
    conversationId,
    conversationUrl,
    sessionApiKey,
    gitPath,
  ];
  const identity = JSON.stringify(queryKey);
  const result = useQuery({
    queryKey,
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

  const currentChanges: GitChange[] = result.data
    ? Array.isArray(result.data)
      ? result.data
      : [result.data]
    : [];
  const currentByPath = new Map(
    currentChanges.map((change) => [change.path, change]),
  );
  const previousPaths = order.identity === identity ? order.paths : [];
  const previousPathSet = new Set(previousPaths);
  const paths = [
    ...currentChanges
      .filter((change) => !previousPathSet.has(change.path))
      .map((change) => change.path),
    ...previousPaths.filter((path) => currentByPath.has(path)),
  ];
  if (
    order.identity !== identity ||
    paths.length !== order.paths.length ||
    paths.some((path, index) => path !== order.paths[index])
  ) {
    setOrder({ identity, paths });
  }

  return {
    data: paths.map((path) => currentByPath.get(path)!),
    isLoading: result.isLoading,
    isFetching: result.isFetching,
    isSuccess: result.isSuccess,
    isError: result.isError,
    error: result.error,
    refetch: result.refetch,
  };
};
