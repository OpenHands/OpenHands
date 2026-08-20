import React from "react";
import { useQuery } from "@tanstack/react-query";
import AgentServerGitService from "#/api/git-service/agent-server-git-service.api";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { getGitPath } from "#/utils/get-git-path";

type UseCommitChangesConfig = {
  enabled: boolean;
};

export const useCommitChanges = (
  sha: string,
  config: UseCommitChangesConfig,
) => {
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

  return useQuery({
    queryKey: [
      "commit_changes",
      conversationId,
      conversationUrl,
      sessionApiKey,
      gitPath,
      sha,
    ],
    queryFn: async () => {
      if (!conversationId) throw new Error("No conversation ID");

      return AgentServerGitService.getCommitChanges(
        conversationUrl,
        sessionApiKey,
        gitPath,
        sha,
      );
    },
    // A commit's change list is sha-addressed and immutable — never stale,
    // and deliberately not invalidated by the bash-observation refresh.
    staleTime: Infinity,
    gcTime: 1000 * 60 * 15, // 15 minutes
    retry: false,
    enabled: config.enabled && runtimeIsReady && !!conversationId,
    meta: {
      disableToast: true,
    },
  });
};
