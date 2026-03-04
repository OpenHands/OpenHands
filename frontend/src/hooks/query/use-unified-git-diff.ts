import React from "react";
import { useQuery } from "@tanstack/react-query";
import GitService from "#/api/git-service/git-service.api";
import V1GitService from "#/api/git-service/v1-git-service.api";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { getGitPath } from "#/utils/get-git-path";
import type { GitChangeStatus } from "#/api/open-hands.types";

/**
 * Check if the URL is a local development URL (relative path without host)
 * In local development, we should use V0 endpoints even for V1 conversations
 */
function isLocalDevelopment(conversationUrl: string | null | undefined): boolean {
  // If conversationUrl is a relative path (starts with /api), it's local development
  // In this case, the backend doesn't have a separate agent-server
  return !conversationUrl || conversationUrl.startsWith('/api');
}

type UseUnifiedGitDiffConfig = {
  filePath: string;
  type: GitChangeStatus;
  enabled: boolean;
};

/**
 * Unified hook to get git diff for both legacy (V0) and V1 conversations
 * - V0: Uses the legacy GitService.getGitChangeDiff API endpoint
 * - V1: Uses the V1GitService.getGitChangeDiff API endpoint with runtime URL
 */
export const useUnifiedGitDiff = (config: UseUnifiedGitDiffConfig) => {
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();

  const isV1Conversation = conversation?.conversation_version === "V1";
  const conversationUrl = conversation?.url;
  const sessionApiKey = conversation?.session_api_key;
  const selectedRepository = conversation?.selected_repository;

  // For V1, we need to convert the relative file path to an absolute path
  // The diff endpoint expects: /workspace/project/RepoName/relative/path
  const absoluteFilePath = React.useMemo(() => {
    if (!isV1Conversation) return config.filePath;

    const gitPath = getGitPath(selectedRepository);
    return `${gitPath}/${config.filePath}`;
  }, [isV1Conversation, selectedRepository, config.filePath]);

  return useQuery({
    queryKey: [
      "file_diff",
      conversationId,
      config.filePath,
      config.type,
      isV1Conversation,
      conversationUrl,
    ],
    queryFn: async () => {
      if (!conversationId) throw new Error("No conversation ID");

      // V1: Use the V1 API endpoint with runtime URL and absolute path
      // But for local development (where conversationUrl is relative), use V0 endpoints
      // because there's no separate agent-server in local mode
      if (isV1Conversation && !isLocalDevelopment(conversationUrl)) {
        return V1GitService.getGitChangeDiff(
          conversationUrl,
          sessionApiKey,
          absoluteFilePath,
        );
      }

      // V0 (Legacy): Use the legacy API endpoint with relative path
      return GitService.getGitChangeDiff(conversationId, config.filePath);
    },
    enabled: config.enabled,
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 15, // 15 minutes
  });
};
