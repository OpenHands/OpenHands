import { useQuery } from "@tanstack/react-query";

import AgentServerRuntimeService from "#/api/runtime-service/agent-server-runtime-service";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useWorkspaceMutationCounter } from "#/stores/use-workspace-mutation-counter";
import { getGitPath } from "#/utils/get-git-path";

/**
 * Reads a workspace file's bytes through the agent server's
 * header-authenticated `/api/file/download` route, for callers that cannot
 * rely on the workspace-session cookie (see the Files tab's "open in new
 * window" link for a backend on another origin, #19038).
 *
 * Pass a falsy `relativePath` to disable the query.
 */
export function useWorkspaceFileBytes(relativePath: string | null) {
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady({ allowAgentError: true });
  const workspaceMutationCount = useWorkspaceMutationCounter(
    (state) => state.count,
  );

  const conversationId = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;
  // `/api/file/download` rejects relative paths; anchor against the working
  // dir the same way `useWorkspaceFileContent` does.
  const gitPath = getGitPath(
    conversation?.selected_repository,
    conversation?.workspace?.working_dir,
  );
  const workspaceRoot = gitPath.startsWith("/") ? gitPath : `/${gitPath}`;
  const absoluteFilePath = relativePath
    ? `${workspaceRoot}/${relativePath}`
    : null;

  return useQuery({
    queryKey: [
      "workspace-file-bytes",
      conversationId,
      conversationUrl,
      sessionApiKey,
      absoluteFilePath,
      workspaceMutationCount,
    ],
    queryFn: () =>
      AgentServerRuntimeService.downloadFile(
        conversationUrl,
        sessionApiKey,
        absoluteFilePath!,
      ),
    enabled: runtimeIsReady && !!conversationId && !!absoluteFilePath,
    retry: false,
    staleTime: 1000 * 5,
    gcTime: 1000 * 60,
    meta: { disableToast: true },
  });
}
