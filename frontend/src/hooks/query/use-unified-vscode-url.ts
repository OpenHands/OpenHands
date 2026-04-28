import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useConversationId } from "#/hooks/use-conversation-id";
import { I18nKey } from "#/i18n/declaration";
import { transformVSCodeUrl } from "#/utils/vscode-url-helper";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useBatchAppConversations } from "./use-batch-app-conversations";
import { useBatchSandboxes } from "./use-batch-sandboxes";

interface VSCodeUrlResult {
  url: string | null;
  error: string | null;
}

/**
 * Unified hook to get VSCode URL for both legacy (V0) and V1 conversations
 * - V0: Uses the legacy getVSCodeUrl API endpoint
 * - V1: Gets the VSCode URL from sandbox exposed_urls
 */
export const useUnifiedVSCodeUrl = () => {
  const { t } = useTranslation();
  const { conversationId } = useConversationId();
  const runtimeIsReady = useRuntimeIsReady({ allowAgentError: true });

  // Fetch V1 app conversation to get sandbox_id
  const appConversationsQuery = useBatchAppConversations(
    conversationId ? [conversationId] : [],
  );
  const appConversation = appConversationsQuery.data?.[0];
  const sandboxId = appConversation?.sandbox_id;

  // Fetch sandbox data for V1 conversations
  const sandboxesQuery = useBatchSandboxes(sandboxId ? [sandboxId] : []);
  const sandbox = sandboxesQuery?.data?.[0];

  const mainQuery = useQuery<VSCodeUrlResult>({
    queryKey: ["unified", "vscode_url", conversationId, sandbox],
    queryFn: async () => {
      if (!conversationId) throw new Error("No conversation ID");

      // V1: Get VSCode URL from sandbox exposed_urls
      if (!sandbox) {
        return {
          url: null,
          error: t(I18nKey.VSCODE$URL_NOT_AVAILABLE),
        };
      }

      const vscodeUrl = sandbox.exposed_urls?.find(
        (url) => url.name === "VSCODE",
      );

      if (!vscodeUrl) {
        return {
          url: null,
          error: t(I18nKey.VSCODE$URL_NOT_AVAILABLE),
        };
      }

      return {
        url: transformVSCodeUrl(vscodeUrl.url),
        error: null,
      };
    },
    enabled: runtimeIsReady && !!conversationId && !!sandboxesQuery.data,
    refetchOnMount: true,
    retry: 3,
  });

  // Calculate overall loading state including dependent queries for V1
  const isLoading =
    appConversationsQuery.isLoading ||
    sandboxesQuery.isLoading ||
    mainQuery.isLoading;

  // Explicitly destructure to avoid excessive re-renders from spreading the entire query object
  return {
    data: mainQuery.data,
    error: mainQuery.error,
    isLoading,
    isError: mainQuery.isError,
    isSuccess: mainQuery.isSuccess,
    status: mainQuery.status,
    refetch: mainQuery.refetch,
  };
};
