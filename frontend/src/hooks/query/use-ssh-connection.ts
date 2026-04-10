import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { I18nKey } from "#/i18n/declaration";
import {
  extractSSHConnectionInfo,
  buildVSCodeRemoteSSHUrl,
} from "#/utils/vscode-url-helper";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useBatchAppConversations } from "./use-batch-app-conversations";
import { useBatchSandboxes } from "./use-batch-sandboxes";

interface SSHConnectionResult {
  host: string | null;
  port: number | null;
  vscodeRemoteUrl: string | null;
  error: string | null;
}

const DEFAULT_WORKSPACE_PATH = "/workspace/project";

/**
 * Hook to get SSH connection info for V1 conversations
 * Used to construct VSCode Remote-SSH URI for opening sandbox in local VSCode
 */
export const useSSHConnection = () => {
  const { t } = useTranslation();
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const runtimeIsReady = useRuntimeIsReady({ allowAgentError: true });

  const isV1Conversation = conversation?.conversation_version === "V1";

  // Fetch V1 app conversation to get sandbox_id
  const appConversationsQuery = useBatchAppConversations(
    isV1Conversation && conversationId ? [conversationId] : [],
  );
  const appConversation = appConversationsQuery.data?.[0];
  const sandboxId = appConversation?.sandbox_id;

  // Fetch sandbox data for V1 conversations
  const sandboxesQuery = useBatchSandboxes(sandboxId ? [sandboxId] : []);
  const sandbox = sandboxesQuery.data?.[0];

  const mainQuery = useQuery<SSHConnectionResult>({
    queryKey: [
      "ssh",
      "connection",
      conversationId,
      isV1Conversation,
      sandboxId,
      sandbox,
    ],
    queryFn: async () => {
      if (!conversationId) throw new Error("No conversation ID");

      // SSH is only available for V1 conversations
      if (!isV1Conversation) {
        return {
          host: null,
          port: null,
          vscodeRemoteUrl: null,
          error: t(I18nKey.SSH$NOT_AVAILABLE_V0),
        };
      }

      if (!sandbox) {
        return {
          host: null,
          port: null,
          vscodeRemoteUrl: null,
          error: t(I18nKey.SSH$URL_NOT_AVAILABLE),
        };
      }

      const sshExposedUrl = sandbox.exposed_urls?.find(
        (url) => url.name === "SSH",
      );

      if (!sshExposedUrl) {
        return {
          host: null,
          port: null,
          vscodeRemoteUrl: null,
          error: t(I18nKey.SSH$URL_NOT_AVAILABLE),
        };
      }

      const connectionInfo = extractSSHConnectionInfo(sshExposedUrl.url);

      if (!connectionInfo) {
        return {
          host: null,
          port: null,
          vscodeRemoteUrl: null,
          error: t(I18nKey.SSH$URL_NOT_AVAILABLE),
        };
      }

      const vscodeRemoteUrl = buildVSCodeRemoteSSHUrl(
        connectionInfo.host,
        connectionInfo.port,
        DEFAULT_WORKSPACE_PATH,
      );

      return {
        host: connectionInfo.host,
        port: connectionInfo.port,
        vscodeRemoteUrl,
        error: null,
      };
    },
    enabled:
      runtimeIsReady && !!conversationId && isV1Conversation && !!sandbox,
    refetchOnMount: true,
    retry: 3,
  });

  // Calculate overall loading state including dependent queries for V1
  const isLoading =
    appConversationsQuery.isLoading ||
    sandboxesQuery.isLoading ||
    mainQuery.isLoading;

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
