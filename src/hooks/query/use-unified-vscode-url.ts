import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useConversationId } from "#/hooks/use-conversation-id";
import { I18nKey } from "#/i18n/declaration";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { transformVSCodeUrl } from "#/utils/vscode-url-helper";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudSandbox } from "#/hooks/query/use-cloud-sandbox";

interface VSCodeUrlResult {
  url: string | null;
}

const VSCODE_EXPOSED_URL_NAME = "VSCODE";

export const useUnifiedVSCodeUrl = () => {
  const { t } = useTranslation("openhands");
  const { conversationId } = useConversationId();
  const runtimeIsReady = useRuntimeIsReady({ allowAgentError: true });
  const { data: conversation } = useActiveConversation();
  const active = useActiveBackend();

  const conversationUrl = conversation?.conversation_url ?? null;
  const sessionApiKey = conversation?.session_api_key ?? null;
  const sandboxId = conversation?.sandbox_id ?? null;
  const isCloud = active.backend.kind === "cloud";

  // Cloud mode: read VSCode URL from the cloud-computed `exposed_urls` on
  // the conversation's sandbox. The runtime's `/api/vscode/url` only
  // knows its internal `localhost:8001`, so calling it returned a URL
  // the user's browser couldn't reach.
  const cloudSandboxQuery = useCloudSandbox(isCloud ? sandboxId : null);

  const localQuery = useQuery<VSCodeUrlResult>({
    // Include conversation host + key in the cache key so different
    // conversations don't share VSCode URL data.
    queryKey: [
      "unified",
      "vscode_url",
      "local",
      conversationId,
      conversationUrl,
      sessionApiKey,
    ],
    queryFn: async () => {
      if (!conversationId) throw new Error("No conversation ID");

      const response = await AgentServerConversationService.getVSCodeUrl(
        conversationId,
        conversationUrl,
        sessionApiKey,
      ).catch(() => ConversationService.getVSCodeUrl(conversationId));

      return { url: transformVSCodeUrl(response.vscode_url) };
    },
    enabled: !isCloud && runtimeIsReady && !!conversationId,
    refetchOnMount: true,
    retry: 3,
  });

  let data: VSCodeUrlResult | undefined;
  let isLoading: boolean;
  let isError: boolean;
  let isSuccess: boolean;
  let status: typeof localQuery.status;
  let error: unknown;
  let refetch: () => Promise<{ data: VSCodeUrlResult | undefined }>;

  if (isCloud) {
    const sandbox = cloudSandboxQuery.data;
    const exposedUrl =
      sandbox?.exposed_urls?.find((u) => u.name === VSCODE_EXPOSED_URL_NAME)
        ?.url ?? null;
    data = cloudSandboxQuery.isSuccess
      ? { url: transformVSCodeUrl(exposedUrl) }
      : undefined;
    isLoading = cloudSandboxQuery.isLoading;
    isError = cloudSandboxQuery.isError;
    isSuccess = cloudSandboxQuery.isSuccess;
    status = cloudSandboxQuery.status;
    error = cloudSandboxQuery.error;
    refetch = async () => {
      const result = await cloudSandboxQuery.refetch();
      const refreshedUrl =
        result.data?.exposed_urls?.find(
          (u) => u.name === VSCODE_EXPOSED_URL_NAME,
        )?.url ?? null;
      return {
        data: result.data
          ? { url: transformVSCodeUrl(refreshedUrl) }
          : undefined,
      };
    };
  } else {
    data = localQuery.data;
    isLoading = localQuery.isLoading;
    isError = localQuery.isError;
    isSuccess = localQuery.isSuccess;
    status = localQuery.status;
    error = localQuery.error;
    refetch = async () => {
      const result = await localQuery.refetch();
      return { data: result.data };
    };
  }

  // Derive the i18n'd "URL unavailable" message outside `queryFn` so the
  // queryKey doesn't have to include `t`.
  const errorMessage =
    data && !data.url ? t(I18nKey.VSCODE$URL_NOT_AVAILABLE) : null;

  return {
    data: data ? { ...data, error: errorMessage } : undefined,
    error,
    isLoading,
    isError,
    isSuccess,
    status,
    refetch,
  };
};
