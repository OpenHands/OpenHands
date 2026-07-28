import { useQuery } from "@tanstack/react-query";
import type { VSCodeStatusResponse } from "@openhands/typescript-client";
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

  // Capability probe. `/api/vscode/status` answers 200 with
  // `enabled: false` when the deployment set `enable_vscode: false`, so a
  // deliberately editor-less deployment is a value here rather than an
  // error — unlike `/api/vscode/url`, which answers 503 and is therefore
  // indistinguishable from an auth, proxy, or server failure.
  //
  // Gating the URL request on this means a disabled editor never issues
  // the 503 in the first place, so the global error toast needs no
  // blanket suppression and genuine failures stay observable.
  const statusQuery = useQuery<VSCodeStatusResponse>({
    queryKey: [
      "unified",
      "vscode_status",
      "local",
      conversationId,
      conversationUrl,
      sessionApiKey,
    ],
    queryFn: () =>
      AgentServerConversationService.getVSCodeStatus(
        conversationUrl,
        sessionApiKey,
      ),
    enabled: !isCloud && runtimeIsReady && !!conversationId,
    refetchOnMount: true,
  });

  // `enabled: false` is the deployment switch. `running: false` alongside
  // `enabled: true` means the process failed to start or has died: the
  // agent-server awaits `VSCodeService.start()` in its lifespan before it
  // serves any request, so this is a terminal state rather than a startup
  // window we would be racing.
  const editorIsAvailable =
    statusQuery.data?.enabled === true && statusQuery.data?.running === true;

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
    enabled:
      !isCloud && runtimeIsReady && !!conversationId && editorIsAvailable,
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
  // True once we know there is nothing to open, so callers can render nothing
  // instead of a control whose activation is a no-op.
  let isUnavailable: boolean;

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
    // Cloud behavior is deliberately unchanged: a sandbox with no VSCODE
    // entry in `exposed_urls` still surfaces the control. Narrowing this
    // change to self-hosted keeps its blast radius off the cloud path.
    isUnavailable = false;
  } else {
    data = localQuery.data;
    // The URL request only starts once the capability probe has cleared it,
    // so the control is "loading" for the probe as well — otherwise it would
    // look ready while there is still nothing to open.
    isLoading = statusQuery.isLoading || localQuery.isLoading;
    isError = statusQuery.isError || localQuery.isError;
    isSuccess = localQuery.isSuccess;
    status = localQuery.status;
    error = statusQuery.error ?? localQuery.error;
    refetch = async () => {
      const result = await localQuery.refetch();
      return { data: result.data };
    };
    // Hide only on an explicit, terminal capability answer:
    //   - the probe succeeded and reports no usable editor (disabled, or
    //     enabled but not running), or
    //   - the editor is there but reports no URL to open.
    // A failed probe is deliberately not "unavailable": transport, auth and
    // server faults stay visible as query errors with their normal retry and
    // toast, rather than silently removing the control.
    isUnavailable =
      (statusQuery.isSuccess && !editorIsAvailable) ||
      (isSuccess && !localQuery.data?.url);
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
    isUnavailable,
    status,
    refetch,
  };
};
