import { useQueries, useQuery } from "@tanstack/react-query";
import axios from "axios";
import React from "react";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useBatchSandboxes } from "./use-batch-sandboxes";
import { useConversationConfig } from "./use-conversation-config";

function appendDefaultQueryParams(url: string): string {
  try {
    const parsed = new URL(url);
    parsed.searchParams.set("mainLayoutScreen", "singleScreenLayout");
    parsed.searchParams.set("initialScreen", "chatPreviewInitialScreen");
    parsed.searchParams.set("colorScheme", "light");
    parsed.searchParams.set("_mcp", "1");
    const previewApp = new URLSearchParams(window.location.search).get(
      "preview_app",
    );
    if (previewApp) {
      parsed.searchParams.set("app", previewApp);
    }
    return parsed.toString();
  } catch {
    return url;
  }
}

/**
 * Unified hook to get active web host for both legacy (V0) and V1 conversations
 * - V0: Uses the legacy getWebHosts API endpoint and polls them
 * - V1: Gets worker URLs from sandbox exposed_urls (WORKER_1, WORKER_2, etc.)
 */
export const useUnifiedActiveHost = () => {
  const [activeHost, setActiveHost] = React.useState<string | null>(null);
  const { conversationId } = useConversationId();
  const runtimeIsReady = useRuntimeIsReady();
  const { data: conversation } = useActiveConversation();
  const { data: conversationConfig, isLoading: isLoadingConfig } =
    useConversationConfig();

  const isV1Conversation = conversation?.conversation_version === "V1";
  const sessionEnvironmentUrl = sessionStorage.getItem(
    `environment-url:${conversationId}`,
  );
  // Treat a sessionStorage env URL as proof of an environment connection so the
  // sandbox-host fallback never wins during the brief window before the
  // conversation (and its `trigger`) finishes loading. Otherwise, in environments
  // where the sandbox exposes a WORKER_* URL (e.g. Codespaces port forwarding),
  // that URL gets latched into `activeHost` and isn't replaced.
  const isEnvironmentConnection =
    conversation?.trigger === "connect_to_environment" ||
    Boolean(sessionEnvironmentUrl);
  const environmentUrl =
    sessionEnvironmentUrl || conversation?.environment_url || null;
  const sandboxId = conversationConfig?.runtime_id;

  // For environment connections, use the environment URL directly
  // No sandbox lookup, no health-check polling needed
  React.useEffect(() => {
    if (isEnvironmentConnection && environmentUrl) {
      setActiveHost(appendDefaultQueryParams(environmentUrl));
    }
  }, [isEnvironmentConnection, environmentUrl]);

  // Fetch sandbox data for V1 conversations (skip for environment connections)
  const sandboxesQuery = useBatchSandboxes(
    sandboxId && !isEnvironmentConnection ? [sandboxId] : [],
  );

  // Get worker URLs from V1 sandbox or legacy web hosts from V0
  const { data, isLoading: hostsQueryLoading } = useQuery({
    queryKey: [conversationId, "unified", "hosts", isV1Conversation, sandboxId],
    queryFn: async () => {
      // V1: Get worker URLs from sandbox exposed_urls
      if (isV1Conversation) {
        if (
          !sandboxesQuery.data ||
          sandboxesQuery.data.length === 0 ||
          !sandboxesQuery.data[0]
        ) {
          return { hosts: [] };
        }

        const sandbox = sandboxesQuery.data[0];
        const workerUrls =
          sandbox.exposed_urls
            ?.filter((url) => url.name.startsWith("WORKER_"))
            .map((url) => url.url) || [];

        return { hosts: workerUrls };
      }

      // V0 (Legacy): Use the legacy API endpoint
      const hosts = await ConversationService.getWebHosts(conversationId);
      return { hosts };
    },
    enabled:
      !isEnvironmentConnection &&
      runtimeIsReady &&
      !!conversationId &&
      (!isV1Conversation || !!sandboxesQuery.data),
    initialData: { hosts: [] },
    meta: {
      disableToast: true,
    },
  });

  // Poll all hosts to find which one is active (skip for environment connections)
  const apps = useQueries({
    queries: isEnvironmentConnection
      ? []
      : data.hosts.map((host) => ({
          queryKey: [conversationId, "unified", "hosts", host],
          queryFn: async () => {
            // Skip XHR health check for cross-origin URLs (e.g., Codespaces port
            // forwarding) since CORS will block the request.  The URL will be
            // loaded in an iframe which doesn't have CORS restrictions.
            try {
              const hostOrigin = new URL(host).origin;
              if (hostOrigin !== window.location.origin) {
                return host;
              }
            } catch {
              // invalid URL — fall through to the normal check
            }
            try {
              await axios.get(host);
              return host;
            } catch (e) {
              return "";
            }
          },
          refetchInterval: 3000,
          meta: {
            disableToast: true,
          },
        })),
  });

  const appsData = apps.map((app) => app.data);

  React.useEffect(() => {
    if (isEnvironmentConnection) return;
    const successfulApp = appsData.find((app) => app);
    setActiveHost(successfulApp ? appendDefaultQueryParams(successfulApp) : "");
  }, [appsData, isEnvironmentConnection]);

  // Calculate overall loading state including dependent queries for V1
  const getLoadingState = () => {
    if (isEnvironmentConnection) return false;
    if (isV1Conversation)
      return isLoadingConfig || sandboxesQuery.isLoading || hostsQueryLoading;
    return hostsQueryLoading;
  };
  const isLoading = getLoadingState();

  return { activeHost, isLoading };
};
