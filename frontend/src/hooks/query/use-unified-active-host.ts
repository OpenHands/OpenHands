import { useQueries, useQuery } from "@tanstack/react-query";
import axios from "axios";
import React from "react";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useRuntimeIsReady } from "#/hooks/use-runtime-is-ready";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useBatchSandboxes } from "./use-batch-sandboxes";

/**
 * Unified hook to get active web host for both legacy (V0) and V1 conversations
 * - V0: Uses the legacy getWebHosts API endpoint and polls them
 * - V1: Gets worker URLs from sandbox exposed_urls (WORKER_1, WORKER_2, etc.)
 */
export const useUnifiedActiveHost = () => {
  const [activeHost, setActiveHost] = React.useState<string | null>(null);
  const { conversationId } = useConversationId();
  const runtimeIsReady = useRuntimeIsReady();
  const { data: conversation, isLoading: isLoadingConversation } =
    useActiveConversation();
  const sandboxId = conversation?.sandbox_id;

  // Fetch sandbox data for V1 conversations
  const sandboxesQuery = useBatchSandboxes(sandboxId ? [sandboxId] : []);
  const sandbox = sandboxesQuery?.data?.[0];

  // Get worker URLs from V1 sandbox or legacy web hosts from V0
  const { data, isLoading: hostsQueryLoading } = useQuery({
    queryKey: [conversationId, "hosts", sandbox],
    queryFn: async () => {
      // V1: Get worker URLs from sandbox exposed_urls
      if (!sandbox) {
        return { hosts: [] };
      }

      const workerUrls =
        sandbox.exposed_urls
          ?.filter((url) => url.name.startsWith("WORKER_"))
          .map((url) => url.url) || [];

      return { hosts: workerUrls };
    },
    enabled: runtimeIsReady && !!conversationId && !!sandboxesQuery.data,
    initialData: { hosts: [] },
    meta: {
      disableToast: true,
    },
  });

  // Poll all hosts to find which one is active
  const apps = useQueries({
    queries: data.hosts.map((host) => ({
      queryKey: [conversationId, "unified", "hosts", host],
      queryFn: async () => {
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
    const successfulApp = appsData.find((app) => app);
    setActiveHost(successfulApp || "");
  }, [appsData]);

  // Calculate overall loading state including dependent queries for V1
  const isLoading =
    isLoadingConversation || sandboxesQuery.isLoading || hostsQueryLoading;

  return { activeHost, isLoading };
};
