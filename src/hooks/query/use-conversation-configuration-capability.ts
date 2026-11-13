import { useQuery } from "@tanstack/react-query";
import { ServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { CONVERSATION_CONFIGURATION_QUERY_KEYS } from "./query-keys";

const CONFIGURATION_ADDITIONS_CAPABILITY =
  "conversation_configuration_additions_v1";
const MCP_INSTALL_CAPABILITY = "conversation_mcp_install_v1";

function useConfigurationCapabilities(enabled: boolean) {
  const { backend } = useActiveBackend();
  // The connection revision changes with host/credentials; keep API keys out of the cache key.
  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  return useQuery({
    queryKey: CONVERSATION_CONFIGURATION_QUERY_KEYS.capability(
      backend.id,
      backend.connectionRevision,
    ),
    queryFn: async () => {
      const info = await new ServerClient(
        getAgentServerClientOptions({
          host: backend.host,
          apiKey: backend.apiKey,
        }),
      ).getServerInfo();
      return info.capabilities ?? [];
    },
    enabled: enabled && backend.kind === "local",
    staleTime: 5 * 60 * 1000,
    retry: false,
    meta: { disableToast: true },
  });
}

export function useConversationConfigurationCapability(enabled: boolean) {
  const { data } = useConfigurationCapabilities(enabled);
  return { data: data?.includes(CONFIGURATION_ADDITIONS_CAPABILITY) };
}

export function useConversationMcpCapabilities(enabled: boolean) {
  const { data } = useConfigurationCapabilities(enabled);
  return { canInstall: data?.includes(MCP_INSTALL_CAPABILITY) === true };
}
