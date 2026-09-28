import { useQuery } from "@tanstack/react-query";
import { ServerClient } from "@openhands/typescript-client/clients";

import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import {
  type AgentServerInfo,
  getCachedAgentServerInfo,
} from "#/api/agent-server-compatibility";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  AGENT_SERVER_INFO_QUERY_KEYS,
  CONFIG_CACHE_OPTIONS,
} from "#/hooks/query/query-keys";

/** `/server_info` of the active local backend; `null` for cloud or no backend. */
export function useAgentServerInfo() {
  const { backend } = useActiveBackend();
  const isLocal = backend.kind === "local" && !isNoBackend(backend);

  return useQuery({
    queryKey: [
      ...AGENT_SERVER_INFO_QUERY_KEYS.all,
      backend.id,
      backend.host,
      backend.connectionRevision ?? 0,
      isLocal,
    ],
    queryFn: async (): Promise<AgentServerInfo | null> => {
      if (!isLocal) return null;
      const client = new ServerClient(
        getAgentServerClientOptions({ host: backend.host }),
      );
      return (await client.getServerInfo()) as AgentServerInfo;
    },
    initialData: () =>
      isLocal
        ? (getCachedAgentServerInfo({ host: backend.host }) ?? undefined)
        : null,
    ...CONFIG_CACHE_OPTIONS,
    meta: { disableToast: true },
  });
}
