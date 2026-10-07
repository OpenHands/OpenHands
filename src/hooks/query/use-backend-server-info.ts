import { useQuery } from "@tanstack/react-query";

import { ServerClient } from "@openhands/typescript-client/clients";
import {
  type AgentServerInfo,
  getDisplayAgentServerVersion,
} from "#/api/agent-server-compatibility";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { type Backend } from "#/api/backend-registry/types";

/**
 * Backend version cache key, shared by every `/server_info` probe so the
 * version badge and the execution-mode badge read one cached response.
 */
export function backendVersionQueryKey(backend: Backend) {
  return ["backend-version", backend.host, backend.apiKey] as const;
}

/**
 * Probe a backend's agent-server `/server_info`.
 *
 * Gated to local backends: cloud backends do not expose `/server_info`, so no
 * request is made and the query stays `undefined`. Callers may pass
 * `enabled: false` to additionally skip the probe (e.g. a backend known to be
 * disabled). Returns the raw server info so callers can derive both the version
 * and the execution mode from a single cached response.
 *
 * @spec BM-004 — Display the active backend's execution mode
 */
export function useBackendServerInfo(
  backend: Backend,
  options?: { enabled?: boolean },
) {
  return useQuery({
    queryKey: backendVersionQueryKey(backend),
    queryFn: async (): Promise<AgentServerInfo | null> => {
      const info = await new ServerClient(
        getAgentServerClientOptions({
          host: backend.host,
          sessionApiKey: backend.apiKey || null,
          timeout: 5000,
        }),
      ).getServerInfo();
      return info as AgentServerInfo;
    },
    retry: false,
    staleTime: 60_000,
    enabled: (options?.enabled ?? true) && backend.kind === "local",
  });
}

/**
 * The backend's displayed agent-server version, or `null` when unavailable.
 *
 * @spec BM-004 — Display the active backend's execution mode
 */
export function useBackendVersion(
  backend: Backend,
  options?: { enabled?: boolean },
): string | null {
  const { data } = useBackendServerInfo(backend, options);
  return data ? getDisplayAgentServerVersion(data) : null;
}
