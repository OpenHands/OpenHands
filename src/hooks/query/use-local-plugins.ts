import { useQuery } from "@tanstack/react-query";
import PluginsService, { type LocalPlugin } from "#/api/plugins-service";
import { PLUGINS_QUERY_KEYS } from "./query-keys";

/**
 * Query hook for the locally-discovered ("ambient") plugins on the local
 * agent-server (e.g. `~/.agents/plugins`). Read-only and local-backend only — a
 * cloud backend yields an empty list. Rendered as the read-only "Local" group on
 * the Plugins page. Mirrors `usePlugins`.
 */
export const useLocalPlugins = () =>
  useQuery<LocalPlugin[]>({
    queryKey: PLUGINS_QUERY_KEYS.local,
    queryFn: () => PluginsService.getLocalPlugins(),
    staleTime: 1000 * 60 * 10, // 10 minutes
    refetchOnWindowFocus: false,
  });
