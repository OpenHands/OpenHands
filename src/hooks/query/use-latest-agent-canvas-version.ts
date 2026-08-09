import { useQuery } from "@tanstack/react-query";
import { fetchLatestAgentCanvasVersion } from "#/api/agent-canvas-updates";
import { APP_UPDATE_QUERY_KEYS } from "./query-keys";

/**
 * Latest published Agent Canvas version (npm `latest` dist-tag).
 *
 * Heimdall fork: disabled by default — this tree has diverged from upstream
 * OpenHands/`@openhands/agent-canvas`, so we do not poll npm for updates.
 * Pass ``enabled: true`` only in unit tests of the legacy update UI.
 *
 * Information-only: failures must stay quiet (`meta.disableToast`).
 */
export function useLatestAgentCanvasVersion({
  enabled = false,
}: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: APP_UPDATE_QUERY_KEYS.latestVersion,
    queryFn: ({ signal }) => fetchLatestAgentCanvasVersion(signal),
    enabled,
    // Offline/air-gapped must settle into the quiet inline state immediately
    // rather than sitting through exponential backoff.
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 1000 * 60 * 60, // the settings sidebar remounts per navigation; don't hammer npm
    meta: { disableToast: true },
  });
}
