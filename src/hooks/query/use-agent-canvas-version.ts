import { useQuery } from "@tanstack/react-query";
import packageJson from "../../../package.json";
import { compareSemanticVersions } from "#/utils/version";
import { AGENT_CANVAS_VERSION_QUERY_KEYS } from "./query-keys";

const NPM_LATEST_URL =
  "https://registry.npmjs.org/@openhands%2Fagent-canvas/latest";

interface NpmLatestResponse {
  version?: unknown;
}

async function fetchLatestAgentCanvasVersion(
  signal?: AbortSignal,
): Promise<string | null> {
  const response = await fetch(NPM_LATEST_URL, {
    headers: { Accept: "application/json" },
    signal,
  });

  if (!response.ok) return null;

  const data = (await response.json()) as NpmLatestResponse;
  return typeof data.version === "string" ? data.version : null;
}

export function useAgentCanvasVersion() {
  const installedVersion = packageJson.version;
  const latestVersionQuery = useQuery({
    queryKey: AGENT_CANVAS_VERSION_QUERY_KEYS.latest,
    queryFn: ({ signal }) => fetchLatestAgentCanvasVersion(signal),
    staleTime: 1000 * 60 * 30,
    gcTime: 1000 * 60 * 60,
    retry: false,
    meta: { disableToast: true },
  });

  const latestVersion = latestVersionQuery.data ?? null;
  const updateAvailable =
    latestVersion !== null &&
    compareSemanticVersions(installedVersion, latestVersion) < 0;

  return {
    installedVersion,
    latestVersion,
    updateAvailable,
    isChecking: latestVersionQuery.isFetching,
    checkForUpdates: latestVersionQuery.refetch,
  };
}
