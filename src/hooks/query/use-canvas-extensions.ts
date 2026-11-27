import { useQuery } from "@tanstack/react-query";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { CANVAS_EXTENSIONS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { useCloudIntegrationsConfig } from "./use-cloud-integrations-config";

/** How often an open tab re-reads the organization's approved Apps. */
const CLOUD_REFETCH_INTERVAL_MS = 60_000;

/**
 * Whether the active backend serves Apps. A local Agent Server owns its own
 * Apps. A cloud backend serves the Apps approved for the selected organization,
 * and only when the instance turns them on (`enable_canvas_apps`).
 * `isLoading` is true until a cloud instance has answered.
 */
export function useCanvasExtensionsSupported() {
  const { backend, orgId } = useActiveBackend();
  const isCloud = !isNoBackend(backend) && backend.kind === "cloud";
  const cloudConfig = useCloudIntegrationsConfig();

  if (!isCloud) {
    return { isSupported: !isNoBackend(backend), isLoading: false };
  }
  return {
    isSupported: !!orgId && cloudConfig.data?.isCanvasAppsEnabled === true,
    isLoading: cloudConfig.isLoading,
  };
}

interface UseCanvasExtensionsOptions {
  enabled?: boolean;
}

export function useCanvasExtensions(options: UseCanvasExtensionsOptions = {}) {
  const active = useActiveBackend();
  const connectionRevision = active.backend.connectionRevision ?? 0;
  const { isSupported } = useCanvasExtensionsSupported();

  return useQuery({
    queryKey: CANVAS_EXTENSIONS_QUERY_KEYS.installed(
      active.backend.id,
      active.orgId,
      connectionRevision,
    ),
    queryFn: () => CanvasExtensionsService.listInstalled(),
    enabled: (options.enabled ?? true) && isSupported,
    retry: false,
    staleTime: 0,
    refetchOnWindowFocus: true,
    // An App removed for the organization has to leave a tab that stays open.
    refetchInterval:
      active.backend.kind === "cloud" ? CLOUD_REFETCH_INTERVAL_MS : false,
    meta: {
      backendId: active.backend.id,
      // An old backend returning 404 is an expected capability result. The
      // management page renders it and the global runtime stays quiet.
      disableToast: true,
    },
  });
}
