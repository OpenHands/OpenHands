import { useQuery } from "@tanstack/react-query";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { CANVAS_EXTENSIONS_QUERY_KEYS } from "#/hooks/query/query-keys";

interface UseCanvasExtensionsOptions {
  enabled?: boolean;
}

export function useCanvasExtensions(options: UseCanvasExtensionsOptions = {}) {
  const active = useActiveBackend();
  const connectionRevision = active.backend.connectionRevision ?? 0;
  const backendSupported =
    !isNoBackend(active.backend) && active.backend.kind === "local";

  return useQuery({
    queryKey: CANVAS_EXTENSIONS_QUERY_KEYS.installed(
      active.backend.id,
      active.orgId,
      connectionRevision,
    ),
    queryFn: () => CanvasExtensionsService.listInstalled(),
    enabled: (options.enabled ?? true) && backendSupported,
    retry: false,
    staleTime: 0,
    refetchOnWindowFocus: true,
    meta: {
      backendId: active.backend.id,
      // An old backend returning 404 is an expected capability result. The
      // management page renders it and the global runtime stays quiet.
      disableToast: true,
    },
  });
}
